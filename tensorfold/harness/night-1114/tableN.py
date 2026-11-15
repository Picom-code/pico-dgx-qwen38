#!/usr/bin/env python3
"""tableN.py RUNS_DIR [P=N] [LOG1=wN1.log] : the night's final tables from the suite's result files. Every number is read from the files;
nothing is typed in. Writes RUNS_DIR/vals.json (every value by name), prints ASCII tables.

  RUNS_DIR/${P}_{off,on,off2}/suite.log, sdrun.log, sq*.jsonl, pred*.jsonl, loop*.jsonl, users.jsonl, at10.json ... and the chain logs."""
import re, sys, os, json, statistics, glob
R = sys.argv[1]; P = sys.argv[2] if len(sys.argv) > 2 else "N"; LOG1 = sys.argv[3] if len(sys.argv) > 3 else "wN1.log"
COLS = ["off", "on", "off2"]
NAME = {"off": "levers off (first)", "on": "recipe default", "off2": "levers off (last)"}
def rd(p):
    try: return open(p, errors="replace").read()
    except OSError: return ""
S = {k: rd(f"{R}/{P}_{k}/suite.log") + rd(f"{R}/{P}_{k}/sdrun.log") for k in COLS}
WL = "".join(rd(p) for p in sorted(glob.glob(f"{R}/wN*.log"))) or rd(f"{R}/{LOG1}")
if not WL: WL = rd(f"{R}/{LOG1}")
V = {}  # name -> {col: value}
def fl(x):
    try: return float(x)
    except (TypeError, ValueError): return None
def blk(k):
    """the chain log's part that belongs to step ${P}_k (between the previous 'done' line and its own)"""
    m = re.search(rf"^{P}_{k} done", WL, re.M)
    if not m: return ""
    head = WL[:m.start()]; prev = [x.end() for x in re.finditer(r"^\w+ done [\d:]+$", head, re.M)]
    return head[prev[-1]:] if prev else head
rows = []
def add(key, name, fn, change=True, lower_better=False):
    v = {k: fn(k) for k in COLS}; V[key] = v
    a = [fl(v[k]) for k in ("off", "off2")]; a = [x for x in a if x]; b = fl(v["on"]); c = ""
    if change and a and b: c = f"{(b / (sum(a) / len(a)) - 1) * 100:+.1f} %"
    rows.append([name] + [v[k] if v[k] is not None else "-" for k in COLS] + [c])
def g(k, pat, n=1):
    m = re.search(pat, S[k], re.M); return m.group(n) if m else None
def row(key, name, pat, n=1, change=True): add(key, name, lambda k: g(k, pat, n), change)
row("ttft_ms", "first token, short prompt (ms)", r"ttft short prompt: median (\d+)")
row("chat3", "3-prompt chat, 9 runs (median)", r"^chat 1 user: median ([\d.]+)")
row("code3", "3-prompt code, 9 runs (median)", r"^code 1 user: median ([\d.]+)")
row("held", "ALL 320 held prompts", r"^held ALL: ([\d.]+)")
row("held_n", "  prompts answered with > 16 tokens", r"^held ALL: [\d.]+ tok/s  n (\d+)", 1, False)
for c in ("chat", "code", "math"): row(f"held_{c}", f"  held: {c}", rf"^held {c}: ([\d.]+)")
row("sdm_prose", "sparkDash-method prose T0", r"sd-method prose T0 think-off: median ([\d.]+)")
row("sdm_prose_t07", "sparkDash-method prose T0.7 thinking", r"sd-method prose T0.7 think-on: median ([\d.]+)")
row("sdm_code", "sparkDash-method code (task 0)", r"sd-method code T0 think-off \(task 0\): median ([\d.]+)")
row("sdm_code8", "sparkDash-method code, 8 tasks", r"sd-method code8 T0 think-off \(tasks 0-7\): median ([\d.]+)")
row("sdm_struct", "sparkDash-method structured", r"sd-method structured T0 think-off: median ([\d.]+)")
def sdreal(k, kind):
    v = [float(x) for x in re.findall(rf"sdrun: c1 {kind} run \d+: ([\d.]+)", rd(f"{R}/{P}_{k}/sdrun.log"))]; return f"{statistics.median(v):.1f}" if v else None
for kind in ("prose", "code", "structured"): add(f"sd_{kind}", f"sparkDash 1.8.8 itself, c1 {kind} (median of 5)", lambda k, kind=kind: sdreal(k, kind))
for kind in ("prose", "structural", "code"): add(f"sdcat_{kind}", f"sparkDash 1.8.8 catalog {kind} (mean)", lambda k, kind=kind: (re.findall(rf"sdrun: catalog {kind}: \d+ prompts, mean ([\d.]+)", rd(f"{R}/{P}_{k}/sdrun.log")) or [None])[-1])
def pre(k, tgt):
    m = re.search(rf"^prefill target {tgt}: prompt_tokens (\d+)\s+ttft ([\d.]+) s\s+(\d+) tok/s", S[k], re.M); return m.group(3) if m else None
for tgt, lab in ((8000, "7.4K"), (32000, "29K"), (128000, "116K")): add(f"prefill_{tgt}", f"prefill, {lab}-token prompt (tok/s)", lambda k, tgt=tgt: pre(k, tgt))
def setline(k, name):
    want = {"edit": "24", "agent": "16"}[name]
    for n, v in re.findall(r"heldpar 1 in flight: (\d+) prompts, \d+ tokens in [\d.]+ s = ([\d.]+) tok/s", blk(k)):
        if n == want: return v
    return None
add("edit", "rewrite-a-file set (24 rewrites), wall tok/s", lambda k: setline(k, "edit"))
add("agent", "agent set (16 turns), wall tok/s", lambda k: setline(k, "agent"))
def ar(k, f, setname):
    m = re.search(rf"/{P}_{k}/{f}\.jsonl\n((?:  .*\n)+)", blk(k))
    if not m: return None
    x = re.search(rf"^  {setname}\s+n\s+(\d+)\s+tokens\s+(\d+)\s+wall tok/s\s+([\d.]+)", m.group(1), re.M); return x.group(3) if x else None
add("pred0_all", "56 rewrite requests, no `prediction` (wall tok/s)", lambda k: ar(k, "pred0", "ALL"))
add("pred1_all", "56 rewrite requests, with `prediction`", lambda k: ar(k, "pred1", "ALL"))
add("pred0_num", "  numbered-lines rewrites (16), no prediction", lambda k: ar(k, "pred0", "numbered"))
add("pred1_num", "  numbered-lines rewrites (16), with prediction", lambda k: ar(k, "pred1", "numbered"))
add("loop0", "agent loop, 46 turns, no prediction (wall tok/s)", lambda k: ar(k, "loop0", "ALL"))
add("loop1", "agent loop, 46 turns, with prediction", lambda k: ar(k, "loop1", "ALL"))
def sq(k, name):
    try:
        rs = [json.loads(l) for l in open(f"{R}/{P}_{k}/{name}.jsonl")]; rs = [r for r in rs if "err" not in r and r.get("toks", 0) > 16]
        T = sum(r["toks"] - 1 for r in rs); s = sum((r["toks"] - 1) / r["tps"] for r in rs); return f"{T / s:.2f}"
    except (OSError, ZeroDivisionError): return None
for name, key, lab in (("sq07off", "samp_chat", "seeded sampled T0.7, 24 held chat prompts"), ("sq07on", "samp_think", "seeded sampled T0.7 with thinking, 24 prompts"), ("sqsd", "samp_sd", "seeded sparkDash-method prose T0.7 thinking, 6")):
    add(key, lab, lambda k, name=name: sq(k, name))
NP = r"^needle 200000: prompt_tokens (\d+)\s+recall (\d/3)\s+ttft ([\d.]+)"; CP = r"^ctx \d+: prompt_tokens (\d+)\s+recall (\d/3)\s+ttft ([\d.]+)"
row("needle_ttft", "182K-token needle: first token (s)", NP, 3); row("needle_recall", "  recall", NP, 2, False)
row("ctx_ttft", "227K-token prompt: first token (s)", CP, 3); row("ctx_tokens", "  prompt tokens", CP, 1, False); row("ctx_recall", "  recall", CP, 2, False)
row("img", "image", r"^img: (\w+)", 1, False)
def qe(k):
    q = re.findall(r"^\s+(\w+)\s+(\d+)/(\d+)\s*$", S[k], re.M); return f"{sum(int(a) for _, a, _ in q)}/{sum(int(b) for _, _, b in q)}" if q else None
add("qeval", "qeval (4 in flight)", qe, False)
add("users3", "3 users distinct, default profile (tok/s)", lambda k: (re.findall(r"distinct 3 users: ([\d.]+) tok/s", blk(k)) or [None])[-1], True)
row("mem_min_mib", "lowest MemAvailable in the suite (MiB)", r"min MemAvailable during suite: (\d+) MiB", 1, False)
row("gpu_max_mib", "highest GPU memory (MiB)", r"max GPU MiB (\d+)", 1, False)
add("window", "run start - end (UTC)", lambda k: (lambda a, b: f"{a[11:16]}-{b[11:16]}" if a and b else None)(g(k, r"suite \S+ mode full start (\S+)"), g(k, r"suite end (\S+)")), False)

def table(H, rows):
    n = len(H); W = [max(len(str(r[i])) for r in rows + [H]) for i in range(n)]
    line = "+" + "+".join("-" * (w + 2) for w in W) + "+"
    out = [line, "| " + " | ".join(str(H[i]).ljust(W[i]) for i in range(n)) + " |", line]
    out += ["| " + " | ".join(str(r[i]).ljust(W[i]) for i in range(n)) + " |" for r in rows] + [line]; return "\n".join(out)
print("ONE USER, spark-923b, final image, bf16 KV, 3 x 262K, one window"); print(table(["bench"] + [NAME[k] for k in COLS] + ["default vs mean off"], rows))

# many users
def users(lab):
    d = {}
    try:
        for l in open(f"{R}/{P}_{lab}/users.jsonl"):
            r = json.loads(l); d.setdefault(r["n"], []).append(r)
    except OSError: pass
    return d
MU = {lab: users(lab) for lab in ("miafp", "mu", "miafp2", "u16s")}; V["users"] = {}
def cell(lab, n):
    rs = MU[lab].get(n)
    if not rs: return "-"
    m = sorted(rs, key=lambda r: r["tps"])[len(rs) // 2]
    V["users"].setdefault(lab, {})[str(n)] = dict(tps=m["tps"], first=m["first"], last=m["last"], runs=[r["tps"] for r in rs], lasts=[r["last"] for r in rs], errors=sum(r["errors"] for r in rs))
    return f"{m['tps']:.1f} ({m['first']:.2f} / {m['last']:.2f} s)" + (f" ERR {sum(r['errors'] for r in rs)}" if sum(r["errors"] for r in rs) else "")
if any(MU.values()):
    print("\nMANY USERS, distinct prompts, 512 tokens, total decode tok/s (first / last user's first token), median of the runs")
    print(table(["users", "MiaAI shipped config, first (16 x 64K, fp8 KV, levers off)", "USERS16 profile (bf16 KV pool)", "MiaAI shipped config, last"],
                [[n, cell("miafp", n), cell("mu", n), cell("miafp2", n)] for n in (1, 4, 8, 12, 16)]))
# agent traces
V["agents"] = {}
for lab in ("agA", "agmia", "agmia5", "agoff"):
    for f in ("at10", "at5"):
        try: rs = json.load(open(f"{R}/{P}_{lab}/{f}.json"))
        except OSError: continue
        ok = [r for r in rs["rows"] if r.get("ptoks")]; fu = [r for r in ok if r["t"] > 0]
        if not fu: continue
        V["agents"][f"{lab}_{f}"] = dict(ok=len(ok), n=len(rs["rows"]), followups=len(fu), resumed=sum(1 for r in fu if (r.get("cached") or 0) > 0.5 * r["ptoks"]),
            fu_med=round(statistics.median(r["ttft"] for r in fu), 2), fu_max=round(max(r["ttft"] for r in fu), 2), wall=round(rs["wall"], 1), ptoks=int(statistics.median(r["ptoks"] for r in ok)))
if V["agents"]:
    print("\nAGENT TRACES (at.py): follow-up turn's first token")
    print(table(["run", "requests ok", "follow-ups resumed from the cache", "follow-up first token median / max (s)", "wall (s)", "prompt tokens (median)"],
                [[k, f"{a['ok']}/{a['n']}", f"{a['resumed']}/{a['followups']}", f"{a['fu_med']} / {a['fu_max']}", a["wall"], a["ptoks"]] for k, a in V["agents"].items()]))
# identity + check lines, straight from the chain logs
print("\nIDENTITY / CHECK LINES (chain logs, as printed on the box)")
keep = []
for l in WL.splitlines():
    if re.search(r"identical \d+/\d+|texts identical|^-- |^### step|DID NOT|NOT UP|SKIPPED|STARTS with|recipe\.sh prepare rc|crash lines: [1-9]|mismatch-like|q38 [a-z0-9 ]*(cert|check|qsa|gdnprod|routeband|scratch|hcdn|snapdelta)|^READY|^FAILED|env knobs|streams at once", l):
        keep.append(l[:330])
print("\n".join(keep))
V["identity_lines"] = [l for l in keep if "identical" in l]
json.dump(V, open(f"{R}/vals.json", "w"), indent=1)
