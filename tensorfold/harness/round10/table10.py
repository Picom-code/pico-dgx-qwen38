#!/usr/bin/env python3
"""table10.py RUNS_DIR [P=F10] [COLS=off,on,off2,fly] : the round-10 final one-user table from RUNS_DIR/${P}_<col>/suite.log (+ sdrun.log, sq*.jsonl) and the
window script's log RUNS_DIR/wF10.log. Every number is read from the logs; nothing is typed in. Last column: on against the mean of off and off2."""
import re, sys, os, json, statistics
R = sys.argv[1]; P = sys.argv[2] if len(sys.argv) > 2 else "F10"; COLS = (sys.argv[3] if len(sys.argv) > 3 else "off,on,off2,fly").split(",")
NAME = {"off": "levers off (first)", "on": "recipe default (ours)", "off2": "levers off (last)", "fly": "custom engine R v0.3.13, EXL3 re-quantised"}
def rd(p):
    try: return open(p, errors="replace").read()
    except OSError: return ""
S = {k: rd(f"{R}/{P}_{k}/suite.log") + rd(f"{R}/{P}_{k}/sdrun.log") for k in COLS}; WL = rd(f"{R}/wF10.log")
def g(k, pat, n=1, d="-"):
    m = re.search(pat, S[k], re.M); return m.group(n) if m else d
def f(x):
    try: return float(x)
    except (TypeError, ValueError): return None
def ch(vals):
    a = [f(vals.get(k)) for k in ("off", "off2")]; a = [x for x in a if x]; b = f(vals.get("on"))
    return f"{(b / (sum(a) / len(a)) - 1) * 100:+.1f} %" if a and b else ""
rows = []
def add(name, fn, change=True):
    v = {k: fn(k) for k in COLS}; rows.append([name] + [v[k] for k in COLS] + [ch(v) if change else ""])
def row(name, pat, n=1, change=True): add(name, lambda k: g(k, pat, n), change)
row("first token, short prompt (ms)", r"ttft short prompt: median (\d+)")
row("3-prompt chat, 9 runs (median)", r"^chat 1 user: median ([\d.]+)")
row("3-prompt code, 9 runs (median)", r"^code 1 user: median ([\d.]+)")
row("sparkDash-method prose T0", r"sd-method prose T0 think-off: median ([\d.]+)")
row("sparkDash-method code (task 0)", r"sd-method code T0 think-off \(task 0\): median ([\d.]+)")
row("sparkDash-method code, 8 tasks", r"sd-method code8 T0 think-off \(tasks 0-7\): median ([\d.]+)")
row("sparkDash-method structured", r"sd-method structured T0 think-off: median ([\d.]+)")
def sdreal(k, kind):
    v = [float(x) for x in re.findall(rf"sdrun: c1 {kind} run \d+: ([\d.]+)", S[k])]; return f"{statistics.median(v):.1f}" if v else "-"
for kind in ("prose", "code", "structured"): add(f"sparkDash 1.8.8 itself, {kind} (median)", lambda k, kind=kind: sdreal(k, kind))
row("ALL 320 held prompts, all", r"^held ALL: ([\d.]+)")
for c in ("chat", "code", "math"): row(f"  held 320: {c}", rf"^held {c}: ([\d.]+)")
def setline(k, name):
    ks = [c for c in COLS]; i = ks.index(k); blk = WL.split(f"{P}_{k} done")[0]
    for j in ks:
        if j != k and f"{P}_{j} done" in blk: blk = blk.split(f"{P}_{j} done")[-1]
    want = {"edit": "24", "agent": "16"}[name]
    for n, v in re.findall(r"heldpar 1 in flight: (\d+) prompts, \d+ tokens in [\d.]+ s = ([\d.]+) tok/s", blk):
        if n == want: return v
    return "-"
for name, lab in (("edit", "edit set (24 file rewrites), wall tok/s"), ("agent", "agent set (16 turns), wall tok/s")): add(lab, lambda k, name=name: setline(k, name))
def sq(k, name):
    try:
        rs = [json.loads(l) for l in open(f"{R}/{P}_{k}/{name}.jsonl")]; rs = [r for r in rs if "err" not in r and r.get("toks", 0) > 16]
        T = sum(r["toks"] - 1 for r in rs); s = sum((r["toks"] - 1) / r["tps"] for r in rs); return f"{T / s:.2f}"
    except (OSError, ZeroDivisionError): return "-"
for name, lab in (("sq07off", "seeded sampled T0.7, 24 held chat prompts"), ("sq07on", "seeded sampled T0.7 with thinking, 24 prompts"), ("sqsd", "seeded sparkDash-method prose T0.7 thinking, 6 rounds")): add(lab, lambda k, name=name: sq(k, name))
add("prefill 7.4K / 29K / 117K prompt (tok/s)", lambda k: " / ".join(re.findall(r"^prefill target \d+: .* (\d+) tok/s", S[k], re.M)) or "-", False)
for lab, pat in (("needle 182K-token prompt: first token s", r"^needle 200000: prompt_tokens \d+  recall (\d/3)  ttft ([\d.]+)"), ("227,621-token prompt: first token s", r"^ctx \d+: prompt_tokens \d+  recall (\d/3)  ttft ([\d.]+)")):
    row(lab, pat, 2); row("  recall", pat, 1, False)
row("image", r"^img: (\w+ ttft [\d.]+ s)", 1, False)
def qe(k):
    q = re.findall(r"^\s+(\w+)\s+(\d+)/(\d+)\s*$", S[k], re.M); return f"{sum(int(a) for _, a, _ in q)}/{sum(int(b) for _, _, b in q)}" if q else "-"
add("qeval (4 in flight)", qe, False)
add("4 users distinct, default profile (tok/s)", lambda k: (re.findall(rf"distinct 4 users: ([\d.]+) tok/s", WL.split(f"{P}_{k} done")[0]) or ["-"])[-1] if k != "fly" else "-", False)
row("min MemAvailable MiB during the suite", r"min MemAvailable during suite: (\d+) MiB", 1, False)
row("max GPU MiB during the suite", r"max GPU MiB (\d+)", 1, False)
H = ["bench"] + [NAME.get(k, k) for k in COLS] + ["ours vs mean off"]; n = len(H); W = [max(len(str(r[i])) for r in rows + [H]) for i in range(n)]
line = "+" + "+".join("-" * (w + 2) for w in W) + "+"
print(line); print("| " + " | ".join(str(H[i]).ljust(W[i]) for i in range(n)) + " |"); print(line)
for r in rows: print("| " + " | ".join(str(r[i]).ljust(W[i]) for i in range(n)) + " |")
print(line)
