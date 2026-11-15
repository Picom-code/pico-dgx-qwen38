#!/usr/bin/env python3
"""table9.py RUNS_DIR [P=F9] : the round-9 final one-user table (levers off vs recipe default) from ${P}_off and ${P}_on (suite.log + the
window script's log lines saved in RUNS_DIR/wF9.log). Prints an ASCII table. Every number is read from the logs; nothing is typed in."""
import re, sys, os, json, statistics
R = sys.argv[1]; P = sys.argv[2] if len(sys.argv) > 2 else "F9"
def rd(p):
    try: return open(p, errors="replace").read()
    except OSError: return ""
S = {k: rd(f"{R}/{P}_{k}/suite.log") for k in ("off", "on")}
def g(k, pat, n=1, d="-"):
    m = re.search(pat, S[k], re.M); return m.group(n) if m else d
def f(x):
    try: return float(x)
    except (TypeError, ValueError): return None
def ch(a, b, inv=False):
    a, b = f(a), f(b)
    if a is None or b is None or a == 0: return ""
    return f"{(b / a - 1) * 100:+.1f} %"
rows = []
def row(name, pat, n=1): a, b = g("off", pat, n), g("on", pat, n); rows.append((name, a, b, ch(a, b)))
row("first token, short prompt (ms)", r"ttft short prompt: median (\d+)")
row("3-prompt chat, 9 runs (median)", r"^chat 1 user: median ([\d.]+)")
row("3-prompt code, 9 runs (median)", r"^code 1 user: median ([\d.]+)")
row("sparkDash-method prose T0", r"sd-method prose T0 think-off: median ([\d.]+)")
row("sparkDash-method prose T0.7 thinking (unseeded)", r"sd-method prose T0.7 think-on: median ([\d.]+)")
row("sparkDash-method code (task 0)", r"sd-method code T0 think-off \(task 0\): median ([\d.]+)")
row("sparkDash-method code, 8 tasks", r"sd-method code8 T0 think-off \(tasks 0-7\): median ([\d.]+)")
row("sparkDash-method structured", r"sd-method structured T0 think-off: median ([\d.]+)")
def sdreal(k, kind):
    v = [float(x) for x in re.findall(rf"sdrun: c1 {kind} run \d+: ([\d.]+)", S[k] + rd(f"{R}/{P}_{k}/sdrun.log"))]
    return f"{statistics.median(v):.1f}" if v else "-"
for kind in ("prose", "code", "structured"):
    a, b = sdreal("off", kind), sdreal("on", kind); rows.append((f"sparkDash 1.8.8 itself, {kind} (median)", a, b, ch(a, b)))
def cat(k):
    t = S[k] + rd(f"{R}/{P}_{k}/sdrun.log"); return " / ".join(m for m in (re.search(rf"catalog {c}: \d+ prompts, mean ([\d.]+)", t) and re.search(rf"catalog {c}: \d+ prompts, mean ([\d.]+)", t).group(1) or "-" for c in ("prose", "structural", "code")))
rows.append(("sparkDash 1.8.8 catalogs prose / structural / code", cat("off"), cat("on"), ""))
row("ALL 320 held prompts, all", r"^held ALL: ([\d.]+)")
for c in ("chat", "code", "math"): row(f"  held 320: {c}", rf"^held {c}: ([\d.]+)")
def setline(k, name):
    t = rd(f"{R}/wF9.log"); blk = t.split(f"{P}_{k} done")[0].split(f"{P}_{'off' if k == 'on' else 'zzz'} done")[-1]
    m = re.findall(r"heldpar 1 in flight: (\d+) prompts, \d+ tokens in [\d.]+ s = ([\d.]+) tok/s", blk)
    want = {"edit": "24", "agent": "16"}[name]
    for n, v in m:
        if n == want: return v
    return "-"
for name, lab in (("edit", "edit set (24 file rewrites), wall tok/s"), ("agent", "agent set (16 turns), wall tok/s")):
    a, b = setline("off", name), setline("on", name); rows.append((lab, a, b, ch(a, b)))
def sq(k, name):
    try:
        rs = [json.loads(l) for l in open(f"{R}/{P}_{k}/{name}.jsonl")]; rs = [r for r in rs if "err" not in r and r.get("toks", 0) > 16]
        T = sum(r["toks"] - 1 for r in rs); s = sum((r["toks"] - 1) / r["tps"] for r in rs); return f"{T / s:.2f}"
    except (OSError, ZeroDivisionError): return "-"
for name, lab in (("sq07off", "seeded sampled T0.7, 24 held chat prompts"), ("sq07on", "seeded sampled T0.7 with thinking, 24 prompts"), ("sqsd", "seeded sparkDash-method prose T0.7 thinking, 6 rounds")):
    a, b = sq("off", name), sq("on", name); rows.append((lab, a, b, ch(a, b)))
def pre(k): return " / ".join(re.findall(r"^prefill target \d+: .* (\d+) tok/s", S[k], re.M)) or "-"
rows.append(("prefill 7.4K / 29K / 117K prompt (tok/s)", pre("off"), pre("on"), ""))
for lab, pat in (("needle 182K-token prompt: recall, first token s", r"^needle 200000: prompt_tokens \d+  recall (\d/3)  ttft ([\d.]+)"), ("227,621-token prompt: recall, first token s", r"^ctx \d+: prompt_tokens \d+  recall (\d/3)  ttft ([\d.]+)")):
    a = g("off", pat, 1) + ", " + g("off", pat, 2); b = g("on", pat, 1) + ", " + g("on", pat, 2); rows.append((lab, a, b, ch(g("off", pat, 2), g("on", pat, 2))))
rows.append(("image", g("off", r"^img: (\w+ ttft [\d.]+ s)"), g("on", r"^img: (\w+ ttft [\d.]+ s)"), ""))
def qe(k):
    q = re.findall(r"^\s+(\w+)\s+(\d+)/(\d+)\s*$", S[k], re.M); return f"{sum(int(a) for _, a, _ in q)}/{sum(int(b) for _, _, b in q)}" if q else "-"
rows.append(("qeval (4 in flight)", qe("off"), qe("on"), ""))
rows.append(("idle MemAvailable GiB / GPU MiB", g("off", r"idle after load: MemAvailable (\d+) GiB, GPU MiB (\d*)") + " / " + g("off", r"idle after load: MemAvailable (\d+) GiB, GPU MiB (\d*)", 2),
             g("on", r"idle after load: MemAvailable (\d+) GiB, GPU MiB (\d*)") + " / " + g("on", r"idle after load: MemAvailable (\d+) GiB, GPU MiB (\d*)", 2), ""))
H = ("bench", "levers off", "recipe default (ours)", "change"); W = [max(len(str(r[i])) for r in rows + [H]) for i in range(4)]
line = "+" + "+".join("-" * (w + 2) for w in W) + "+"
print(line); print("| " + " | ".join(str(H[i]).ljust(W[i]) for i in range(4)) + " |"); print(line)
for r in rows: print("| " + " | ".join(str(r[i]).ljust(W[i]) for i in range(4)) + " |")
print(line)
