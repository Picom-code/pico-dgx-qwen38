#!/usr/bin/env python3
# tfsteps.py ENGINE.log [HELD.jsonl] : step budget from TensorFold's per-request "done" lines (rounds, accepted=a/b, decode seconds).
# Only one-stream requests are used (no overlap in time is assumed for the c1 parts: requests with tokens >= 64, greedy).
import re, sys, json, statistics as st
R = re.compile(r"done \S+ prompt=(\d+) cached=(\d+) thinking=(\w+) effort=\S+ tokens=(\d+) sha=(\w+) finish=(\w+) rounds=(\d+) accepted=(\d+)/(\d+) ttft=([\d.]+) decode=([\d.]+) elapsed=([\d.]+) queued=([\d.]+) tok/s=([\d.]+)")
rows = []
for l in open(sys.argv[1], errors="replace"):
    m = R.search(l)
    if m:
        p, c, th, tok, sha, fin, rd, a, b, ttft, dec, el, qd, tps = m.groups()
        rows.append(dict(prompt=int(p), tok=int(tok), rounds=int(rd), acc=int(a), prop=int(b), dec=float(dec), tps=float(tps), sha=sha, think=th))
print("requests", len(rows))
def rep(name, rs):
    rs = [r for r in rs if r["tok"] >= 64 and r["rounds"] > 0]
    if not rs: return
    T = sum(r["tok"] for r in rs); RD = sum(r["rounds"] for r in rs); A = sum(r["acc"] for r in rs); P = sum(r["prop"] for r in rs); D = sum(r["dec"] for r in rs)
    print(f"{name}: n {len(rs)} tok/s {T/D:.1f}  tok/round {T/RD:.3f}  drafts proposed/round {P/RD:.3f}  accepted/round {A/RD:.3f}  accept rate {A/max(P,1):.3f}  ms/round {1000*D/RD:.2f}  rows/round {1+P/RD:.2f}")
    return rs
nd = [r for r in rows if r["prop"] == 0 and r["tok"] >= 64]
dr = [r for r in rows if r["prop"] > 0 and r["tok"] >= 64]
rep("no-draft requests", nd); rs = rep("drafted requests", dr)
# least squares ms/round = a + b * verified drafts per round, over drafted requests (weights: rounds)
if rs:
    xs = [r["prop"] / r["rounds"] for r in rs]; ys = [1000 * r["dec"] / r["rounds"] for r in rs]; w = [r["rounds"] for r in rs]
    W = sum(w); mx = sum(a * b for a, b in zip(xs, w)) / W; my = sum(a * b for a, b in zip(ys, w)) / W
    b = sum(wi * (x - mx) * (y - my) for x, y, wi in zip(xs, ys, w)) / sum(wi * (x - mx) ** 2 for x, wi in zip(xs, w)); a = my - b * mx
    print(f"fit over drafted requests: ms/round = {a:.2f} + {b:.2f} x drafts verified per round  (x range {min(xs):.2f}..{max(xs):.2f})")
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 7)):
        g = [r for r in rs if lo <= r["prop"] / r["rounds"] < hi]
        if g: rep(f"  drafts/round in [{lo},{hi})", g)
