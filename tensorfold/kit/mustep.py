#!/usr/bin/env python3
# mustep.py ENGINE.log : shared-round budget under N users, from TensorFold's "done" lines of the distinct/copies runs (512-token greedy requests).
# Requests are grouped into runs of consecutive lines with similar per-request speed; N is inferred from the group size.
import re, sys
R = re.compile(r"done \S+ prompt=(\d+) cached=(\d+) thinking=(\w+) effort=\S+ tokens=(\d+) sha=(\w+) finish=(\w+) rounds=(\d+) accepted=(\d+)/(\d+) ttft=([\d.]+) decode=([\d.]+) elapsed=([\d.]+) queued=([\d.]+) tok/s=([\d.]+)")
rows = []
for l in open(sys.argv[1], errors="replace"):
    m = R.search(l)
    if m:
        p, c, th, tok, sha, fin, rd, a, b, ttft, dec, el, qd, tps = m.groups()
        rows.append(dict(tok=int(tok), rounds=int(rd), acc=int(a), prop=int(b), dec=float(dec), tps=float(tps), el=float(el)))
# groups: consecutive requests that finished within the same wave (elapsed within 25 % and tokens >= 400)
groups = []; cur = []
for r in rows:
    if r["tok"] < 400: 
        if cur: groups.append(cur); cur = []
        continue
    if cur and abs(r["el"] - cur[0]["el"]) / cur[0]["el"] > 0.35: groups.append(cur); cur = []
    cur.append(r)
if cur: groups.append(cur)
print("users  tok/s total  ms/round  rows/round(all users)  drafts verified/user/round  accepted/user/round  tok/user/round  ms per row")
for g in groups:
    n = len(g)
    if n not in (1, 4, 5, 8, 16): continue
    T = sum(r["tok"] for r in g); wall = max(r["dec"] for r in g); rd = sum(r["rounds"] for r in g) / n
    P = sum(r["prop"] for r in g) / sum(r["rounds"] for r in g); A = sum(r["acc"] for r in g) / sum(r["rounds"] for r in g)
    msr = 1000 * (sum(r["dec"] for r in g) / n) / rd
    print(f"{n:5d}  {T / wall:10.1f}  {msr:8.1f}  {n * (1 + P):8.1f}               {P:6.2f}                      {A:6.2f}              {T / n / rd:6.2f}        {msr / (n * (1 + P)):6.2f}")
