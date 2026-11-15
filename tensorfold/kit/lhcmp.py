#!/usr/bin/env python3
# lhcmp.py ENGINE_A.log ENGINE_B.log : compare the "q38 lhash" lines (TF_FLASHNEXT_Q38_LHASH=1) of two runs of the same prompts:
# for every prompt (pid) the hash of its scores after the prompt pass ("first") and at release ("end", with the tokens fed).
import sys, re, collections
R = re.compile(r"q38 lhash (first|end) pid=(\w+) plen=(\d+) fed=(\d+) h=(\w+)")
def load(p):
    d = {"first": collections.defaultdict(set), "end": collections.defaultdict(set)}
    for l in open(p, errors="replace"):
        m = R.search(l)
        if m: d[m.group(1)][m.group(2)].add((m.group(4), m.group(5)))
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
for k in ("first", "end"):
    ks = [p for p in a[k] if p in b[k]]; same = sum(a[k][p] == b[k][p] for p in ks)
    print(f"lhash {k}: prompts in both {len(ks)}  identical {same}  differ {len(ks) - same}  (only A {len(a[k]) - len(ks)}, only B {len(b[k]) - len(ks)})")
