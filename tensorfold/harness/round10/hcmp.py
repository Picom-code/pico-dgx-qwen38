#!/usr/bin/env python3
# hcmp.py A.jsonl B.jsonl : paired per-prompt speed ratio of two held runs (qk.py held output), mean with a 95 % interval.
import sys, json, math
def load(p):
    d = {}
    for l in open(p):
        r = json.loads(l)
        if "err" in r: continue
        t = r.get("tps") or r.get("tok_s") or (r.get("tokens", 0) / r["decode_s"] if r.get("decode_s") else None)
        if t and r.get("tokens", r.get("completion_tokens", 99)) > 16: d[r.get("id", len(d))] = t
    return d
a, b = load(sys.argv[1]), load(sys.argv[2]); k = [i for i in a if i in b]
if not k: print("hcmp: no common rows; keys", list(json.loads(open(sys.argv[1]).readline()).keys())); sys.exit(0)
r = [b[i] / a[i] - 1 for i in k]; m = sum(r) / len(r); sd = math.sqrt(sum((x - m) ** 2 for x in r) / max(1, len(r) - 1))
print("hcmp: n %d per-prompt ratio %+.2f %% +- %.2f (95 %%)" % (len(k), 100 * m, 100 * 1.96 * sd / math.sqrt(len(k))))
