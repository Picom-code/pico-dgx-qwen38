#!/usr/bin/env python3
# kprof.py ENGINE.log : per-kernel time per decode step from TF_KPROF tables (serialised launches). Uses the last 60% of tables.
import sys, re, collections
tabs=[]; cur=None
for l in open(sys.argv[1], errors="replace"):
    if l.startswith("kprof-begin"): cur={}
    elif l.startswith("kprof-end"):
        if cur is not None: tabs.append(cur); cur=None
    elif l.startswith("kprof ") and cur is not None:
        p=l.split(); cur[p[1]]=(int(p[2]),int(p[3]),int(p[4]))
skip=int(sys.argv[2]) if len(sys.argv)>2 else len(tabs)*4//10
use=tabs[skip:]
agg=collections.defaultdict(lambda:[0,0,0])
for t in use:
    for k,v in t.items():
        for i in range(3): agg[k][i]+=v[i]
def short(k):
    m=re.match(r"_ZN\d+(\w+?)\d+(\w+_kernel)ILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELb(\d)ELb(\d)ELb(\d)",k)
    if m: g=m.groups(); return f"{g[1]}<{g[2]},{g[3]},{g[4]},{g[5]},{g[6]},{g[7]},{g[8]}{g[9]}{g[10]}>"
    m=re.match(r"_ZN\d+(\w+?)\d+(\w+_kernel)ILi(\d+)ELi(\d+)ELi(\d+)",k)
    if m: g=m.groups(); return f"{g[1]}<{g[2]},{g[3]},{g[4]}>"
    m=re.match(r"_ZN\d+tf_experts\d+(\w+)E",k)
    return m.group(1) if m else k
# steps: route_topk is launched once per layer per pass (40 layers?) -> estimate passes from tf_route_topk / layers
layers=None
steps=agg.get("tf_embed_rows_bf16_copies",[0])[0]
tot=sum(v[1] for v in agg.values())
print(f"tables {len(tabs)} used {len(use)}; embed launches (steps/passes) {steps}; total serialised ms {tot/1e6:.1f}")
N=agg["tf_route_topk"][0]
print(f"route_topk launches {N}; per embed launch {N/max(steps,1):.2f}")
print(f"{'kernel':52s} {'calls/pass':>10s} {'ms/pass':>8s} {'%':>6s} {'us/call':>8s} {'blocks/call':>11s}")
for k,v in sorted(agg.items(), key=lambda kv:-kv[1][1]):
    print(f"{short(k):52s} {v[0]/max(steps,1):10.2f} {v[1]/1e6/max(steps,1):8.3f} {100*v[1]/tot:6.2f} {v[1]/1e3/v[0]:8.1f} {v[2]/v[0]:11.1f}")
print(f"TOTAL ms per pass {tot/1e6/max(steps,1):.2f}")
