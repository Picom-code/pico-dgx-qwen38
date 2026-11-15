#!/usr/bin/env python3
# syncsites.py ENGINE.log NM.txt : stream-sync call sites ("hostprofs addr n ns", TF_HOSTPROF_K=1) as function names, via the
# "hostprofbase" line (runtime address of cuda.launch.launch) and `nm -C` of the binary.
import sys, bisect, collections
syms = []
for l in open(sys.argv[2], errors="replace"):
    p = l.split(None, 2)
    if len(p) == 3 and p[1] in "tTwW":
        try: syms.append((int(p[0], 16), p[2].strip()))
        except ValueError: pass
syms.sort(); addrs = [a for a, _ in syms]
base = None; d = collections.defaultdict(lambda: [0, 0])
for l in open(sys.argv[1], errors="replace"):
    if l.startswith("hostprofbase "): base = int(l.split()[1], 16)
    elif l.startswith("hostprofs "):
        p = l.split(); a = d[int(p[1], 16)]; a[0] += int(p[2]); a[1] += int(p[3])
cand = [a for a, n in syms if n.endswith("launch.launch") or n == "cuda.launch.launch" or n.endswith(".launch.launch")]
print("launch symbol candidates", [(hex(a)) for a in cand][:4], "runtime", hex(base or 0))
slide = (base - cand[0]) if cand and base else 0
agg = collections.defaultdict(lambda: [0, 0])
for ra, (n, ns) in d.items():
    i = bisect.bisect_right(addrs, ra - slide) - 1
    name = syms[i][1] if i >= 0 else hex(ra)
    a = agg[name]; a[0] += n; a[1] += ns
tot = sum(a[0] for a in agg.values())
print("| sync call site | syncs | share | mean wait us | total ms |")
for name, (n, ns) in sorted(agg.items(), key=lambda x: -x[1][0])[:14]:
    print(f"| {name[:90]} | {n} | {100*n/tot:.1f} % | {ns/n/1000:.1f} | {ns/1e6:.0f} |")
