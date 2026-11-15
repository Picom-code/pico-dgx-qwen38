#!/usr/bin/env python3
# hpk.py ENGINE.log [ROUNDS] : per-kernel host launch cost from "hostprofk name n ns slow" lines (TF_HOSTPROF + TF_HOSTPROF_K=1).
import sys, re, collections
d = collections.defaultdict(lambda: [0, 0, 0]); tot = [0, 0.0, 0.0, 0, 0.0]
for l in open(sys.argv[1], errors="replace"):
    if l.startswith("hostprofk "):
        p = l.split(); a = d[p[1]]; a[0] += int(p[2]); a[1] += int(p[3]); a[2] += int(p[4])
    elif l.startswith("hostprof launches"):
        p = l.split(); tot[0] += int(p[2]); tot[1] += float(p[4]); tot[2] += float(p[6]); tot[3] += int(p[8]); tot[4] += float(p[10])
N = sum(a[0] for a in d.values()); T = sum(a[1] for a in d.values())
print(f"launches {tot[0]} in-driver {tot[1]:.0f} ms ({1000*tot[1]/max(1,tot[0]):.2f} us each) wall {tot[2]:.0f} ms; syncs {tot[3]} waiting {tot[4]:.0f} ms ({100*tot[4]/max(1,tot[2]):.1f} % of wall), launches per sync {tot[0]/max(1,tot[3]):.1f}")
def short(n):
    m = re.search(r"qmmf_kernelILi(\d)ELi(\d+)ELi64ELi1ELi4ELi(\d+)ELb(\d)ELb(\d)ELb(\d)", n)
    if m: return "qmmf mode%s bm%s st%s f32=%s cl=%s fuse=%s" % m.groups()
    return n[:60]
print("| kernel | launches | share | us/launch | >20us | ms total |")
for k, a in sorted(d.items(), key=lambda x: -x[1][1])[:int(sys.argv[2]) if len(sys.argv) > 2 else 28]:
    print(f"| {short(k)} | {a[0]} | {100*a[0]/N:.1f} % | {a[1]/a[0]/1000:.2f} | {a[2]} ({100*a[2]/a[0]:.2f} %) | {a[1]/1e6:.0f} |")
