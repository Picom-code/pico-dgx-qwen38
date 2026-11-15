#!/usr/bin/env python3
# gatesim.py ENGINE.log(always-8 step log) : simulate draft-gate policies on logged rounds (iid-round approximation).
# cost model (measured fit, 923b): ms/round = A + B*drafts verified. tokens/round = min(kept, d) + 1.
import sys, math, collections
A=float(sys.argv[2]) if len(sys.argv)>2 else 44.5; B=float(sys.argv[3]) if len(sys.argv)>3 else 4.25
rounds=[]; chain=None
for l in open(sys.argv[1], errors="replace"):
    if l.startswith("QCHAIN"):
        p=l.split(); chain=[float(x) for x in p[2:]]
    elif l.startswith("QSTEP"):
        p=l.split()
        if p[1]=="head" and chain is not None and len(chain)==int(p[2])-1:
            rounds.append((chain,int(p[3])))
        chain=None
n=len(rounds); print("rounds",n,"mean kept",sum(k for _,k in rounds)/n,"mean chain",sum(len(c) for c,_ in rounds)/n)
def run(policy):
    T=C=D=0
    for c,k in rounds:
        d=policy(c,k); T+=min(k,d)+1; C+=A+B*d; D+=d
    return 1000*T/C, T/n, D/n
def prod_gate(th,mx):
    def f(c,k):
        pr=1.0; d=0
        for x in c[:mx]:
            pr*=x
            if pr<th: break
            d+=1
        return d
    return f
def prod_gate_incl(th,mx):  # verifies the draft that crosses the threshold too? (engine semantics unknown) 
    def f(c,k):
        pr=1.0; d=0
        for x in c[:mx]:
            pr*=x; d+=1
            if pr<th: break
        return d
    return f
base=None
print("policy                         tok/s   tok/round drafts/round")
for th in (0.05,0.12,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9):
    for mx in (6,8):
        r=run(prod_gate(th,mx)); print(f"product>={th:<4} max{mx}            {r[0]:6.2f}  {r[1]:.3f}  {r[2]:.3f}")
r=run(lambda c,k:min(k,len(c))); print(f"oracle d=kept                  {r[0]:6.2f}  {r[1]:.3f}  {r[2]:.3f}")
r=run(lambda c,k:min(k+1,len(c))); print(f"oracle d=kept+1                {r[0]:6.2f}  {r[1]:.3f}  {r[2]:.3f}")
r=run(lambda c,k:0); print(f"no drafts                      {r[0]:6.2f}  {r[1]:.3f}  {r[2]:.3f}")
r=run(lambda c,k:len(c)); print(f"all 8                          {r[0]:6.2f}  {r[1]:.3f}  {r[2]:.3f}")
# calibration: empirical P(prefix i accepted) by product bucket
bk=collections.defaultdict(lambda:[0,0])
for c,k in rounds:
    pr=1.0
    for i,x in enumerate(c):
        pr*=x; b=min(int(pr*10),9); bk[b][0]+=1; bk[b][1]+= (k>=i+1)
print("calibration: product bucket -> empirical P(prefix accepted), n")
for b in sorted(bk): print(f"  [{b/10:.1f},{(b+1)/10:.1f}) {bk[b][1]/bk[b][0]:.3f} {bk[b][0]}")
# learned gate bound: logistic regression on features of the prefix (log prod, log last p, min p, depth), 2-fold; policy: stop when calibrated P < t
import random
feats=[]
for ri,(c,k) in enumerate(rounds):
    pr=1.0; mn=1.0
    for i,x in enumerate(c):
        pr*=x; mn=min(mn,x)
        feats.append((ri,i,[1.0,math.log(max(pr,1e-6)),math.log(max(x,1e-6)),mn,i/8.0,pr], 1.0 if k>=i+1 else 0.0))
def fit(rows,it=300,lr=0.5):
    w=[0.0]*6
    for _ in range(it):
        g=[0.0]*6
        for _,_,f,y in rows:
            z=sum(a*b for a,b in zip(w,f)); p=1/(1+math.exp(-max(-30,min(30,z))))
            for j in range(6): g[j]+=(p-y)*f[j]
        for j in range(6): w[j]-=lr*g[j]/len(rows)
    return w
random.seed(1)
fold={ri:random.random()<0.5 for ri in range(n)}
wa=fit([r for r in feats if fold[r[0]]]); wb=fit([r for r in feats if not fold[r[0]]])
pred={}
for ri,i,f,y in feats:
    w=wb if fold[ri] else wa
    pred[(ri,i)]=1/(1+math.exp(-sum(a*b for a,b in zip(w,f))))
for t in (0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6):
    T=C=D=0
    for ri,(c,k) in enumerate(rounds):
        d=0
        for i in range(len(c)):
            if pred[(ri,i)]<t: break
            d+=1
        T+=min(k,d)+1; C+=A+B*d; D+=d
    print(f"logistic gate P>={t:<4}           {1000*T/C:6.2f}  {T/n:.3f}  {D/n:.3f}")
