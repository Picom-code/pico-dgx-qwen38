#!/usr/bin/env python3
"""evalrot.py MODELDIR VOCAB.txt ALL.mix ENGINE.log NFIT NFRESH OLD.rot OUTPREFIX [FRESH.jsonl]  (round 9, builder A; pytorch container, GPU)
The dump holds the draft-head input rows of three request groups in order: NFIT fitting-corpus requests, NFRESH fresh-set requests, then
the held prompts. Rows are assigned to requests as builder B's mkhead2.py does (one row a QTOP line, a request a 'prompt cache' line).
Writes OUTPREFIX_fit.mix (the fitting rows only: the input of mkhead2.py) and prints, for the OLD rotation (fitted on the held prompts)
and a NEW one fitted here on the fitting rows only: how often the two-stage pick equals the full 4-bit draft head's pick, block form
(3 best of each 256-row block re-scored, what the engine runs) and top-64 form, on each group. Drafts only: nothing here changes a reply."""
import sys, json, os, struct, numpy as np, torch
dev = "cuda" if torch.cuda.is_available() else "cpu"; D = 2560; r = 512
mdir, vocab, dump, log, nfit, nfresh, oldrot, outp = sys.argv[1:9]; nfit = int(nfit); nfresh = int(nfresh)
ids = sorted(set(int(x) for x in open(vocab).read().split() if int(x) < 248320)); V = len(ids)
def find_tensor(mdir, names):
    for sub in ("", "tensors"):
        d = os.path.join(mdir, sub)
        if not os.path.isdir(d): continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".safetensors"): continue
            with open(os.path.join(d, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
                for name in names:
                    if name in h: return os.path.join(d, f), 8 + n + h[name]["data_offsets"][0], h[name]
    raise SystemExit("lm_head not found")
path, off, meta = find_tensor(mdir, ["language_model.lm_head.weight", "lm_head.weight"])
mm = np.memmap(path, dtype=np.uint16, mode="r", offset=off, shape=(248320, D))
W = torch.from_numpy((np.asarray(mm[ids]).astype(np.uint32) << 16).view(np.float32).copy()).to(dev)
def bf16bits(t):
    b = t.contiguous().view(torch.int32); return ((b + 0x7fff + ((b >> 16) & 1)) >> 16)
def bf16r(t): return (bf16bits(t) << 16).view(torch.float32)
def affine4(Wm):
    G = Wm.reshape(Wm.shape[0], -1, 32); lo = G.amin(2, keepdim=True); hi = G.amax(2, keepdim=True)
    sc = ((hi - lo) / 15.0).clamp_min(1e-8); q = ((G - lo) / sc).clamp(0, 15)
    fl = q.floor(); fr = q - fl; code = fl + ((fr > 0.5) | ((fr == 0.5) & (fl.long() % 2 == 1))).float()
    return code, sc, lo
def deq(code, sc, lo): return (code * bf16r(sc) + bf16r(lo)).reshape(code.shape[0], -1)
W4 = deq(*affine4(W)); del W
raw = np.fromfile(dump, dtype=np.uint16); raw = raw[: (len(raw) // D) * D].reshape(-1, D)
req = []; q = -1; nq = 0
for l in open(log, errors="replace"):
    if l.startswith("info: prompt cache:"): q += 1
    elif l.startswith("QTOP "): req.append(q); nq += 1
n = min(len(req), raw.shape[0]); req = np.array(req[:n])
print(f"dump rows {raw.shape[0]}, QTOP lines {nq}, requests {q + 1} (expected fit {nfit} + fresh {nfresh} + held)")
g = {"fit": req < nfit, "fresh": (req >= nfit) & (req < nfit + nfresh), "held": req >= nfit + nfresh}
raw[:n][g["fit"]].tofile(outp + "_fit.mix"); print("fitting rows", int(g["fit"].sum()), "->", outp + "_fit.mix")
def rows(mask): return torch.from_numpy((raw[:n][mask].astype(np.uint32) << 16).view(np.float32).copy()).to(dev)
def rotation(x):
    e, Q = torch.linalg.eigh((x.T.double() @ x.double()) / x.shape[0]); return Q[:, torch.argsort(-e)].float()
ro = open(oldrot, "rb").read(); assert ro[:8] == b"Q38ROT01"
Qold = torch.from_numpy(np.frombuffer(ro, dtype="<f4", offset=16).reshape(-1, D)[:r].copy()).to(dev).T.contiguous()
Xfit = rows(g["fit"]); Qnew = rotation(Xfit)[:, :r].contiguous(); del Xfit
NB = (V + 255) // 256
def agree(x, Q, P):
    xr = x @ Q; prev = bf16r(xr @ P.T); full = x @ W4.T; pf = full.argmax(1)
    cand = prev.topk(64, 1).indices; m = prev.clone(); m.scatter_(1, cand, bf16r(full.gather(1, cand).contiguous())); a64 = (m.argmax(1) == pf)
    pad = torch.full((x.shape[0], NB * 256 - V), -1e30, device=dev); cb = torch.cat([prev, pad], 1).view(-1, NB, 256).topk(3, 2).indices
    cb = (cb + (torch.arange(NB, device=dev) * 256).view(1, NB, 1)).view(x.shape[0], -1).clamp_max(V - 1)
    m = prev.clone(); m.scatter_(1, cb, bf16r(full.gather(1, cb).contiguous())); ab = (m.argmax(1) == pf)
    p2 = (m.max(1).values - torch.logsumexp(m, 1)).exp(); pfp = (full.max(1).values - torch.logsumexp(full, 1)).exp()
    return ab, a64, ((p2 >= 0.4) != (pfp >= 0.4)), (prev.argmax(1) == pf)
cats = None
if len(sys.argv) > 9: cats = [json.loads(l)["cat"] for l in open(sys.argv[9])]
for name, Q in (("OLD rotation (fitted on the held prompts, round 8)", Qold), ("NEW rotation (fitted on the round-9 corpus rows only)", Qnew)):
    P = deq(*affine4(W4 @ Q)); print(name)
    for gname in ("fit", "fresh", "held"):
        idx = np.nonzero(g[gname])[0]; R = [[], [], [], []]
        for s in range(0, len(idx), 2048):
            x = torch.from_numpy((raw[idx[s:s + 2048]].astype(np.uint32) << 16).view(np.float32).copy()).to(dev)
            for k, v in enumerate(agree(x, Q, P)): R[k].append(v)
        R = [torch.cat(v).float() for v in R]
        if not len(idx): continue
        print(f"  {gname:5s} rows {len(idx):6d}: pick == full 4-bit head: block form {R[0].mean():.4f}, top-64 form {R[1].mean():.4f}, preview alone {R[3].mean():.4f}; gate (p >= 0.4) flips {R[2].mean():.4f}")
        if gname == "fresh" and cats:
            rq = req[idx] - nfit
            for c in sorted(set(cats)):
                msk = torch.tensor([cats[i] == c for i in rq], device=dev)
                if msk.any(): print(f"        fresh {c:6s} rows {int(msk.sum()):5d}: block form {R[0][msk].mean():.4f}")
    del P
