#!/usr/bin/env python3
"""calB.py MODELDIR VOCAB.txt MIXED.dump HEAD2.bin : calibration of the two-stage head's probability (builder B, CPU ok).
p2 = two-stage probability (preview log-sum-exp, best K exact), pf = the full 4-bit head's. Fits c in p' = p2 / (p2 + c (1 - p2))."""
import sys, json, os, struct, numpy as np, torch
torch.set_num_threads(int(os.environ.get("CAL_THREADS", "6")))
D = 2560; mdir, vocab, dump, h2 = sys.argv[1:5]
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
path, off, meta = find_tensor(mdir, ["language_model.lm_head.weight", "lm_head.weight"])
mm = np.memmap(path, dtype=np.uint16, mode="r", offset=off, shape=(248320, D))
def f16(a): return torch.from_numpy((a.astype(np.uint32) << 16).view(np.float32).copy())
W = f16(np.asarray(mm[ids]))
def bf16r(t):
    b = t.contiguous().view(torch.int32); return (((b + 0x7fff + ((b >> 16) & 1)) >> 16) << 16).view(torch.float32)
G = W.reshape(V, -1, 32); lo = G.amin(2, keepdim=True); sc = ((G.amax(2, keepdim=True) - lo) / 15.0).clamp_min(1e-8); q = ((G - lo) / sc).clamp(0, 15)
fl = q.floor(); fr = q - fl; code = fl + ((fr > 0.5) | ((fr == 0.5) & (fl.long() % 2 == 1))).float()
W4 = (code * bf16r(sc) + bf16r(lo)).reshape(V, D); del W, G, q, fl, fr, code
raw = open(h2, "rb").read(); magic, ver, d0, r, v0 = struct.unpack("<8sIIII", raw[:24]); assert v0 == V
o = 24; At = torch.from_numpy(np.frombuffer(raw, "<f4", r * D, o).reshape(r, D).copy()); o += r * D * 4
pk = np.frombuffer(raw, np.uint8, V * r // 2, o).reshape(V, r // 2); o += V * r // 2
S = f16(np.frombuffer(raw, "<u2", V * r // 32, o).reshape(V, r // 32)); o += V * r // 32 * 2
B = f16(np.frombuffer(raw, "<u2", V * r // 32, o).reshape(V, r // 32))
codes = np.empty((V, r), np.float32); codes[:, 0::2] = pk & 15; codes[:, 1::2] = pk >> 4
P = (torch.from_numpy(codes).reshape(V, r // 32, 32) * S[:, :, None] + B[:, :, None]).reshape(V, r)
X = np.fromfile(dump, dtype=np.uint16); X = f16(X[: (len(X) // D) * D].reshape(-1, D))
X = X[:: int(os.environ.get("CAL_STRIDE", "4"))]
print("rows", X.shape[0], "rank", r)
print("block candidates: the best M preview rows of every 256-row block are re-scored (311 blocks); pick = best exact among them;")
print("p = exp(top) / (sum exp(exact candidates) + sum exp(preview of the rest))")
NB = (V + 255) // 256; pad = NB * 256 - V
for M in (1, 2, 3):
    p2s = []; pfs = []; agree = 0
    for a in range(0, X.shape[0], 1024):
        x = X[a:a + 1024]; full = x @ W4.T; prev = bf16r((x @ At.T) @ P.T)
        pp = torch.nn.functional.pad(prev, (0, pad), value=-1e30).reshape(-1, NB, 256)
        ti = pp.topk(M, 2).indices + (torch.arange(NB) * 256)[None, :, None]; cand = ti.reshape(x.shape[0], -1).clamp_max(V - 1)
        ex = full.gather(1, cand); top = ex.max(1); pick = cand.gather(1, top.indices[:, None]).squeeze(1)
        rest = prev.clone(); rest.scatter_(1, cand, -1e30)
        lse = torch.logsumexp(torch.cat([ex, rest], 1), 1)
        p2s.append((top.values - lse).exp()); pfs.append((full.max(1).values - torch.logsumexp(full, 1)).exp()); agree += int((pick == full.argmax(1)).sum())
    p2 = torch.cat(p2s).double(); pf = torch.cat(pfs).double()
    line = f"M={M} ({NB * M} rows re-scored): pick agreement {agree / X.shape[0]:.4f}; mean (p2 - pf) {float((p2 - pf).mean()):+.4f}; up-flips {float(((p2 >= 0.4) & (pf < 0.4)).double().mean()):.4f} down-flips {float(((p2 < 0.4) & (pf >= 0.4)).double().mean()):.4f}"
    best = None
    for c in [1.0 + 0.02 * i for i in range(0, 60)]:
        pc = p2 / (p2 + c * (1 - p2)); m = float((pc - pf).mean())
        if best is None or abs(m) < abs(best[1]): best = (c, m)
    print(line + f"; centred by c = {best[0]:.2f}")
for K in (32, 64):
    p2s = []; pfs = []; agree = 0
    for a in range(0, X.shape[0], 1024):
        x = X[a:a + 1024]; full = x @ W4.T; prev = bf16r((x @ At.T) @ P.T)
        cand = prev.topk(K, 1).indices; mixed = prev.clone(); mixed.scatter_(1, cand, bf16r(full.gather(1, cand).contiguous()))
        p2s.append((mixed.max(1).values - torch.logsumexp(mixed, 1)).exp()); pfs.append((full.max(1).values - torch.logsumexp(full, 1)).exp())
        agree += int((mixed.argmax(1) == full.argmax(1)).sum())
    p2 = torch.cat(p2s).double(); pf = torch.cat(pfs).double()
    print(f"K={K}: pick agreement {agree / X.shape[0]:.4f}; mean p2 {p2.mean():.4f} mean pf {pf.mean():.4f}; mean (p2 - pf) {float((p2 - pf).mean()):+.4f}; share p2 > pf {float((p2 > pf).double().mean()):.3f}; (p>=0.4) up-flips {float(((p2 >= 0.4) & (pf < 0.4)).double().mean()):.4f} down-flips {float(((p2 < 0.4) & (pf >= 0.4)).double().mean()):.4f}")
    best = None
    for c in [1.0 + 0.02 * i for i in range(0, 60)]:
        pc = p2 / (p2 + c * (1 - p2)); e = float(((pc.clamp_min(1e-6).log() - pf.clamp_min(1e-6).log()) ** 2).mean()); m = float((pc - pf).mean())
        if best is None or abs(m) < abs(best[2]): best = (c, e, m)
    c = best[0]; pc = p2 / (p2 + c * (1 - p2))
    print(f"   c = {c:.2f} makes the mean difference {best[2]:+.5f}; then |p' - pf| mean {float((pc - pf).abs().mean()):.4f} (before {float((p2 - pf).abs().mean()):.4f}); up-flips {float(((pc >= 0.4) & (pf < 0.4)).double().mean()):.4f} down-flips {float(((pc < 0.4) & (pf >= 0.4)).double().mean()):.4f}")
