#!/usr/bin/env python3
# casc.py MODEL_DIR FIT_PREFIX TEST_PREFIX [MAXWIN=3000] [MARKS=name:calls,...] : offline study for a CASCADE certified LM head (builder A, round 10).
# Rows = LM-head input rows dumped by TF_FLASHNEXT_Q38_HDUMP (bf16 [n, 2560]; PREFIX.x rows, PREFIX.n u32 rows a call = one verify window).
# Head = the checkpoint's BF16 [248320, 2560] tensor. For each scheme: per window, entries that can be or tie with a row's best after the
# stage's rigorous bound, the 64-entry tiles of the window's union, bytes read. Stage 2 = the engine's SM8 copy on whole surviving tiles,
# stage 3 = exact BF16 tiles. A rotation (PCA of the FIT rows, uncentred) is evaluated on the TEST rows only.
import sys, os, json, struct, glob, math
import torch
D = 2560; V = 248320; TILE = 64; NT = V // TILE
mdir, fitp, testp = sys.argv[1], sys.argv[2], sys.argv[3]
maxwin = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
marks = [(m.split(":")[0], int(m.split(":")[1])) for m in sys.argv[5].split(",")] if len(sys.argv) > 5 and sys.argv[5] else []
dev = "cuda"; torch.manual_seed(1)
def find_head():
    for f in sorted(glob.glob(os.path.join(mdir, "*.safetensors")) + glob.glob(os.path.join(mdir, "tensors", "*.safetensors")))[::-1]:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
            for k, v in hdr.items():
                if k != "__metadata__" and v.get("shape") == [V, D] and "head" in k:
                    a, b = v["data_offsets"]; fh.seek(8 + n + a); return k, v["dtype"], fh.read(b - a)
    raise SystemExit("head not found")
name, dt, raw = find_head(); print("head tensor", name, dt, flush=True)
W = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(V, D).to(dev).float(); del raw
wn = W.double().norm(dim=1).float()
def counts(p):
    c = torch.frombuffer(bytearray(open(p + ".n", "rb").read()), dtype=torch.int32).long()
    n = os.path.getsize(p + ".x") // (D * 2); cs = torch.cumsum(c, 0); k = int((cs <= n).sum()); return c[:k], cs[:k] - c[:k]
# ---- fit: second moment of the fit rows
cf, sf = counts(fitp); nfit = int(cf.sum()); M = torch.zeros(D, D, dtype=torch.float64, device=dev); xs = torch.zeros(D, dtype=torch.float64, device=dev)
with open(fitp + ".x", "rb") as fh:
    CH = 16384; done = 0
    while done < nfit:
        b = fh.read(min(CH, nfit - done) * D * 2); x = torch.frombuffer(bytearray(b), dtype=torch.bfloat16).view(-1, D).to(dev).double()
        M += x.T @ x; xs += x.sum(0); done += x.shape[0]
ev, Q = torch.linalg.eigh(M / nfit); Q = Q.flip(1).float().contiguous(); ev = ev.flip(0)
print("fit rows %d calls %d; second-moment energy in the first r directions:" % (nfit, len(cf)), " ".join("%d:%.4f" % (r, float(ev[:r].sum() / ev.sum())) for r in (32, 64, 128, 256, 512, 1024, 2048)), flush=True)
# ---- test windows (calls of 2..64 rows = verify windows), evenly spaced
ct, st = counts(testp); ok = [i for i in range(len(ct)) if 2 <= int(ct[i]) <= 64]
step = max(1, len(ok) // maxwin); pick = ok[::step][:maxwin]
X = []; win = []; src = []
def srcof(i):
    for nm, lim in marks:
        if i < lim: return nm
    return "rest"
with open(testp + ".x", "rb") as fh:
    for j, i in enumerate(pick):
        s, c = int(st[i]), int(ct[i]); fh.seek(s * D * 2)
        X.append(torch.frombuffer(bytearray(fh.read(c * D * 2)), dtype=torch.bfloat16).view(c, D)); win += [j] * c; src.append(srcof(i))
X = torch.cat(X).to(dev).float(); win = torch.tensor(win, device=dev); R = X.shape[0]; NW = len(pick)
print("test: calls %d rows %d (all sizes); verify windows used %d rows %d (mean %.2f rows a window); sources %s" % (len(ct), int(ct.sum()), NW, R, R / NW, {s: src.count(s) for s in sorted(set(src))}), flush=True)
hn = X.norm(dim=1)
XQ = X @ Q
print("test rows: norm outside the first r fitted directions / row norm (mean, p99):", " ".join("%d:%.3f/%.3f" % (r, float((XQ[:, r:].norm(dim=1) / hn).mean()), float((XQ[:, r:].norm(dim=1) / hn).quantile(0.99))) for r in (64, 128, 256, 512, 1024, 2048)), flush=True)
wbar = W.mean(0); print("row-mean shift: |mean head row| %.4f, mean |head row| %.4f, mean |head row - mean row| %.4f" % (float(wbar.norm()), float(wn.mean()), float((W - wbar).norm(dim=1).mean())), flush=True)
def quant(Wb, bits, pow2):  # sign-magnitude integer code, one scale per 32 inputs
    n = 2 ** (bits - 1) - 1; Vv, Dd = Wb.shape; Y = Wb.reshape(Vv, Dd // 32, 32); m = Y.abs().amax(dim=2, keepdim=True).clamp(min=1e-30)
    s = torch.pow(2.0, torch.ceil(torch.log2(m / n))) if pow2 else m / n
    return (torch.round(Y / s).clamp(-n, n) * s).reshape(Vv, Dd)
def bfstep(t): return torch.pow(2.0, torch.floor(torch.log2(t.abs().clamp(min=1e-6))) - 7)
GEDGES = [0, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 2560]
def stage(Wq, Wt, Xt, groups):
    """coarse copy Wq of Wt (both [V, D] in the basis of Xt). groups: list of (a, b) column ranges for the group-wise bound, None = one norm."""
    E = Wt - Wq
    if groups is None: C = (E.double().norm(dim=1).float() * 1.0001 + 1.0e-3 * wn)[:, None]; H = Xt.norm(dim=1, keepdim=True) * 1.0001
    else:
        C = torch.stack([E[:, a:b].double().norm(dim=1).float() for a, b in groups], 1) * 1.0001; C[:, 0] += 1.0e-3 * wn
        H = torch.stack([Xt[:, a:b].norm(dim=1) for a, b in groups], 1) * 1.0001
    del E
    surv_tiles = []; ents = []; bmean = []
    B = 384
    for a in range(0, R, B):
        x = Xt[a:a + B]; A = x @ Wq.T; bd = H[a:a + B] @ C.T
        lo = A - bd; th = lo.amax(dim=1, keepdim=True); keep = (A + bd) >= (th - 2 * bfstep(th))
        ents.append(keep.sum(1)); bmean.append(bd.mean(1))
        t = keep.view(x.shape[0], NT, TILE).any(dim=2); surv_tiles.append(t)
        del A, bd, lo, keep
    T = torch.cat(surv_tiles); ents = torch.cat(ents).float(); bmean = torch.cat(bmean)
    wt = torch.zeros(NW, NT, dtype=torch.bool, device=dev); wt.index_put_((win[T.nonzero()[:, 0]], T.nonzero()[:, 1]), torch.tensor(True, device=dev))
    return wt, ents, bmean
def rep(label, bytes1, wt1, ents, bmean, wt_ex):
    t1 = wt1.sum(1).float(); tx = wt_ex.sum(1).float()
    b2 = t1 * TILE * D if bytes1 is not None else 0 * t1
    b1 = bytes1 if bytes1 is not None else V * D
    over = (t1 > 1500)
    tot = torch.where(over, torch.full_like(t1, float(b1 + V * D)), b1 + b2) + tx * TILE * D * 2 if bytes1 is not None else b1 + tx * TILE * D * 2
    base = V * D + tx * TILE * D * 2
    q = lambda v, p: float(v.quantile(p))
    print("| %-44s | %6.1f | %7.2f | %8.0f | %6.0f / %6.0f / %6.0f / %6.0f | %5.1f %% | %6.1f | %5.1f %% |" % (label, b1 / 2 ** 20, float(bmean.mean()), float(ents.mean()), float(t1.mean()), q(t1, 0.5), q(t1, 0.9), q(t1, 0.99), 100 * float(over.float().mean()), float(tot.mean()) / 2 ** 20, 100 * (1 - float(tot.mean()) / float(base.mean()))), flush=True)
    return tot
print("| scheme (stage 1)                             | MiB s1 | bound   | entries  | tiles a window mean / p50 / p90 / p99 | >1500  | MiB all | saved   |", flush=True)
# (a) the engine's SM8 copy (power-of-two scale), whole-row bound = today's stage; its survivors = exact tiles
Wq8 = quant(W, 8, True); wt_ex, e8, b8 = stage(Wq8, W, X, None)
tx = wt_ex.sum(1).float(); print("(a) SM8 today: exact tiles a window mean %.1f p50 %.0f p90 %.0f p99 %.0f; entries a row %.1f; bound +-%.3f logits; residual / row norm %.4f" % (float(tx.mean()), float(tx.quantile(0.5)), float(tx.quantile(0.9)), float(tx.quantile(0.99)), float(e8.mean()), float(b8.mean()), float(((W - Wq8).norm(dim=1) / wn).mean())), flush=True)
rep("(a) sm8 /32 pow2, one norm [today]", None, wt_ex, e8, b8, wt_ex)
# (b) row-mean shift on the SM8 copy
Ws = W - wbar; Wq = quant(Ws, 8, True); wtb, eb, bb = stage(Wq, Ws, X, None)
print("(b) residual / row norm with the shift %.4f" % float(((Ws - Wq).norm(dim=1) / wn).mean()), flush=True)
rep("(b) sm8 + row-mean shift (exact tiles)", None, wtb, eb, bb, wtb); del Wq
# (b2) group-wise bound on today's copy (80 groups of 32 would be the limit; 8 groups of 320 here)
g8 = [(i * 320, (i + 1) * 320) for i in range(8)]; wtg, eg, bg = stage(Wq8, W, X, g8); rep("(b2) sm8, 8-group bound (exact tiles)", None, wtg, eg, bg, wtg)
del Wq8
def tiers(label, base, Xt, spec, pow2=False, grp=True):
    """spec: [(ndims, bits)] from column 0; the rest of the columns are dropped (bound through their norm)."""
    Wq = torch.zeros_like(base); a = 0; bits_total = 0
    for nd, b in spec:
        Wq[:, a:a + nd] = quant(base[:, a:a + nd], b, pow2); a += nd; bits_total += nd * b
    edges = sorted(set([e for e in GEDGES if e < D] + [D] + [sum(n for n, _ in spec[:i + 1]) for i in range(len(spec))]))
    groups = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)] if grp else None
    wt1, e1, b1 = stage(Wq, base, Xt, groups); del Wq
    return rep(label, V * bits_total / 8 + (V * len(groups) if groups else V), wt1, e1, b1, wt_ex)
SET2 = os.environ.get("CASC_SET") == "2"
if SET2:
    # permuted plain basis (exact: a permutation of the input dims), heavy dims first: order by rms(x_j) * rms(w_j) on the FIT rows
    xr = torch.sqrt(torch.diag(M / nfit)).float(); wr = W.pow(2).mean(0).sqrt(); order = torch.argsort(xr * wr, descending=True)
    Wp = W[:, order].contiguous(); Xp = X[:, order].contiguous()
    print("permuted basis: share of sum x_j^2 w_j^2 in the first n dims:", " ".join("%d:%.3f" % (n, float(((xr * wr)[order][:n] ** 2).sum() / ((xr * wr) ** 2).sum())) for n in (256, 512, 1024, 1536)), flush=True)
    for lab, spec in (("perm 6b x 2560", [(2560, 6)]), ("perm 8b x 512 + 5b x 2048", [(512, 8), (2048, 5)]), ("perm 7b x 512 + 6b x 1024 + 5b x 1024", [(512, 7), (1024, 6), (1024, 5)]),
                      ("perm 6b x 1024 + 5b x 1536", [(1024, 6), (1536, 5)]), ("perm 6b x 512 + 5b x 2048", [(512, 6), (2048, 5)]), ("perm 6b x 1280 + 4b x 1280", [(1280, 6), (1280, 4)]),
                      ("perm 7b x 256 + 6b x 768 + 5b x 1536", [(256, 7), (768, 6), (1536, 5)])):
        tiers("(p) " + lab, Wp, Xp, spec)
    del Wp, Xp
    WR = W @ Q
    for lab, spec in (("rot 6b x 1024 + 5b x 1536", [(1024, 6), (1536, 5)]), ("rot 7b x 512 + 6b x 1024 + 4b x 1024", [(512, 7), (1024, 6), (1024, 4)]),
                      ("rot 6b x 512 + 5b x 1024 + 4b x 1024", [(512, 6), (1024, 5), (1024, 4)]), ("rot 7b x 256 + 6b x 768 + 4b x 1536", [(256, 7), (768, 6), (1536, 4)]),
                      ("rot 6b x 1024 + 4b x 1536", [(1024, 6), (1536, 4)]), ("rot 6b x 512 + 4b x 2048", [(512, 6), (2048, 4)]), ("rot 5b x 1536 + 4b x 1024", [(1536, 5), (1024, 4)]),
                      ("rot 8b x 256 + 6b x 768 + 5b x 1536", [(256, 8), (768, 6), (1536, 5)]), ("rot 7b x 1024 + 5b x 1536", [(1024, 7), (1536, 5)]), ("rot 5b x 2560", [(2560, 5)]),
                      ("rot 7b x 2560", [(2560, 7)])):
        tiers("(c) " + lab, WR, XQ, spec)
    # the same, with power-of-two scales (what a kernel that reuses MXFP8's scale bytes would hold)
    tiers("(c0) plain 6b x 2560, POWER-OF-TWO scale", W, X, [(2560, 6)], pow2=True)
    tiers("(c0) plain 6b x 2560, one norm (no group bound)", W, X, [(2560, 6)], grp=False)
    tiers("(c0) plain 7b x 2560 ideal scale", W, X, [(2560, 7)])
    sys.exit(0)
# (c) first stages. plain basis:
tiers("(c0) plain basis, 4-bit all, group bound", W, X, [(2560, 4)])
tiers("(c0) plain + shift, 4-bit all", Ws, X, [(2560, 4)])
tiers("(c0) plain basis, 5-bit all", W, X, [(2560, 5)])
tiers("(c0) plain basis, 6-bit all", W, X, [(2560, 6)])
del Ws
# rotated basis (fitted on the fit rows)
WR = W @ Q
for lab, spec in (("rot 8b x 2560 (same bytes)", [(2560, 8)]), ("rot 8b x 1024", [(1024, 8)]), ("rot 8b x 512", [(512, 8)]), ("rot 8b x 256", [(256, 8)]),
                  ("rot 4b x 2560", [(2560, 4)]), ("rot 4b x 1024", [(1024, 4)]), ("rot 8b x 256 + 4b x 768", [(256, 8), (768, 4)]),
                  ("rot 8b x 512 + 4b x 2048", [(512, 8), (2048, 4)]), ("rot 8b x 256 + 4b x 2304", [(256, 8), (2304, 4)]),
                  ("rot 8b x 128 + 6b x 384 + 4b x 2048", [(128, 8), (384, 6), (2048, 4)]), ("rot 8b x 512 + 2b x 2048", [(512, 8), (2048, 2)]),
                  ("rot 6b x 2560", [(2560, 6)]), ("rot 8b x 1024 + 4b x 1536", [(1024, 8), (1536, 4)])):
    tiers("(c) " + lab, WR, XQ, spec)
# rotated + shift (shift in the rotated basis is the same vector rotated)
WRs = WR - WR.mean(0)
tiers("(c) rot + shift, 8b x 512 + 4b x 2048", WRs, XQ, [(512, 8), (2048, 4)])
tiers("(c) rot + shift, 4b x 2560", WRs, XQ, [(2560, 4)])
print("notes: MiB s1 = stage-1 bytes (codes + bound table); MiB all = stage 1 + SM8 on the surviving whole tiles (whole SM8 copy when > 1500 tiles) + exact tiles; saved = against today's (a). Stage-1 scales are ideal (not power of two): generous.", flush=True)
