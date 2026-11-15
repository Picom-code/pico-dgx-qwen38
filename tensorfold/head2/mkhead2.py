#!/usr/bin/env python3
"""mkhead2.py MODELDIR VOCAB.txt MIXED.dump|ROTATION.rot OUT.bin [RANK=512] [ENGINE.log] : the two-stage draft head's file (TF_FLASHNEXT_Q38_HEAD2).
With a .rot file (what an earlier run wrote next to OUT.bin: "Q38ROT01", u32 dims, u32 R, then [R, dims] fp32) no dump is needed: the
rotation is read, and only the checkpoint's own head rows are projected on it.
Runs in a pytorch container (GPU or CPU). Inputs: the checkpoint's own lm_head rows (BF16), the draft vocabulary (the engine's
draft_vocab_default.txt), and a dump of draft-head input rows ([n, 2560] bf16, TF_FLASHNEXT_Q38_MTPDUMP2 of a TF_STEP_LOG=2 run) that
gives the rotation: the eigenvectors of the inputs' second-moment matrix (PCA of what the head multiplies).
Output (little endian): "Q38HEAD2", u32 2, u32 dims, u32 R, u32 V; At [R, dims] fp32; codes [V, R/2] u8 (4-bit, low nibble first);
scales [V, R/32] bf16; biases [V, R/32] bf16: the first R rotated coordinates of the 4-bit draft rows, group-32 affine 4-bit again.
With ENGINE.log (the dump's own log): agreement of the two-stage pick with the full 4-bit head on the last 30 % of the requests, with a
rotation fitted on the first 70 % only. Drafts only: nothing here can change a reply."""
import sys, json, os, struct, numpy as np, torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
D = 2560; mdir, vocab, dump, out = sys.argv[1:5]; r = int(sys.argv[5]) if len(sys.argv) > 5 else 512
assert r in (128, 256, 512)
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
assert meta["dtype"] == "BF16" and meta["shape"] == [248320, D]
mm = np.memmap(path, dtype=np.uint16, mode="r", offset=off, shape=(248320, D))
W = torch.from_numpy((np.asarray(mm[ids]).astype(np.uint32) << 16).view(np.float32).copy()).to(dev)
def bf16bits(t):
    b = t.contiguous().view(torch.int32); return ((b + 0x7fff + ((b >> 16) & 1)) >> 16)
def bf16r(t): return (bf16bits(t) << 16).view(torch.float32)
def affine4(Wm):  # cuda_draft_affine.zig's recipe: codes, bf16 scale and bias a group of 32
    G = Wm.reshape(Wm.shape[0], -1, 32); lo = G.amin(2, keepdim=True); hi = G.amax(2, keepdim=True)
    sc = ((hi - lo) / 15.0).clamp_min(1e-8); q = ((G - lo) / sc).clamp(0, 15)
    fl = q.floor(); fr = q - fl; code = fl + ((fr > 0.5) | ((fr == 0.5) & (fl.long() % 2 == 1))).float()
    return code, sc, lo
def deq(code, sc, lo): return (code * bf16r(sc) + bf16r(lo)).reshape(code.shape[0], -1)
W4 = deq(*affine4(W))
have_rot = dump.endswith(".rot")
if have_rot:
    raw = open(dump, "rb").read(); assert raw[:8] == b"Q38ROT01"; d0, r0 = struct.unpack("<II", raw[8:16]); assert d0 == D and r0 >= r
    Qr = torch.from_numpy(np.frombuffer(raw, dtype="<f4", offset=16).reshape(r0, D)[:r].copy()).to(dev).T.contiguous()   # [D, r]
    print("draft rows", V, "rotation read from", dump, "rank", r)
else:
    X = np.fromfile(dump, dtype=np.uint16); X = X[: (len(X) // D) * D].reshape(-1, D)
    X = torch.from_numpy((X.astype(np.uint32) << 16).view(np.float32).copy()).to(dev)
    print("draft rows", V, "head-input rows", X.shape[0], "rank", r)
def rotation(x):
    e, Q = torch.linalg.eigh((x.T.double() @ x.double()) / x.shape[0]); return Q[:, torch.argsort(-e)].float()
def preview(Q):
    code, sc, lo = affine4(W4 @ Q[:, :r]); return code, sc, lo
def two_stage(x, Q, P, K):
    xr = x @ Q[:, :r]; prev = bf16r(xr @ P.T); full = x @ W4.T
    cand = prev.topk(K, 1).indices; mixed = prev.clone(); mixed.scatter_(1, cand, bf16r(full.gather(1, cand).contiguous()))
    return mixed.argmax(1), full.argmax(1), (mixed.max(1).values - torch.logsumexp(mixed, 1)).exp(), (full.max(1).values - torch.logsumexp(full, 1)).exp()
if len(sys.argv) > 6 and not have_rot:
    # which request a dump row belongs to: one row a QTOP line
    req = []; q = -1
    for l in open(sys.argv[6], errors="replace"):
        if l.startswith("info: prompt cache:"): q += 1
        elif l.startswith("QTOP "): req.append(q)
    n = min(len(req), X.shape[0]); req = torch.tensor(req[:n], device=dev); cut = int(req.max().item() * 0.7)
    fit = req <= cut; te = ~fit; Q = rotation(X[:n][fit]); P = deq(*preview(Q))
    for K in (16, 32, 64):
        a = b = c = m = 0
        for s in range(0, int(te.sum()), 4096):
            x = X[:n][te][s:s + 4096]; p2, pf, q2, qf = two_stage(x, Q, P, K)
            a += int((p2 == pf).sum()); b += int(((q2 >= 0.4) != (qf >= 0.4)).sum()); c += float((q2 - qf).abs().sum()); m += x.shape[0]
        print(f"held-out requests (rotation fitted on the first 70 % of requests, {int(fit.sum())} rows; tested on {m}): K={K}: two-stage pick == full 4-bit head pick {a / m:.4f}; |p - p_full| mean {c / m:.4f}; (p >= 0.4) flips {b / m:.4f}")
Q = Qr if have_rot else rotation(X); code, sc, lo = preview(Q)
if not have_rot:
    with open(out + ".rot", "wb") as f:
        f.write(b"Q38ROT01" + struct.pack("<II", D, 512)); f.write(Q[:, :512].T.contiguous().cpu().numpy().astype("<f4").tobytes())
codes = code.reshape(V, r).to(torch.uint8).cpu().numpy(); packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
with open(out, "wb") as f:
    f.write(struct.pack("<8sIIII", b"Q38HEAD2", 2, D, r, V))
    f.write(Q[:, :r].T.contiguous().cpu().numpy().astype("<f4").tobytes())
    f.write(packed.tobytes())
    f.write(bf16bits(sc.reshape(V, r // 32)).to(torch.int16).cpu().numpy().astype("<u2").tobytes())
    f.write(bf16bits(lo.reshape(V, r // 32)).to(torch.int16).cpu().numpy().astype("<u2").tobytes())
print("wrote", out, os.path.getsize(out), "bytes")
