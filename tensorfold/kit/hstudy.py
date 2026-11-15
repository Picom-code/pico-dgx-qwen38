#!/usr/bin/env python3
# hstudy.py MODEL_DIR DUMP_PREFIX [MAXROWS] : offline study for a certified exact LM head (runs on a GPU with torch, model NOT loaded).
# Rows = the LM head's input rows dumped by TF_FLASHNEXT_Q38_HDUMP (bf16 [n, 2560]); head = the checkpoint's BF16 [248320, 2560] tensor.
# For each coarse code of the head: a rigorous bound |(w - w^).h| <= sum_g ||e_g|| ||h_g|| (Cauchy-Schwarz, whole row or per 64-column
# group), the rows a coarse pass cannot rule out (upper bound >= best lower bound - one bf16 step), as elements, 64-row tiles, and tiles
# of a verify window's union. Prints one table; saves a sample (rows, exact top-8) for later CPU work.
import sys, os, json, struct, glob, math, time
import torch
D = 2560; V = 248320
mdir, pre = sys.argv[1], sys.argv[2]; maxrows = int(sys.argv[3]) if len(sys.argv) > 3 else 120000
dev = "cuda"
def find_head():
    for f in sorted(glob.glob(os.path.join(mdir, "*.safetensors")) + glob.glob(os.path.join(mdir, "tensors", "*.safetensors")))[::-1]:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
            for k, v in hdr.items():
                if k != "__metadata__" and v.get("shape") == [V, D] and "head" in k:
                    a, b = v["data_offsets"]; fh.seek(8 + n + a); return k, v["dtype"], fh.read(b - a)
    raise SystemExit("head not found")
name, dt, raw = find_head(); print("head tensor", name, dt, len(raw), flush=True)
W = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(V, D).to(dev).float(); del raw
cnt = torch.frombuffer(bytearray(open(pre + ".n", "rb").read()), dtype=torch.int32)
nrows = min(int(cnt.sum()), os.path.getsize(pre + ".x") // (D * 2))
print("dump calls", len(cnt), "rows", nrows, "rows/call mean %.2f" % (nrows / max(1, len(cnt))), "hist", torch.bincount(cnt.clamp(max=20)).tolist(), flush=True)
# rows: a spread sample of whole calls (verify windows kept together)
starts = torch.cumsum(cnt, 0) - cnt
step = max(1, int(nrows / maxrows)); pick = list(range(0, len(cnt), step))
X = []; win = []
with open(pre + ".x", "rb") as fh:
    for i in pick:
        s, c = int(starts[i]), int(cnt[i])
        if s + c > nrows: break
        fh.seek(s * D * 2); X.append(torch.frombuffer(bytearray(fh.read(c * D * 2)), dtype=torch.bfloat16).view(c, D)); win += [i] * c
X = torch.cat(X).to(dev).float(); win = torch.tensor(win, device=dev); R = X.shape[0]
print("study rows", R, "windows", len(pick), flush=True)
hn = X.norm(dim=1); hg = X.view(R, D // 64, 64).norm(dim=2)                     # ||h||, ||h_g||
wn = W.double().norm(dim=1).float()
B = 1024
def bf16(x): return x.to(torch.bfloat16).float()
def ulp(x):  # one bf16 step at |x| (8 significant bits)
    return torch.pow(2.0, torch.floor(torch.log2(x.abs().clamp(min=1e-20))) - 7)
# exact pass: top-8 and bf16 ties
top = torch.empty(R, 8, device=dev); topi = torch.empty(R, 8, dtype=torch.long, device=dev)
for a in range(0, R, B):
    L = X[a:a + B] @ W.T; t = L.topk(8, dim=1); top[a:a + B] = t.values; topi[a:a + B] = t.indices
gap = top[:, 0] - top[:, 1]
q = torch.tensor([.01, .05, .1, .25, .5, .75, .9], device=dev)
print("top logit quantiles", [round(v, 2) for v in torch.quantile(top[:, 0], q).tolist()])
print("top1-top2 gap quantiles", [round(v, 3) for v in torch.quantile(gap, q).tolist()], " bf16 tie top1==top2: %.4f" % (bf16(top[:, 0]) == bf16(top[:, 1])).float().mean().item())
print("||h|| mean %.2f  ||w|| mean %.3f  ||w||*||h|| mean %.1f" % (hn.mean().item(), wn.mean().item(), (wn.mean() * hn.mean()).item()), flush=True)
torch.save({"X": X[:20000].to(torch.bfloat16).cpu(), "top": top[:20000].cpu(), "topi": topi[:20000].cpu()}, pre + ".sample.pt")
def q_mxfp8(W):   # E4M3 values with a power-of-two scale per 32 columns (the engine's MXFP8 lane format)
    Wb = W.view(V, D // 32, 32); m = Wb.abs().amax(dim=2, keepdim=True).clamp(min=1e-30)
    e = torch.ceil(torch.log2(m / 448.0)); s = torch.pow(2.0, e); y = (Wb / s)
    a = y.abs().clamp(min=2.0 ** -9); ex = torch.floor(torch.log2(a)).clamp(min=-6, max=8); stepv = torch.pow(2.0, ex - 3)
    r = torch.round(y / stepv) * stepv; r = r.clamp(-448, 448)
    return (r * s).view(V, D)
def q_int(W, bits, blk):  # symmetric integers, scale per `blk` columns (blk = D: per row)
    Wb = W.view(V, D // blk, blk); m = Wb.abs().amax(dim=2, keepdim=True).clamp(min=1e-30); n = 2 ** (bits - 1) - 1
    return (torch.round(Wb / m * n) * m / n).view(V, D)
variants = [("mxfp8 e4m3 /32", lambda: q_mxfp8(W), 8 + 8 / 32), ("int8 /row", lambda: q_int(W, 8, D), 8), ("int8 /32", lambda: q_int(W, 8, 32), 8.25),
            ("int6 /32", lambda: q_int(W, 6, 32), 6.25), ("int5 /32", lambda: q_int(W, 5, 32), 5.25), ("int4 /32", lambda: q_int(W, 4, 32), 4.25), ("int4 /16", lambda: q_int(W, 4, 16), 4.5)]
print("| coarse code | bits | resid/row-norm | bound mode | mean bound (logits) | rows certified alone (1 survivor) | mean survivors | p99 survivors | mean tiles/row | mean tiles/window | p99 tiles/window | windows > 256 tiles | argmax in survivors |")
for vname, fn, bits in variants:
    Wq = fn(); E = (W - Wq)
    en = E.double().norm(dim=1).float() * 1.0005 + 4e-4 * wn                                   # whole-row residual norm (+ float slack)
    eg = E.view(V, D // 64, 64).double().norm(dim=2).float() * 1.0005                          # per 64-column group
    del E
    for mode in ("row", "group64"):
        surv = torch.empty(R, device=dev); tiles = torch.empty(R, device=dev); ok = 0; bsum = 0.0
        wt = {}
        for a in range(0, R, B):
            x = X[a:a + B]; A = x @ Wq.T
            if mode == "row": bd = hn[a:a + B, None] * en[None, :]
            else: bd = hg[a:a + B] @ eg.T + 4e-4 * hn[a:a + B, None] * wn[None, :]
            lo = (A - bd).amax(dim=1, keepdim=True); thr = lo - ulp(lo) - 1e-3
            s = (A + bd) >= thr; surv[a:a + B] = s.sum(dim=1).float()
            st = s.view(-1, V // 64, 64).any(dim=2); tiles[a:a + B] = st.sum(dim=1).float()
            ok += s.gather(1, topi[a:a + B, :1]).sum().item(); bsum += bd.gather(1, topi[a:a + B, :1]).sum().item()
            w_ = win[a:a + B]
            for wid in torch.unique(w_).tolist():
                m = st[w_ == wid].any(dim=0); wt[wid] = (wt[wid] | m) if wid in wt else m
        tw = torch.tensor([float(m.sum().item()) for m in wt.values()])
        print("| %s | %.2f | %.4f | %s | %.2f | %.4f | %.1f | %.0f | %.1f | %.1f | %.0f | %.4f | %.6f |" % (vname, bits, (en / wn).mean().item(), mode, bsum / R,
              (surv == 1).float().mean().item(), surv.mean().item(), torch.quantile(surv, .99).item(), tiles.mean().item(), tw.mean().item(), torch.quantile(tw, .99).item(), (tw > 256).float().mean().item(), ok / R), flush=True)
    del Wq
print("HSTUDY_DONE")
