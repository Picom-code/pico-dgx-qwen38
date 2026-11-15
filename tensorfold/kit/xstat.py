#!/usr/bin/env python3
"""xstat.py FILE [LABEL] : expert statistics from a TF_FLASHNEXT_Q38_XSTAT dump (builder D). numpy only.
Record: u32 magic, model layers, layer, rows, slots, lanes, lanes x u32 rows-of-lane, picks i32 [rows, slots], weights f32 [rows, slots].
Only the 48-layer main model's blocks are used. A pass = 48 consecutive blocks (layer 0..47) with the same rows."""
import sys, numpy as np
MB_EXPERT = (3 * 640 * 2560 / 2) * 36 / 32 / 1e6      # NVFP4 codes + e4m3 scales (BLOCK4 = 36 words per 32): 2.7648 MB an expert
NE = 512
buf = np.fromfile(sys.argv[1], dtype=np.uint8); label = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]
u32 = lambda o, n=1: buf[o:o + 4 * n].view(np.uint32)
o = 0; blocks = []
while o + 24 <= len(buf):
    h = u32(o, 6)
    if h[0] != 0x58333851: raise SystemExit("bad magic at %d" % o)
    L, layer, n, slots, nl = int(h[1]), int(h[2]), int(h[3]), int(h[4]), int(h[5])
    o += 24; lanes = u32(o, nl).astype(int); o += 4 * nl
    if o + 8 * n * slots > len(buf): break
    picks = buf[o:o + 4 * n * slots].view(np.int32).reshape(n, slots); o += 4 * n * slots
    wts = buf[o:o + 4 * n * slots].view(np.float32).reshape(n, slots); o += 4 * n * slots
    if L == 48: blocks.append((layer, n, lanes, picks, wts))
# passes
passes = []; cur = []
for b in blocks:
    if b[0] == 0: cur = [b]
    elif cur and b[0] == len(cur) and b[1] == cur[0][1]: cur.append(b)
    else: cur = []
    if len(cur) == 48: passes.append(cur); cur = []
print("== %s: %d main-model blocks, %d whole passes" % (label, len(blocks), len(passes)))
def lane_rows(lanes, n):
    """row -> lane id, rows contiguous in lane order (checked: sum of lane rows == n; else None)"""
    if len(lanes) == 0 or lanes.sum() != n: return None
    return np.repeat(np.arange(len(lanes)), lanes)
zero_w = 0; tot_w = 0; min_w = 1.0
rows_by = {}
for p in passes:
    n = p[0][1]; lanes = p[0][2]; nl = len(lanes)
    key = "1 lane" if nl <= 1 else "2-4" if nl <= 4 else "5-8" if nl <= 8 else "9-12" if nl <= 12 else "13-16"
    rl = lane_rows(lanes, n)
    U = []; I16 = []; I32 = []; mx = []; lane_u = []; pair_j = []; over16 = []; adj = []; hot = []
    for (layer, n_, lanes_, picks, wts) in p:
        r = picks[:, :10]; w = wts[:, :10]
        zero_w += int((w == 0).sum()); tot_w += w.size; min_w = min(min_w, float(w.min()))
        cnt = np.bincount(r.ravel(), minlength=NE + 1)[:NE]
        used = cnt > 0; u = int(used.sum()); U.append(u)
        I16.append(int(np.ceil(cnt[used] / 16).sum())); I32.append(int(np.ceil(cnt[used] / 32).sum())); mx.append(int(cnt.max())); over16.append(int((cnt > 16).sum()))
        ids = np.nonzero(used)[0]; adj.append(float((np.diff(ids) == 1).mean()) if u > 1 else 0.0)
        if rl is not None and nl > 1:
            sets = [np.unique(r[rl == l]) for l in range(nl)]
            lane_u.append(float(np.mean([len(s) for s in sets])))
            m = np.zeros((nl, NE), bool)
            for l, s in enumerate(sets): m[l, s] = True
            inter = (m[:, None, :] & m[None, :, :]).sum(-1); uni = (m[:, None, :] | m[None, :, :]).sum(-1)
            iu = np.triu_indices(nl, 1); pair_j.append(float((inter[iu] / np.maximum(uni[iu], 1)).mean()))
            hot.append(int((m.sum(0) >= max(2, nl // 2)).sum()))
    rows_by.setdefault(key, []).append(dict(n=n, nl=nl, U=np.mean(U), I16=np.mean(I16), I32=np.mean(I32), mx=np.mean(mx), over16=np.mean(over16), adj=np.mean(adj),
        lane_u=np.mean(lane_u) if lane_u else float("nan"), pj=np.mean(pair_j) if pair_j else float("nan"), hot=np.mean(hot) if hot else float("nan"), Ul=np.array(U, float)))
print("routing weights: %d of %d exactly zero; smallest %.3g" % (zero_w, tot_w, min_w))
print("+--------+--------+-------+-------+----------+--------+---------+---------+---------+--------+--------+---------+----------+--------+")
print("| lanes  | passes | rows  | lanes | distinct | of 512 | rows an | items   | experts | most   | GiB a  | ms at   | distinct | pair   |")
print("| a pass |        | a pass| mean  | experts  |        | expert  | of 16 / | with    | rows on| pass   | 225 GB/s| a lane   | Jaccard|")
print("|        |        |       |       | a layer  |        |         | distinct| > 16 r. | one    | (items)|         | (alone)  |        |")
print("+--------+--------+-------+-------+----------+--------+---------+---------+---------+--------+--------+---------+----------+--------+")
for key in ["1 lane", "2-4", "5-8", "9-12", "13-16"]:
    v = rows_by.get(key)
    if not v: continue
    m = lambda k: float(np.nanmean([x[k] for x in v]))
    gib = m("I16") * MB_EXPERT * 48 / 1024
    print("| %-6s | %6d | %5.1f | %5.1f | %8.1f | %5.1f%% | %7.2f | %7.3f | %7.2f | %6.1f | %6.2f | %7.1f | %8.1f | %6.3f |" % (key, len(v), m("n"), m("nl"), m("U"), 100 * m("U") / NE,
          10 * m("n") / m("U"), m("I16") / m("U"), m("over16"), m("mx"), gib, gib * 1.0737 / 225 * 1000, m("lane_u"), m("pj")))
print("+--------+--------+-------+-------+----------+--------+---------+---------+---------+--------+--------+---------+----------+--------+")
# marginal cost of a row: least squares of mean distinct experts on rows, within the widest bucket
for key in ["5-8", "13-16"]:
    v = rows_by.get(key)
    if not v or len(v) < 8: continue
    x = np.array([p["n"] for p in v], float); y = np.array([p["U"] for p in v])
    if x.std() == 0: continue
    a, b = np.polyfit(x, y, 1)
    print("%s lanes: distinct experts a layer = %.1f + %.3f x rows (rows %d..%d, r %.2f) => an extra row reads %.2f experts a layer = %.1f MB a pass = %.3f ms at 225 GB/s"
          % (key, b, a, x.min(), x.max(), np.corrcoef(x, y)[0, 1], a, a * MB_EXPERT * 48, a * MB_EXPERT * 48 / 225))
    Ul = np.mean([p["Ul"] for p in v], 0)
    print("  distinct experts by layer (mean): min %.0f (layer %d), max %.0f (layer %d); layers 0-11 %.0f, 12-23 %.0f, 24-35 %.0f, 36-47 %.0f" % (Ul.min(), Ul.argmin(), Ul.max(), Ul.argmax(), Ul[:12].mean(), Ul[12:24].mean(), Ul[24:36].mean(), Ul[36:].mean()))
    print("  32-pair items would read %.3f x the distinct experts (16-pair: %.3f); adjacent-id share of used experts %.3f; experts used by at least half the lanes: %.1f a layer"
          % (np.mean([p["I32"] / p["U"] for p in v]), np.mean([p["I16"] / p["U"] for p in v]), np.mean([p["adj"] for p in v]), np.nanmean([p["hot"] for p in v])))
# sub-pass simulation on the widest passes: split the lanes in two groups (best of greedy by overlap, and worst = random), bytes = sum of the groups' unions
v = [p for p in passes if len(p[0][2]) >= 13 and lane_rows(p[0][2], p[0][1]) is not None]
if v:
    rng = np.random.default_rng(1); whole = []; greedy = []; rand = []; sumlane = []; temporal = []
    prev = None
    for p in v[:: max(1, len(v) // 120)]:
        nl = len(p[0][2]); rl = lane_rows(p[0][2], p[0][1])
        for (layer, n_, lanes_, picks, wts) in p[::4]:
            m = np.zeros((nl, NE), bool)
            for l in range(nl): m[l, np.unique(picks[rl == l, :10])] = True
            whole.append(m.any(0).sum()); sumlane.append(m.sum())
            perm = rng.permutation(nl); rand.append(m[perm[: nl // 2]].any(0).sum() + m[perm[nl // 2:]].any(0).sum())
            # greedy: seed two groups with the least-overlapping pair, add each lane to the group whose union grows least (balanced)
            inter = (m[:, None, :] & m[None, :, :]).sum(-1).astype(float); np.fill_diagonal(inter, 1e9)
            a, b = np.unravel_index(inter.argmin(), inter.shape); g = [[a], [b]]; un = [m[a].copy(), m[b].copy()]
            for l in sorted(set(range(nl)) - {a, b}, key=lambda l: -m[l].sum()):
                grow = [int((m[l] & ~un[k]).sum()) if len(g[k]) < (nl + 1) // 2 else 10 ** 9 for k in (0, 1)]
                k = int(np.argmin(grow)); g[k].append(l); un[k] |= m[l]
            greedy.append(un[0].sum() + un[1].sum())
    print("sub-pass simulation (13-16 lanes, %d layer samples): one pass reads %.1f distinct experts a layer; two sub-passes of half the lanes, grouped by expert overlap (greedy) %.1f (x%.2f), random halves %.1f (x%.2f); every lane alone %.1f (x%.2f)"
          % (len(whole), np.mean(whole), np.mean(greedy), np.mean(greedy) / np.mean(whole), np.mean(rand), np.mean(rand) / np.mean(whole), np.mean(sumlane), np.mean(sumlane) / np.mean(whole)))
