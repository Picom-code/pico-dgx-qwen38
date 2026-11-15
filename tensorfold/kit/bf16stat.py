#!/usr/bin/env python3
# bf16stat.py DIR [MB] : is the BF16 weight data of the official checkpoint compressible by an EXACT fixed-length re-encoding?
# For each BF16 group (hyper-connection, LM head, router, MTP dense) it samples tensors (up to MB MiB each, default 8) and prints:
# distinct 16-bit values, zero-order entropy (bits a value), entropy of the exponent field and of the 7-bit mantissa, how many low
# mantissa bits are always zero, and the exponent range inside 32-value blocks (bits a fixed block-exponent code would need).
import json, os, re, struct, sys, collections, array, math
d = sys.argv[1]; mb = int(sys.argv[2]) if len(sys.argv) > 2 else 8
def ent(c):
    n = sum(c.values()); return -sum(v / n * math.log2(v / n) for v in c.values())
groups = collections.defaultdict(list)
for root, _, files in os.walk(d):
    for f in sorted(files):
        if not f.endswith(".safetensors"): continue
        p = os.path.join(root, f); fh = open(p, "rb"); (hl,) = struct.unpack("<Q", fh.read(8)); h = json.loads(fh.read(hl))
        for k, v in h.items():
            if k == "__metadata__" or v["dtype"] != "BF16": continue
            b = v["data_offsets"][1] - v["data_offsets"][0]
            if b < 1 << 20: continue
            g = "mtp" if k.startswith("mtp.") or ".mtp." in k else "lm_head" if "lm_head" in k else "embed" if "embed_tokens" in k else "hyper_connection" if "hyper_connection" in k else "router" if ".mlp.gate" in k else "vision" if "visual" in k else "other"
            groups[g].append((k, p, 8 + hl + v["data_offsets"][0], b, v["shape"]))
for g, ts in sorted(groups.items()):
    tot = sum(t[3] for t in ts); pick = ts[:: max(1, len(ts) // 4)][:4]
    print(f"== {g}: {len(ts)} tensors, {tot / 2**30:.3f} GiB")
    for k, p, off, b, shp in pick:
        fh = open(p, "rb"); fh.seek(off + (b // 2 // 4096) * 4096); a = array.array("H"); a.frombytes(fh.read(min(b // 2, mb << 20) // 64 * 64))
        c = collections.Counter(a); ce = collections.Counter(x >> 7 & 0xFF for x in c.elements()) if len(a) < 1 << 22 else None
        ex = collections.Counter(); ma = collections.Counter(); orm = 0
        for x, n in c.items(): ex[x >> 7 & 0xFF] += n; ma[x & 0x7F] += n; orm |= x & 0x7F
        low0 = 0
        while low0 < 7 and not (orm >> low0) & 1: low0 += 1
        # exponent spread inside blocks of 32 values (zeros ignored)
        spread = collections.Counter()
        for i in range(0, min(len(a), 1 << 20), 32):
            e = [x >> 7 & 0xFF for x in a[i:i + 32] if x & 0x7FFF]
            spread[(max(e) - min(e)) if e else 0] += 1
        ns = sum(spread.values()); cum = 0; need = {}
        for s in sorted(spread):
            cum += spread[s]
            for q in (0.5, 0.99, 1.0):
                if q not in need and cum >= q * ns: need[q] = s
        print(f"  {k[-70:]:70s} {str(shp):18s} distinct {len(c):6d}  H16 {ent(c):5.2f} b  Hexp {ent(ex):4.2f}  Hmant {ent(ma):4.2f}  low zero mantissa bits {low0}  zeros {c.get(0, 0) / len(a):.4f}  block-32 exponent spread p50/p99/max {need[0.5]}/{need[0.99]}/{need[1.0]}")
