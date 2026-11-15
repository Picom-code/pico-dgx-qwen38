#!/usr/bin/env python3
# lanehist.py FILE : rows-a-lane histogram of decode passes in an XSTAT dump (layer 0 blocks of the 48-layer model). stdlib only.
import sys, struct, collections
f = open(sys.argv[1], "rb"); h = collections.Counter(); hp = collections.Counter(); passes = 0; capped = 0
while True:
    b = f.read(24)
    if len(b) < 24: break
    magic, L, layer, n, slots, nl = struct.unpack("<6I", b)
    lanes = struct.unpack("<%dI" % nl, f.read(4 * nl)) if nl else ()
    f.seek(8 * n * slots, 1)
    if L == 48 and layer == 0 and nl >= 13:
        passes += 1; hp[n // 16 * 16] += 1
        for r in lanes: h[r] += 1
tot = sum(h.values())
print("passes with 13-16 lanes:", passes, " lanes:", tot)
print("rows a lane (1 = no draft):", " ".join("%d:%.1f%%" % (k, 100 * v / tot) for k, v in sorted(h.items())))
print("mean rows a lane %.2f" % (sum(k * v for k, v in h.items()) / tot))
print("rows a pass (16-wide bins):", " ".join("%d-%d:%.1f%%" % (k, k + 15, 100 * v / passes) for k, v in sorted(hp.items())))
