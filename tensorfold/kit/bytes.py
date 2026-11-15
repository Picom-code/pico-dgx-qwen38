#!/usr/bin/env python3
# bytes.py DIR : bytes by tensor group from the safetensors shard headers of the official checkpoint (no weights read).
import json, os, re, struct, sys, collections
d = sys.argv[1]; g = collections.Counter(); ex = collections.defaultdict(list); n = 0; lay0 = collections.Counter(); dt = collections.defaultdict(collections.Counter)
for root, _, files in os.walk(d):
    for f in sorted(files):
        if not f.endswith(".safetensors"): continue
        fh = open(os.path.join(root, f), "rb"); (hl,) = struct.unpack("<Q", fh.read(8)); h = json.loads(fh.read(hl))
        for k, v in h.items():
            if k == "__metadata__": continue
            b = v["data_offsets"][1] - v["data_offsets"][0]; n += 1
            m = re.search(r"layers\.(\d+)\.", k); L = int(m.group(1)) if m else -1
            if k.startswith("mtp.") or ".mtp." in k: grp = "mtp." + ("experts" if ".mlp.experts" in k else "lm/emb" if "embed" in k or "head" in k else "dense")
            elif "visual" in k: grp = "vision"
            elif "lm_head" in k: grp = "lm_head"
            elif "embed_tokens" in k: grp = "embed_tokens"
            elif ".ple." in k or "ngram" in k: grp = "ple_ngram"
            elif ".mlp.experts" in k: grp = "experts"
            elif ".mlp.shared_expert" in k: grp = "shared_expert"
            elif ".mlp.gate" in k: grp = "router_gate"
            elif "linear_attn" in k: grp = "gdn_linear_attn"
            elif "self_attn" in k: grp = "qsa_self_attn"
            elif "hyper_connection" in k: grp = "hyper_connection"
            else: grp = "other"
            g[grp] += b; dt[grp][v["dtype"]] += b
            if L in (0, 3): lay0[(L, grp, re.sub(r".*layers\.\d+\.", "", k)[:60], v["dtype"], tuple(v["shape"]))] += b
tot = sum(g.values()); print(f"tensors {n} total {tot/2**30:.2f} GiB")
for k, v in g.most_common(): print(f"  {k:18s} {v/2**30:8.3f} GiB  {dict((a, round(b/2**30,3)) for a,b in dt[k].items())}")
print("layer 0 and 3 tensors (bytes):")
for (L, grp, name, dty, shp), b in sorted(lay0.items()): print(f"  L{L} {grp:16s} {name:60s} {dty:6s} {str(shp):24s} {b}")
