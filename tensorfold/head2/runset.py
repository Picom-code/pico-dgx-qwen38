#!/usr/bin/env python3
# runset.py SET.jsonl OUT.jsonl : one request in flight, greedy; per-item max_tokens and think (round 9 fitting corpus). Uses qk.req.
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); K = os.environ.get("QKDIR", os.path.expanduser("~/q38/kit-A"))
NS = {"__file__": os.path.join(K, "qk.py"), "__name__": "qk"}; exec(compile(open(NS["__file__"]).read().split("\ndef main():")[0], "qk.py", "exec"), NS)
o = open(sys.argv[2], "w"); T = 0; S = 0.0; E = 0; t0 = time.time()
for l in open(sys.argv[1]):
    x = json.loads(l)
    try: r = NS["req"](x["prompt"], x.get("max_tokens", 384), think=x.get("think", False))
    except Exception as e: r = {"err": str(e)[:200]}; E += 1
    o.write(json.dumps(dict(id=x["id"], cat=x["cat"], **r), ensure_ascii=False) + "\n"); o.flush()
    if r.get("toks", 0) > 16: T += r["toks"] - 1; S += (r["toks"] - 1) / r["tps"]
print(f"runset {sys.argv[1]}: {T} tokens, {T / max(S, 1e-9):.2f} tok/s decode, wall {time.time() - t0:.0f} s, errors {E}", flush=True)
