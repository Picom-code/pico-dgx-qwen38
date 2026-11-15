#!/usr/bin/env python3
"""nu.py N OUT.jsonl [RUNS=3] : N users at once, a DIFFERENT coding prompt each (qk.py's `distinct` prompts and request shape: 512 tokens,
temperature 0, thinking off), RUNS times. Per run: total decode tok/s (all completion tokens / wall, as qk.py distinct), the FIRST and the
LAST user's first-token time. Prints one line with the median run and every run; appends every stream of every run to OUT.jsonl."""
import sys, os, json, threading, time, statistics
HERE = os.path.dirname(os.path.abspath(__file__)); QK = os.environ.get("NU_QK", os.path.join(os.path.dirname(HERE), "kit-A", "qk.py"))
NS = {"__file__": QK, "__name__": "qk"}; exec(compile(open(QK).read().split("\ndef main():")[0], "qk.py", "exec"), NS)
n = int(sys.argv[1]); out = sys.argv[2]; runs = int(sys.argv[3]) if len(sys.argv) > 3 else 3; P = NS["CODE"]; rows = []
NS["req"]("hi", 8)
for k in range(runs):
    res = [None] * n
    def w(i):
        try: res[i] = NS["req"](NS["rnd"](P[i % len(P)]))
        except Exception as e: res[i] = {"toks": 0, "err": str(e)[:100], "ttft": 0, "text": ""}
    th = [threading.Thread(target=w, args=(i,)) for i in range(n)]; t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]; wall = time.time() - t0
    tot = sum(r["toks"] for r in res); tt = [r["ttft"] for r in res if "err" not in r] or [0]
    rows.append(dict(n=n, run=k, tps=round(tot / wall, 1), toks=tot, wall=round(wall, 2), first=round(min(tt), 3), last=round(max(tt), 3), errors=sum("err" in r for r in res)))
    with open(out, "a") as o: o.write(json.dumps(dict(rows[-1], streams=[{x: r.get(x) for x in ("toks", "tps", "ttft", "wall", "err")} for r in res])) + "\n")
    time.sleep(1)
m = sorted(rows, key=lambda r: r["tps"])[len(rows) // 2]
print(f"users {n}: median {m['tps']} tok/s total, first / last first-token {m['first']:.2f} / {m['last']:.2f} s; runs {[r['tps'] for r in rows]} last-ttft {[r['last'] for r in rows]} first-ttft {[r['first'] for r in rows]} errors {sum(r['errors'] for r in rows)}", flush=True)
