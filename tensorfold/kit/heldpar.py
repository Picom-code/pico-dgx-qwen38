#!/usr/bin/env python3
# heldpar.py N OUT.jsonl [MAXTOK] [STRIDE] [REPEAT] : the held prompts with N requests in flight (for dumps and many-user checks).
# Prints total tokens / wall. Uses qk.req (same request shape as every other bench).
import sys, os, json, threading, time
HERE = os.path.dirname(os.path.abspath(__file__))
NS = {"__file__": os.path.join(HERE, "qk.py"), "__name__": "qk"}; exec(compile(open(NS["__file__"]).read().split("\ndef main():")[0], "qk.py", "exec"), NS)
n = int(sys.argv[1]); out = sys.argv[2]; mt = int(sys.argv[3]) if len(sys.argv) > 3 else 384
stride = int(sys.argv[4]) if len(sys.argv) > 4 else 1; rep = int(sys.argv[5]) if len(sys.argv) > 5 else 1
items = [json.loads(l) for l in open(os.environ.get("QK_HELD", os.path.join(HERE, "prompts.jsonl")))][::stride] * rep
lock = threading.Lock(); it = iter(enumerate(items)); o = open(out, "w"); tot = [0, 0]
def w():
    while True:
        with lock:
            try: i, x = next(it)
            except StopIteration: return
        try: r = NS["req"](x["prompt"], mt)
        except Exception as e: r = {"err": str(e)[:200]}
        with lock:
            o.write(json.dumps(dict(id=x.get("id", i), cat=x.get("cat", "x"), **r), ensure_ascii=False) + "\n"); o.flush()
            tot[0] += r.get("toks", 0); tot[1] += "err" in r
th = [threading.Thread(target=w) for _ in range(n)]; t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]; wall = time.time() - t0
print(f"heldpar {n} in flight: {len(items)} prompts, {tot[0]} tokens in {wall:.1f} s = {tot[0] / wall:.1f} tok/s total; errors {tot[1]}", flush=True)
