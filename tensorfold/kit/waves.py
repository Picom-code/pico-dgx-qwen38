#!/usr/bin/env python3
# waves.py N OUT.jsonl [MAXTOK] [STRIDE] [WAVES] : the held prompts in WAVES of N requests that all arrive together (each wave waits for
# the one before). Prints total tok/s over the waves, and the first-token time of the LAST stream of a wave (mean and max over waves).
# Texts are saved for qk.py cmp. Set QK_HELD for another prompt file.
import sys, os, json, threading, time
HERE = os.path.dirname(os.path.abspath(__file__))
NS = {"__file__": os.path.join(HERE, "qk.py"), "__name__": "qk"}; exec(compile(open(NS["__file__"]).read().split("\ndef main():")[0], "qk.py", "exec"), NS)
n = int(sys.argv[1]); out = sys.argv[2]; mt = int(sys.argv[3]) if len(sys.argv) > 3 else 384
stride = int(sys.argv[4]) if len(sys.argv) > 4 else 1; waves = int(sys.argv[5]) if len(sys.argv) > 5 else 0
items = [json.loads(l) for l in open(os.environ.get("QK_HELD", os.path.join(HERE, "prompts.jsonl")))][::stride]
if waves: items = items[: n * waves]
o = open(out, "w"); toks = 0; wall = 0.0; lasts = []; errs = 0; firsts = []
for w0 in range(0, len(items), n):
    grp = list(enumerate(items[w0:w0 + n], w0)); res = [None] * len(grp)
    def run(k, i, x):
        try: res[k] = NS["req"](x["prompt"], mt)
        except Exception as e: res[k] = {"err": str(e)[:200]}
    th = [threading.Thread(target=run, args=(k, i, x)) for k, (i, x) in enumerate(grp)]
    t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]; wall += time.time() - t0
    for (i, x), r in zip(grp, res):
        o.write(json.dumps(dict(id=x.get("id", i), cat=x.get("cat", "x"), **r), ensure_ascii=False) + "\n")
        toks += r.get("toks", 0); errs += "err" in r
    tt = [r["ttft"] for r in res if "ttft" in r]
    if tt: lasts.append(max(tt)); firsts.append(min(tt))
o.close()
print(f"waves of {n}: {len(items)} prompts, {toks} tokens in {wall:.1f} s = {toks / max(wall, 1e-9):.1f} tok/s total; last first-token of a wave mean {sum(lasts) / max(1, len(lasts)):.2f} s max {max(lasts or [0]):.2f} s (first {sum(firsts) / max(1, len(firsts)):.2f} s); errors {errs}", flush=True)
