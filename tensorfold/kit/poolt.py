#!/usr/bin/env python3
# poolt.py SCENARIO... : mixed-size requests against one engine (pool admission test). A scenario is LONGTOK x COUNT + SHORTS, e.g.
# 150000x1+15 = one ~150K-token needle prompt and 15 short code prompts (256 tokens), all sent together. Prints per scenario: errors,
# needle recall of every long request, its first-token time, the short streams' total tok/s and last first token, and the lowest
# MemAvailable seen while it ran.
import sys, os, json, threading, time, random
HERE = os.path.dirname(os.path.abspath(__file__))
NS = {"__file__": os.path.join(HERE, "qk.py"), "__name__": "qk"}; exec(compile(open(NS["__file__"]).read().split("\ndef main():")[0], "qk.py", "exec"), NS)
req, filler, CODE = NS["req"], NS["filler"], NS["CODE"]
FACTS = [("the vault code", "MARLIN-7731"), ("the name of the lead surveyor", "Odalys Brenner"), ("the bridge inspection number", "QX-90214")]
def needle(tk, seed):
    doc = filler(tk, seed); L = len(doc); pos = [int(L * 0.02), int(L * 0.5), int(L * 0.97)]
    for (k, v), p in sorted(zip(FACTS, pos), key=lambda z: -z[1]):
        p = doc.rfind(". ", 0, p) + 2; doc = doc[:p] + f"Note for the record: {k} is {v}. " + doc[p:]
    return f"[{random.random():.8f}]\n" + doc + "\n\nFrom the document above, state: the vault code, the name of the lead surveyor, and the bridge inspection number. Answer in one line."
def avail(): return int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) // 1048576
for sc in sys.argv[1:]:
    a, shorts = sc.split("+"); tk, cnt = a.split("x"); tk, cnt, shorts = int(tk), int(cnt), int(shorts)
    jobs = [("long", needle(tk, 100 + i), 120) for i in range(cnt)] + [("short", f"[{random.random():.8f}] " + CODE[i % len(CODE)], 256) for i in range(shorts)]
    res = [None] * len(jobs); low = [avail()]; stop = []
    def mon():
        while not stop: low[0] = min(low[0], avail()); time.sleep(0.5)
    def run(i):
        try: res[i] = req(jobs[i][1], jobs[i][2], timeout=3600)
        except Exception as e: res[i] = {"err": str(e)[:160]}
    m = threading.Thread(target=mon); m.start()
    th = [threading.Thread(target=run, args=(i,)) for i in range(len(jobs))]; t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]; wall = time.time() - t0
    stop.append(1); m.join()
    L = [r for (k, _, _), r in zip(jobs, res) if k == "long"]; S = [r for (k, _, _), r in zip(jobs, res) if k == "short"]
    errs = [r["err"] for r in res if "err" in r]
    rec = [sum(v in r.get("text", "") for _, v in FACTS) for r in L]
    st = sum(r.get("toks", 0) for r in S); sw = max([r.get("wall", 0) for r in S] or [0])
    print(f"pool {sc}: wall {wall:.1f} s; errors {len(errs)} {errs[:2]}; long prompt_tokens {[r.get('ptoks') for r in L]} recall {rec} ttft {[round(r.get('ttft', -1), 1) for r in L]} s; "
          f"shorts {len(S)}: {st} tokens, {st / max(sw, 1e-9):.1f} tok/s while they ran ({sw:.1f} s), last first token {max([r.get('ttft', 0) for r in S] or [0]):.2f} s; lowest MemAvailable {low[0]} GiB", flush=True)
