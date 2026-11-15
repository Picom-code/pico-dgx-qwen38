#!/usr/bin/env python3
"""sq.py: seeded SAMPLED ruler for the Qwen3.8 build (builder D, round 9). Stdlib only; uses qk.req (same request shape as every bench).

  sq.py run OUT.jsonl MODE [N=40] [MAXTOK=256] [CAT=chat]
        MODE: s07on  = temperature 0.7, thinking on   (as MiaAI reports sparkDash prose; top_p / top_k = the server's defaults)
              s07off = temperature 0.7, thinking off
              s10on / s10off = temperature 1.0
              g0off  = greedy, thinking off (reference shape)
        The first N held prompts of category CAT (kit/prompts.jsonl), request i sent with "seed": 9000 + id, so two engines
        (levers on / off) get the same keyed draws. Prints tokens / decode time.
  sq.py sd OUT.jsonl [ROUNDS=8] [temp=0.7] [think=1]   sparkDash METHOD prose (32-token warm-up, 256-token stream with ignore_eos), round r seeded 7000 + r
  sq.py cmp A.jsonl B.jsonl      identical texts (content + reasoning), paired speed: totals, mean of per-prompt ratios with a 95 % interval
  sq.py stat A.jsonl             tokens / decode time, mean and sd of per-prompt tok/s
Env: QK_URL, QK_MODEL, SQ_EXTRA (JSON merged into every body, e.g. {"top_k": 20, "top_p": 0.95}).
"""
import sys, os, json, math, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
QK = os.environ.get("SQ_QK", os.path.join(os.path.dirname(HERE), "kit", "qk.py"))
NS = {"__file__": QK, "__name__": "qk"}
exec(compile(open(QK).read().split("\ndef main():")[0], "qk.py", "exec"), NS)
req = NS["req"]
MODES = {"s07on": (0.7, True), "s07off": (0.7, False), "s10on": (1.0, True), "s10off": (1.0, False), "g0off": (0.0, False), "g0on": (0.0, True)}
EXTRA = json.loads(os.environ.get("SQ_EXTRA", "{}"))


def load(p):
    return {r["id"]: r for r in (json.loads(l) for l in open(p)) if "err" not in r}


def tot(rows):
    T = sum(r["toks"] - 1 for r in rows if r["toks"] > 16); S = sum((r["toks"] - 1) / r["tps"] for r in rows if r["toks"] > 16)
    return T, S


def main():
    m = sys.argv[1]
    if m == "run":
        out, mode = sys.argv[2], sys.argv[3]; n = int(sys.argv[4]) if len(sys.argv) > 4 else 40
        mt = int(sys.argv[5]) if len(sys.argv) > 5 else 256; cat = sys.argv[6] if len(sys.argv) > 6 else "chat"
        temp, think = MODES[mode]
        items = [x for x in (json.loads(l) for l in open(os.path.join(os.path.dirname(QK), "prompts.jsonl"))) if x.get("cat") == cat or cat == "all"][:n]
        rows = []
        with open(out, "w") as o:
            for x in items:
                ex = dict(EXTRA); ex["seed"] = 9000 + int(x["id"])
                try: r = req(x["prompt"], mt, temp, think, extra=ex)
                except Exception as e: r = {"err": str(e)[:200]}
                r = dict(id=x["id"], cat=x.get("cat"), mode=mode, **r); rows.append(r)
                o.write(json.dumps(r, ensure_ascii=False) + "\n"); o.flush()
        ok = [r for r in rows if "err" not in r]; T, S = tot(ok)
        v = [r["tps"] for r in ok if r["toks"] > 16]
        print(f"sq {mode}: {T / max(S, 1e-9):.2f} tok/s (tokens/decode-time)  mean-of-prompts {statistics.mean(v):.2f} sd {statistics.pstdev(v):.2f}  n {len(v)}  errors {len(rows) - len(ok)}  ttft median {statistics.median([r['ttft'] for r in ok]) * 1000:.0f} ms", flush=True)
    elif m == "sd":
        out = sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        temp = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7; think = (sys.argv[5] if len(sys.argv) > 5 else "1") == "1"
        v = []
        with open(out, "w") as o:
            for i in range(n):
                ex = dict(EXTRA); ex["seed"] = 7000 + i
                req(NS["SD_PROSE"], 32, temp, think, extra=ex)
                r = req(NS["SD_PROSE"], 256, temp, think, extra=ex, fill=True); r["id"] = i; v.append(r["tps"])
                o.write(json.dumps(r, ensure_ascii=False) + "\n"); o.flush()
        print(f"sq sd prose T{temp} think-{'on' if think else 'off'}: mean {statistics.mean(v):.2f} median {statistics.median(v):.2f} min {min(v):.1f} max {max(v):.1f}  n {n}  {[round(x, 1) for x in v]}", flush=True)
    elif m == "cmp":
        a, b = load(sys.argv[2]), load(sys.argv[3]); ids = sorted(set(a) & set(b))
        same = sum(a[i]["text"] == b[i]["text"] and a[i].get("reasoning", "") == b[i].get("reasoning", "") for i in ids)
        Ta, Sa = tot([a[i] for i in ids]); Tb, Sb = tot([b[i] for i in ids])
        rat = [b[i]["tps"] / a[i]["tps"] for i in ids if a[i]["toks"] > 16 and b[i]["toks"] > 16 and a[i]["tps"] > 0]
        mu = statistics.mean(rat); se = statistics.stdev(rat) / math.sqrt(len(rat)) if len(rat) > 1 else 0
        # ratios restricted to identical texts (a valid pair: the same tokens)
        rs = [b[i]["tps"] / a[i]["tps"] for i in ids if a[i]["text"] == b[i]["text"] and a[i].get("reasoning", "") == b[i].get("reasoning", "") and a[i]["toks"] > 16]
        ms = statistics.mean(rs) if rs else float("nan")
        print(f"cmp: identical {same}/{len(ids)}  A {Ta / Sa:.2f} -> B {Tb / Sb:.2f} tok/s ({(Tb / Sb) / (Ta / Sa) * 100 - 100:+.2f} %)  per-prompt ratio {mu * 100 - 100:+.2f} % +- {1.96 * se * 100:.2f} (95 %)  identical-only {ms * 100 - 100:+.2f} % n {len(rs)}", flush=True)
        if same != len(ids):
            for i in ids:
                if a[i]["text"] != b[i]["text"] or a[i].get("reasoning", "") != b[i].get("reasoning", ""):
                    x, y = a[i].get("reasoning", "") + "|" + a[i]["text"], b[i].get("reasoning", "") + "|" + b[i]["text"]
                    k = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
                    print(f"  differ id {i}: first difference at char {k} of {len(x)} / {len(y)}")
    elif m == "stat":
        a = load(sys.argv[2]); T, S = tot(a.values()); v = [r["tps"] for r in a.values() if r["toks"] > 16]
        print(f"stat: {T / S:.2f} tok/s  mean-of-prompts {statistics.mean(v):.2f} sd {statistics.pstdev(v):.2f} n {len(v)} tokens {T}")


main()
