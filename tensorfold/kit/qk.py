#!/usr/bin/env python3
"""qk.py: the Qwen3.8 single-Spark measurement kit (stdlib only). Same prompts and settings for every engine.

  qk.py c1 [RUNS=9]            our bench (pico-dgx-ds4 tools/bench/bench.py): 3 chat + 3 code prompts, 512 tok, temp 0, thinking off
  qk.py distinct N | copies N  N users at once (distinct coding prompts, or 3 prompts repeated)
  qk.py sd [ROUNDS=5]          sparkDash decode METHOD re-implemented (32-tok warm-up, 256-tok stream, (n-1)/(t_last-t_first)):
                               prose T0 think-off, prose T0.7 think-on (as MiaAI reports), code T0 think-off
  qk.py sdn N                  sparkDash concurrent wave method, N streams (prose T0.7 think-on, code T0 think-off)
  qk.py held OUT.jsonl [N] [MAXTOK=384]   the 320 held prompts, one user, greedy, thinking off; per-category tok/s; texts saved
  qk.py ident OUT.json [nodraft]         54-prompt identity set (head ~/drafter2/ident.py prompts), greedy; 'nodraft' adds "draft": false
  qk.py cmp A.json B.json                identical count + mean shared-prefix fraction (ident or held files)
  qk.py prefill 8000,32000,128000        cold prefill tok/s = prompt_tokens / TTFT, unique prefix each
  qk.py needle TOKENS                    3 facts (start/middle/end) in a haystack of about TOKENS tokens
  qk.py ttft                             short-prompt TTFT, median of 7
  qk.py img                              one generated PNG (red circle left, blue square right)
  qk.py ctx TOKENS                       does a prompt of about TOKENS tokens get admitted and answered

Env: QK_URL (default http://127.0.0.1:18301/v1/chat/completions), QK_MODEL, QK_KEY, QK_THINK=enable_thinking,thinking (kwarg names),
QK_EXTRA (JSON merged into every body), QK_HELD (prompts.jsonl path).
"""
import base64, json, os, random, statistics, struct, sys, threading, time, urllib.request, zlib

URL = os.environ.get("QK_URL", "http://127.0.0.1:18301/v1/chat/completions")
MODEL = os.environ.get("QK_MODEL", "qwen3.8-flash-next")
KEY = os.environ.get("QK_KEY", "")
THINK = [k for k in os.environ.get("QK_THINK", "enable_thinking,thinking").split(",") if k]
EXTRA = json.loads(os.environ.get("QK_EXTRA", "{}"))
HERE = os.path.dirname(os.path.abspath(__file__))

CHAT = ["Explain how a refrigerator works, in plain language, with a few paragraphs.",
        "Write a short story about a lighthouse keeper who finds a message in a bottle.",
        "Compare the Roman and Han empires: government, economy, military, legacy."]
CODE = ["Write a Python LRU cache class with get/put, type hints, docstrings and unit tests.",
        "Write a Rust function that parses a CSV line with quoted fields, plus tests.",
        "Implement Dijkstra's algorithm in TypeScript with a binary heap, with comments.",
        "Write a Go HTTP server with graceful shutdown, a /health route and middleware for request logging.",
        "Implement a thread-safe bounded blocking queue in Java with put/take and timeouts.",
        "Write a C function that reverses a singly linked list in place, plus a small test harness.",
        "Write a SQL schema for a library system (books, members, loans) with indexes and three example queries.",
        "Implement a trie in C++ with insert, search, prefix count and delete.",
        "Write a bash script that rotates log files older than 7 days and compresses them with gzip.",
        "Write a React component in TSX for a paginated, sortable table with a search box.",
        "Implement the A* pathfinding algorithm in Python on a 2D grid with obstacles.",
        "Write a Kotlin data class and repository for a todo app using coroutines and Flow.",
        "Implement a token-bucket rate limiter in Rust that is safe across threads.",
        "Write a Python asyncio web scraper that fetches 50 URLs with a concurrency limit of 5 and retries.",
        "Implement a Swift struct for a matrix with addition, multiplication and transpose, plus tests.",
        "Write a Haskell function for run-length encoding and decoding of strings with QuickCheck properties."]
# sparkDash 1.8.8 src/shared/llmPrompts.js, copied verbatim: DECODE_PROSE_PROMPT, DECODE_STRUCTURED_PROMPT, CODE_TASK_TAIL, first DECODE_CODE_TASKS
SD_PROSE = "Write a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity. Be thorough."
SD_STRUCT = "Count from 1 to 200. Output only the numbers, separated by spaces. No other text."
SD_CODE_TAIL = "Output only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this needs. Keep writing code."
SD_TASKS = [("binary_search", "def binary_search(nums, target) -> int: index of target in a sorted list, or -1."),
            ("merge_sort", "def merge_sort(nums) -> list: stable sort of a list of ints, returning a new list."),
            ("lru_cache", "class LRUCache: get(key) and put(key, value) with a fixed capacity, evicting the least recently used."),
            ("token_bucket", "class TokenBucket: allow(n) consumes n tokens refilled at a fixed rate, else returns False."),
            ("ring_buffer", "class RingBuffer: push and pop over a fixed-capacity array, raising on overflow and underflow."),
            ("dijkstra", "def dijkstra(graph, src) -> dict: shortest path weights from src on a non-negative weighted graph."),
            ("edit_distance", "def edit_distance(a, b) -> int: Levenshtein distance between two strings."),
            ("semver_cmp", "def semver_cmp(a, b) -> int: compare dotted numeric versions, negative if a < b.")]
SD_CODE = [f"{n}\n{sp}\n{SD_CODE_TAIL}" for n, sp in SD_TASKS]

IDENT = {"chat": [
 "Explain how a refrigerator works to a 12 year old.",
 "Write a short story about a lighthouse keeper who finds a message in a bottle.",
 "What are the pros and cons of living in Austin, Texas?",
 "Why did the Roman Empire fall? Give the main theories.",
 "Write a heartfelt toast for my best friend's wedding.",
 "Explain the difference between stocks, bonds and index funds.",
 "How do I become a better public speaker? Practical tips.",
 "Compare electric cars and gas cars for a daily commuter.",
 "Write an email to my landlord asking to fix the AC, polite but firm.",
 "Explain what inflation is and why central banks raise interest rates.",
 "Describe a perfect autumn day in Vermont.",
 "What should I pack for a week-long hiking trip in the Alps?",
 "Summarize the causes and effects of the French Revolution.",
 "Give me advice on negotiating a higher salary.",
 "Write a poem about the ocean at night.",
 "Explain photosynthesis step by step.",
 "What is the difference between a virus and a bacterium?",
 "Plan a three-day itinerary for Tokyo for a first-time visitor.",
 "Why is sleep important? Explain the science.",
 "Write a product description for a handmade leather wallet."],
 "code": [
 "Write a Python LRU cache class with get/put, type hints, docstrings and unit tests.",
 "Write a Rust function that parses a CSV line with quoted fields, plus tests.",
 "Implement Dijkstra's algorithm in TypeScript with a binary heap, with comments.",
 "Implement a thread-safe LRU cache in Go with tests.",
 "Write a React component for a todo list with add, delete and toggle, using hooks.",
 "Write a bash script that backs up a directory to a timestamped tar.gz and keeps only the last 7.",
 "Write a C function that reverses a singly linked list, with a test main.",
 "Write a SQL schema for a library system with books, members and loans, plus three example queries.",
 "Implement a trie in Java with insert, search and startsWith.",
 "Write a Python asyncio web crawler limited to 10 concurrent requests.",
 "Write a Haskell function for merge sort with type signatures.",
 "Write a C++ matrix class with addition and multiplication operators."],
 "math": [
 "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. Walk through the reasoning carefully.",
 "Prove that the square root of 2 is irrational.",
 "If a train leaves at 3pm going 60 mph and another at 4pm going 80 mph on the same track, when does the second catch up? Explain.",
 "How many weekdays are there between March 1 and April 15 of a non-leap year? Show your work.",
 "Solve x^2 - 5x + 6 = 0 and explain each step.",
 "What is the sum of the first 100 positive integers? Explain Gauss's method.",
 "Compute the derivative of x^3 * sin(x) and explain the product rule.",
 "A rectangle has perimeter 40 and area 96. Find its sides.",
 "Explain Bayes' theorem with a medical test example and numbers.",
 "What is 17*23? Just the number."],
 "zh": ["用三段话解释什么是机器学习。", "写一首关于春天的短诗。", "介绍一下北京的历史。"],
 "json": ["Return a JSON array of 8 fictional users with id, name, email, age and city. Only JSON.",
          "Convert this to YAML: {\"server\": {\"port\": 8080, \"hosts\": [\"a\", \"b\"], \"tls\": true}}"],
}
_base = " ".join(f"Section {i}: the committee reviewed item {i*7%113} and noted figure {i*13%997}." for i in range(6000))
IDENT_LONG = [("long:8k", _base[:40000] + "\n\nThe secret code is PELICAN-42.\n\n" + _base[40000:48000] + "\n\nWhat is the secret code, and summarise the document in 5 bullet points."),
        ("long:30k", _base[:150000] + "\n\nWhat figure did section 4321 note? Then explain what a committee review usually involves."),
        ("long:code", "Here is a log file:\n" + "\n".join(f"2026-11-0{i%9+1} 12:{i%60:02d} INFO worker {i%17} processed job {i} in {i*37%900} ms" for i in range(1500)) + "\n\nWrite a Python script that parses this log format and reports p50/p95 latency per worker."),
        ("long:qa", _base[:80000] + "\n\nList three section numbers that noted figure 0, if any, and explain how you searched.")]


def req(content, max_tokens=512, temp=0.0, think=False, extra=None, timeout=1800, fill=False):
    """One streamed request. Returns dict(text, reasoning, toks, tps, ttft, ptoks, wall, finish)."""
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": temp, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {k: bool(think) for k in THINK}}
    if temp == 0: body["top_p"] = 1
    if fill: body.update(min_tokens=max_tokens, ignore_eos=True, stop=[])
    body.update(EXTRA); body.update(extra or {})
    head = {"Content-Type": "application/json"}
    if KEY: head["Authorization"] = "Bearer " + KEY
    r = urllib.request.Request(URL, json.dumps(body).encode(), head)
    t0 = time.time(); first = last = None; txt = []; rsn = []; usage = None; n = 0; fin = None
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        for line in resp:
            line = line.strip()
            if not line.startswith(b"data:") or line.endswith(b"[DONE]"): continue
            try: d = json.loads(line[5:])
            except ValueError: continue
            usage = d.get("usage") or usage
            for c in d.get("choices", []):
                de = c.get("delta", {}); fin = c.get("finish_reason") or fin
                a, b = de.get("content") or "", de.get("reasoning_content") or de.get("reasoning") or ""
                if a or b:
                    last = time.time(); first = first or last; n += 1
                    if a: txt.append(a)
                    if b: rsn.append(b)
    t1 = time.time(); toks = (usage or {}).get("completion_tokens") or n
    return {"text": "".join(txt), "reasoning": "".join(rsn), "toks": toks, "ptoks": (usage or {}).get("prompt_tokens"),
            "tps": round((toks - 1) / max(1e-6, (last or t1) - (first or t0)), 2), "ttft": round((first or t1) - t0, 4),
            "wall": round(t1 - t0, 3), "finish": fin}


def rnd(p): return f"[{random.random():.8f}] " + p


def png_shapes(w=256, h=128):
    rows = []
    for y in range(h):
        row = bytearray([0])
        for x in range(w):
            c = (255, 255, 255)
            if (x - 64) ** 2 + (y - 64) ** 2 < 40 ** 2: c = (220, 20, 20)
            if 152 <= x < 232 and 24 <= y < 104: c = (20, 40, 220)
            row += bytes(c)
        rows.append(bytes(row))
    def ch(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(b"".join(rows))) + ch(b"IEND", b"")


def filler(tokens, seed):
    """Real-ish English text of about `tokens` tokens (about 4.6 chars per token for this generator), unique per seed."""
    rng = random.Random(seed)
    W = ("the committee reviewed several proposals about river management and decided that further study was needed before "
         "any funding could be released while engineers measured flow rates along northern valley farms where irrigation "
         "channels had silted during winter storms and local councils argued over maintenance budgets schedules contractors "
         "equipment surveys reports deadlines permits inspections bridges culverts pumps gates sensors records maps").split()
    out = []; n = 0
    while n < tokens * 4.6:
        s = " ".join(rng.choice(W) for _ in range(rng.randint(8, 18))).capitalize() + f" ({rng.randint(100, 99999)}). "
        out.append(s); n += len(s)
    return "".join(out)


def main():
    mode = sys.argv[1]
    if mode == "cmp":
        def load(p):
            if p.endswith(".jsonl"): return {str(d["id"]): d for d in map(json.loads, open(p)) if "text" in d}
            return json.load(open(p))["out"]
        a, b = load(sys.argv[2]), load(sys.argv[3]); ks = [k for k in a if k in b]
        same = [k for k in ks if a[k]["text"] == b[k]["text"]]; fr = []
        for k in ks:
            x, y = a[k]["text"], b[k]["text"]; i = 0; m = min(len(x), len(y))
            while i < m and x[i] == y[i]: i += 1
            fr.append(i / max(1, max(len(x), len(y))))
        print(f"identical {len(same)}/{len(ks)}  mean shared-prefix fraction {statistics.mean(fr):.3f}  differ: {[k for k in ks if k not in same][:12]}")
        return
    req("hi", 8)
    if mode == "c1":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 9
        for kind, P in (("chat", CHAT), ("code", CODE[:3])):
            v = [req(rnd(P[i % 3]))["tps"] for i in range(n)]
            print(f"{kind} 1 user: median {statistics.median(v):.1f} tok/s  mean {statistics.mean(v):.1f}  {v}", flush=True)
    elif mode in ("distinct", "copies"):
        n = int(sys.argv[2]); P = CODE if mode == "distinct" else CODE[:3]; res = [None] * n
        def w(i):
            try: res[i] = req(rnd(P[i % len(P)]))
            except Exception as e: res[i] = {"toks": 0, "err": str(e)[:100], "ttft": 0}
        th = [threading.Thread(target=w, args=(i,)) for i in range(n)]
        t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]; wall = time.time() - t0
        tot = sum(r["toks"] for r in res); errs = [r["err"] for r in res if "err" in r]
        print(f"{mode} {n} users: {tot / wall:.1f} tok/s total  ({tot} tokens in {wall:.1f} s; max ttft {max(r['ttft'] for r in res):.2f} s; errors {len(errs)} {errs[:2]})", flush=True)
    elif mode == "sd":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        def one(p, temp, think):
            req(p, 32, temp, think); return req(p, 256, temp, think, fill=True)["tps"]
        # c1 uses the first code task (DECODE_CODE_PROMPT), like sparkDash; "codeN" cycles the first 8 tasks as an extra
        for name, ps, temp, think in (("prose T0 think-off", [SD_PROSE], 0, False), ("prose T0.7 think-on", [SD_PROSE], 0.7, True),
                                      ("code T0 think-off (task 0)", SD_CODE[:1], 0, False), ("code8 T0 think-off (tasks 0-7)", SD_CODE, 0, False),
                                      ("structured T0 think-off", [SD_STRUCT], 0, False)):
            v = [one(ps[i % len(ps)], temp, think) for i in range(n if len(ps) == 1 else len(ps))]
            print(f"sd-method {name}: median {statistics.median(v):.1f}  mean {statistics.mean(v):.1f}  {v}", flush=True)
    elif mode == "sdn":
        # sparkDash concurrent wave: N streams at once, distinct code tasks or "(stream i/n)" prose; aggregate = sum of per-request decode speeds
        n = int(sys.argv[2])
        for name, ps, temp, think in (("prose T0.7 think-on", [SD_PROSE + (f" (stream {i + 1}/{n})" if n > 1 else "") for i in range(n)], 0.7, True),
                                      ("code T0 think-off", [SD_CODE[i % len(SD_CODE)] for i in range(n)], 0, False)):
            res = [None] * n
            def w(i):
                try: res[i] = req(ps[i], 256, temp, think, fill=True)
                except Exception as e: res[i] = {"tps": 0, "ttft": 0, "err": str(e)[:80]}
            th = [threading.Thread(target=w, args=(i,)) for i in range(n)]; [t.start() for t in th]; [t.join() for t in th]
            print(f"sd-method wave {n} {name}: aggregate {sum(r['tps'] for r in res):.1f}  per-request {statistics.mean(r['tps'] for r in res):.1f}  ttft {statistics.median(r['ttft'] for r in res) * 1000:.0f} ms  errors {sum('err' in r for r in res)}", flush=True)
    elif mode == "held":
        out = sys.argv[2]; lim = int(sys.argv[3]) if len(sys.argv) > 3 else 0; mt = int(sys.argv[4]) if len(sys.argv) > 4 else 384
        src = os.environ.get("QK_HELD", os.path.join(HERE, "prompts.jsonl")); o = open(out, "w"); agg = {}
        for n, l in enumerate(open(src)):
            if lim and n >= lim: break
            if n % int(os.environ.get("QK_HELD_STRIDE", "1")): continue
            it = json.loads(l)
            try: r = req(it["prompt"], mt)
            except Exception as e: r = {"err": str(e)[:200]}
            o.write(json.dumps(dict(id=it.get("id", n), cat=it.get("cat", "x"), **r), ensure_ascii=False) + "\n"); o.flush()
            if "tps" in r and r["toks"] > 16:
                a = agg.setdefault(it.get("cat", "x"), [0, 0.0, []]); a[0] += r["toks"] - 1; a[1] += (r["toks"] - 1) / r["tps"]; a[2].append(r["tps"])
        for c, (t, s, v) in sorted(agg.items()):
            print(f"held {c}: {t / s:.2f} tok/s (tokens/decode-time)  mean-of-prompts {statistics.mean(v):.2f}  n {len(v)}", flush=True)
        T = sum(a[0] for a in agg.values()); S = sum(a[1] for a in agg.values())
        print(f"held ALL: {T / S:.2f} tok/s  n {sum(len(a[2]) for a in agg.values())}", flush=True)
    elif mode == "ident":
        out = sys.argv[2]; ex = {"draft": False} if len(sys.argv) > 3 and sys.argv[3] == "nodraft" else None; res = {}
        items = [(f"{c}:{i}", m) for c, L in IDENT.items() for i, m in enumerate(L)] + IDENT_LONG
        for k, m in items:
            try: res[k] = req(m, 256, extra=ex)
            except Exception as e: res[k] = {"text": "ERR " + str(e)[:100], "toks": 0, "tps": 0}
            print(k, res[k]["toks"], res[k]["tps"], flush=True)
        json.dump({"url": URL, "ts": time.strftime("%FT%TZ", time.gmtime()), "nodraft": bool(ex), "out": res}, open(out, "w"), ensure_ascii=False)
        v = [r["tps"] for k, r in res.items() if r["toks"] > 16]
        print(f"ident saved {out} n {len(res)} median tps {statistics.median(v):.1f}")
    elif mode == "prefill":
        for tk in [int(x) for x in sys.argv[2].split(",")]:
            p = f"[prefill {random.random():.10f}] Ignore the text below. Reply with the single word OK.\n" + filler(tk, random.random()) + "\nReply OK."
            try:
                r = req(p, 8)
                print(f"prefill target {tk}: prompt_tokens {r['ptoks']}  ttft {r['ttft']:.2f} s  {r['ptoks'] / r['ttft']:.0f} tok/s", flush=True)
            except Exception as e: print(f"prefill target {tk}: FAILED {str(e)[:200]}", flush=True)
    elif mode in ("needle", "ctx"):
        tk = int(sys.argv[2]); doc = filler(tk, 7); L = len(doc)
        facts = [("the vault code", "MARLIN-7731"), ("the name of the lead surveyor", "Odalys Brenner"), ("the bridge inspection number", "QX-90214")]
        pos = [int(L * 0.02), int(L * 0.5), int(L * 0.97)]
        for (k, v), p in sorted(zip(facts, pos), key=lambda z: -z[1]):
            p = doc.rfind(". ", 0, p) + 2; doc = doc[:p] + f"Note for the record: {k} is {v}. " + doc[p:]
        q = "\n\nFrom the document above, state: the vault code, the name of the lead surveyor, and the bridge inspection number. Answer in one line."
        try:
            r = req(f"[{random.random():.8f}]\n" + doc + q, 120, timeout=3600)
            hit = sum(v in r["text"] for _, v in facts)
            print(f"{mode} {tk}: prompt_tokens {r['ptoks']}  recall {hit}/3  ttft {r['ttft']:.1f} s  prefill {r['ptoks'] / r['ttft']:.0f} tok/s  answer {r['text'][:160]!r}", flush=True)
        except Exception as e: print(f"{mode} {tk}: FAILED {str(e)[:300]}", flush=True)
    elif mode == "ttft":
        v = [req(rnd("Say hello in five words."), 16)["ttft"] for _ in range(7)]
        print(f"ttft short prompt: median {statistics.median(v) * 1000:.0f} ms  {[round(x * 1000) for x in v]}", flush=True)
    elif mode == "img":
        url = "data:image/png;base64," + base64.b64encode(png_shapes()).decode()
        c = [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "What two shapes are in this image, and what colour is each? One short sentence."}]
        try:
            r = req(c, 80); t = r["text"].lower(); ok = "red" in t and "blue" in t and "circle" in t and "square" in t
            print(f"img: {'PASS' if ok else 'FAIL'} ttft {r['ttft']:.2f} s  {r['text'][:160]!r}", flush=True)
        except Exception as e: print(f"img: FAILED {str(e)[:200]}", flush=True)


main()
