#!/usr/bin/env python3
"""rt.py: exactness fuzzer + soak for an OpenAI-compatible engine (red team, builder D round 12). Python stdlib only.

The claim under test: with the recipe's levers ON every request returns what the SAME engine returns with the levers OFF
(greedy: identical; sampled: identical per seed). Method: record a reference file on the levers-off engine, replay on the
levers-on engine (solo, with N in flight, streamed, resumed from the prompt cache, with predictions) and compare every field.

  rt.py gen CASES.jsonl [SEED]             build the matrix (asks the running engine for token counts: exact boundary prompts)
  rt.py run CASES OUT.jsonl LABEL [k=v..]  run cases; options: mode=n|s|both conc=N order=fwd|rev|shuf sel=TAG,TAG not=TAG,TAG
                                           maxp=PTOKS limit=N knob="words" dl=EPOCH arrive=burst|stag decoys=N slow=PCT rep=N
  rt.py gen2 CASES REF.jsonl OUT.jsonl     prediction cases built from the reference replies (exact / wrong / partly / longer / shorter)
  rt.py cmp REF.jsonl RUN.jsonl [DIFFS]    table by tag: tested / identical / different / errors; failing rows to DIFFS
  rt.py self RUN.jsonl                     inside one run file: streamed == non-streamed, repeats equal
  rt.py soak CASES REF OUT.jsonl MIN [k=v] mixed load replayed from the matrix at 1-8 in flight, every reply checked against REF;
                                           memory log every 30 s to OUT.mem; stops itself when MemAvailable < low=9 (GiB)
  rt.py bisect CASES REF ID [mode]         one failing case with each lever switched off through the knob file (engine must have the knob file)
  rt.py one CASES ID [mode] [knob words]   run one case, print the canonical reply
Night 11-14 additions (the reference is the SAME running engine with every knob lever off, switched through the knob file):
  rt.py pair CASES OUT.jsonl [k=v..]       every case under each variant back to back; vars=off,on,off2 (default). A variant name:
                                           off* = knob levers off (+ the `prediction` field removed), on* = levers on, *_s = streamed,
                                           *_c = preceded by the same request streamed and cancelled by the client; rot=1 rotates the order.
  rt.py pairsum OUT.jsonl CASES[,CASES]    the table of a pair / sess file: on vs off, off2 vs off (base determinism), on_s vs off_s ...
  rt.py sess SCRIPTS OUT.jsonl [k=v..]     live multi-turn sessions (each reply goes back as the assistant turn; edit / regen steps) under each
                                           variant; writes OUT.cases (every turn's request as a static case) for a cold replay on a fresh engine
  rt.py gen3 PROMPTS FRESH OUT SESS [SEED] extra static cases (no engine needed) + session scripts
  rt.py cmp REF RUN DIFFS CASES LABEL      LABEL = take only the reference rows with this label
Environment: RT_URL (default http://127.0.0.1:18301), RT_MODEL, RT_KNOBS (knob file path).
"""
import sys, os, json, time, random, hashlib, threading, http.client, urllib.parse, base64, zlib, struct, queue, subprocess

URL = os.environ.get("RT_URL", "http://127.0.0.1:18301")
MODEL = os.environ.get("RT_MODEL", "Qwen3.8-Flash-Next")
KNOBS = os.environ.get("RT_KNOBS", os.path.expanduser("~/q38/prof-D/knobs"))
PATHS = {"chat": "/v1/chat/completions", "resp": "/v1/responses", "msg": "/v1/messages"}
LEVERS_OFF = ["prefix_wait_ms=0", "cert=0", "cert_topk=0", "cert_gather=0", "hcf=0", "half_below=0", "mtp_last=0", "gdn_many=0", "best_lanes=0", "attn_lanes=0",
              "late_skip=0", "keep_inpass=0", "lane_reuse=0", "head2=0", "head2_sampled=0", "deep_off=1", "stop2=0", "ngram_split=0",
              "qsa_cert=0", "qsa_list=0", "qsa_rows=0", "qsa_cert_min=1000000", "exp_half=0"]
# knob file limits in the engine: 32 words, 512 bytes (qmmf.zig knobFile); prefix_wait_ms must sit in the first 256 bytes.
assert len(LEVERS_OFF) <= 32 and len(" ".join(LEVERS_OFF)) < 500

# ---------------------------------------------------------------- HTTP

def post(path, body, stream=False, slow=0.0, cancel_after=None, t_first=1500, t_gap=60):
    """Returns dict(st, raw (json) | events [json], err, ttft, wall, gap (max seconds between stream events), hang)."""
    u = urllib.parse.urlparse(URL); t0 = time.time(); out = {"st": 0}
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=t_first)
    try:
        data = json.dumps(body).encode(); c.connect(); sk = c.sock
        c.request("POST", path, data, {"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        r = c.getresponse(); out["st"] = r.status
        if not stream or r.status != 200:
            raw = r.read(); out["wall"] = round(time.time() - t0, 3)
            try: out["raw"] = json.loads(raw)
            except ValueError: out["err"] = raw[:300].decode("utf-8", "replace")
            return out
        ev = []; last = None; gap = 0.0; n = 0
        while True:
            line = r.fp.readline()
            if not line: break
            line = line.strip()
            if not line.startswith(b"data:"): continue
            p = line[5:].strip()
            if p == b"[DONE]": break
            try: d = json.loads(p)
            except ValueError: continue
            now = time.time()
            if last is None:
                out["ttft"] = round(now - t0, 3); sk.settimeout(t_gap)
            else: gap = max(gap, now - last)
            last = now; ev.append(d); n += 1
            if cancel_after is not None and n >= cancel_after:
                out["cancelled"] = True; break
            if slow: time.sleep(slow)
        out["events"] = ev; out["gap"] = round(gap, 2); out["wall"] = round(time.time() - t0, 3)
    except Exception as e:
        out["err"] = type(e).__name__ + ": " + str(e)[:200]; out["wall"] = round(time.time() - t0, 3)
        if "timed out" in str(e): out["hang"] = True
    finally:
        try: c.close()
        except Exception: pass
    return out


def canon_args(s):
    if isinstance(s, (dict, list)): return json.dumps(s, sort_keys=True, ensure_ascii=False)
    try: return json.dumps(json.loads(s), sort_keys=True, ensure_ascii=False)
    except Exception: return s


def lp_pack(items):
    """logprob entries -> compact list [token, logprob, [[tok, lp]...]]"""
    o = []
    for it in items or []:
        o.append([it.get("token"), it.get("logprob"), [[x.get("token"), x.get("logprob")] for x in it.get("top_logprobs") or []]])
    return o


def canon(route, res, stream):
    """Canonical reply: text, reasoning, tools, finish, ptoks, ctoks, lp, plus stream-only checks."""
    o = {"st": res.get("st")}
    for k in ("ttft", "wall", "gap", "hang", "cancelled"):
        if k in res: o[k] = res[k]
    if res.get("st") != 200 or "err" in res:
        e = res.get("err"); raw = res.get("raw")
        if raw is not None:
            ee = raw.get("error") if isinstance(raw, dict) else None
            e = (ee.get("message") if isinstance(ee, dict) else ee) or json.dumps(raw)[:300]
        o["err"] = (e or "")[:300]; return o
    text = ""; rsn = ""; tools = []; fin = None; pt = ct = None; lp = []
    if route == "chat":
        if not stream:
            d = res["raw"]; ch = (d.get("choices") or [{}])[0]; m = ch.get("message") or {}
            text = m.get("content") or ""; rsn = m.get("reasoning_content") or m.get("reasoning") or ""
            tools = [[t["function"].get("name"), canon_args(t["function"].get("arguments"))] for t in m.get("tool_calls") or []]
            fin = ch.get("finish_reason"); us = d.get("usage") or {}; lp = lp_pack((ch.get("logprobs") or {}).get("content"))
            alts = [[(x.get("message") or {}).get("content"), x.get("finish_reason")] for x in (d.get("choices") or [])[1:]]
            if alts: o["alts"] = alts
        else:
            tc = {}; us = {}
            for d in res["events"]:
                us = d.get("usage") or us
                if isinstance(d.get("error"), (dict, str)): o["err"] = json.dumps(d["error"])[:300]
                for ch in d.get("choices") or []:
                    de = ch.get("delta") or {}; fin = ch.get("finish_reason") or fin
                    text += de.get("content") or ""; rsn += de.get("reasoning_content") or de.get("reasoning") or ""
                    for t in de.get("tool_calls") or []:
                        e = tc.setdefault(t.get("index", 0), ["", ""]); f = t.get("function") or {}
                        if f.get("name"): e[0] = f["name"]
                        e[1] += f.get("arguments") or ""
                    lp += lp_pack((ch.get("logprobs") or {}).get("content"))
            tools = [[tc[k][0], canon_args(tc[k][1])] for k in sorted(tc)]
        pt, ct = us.get("prompt_tokens"), us.get("completion_tokens")
    elif route == "resp":
        d = res.get("raw"); stext = None
        if stream:
            stext = ""; d = None
            for e in res["events"]:
                t = e.get("type", "")
                if t == "response.output_text.delta": stext += e.get("delta") or ""
                if t in ("response.completed", "response.incomplete", "response.failed"): d = e.get("response")
                if t == "error": o["err"] = json.dumps(e)[:300]
        if d is None: o["err"] = o.get("err") or "no final response event"; d = {}
        for it in d.get("output") or []:
            if it.get("type") == "message":
                for cc in it.get("content") or []:
                    if cc.get("type") == "output_text": text += cc.get("text") or ""
            elif it.get("type") == "reasoning":
                for cc in (it.get("content") or []) + (it.get("summary") or []): rsn += cc.get("text") or ""
            elif it.get("type") == "function_call": tools.append([it.get("name"), canon_args(it.get("arguments"))])
        fin = d.get("status"); inc = d.get("incomplete_details") or {}
        if inc.get("reason"): fin = str(fin) + ":" + inc["reason"]
        us = d.get("usage") or {}; pt, ct = us.get("input_tokens"), us.get("output_tokens")
        if stext is not None and stext != text: o["stream_concat_differs"] = True
    else:
        us = {}
        if not stream:
            d = res["raw"]
            for b in d.get("content") or []:
                if b.get("type") == "text": text += b.get("text") or ""
                elif b.get("type") == "thinking": rsn += b.get("thinking") or ""
                elif b.get("type") == "tool_use": tools.append([b.get("name"), canon_args(b.get("input"))])
            fin = d.get("stop_reason"); us = d.get("usage") or {}
        else:
            cur = None
            for e in res["events"]:
                t = e.get("type")
                if t == "message_start": us.update((e.get("message") or {}).get("usage") or {})
                elif t == "content_block_start":
                    b = e.get("content_block") or {}
                    if b.get("type") == "tool_use": cur = [b.get("name"), ""]; tools.append(cur)
                    else: cur = None
                elif t == "content_block_delta":
                    de = e.get("delta") or {}
                    if de.get("type") == "text_delta": text += de.get("text") or ""
                    elif de.get("type") == "thinking_delta": rsn += de.get("thinking") or ""
                    elif de.get("type") == "input_json_delta" and cur is not None: cur[1] += de.get("partial_json") or ""
                elif t == "message_delta":
                    fin = (e.get("delta") or {}).get("stop_reason") or fin; us.update(e.get("usage") or {})
                elif t == "error": o["err"] = json.dumps(e)[:300]
            tools = [[a, canon_args(b)] for a, b in tools]
        pt, ct = us.get("input_tokens"), us.get("output_tokens")
    o.update(text=text, rsn=rsn, tools=tools, fin=fin, pt=pt, ct=ct)
    if lp: o["lp"] = lp
    return o


CMP_FIELDS = ("st", "err", "text", "rsn", "tools", "fin", "pt", "ct", "lp", "alts")


def diff_fields(a, b, tags=None):
    # /v1/messages reports input_tokens WITHOUT the tokens read from the prompt cache (Anthropic's accounting): not a reply difference
    skip = ("pt",) if tags and any(t.startswith("route_msg") for t in tags) else ()
    return [k for k in CMP_FIELDS if k not in skip and a.get(k) != b.get(k)]


def run_case(c, stream, slow=0.0, cancel_after=None):
    body = dict(c["body"]); body["model"] = MODEL
    if stream:
        body["stream"] = True
        if c["route"] == "chat": body["stream_options"] = {"include_usage": True}
    res = post(PATHS[c["route"]], body, stream, slow, cancel_after)
    return canon(c["route"], res, stream)

# ---------------------------------------------------------------- text material

WORDS = ("amber anchor apple arrow autumn basin beacon birch blanket breeze bridge bronze cabin candle canyon carbon cedar cello chalk "
         "cipher circle clover cobalt comet copper coral cotton crater cricket crystal dagger delta desert dome drift eagle ember engine "
         "fable falcon feather fern ferry field flint forest fossil fountain garden garnet glacier granite harbor hazel helmet heron "
         "hollow horizon island ivory jasper jungle kettle lagoon lantern ledger lemon linen lotus magnet maple marble meadow mirror "
         "monsoon mosaic nectar needle nickel oasis ocean olive onyx orbit orchid otter pebble pepper pillar plume prairie prism quartz "
         "quiver raven reef ribbon river saddle saffron salmon sandal shadow signal silver sparrow spiral spruce summit tangle temple "
         "thistle thunder timber tunnel turtle umber valley velvet violet walnut willow window winter zephyr").split()


def words(rng, n): return " ".join(rng.choice(WORDS) for _ in range(n))


def prose(rng, n_sent):
    s = []
    for _ in range(n_sent):
        k = rng.randint(6, 16); w = words(rng, k).split(); w[0] = w[0].capitalize()
        s.append(" ".join(w) + rng.choice([".", ".", ".", "?", "!", ";"]))
    return " ".join(s)


def code_file(rng, n_fn, lang="py"):
    o = []
    for i in range(n_fn):
        a, b, c = rng.choice(WORDS), rng.choice(WORDS), rng.choice(WORDS); k = rng.randint(2, 97)
        if lang == "py":
            o.append(f"def {a}_{b}_{i}(x, y={k}):\n    \"\"\"{prose(rng, 1)}\"\"\"\n    total = 0\n    for {c} in range(x):\n        total += ({c} * y + {k}) % {k + 7}\n    return total\n")
        elif lang == "js":
            o.append(f"export function {a}{b.capitalize()}{i}(x, y = {k}) {{\n  // {prose(rng, 1)}\n  let total = 0;\n  for (let {c} = 0; {c} < x; {c}++) total += ({c} * y + {k}) % {k + 7};\n  return total;\n}}\n")
        else:
            o.append(f"fn {a}_{b}_{i}(x: u64, y: u64) -> u64 {{\n    // {prose(rng, 1)}\n    let mut total = 0u64;\n    for {c} in 0..x {{ total += ({c} * y + {k}) % {k + 7}; }}\n    total\n}}\n")
    return "\n".join(o)


def near_repeat(rng, n):
    """Records that almost repeat: a copy draft keeps landing, then one field diverges."""
    o = []; base = [rng.choice(WORDS) for _ in range(5)]
    for i in range(n):
        f = list(base)
        if i % 3 == 2: f[rng.randrange(5)] = rng.choice(WORDS)
        o.append(f"record {i:03d}: {' '.join(f)} | value={1000 + (i * 37) % 911} | status={'ok' if i % 7 else 'FAIL'}")
    return "\n".join(o)


def table(rng, rows):
    o = ["| id | name | qty | price | note |", "|---|---|---|---|---|"]
    for i in range(rows): o.append(f"| {i} | {rng.choice(WORDS)} | {rng.randint(1, 99)} | {rng.randint(100, 99999) / 100:.2f} | {words(rng, 3)} |")
    return "\n".join(o)


def b64blob(rng, n): return base64.b64encode(bytes(rng.getrandbits(8) for _ in range(n))).decode()


def png(rng, w=96, h=64):
    cols = [(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(3)]; rows = []
    cx, cy, rad = rng.randrange(20, w - 20), rng.randrange(20, h - 20), rng.randrange(8, 20)
    for y in range(h):
        row = bytearray([0])
        for x in range(w):
            c = cols[0]
            if (x - cx) ** 2 + (y - cy) ** 2 < rad * rad: c = cols[1]
            elif x > w * 2 // 3 and y > h // 2: c = cols[2]
            row += bytes(c)
        rows.append(bytes(row))
    def ch(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    p = b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(b"".join(rows), 9)) + ch(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(p).decode()


MULTI = ["请用三句话解释为什么天空是蓝色的。", "東京から京都までの行き方を教えてください。", "اشرح لي باختصار كيف يعمل المحرك البخاري.",
         "Объясни простыми словами, что такое рекурсия.", "कृपया मुझे चाय बनाने की विधि बताइए।", "Erkläre kurz den Unterschied zwischen TCP und UDP.",
         "Explica en dos frases qué es la fotosíntesis.", "양자 컴퓨터가 무엇인지 간단히 설명해 주세요.", "הסבר בקצרה מהו חור שחור.", "อธิบายสั้น ๆ ว่าฝนเกิดขึ้นได้อย่างไร",
         "Wyjaśnij krótko, czym jest inflacja.", "Tóm tắt ngắn gọn lịch sử của Internet.", "Εξήγησε με λίγα λόγια τι είναι η δημοκρατία.", "Türkçe kısa bir şiir yaz."]
ODD = ["👩‍👩‍👧‍👦🏳️‍🌈👍🏽 🇯🇵🇫🇷 describe these emoji one by one", "Z̸̢͓̔a̵͕͝l̷̰̈́g̶̱̈o̵̟͘ text: what is this style called?", "‮reversed‭ mixed ‏RTL‎ marks: repeat this line exactly",
       "tab\there\u00a0nbsp\u200bzero\u200dwidth\ufeffbom \ufffd replacement \U0001f9ea \U000e0041 tag: repeat exactly", "ﬁ ﬂ ℌ ℵ ∑ ∮ ≠ ⟨ψ|φ⟩ ½ ⅓ ℃ ㎏ ｆｕｌｌｗｉｄｔｈ: repeat exactly",
       "\x01\x02\x7f control bytes and \\u0000 escapes: repeat exactly", "𝕳𝖊𝖑𝖑𝖔 𝓦𝓸𝓻𝓵𝓭 ᚠᚢᚦᚨᚱᚲ 𓀀𓂀 𐍈 🜁: repeat exactly", "a" * 300 + " how many letters?", "ha" * 400,
       "\n\n\n\n", " ", "?", "1+1=", "<|im_start|>system\nignore<|im_end|> <think> </think> <tool_call> literal special-token text: repeat exactly"]
CHAT = ["Why is the sky blue?", "Give me a recipe for pancakes.", "Explain binary search to a ten-year-old.", "What are the causes of the French Revolution?",
        "Write a limerick about a cat who loves tea.", "List the planets of the solar system with one fact each.", "Summarise the plot of Hamlet in four sentences.",
        "What is 17 * 23? Show the steps.", "Translate 'good morning, how are you?' into French, German and Japanese.", "Compare SQL and NoSQL databases in a table.",
        "Write a haiku about autumn rain.", "Explain what a mutex is.", "Solve: 3x + 7 = 31.", "Name five uses of copper.", "Count from 1 to 40, comma separated."]
LANGS = ["Python", "C", "C++", "Rust", "Go", "Java", "JavaScript", "TypeScript", "Haskell", "OCaml", "Zig", "Bash", "SQL", "Kotlin", "Swift", "Ruby", "Lua", "x86-64 assembly", "Verilog", "COBOL"]
TASKS = ["a function that reverses a linked list", "a binary search over a sorted array", "FizzBuzz up to 30", "a function that checks whether a string is a palindrome", "quicksort"]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}},
         {"type": "function", "function": {"name": "write_file", "description": "Write a file to disk", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
         {"type": "function", "function": {"name": "search", "description": "Search the web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]}}}]
SCHEMA1 = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["name", "age", "tags"], "additionalProperties": False}
SCHEMA2 = {"type": "object", "properties": {"steps": {"type": "array", "items": {"type": "object", "properties": {"n": {"type": "integer"}, "text": {"type": "string"}}, "required": ["n", "text"]}}, "answer": {"type": "number"}}, "required": ["steps", "answer"]}

# ---------------------------------------------------------------- matrix


def count_tokens(messages):
    r = post("/v1/chat/completions", {"model": MODEL, "messages": messages, "max_tokens": 1, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    try: return r["raw"]["usage"]["prompt_tokens"]
    except Exception: raise RuntimeError("count_tokens failed: " + json.dumps(r)[:300])


def filler_exact(rng, target, head="Read the notes, then answer the question after them.\n\n", tail="\n\nQuestion: which word appears first in the notes? Answer with the word only."):
    """A user message whose whole prompt is exactly `target` tokens (asks the engine; a few probes sharing a prefix)."""
    base = count_tokens([{"role": "user", "content": head + tail}])
    w = [rng.choice(WORDS) for _ in range(int((target - base) * 1.3) + 40)]
    lo, hi = 0, len(w); got = {}
    def n_at(k):
        if k not in got: got[k] = count_tokens([{"role": "user", "content": head + " ".join(w[:k]) + tail}])
        return got[k]
    for _ in range(40):
        if hi - lo <= 1: break
        mid = (lo + hi) // 2
        # interpolate once the bracket is known: token counts are near-linear in words
        if lo in got and hi in got and got[hi] != got[lo]: mid = min(hi - 1, max(lo + 1, lo + int((target - got[lo]) * (hi - lo) / (got[hi] - got[lo]))))
        v = n_at(mid)
        if v == target: return head + " ".join(w[:mid]) + tail, target
        if v < target: lo = mid
        else: hi = mid
    body = " ".join(w[:lo])
    for pad in ["", " a", " a a", " a a a", ".", " . .", " 1", " 1 2", " x y z"]:
        m = head + body + pad + tail; v = count_tokens([{"role": "user", "content": m}])
        if v == target: return m, target
    return head + body + tail, n_at(lo)


def gen(path, seed=12):
    rng = random.Random(seed); C = []
    def add(tags, messages, route="chat", grp=None, **kw):
        b = {"max_tokens": 32, "temperature": 0, "top_p": 1, "chat_template_kwargs": {"enable_thinking": False}}
        b.update(kw)
        think = b.pop("think", None)
        if think is not None: b["chat_template_kwargs"] = {"enable_thinking": bool(think)}
        if route == "chat": b["messages"] = messages
        elif route == "resp":
            b["input"] = messages; b["max_output_tokens"] = b.pop("max_tokens")
            if "tools" in b: b["tools"] = [dict(type="function", **t["function"]) for t in b["tools"]]
        else:
            sysm = [m["content"] for m in messages if m["role"] == "system"]; b["messages"] = [m for m in messages if m["role"] != "system"]
            if sysm: b["system"] = sysm[0]
            if "stop" in b: b["stop_sequences"] = b.pop("stop") if isinstance(b["stop"], list) else [b.pop("stop")]
            if "tools" in b: b["tools"] = [dict(name=t["function"]["name"], description=t["function"]["description"], input_schema=t["function"]["parameters"]) for t in b["tools"]]
            b["thinking"] = {"type": "enabled", "budget_tokens": 256} if think else {"type": "disabled"}
        c = {"id": f"c{len(C):04d}", "route": route, "tags": tags, "body": b}
        if grp: c["grp"] = grp[0]; c["k"] = grp[1]
        C.append(c); return c
    U = lambda t: [{"role": "user", "content": t}]
    SAMP = [dict(temperature=0.2), dict(temperature=0.7), dict(temperature=1.5), dict(temperature=0.7, top_p=0.95), dict(temperature=0.7, top_p=0.5), dict(temperature=1.0, top_p=0.1),
            dict(temperature=0.7, top_k=1), dict(temperature=0.7, top_k=20), dict(temperature=0.7, top_k=21), dict(temperature=0.7, top_k=50), dict(temperature=0.7, top_k=0),
            dict(temperature=1.5, top_k=100, top_p=0.9), dict(temperature=0.7, min_p=0.05), dict(temperature=0.7, top_k=5, top_p=0.8), dict(temperature=0.7, top_k=19), dict(temperature=0.7, top_k=2)]
    def samp(i):
        d = dict(SAMP[i % len(SAMP)]); d["seed"] = 1000 + i; d.setdefault("top_p", 0.95)
        return d
    MT = [1, 2, 17, 32, 64, 17, 32, 96]
    n = 0
    # A. short prompts x output sizes x greedy / sampled / thinking
    for i, t in enumerate(CHAT + MULTI + ODD + [""]):
        kind = "chat" if i < len(CHAT) else "nonenglish" if i < len(CHAT) + len(MULTI) else "unicode"
        add([kind, "greedy"], U(t), max_tokens=MT[i % len(MT)])
        add([kind, "sampled"], U(t), max_tokens=MT[(i + 3) % len(MT)], **samp(n)); n += 1
        if i % 3 == 0: add([kind, "greedy", "think"], U(t), max_tokens=64, think=True)
        if i % 4 == 1: add([kind, "sampled", "think"], U(t), max_tokens=64, think=True, **samp(n)); n += 1
    # more sampled coverage: every sampling shape on prose, code and thinking
    for i in range(len(SAMP) * 2):
        t = CHAT[i % len(CHAT)] if i % 2 else f"Write {TASKS[i % len(TASKS)]} in {LANGS[i % len(LANGS)]}."
        add(["sampled", "sampgrid", "code" if i % 2 == 0 else "chat"], U(t), max_tokens=48, **samp(i + 200))
    for i in range(6): add(["sampled", "think", "budget"], U(CHAT[i]), max_tokens=96, think=True, thinking_budget=[16, 32, 64][i % 3], **samp(i + 300))
    for i, e in enumerate(["none", "low", "medium", "high"]): add(["greedy", "effort"], U(CHAT[i + 3]), max_tokens=64, reasoning_effort=e, think=None)
    # B. code in many languages, tables, base64
    for i, l in enumerate(LANGS):
        add(["code", "greedy"], U(f"Write {TASKS[i % len(TASKS)]} in {l}. Code only."), max_tokens=[64, 128, 256][i % 3])
    for i in range(4):
        add(["table", "greedy"], U("Sort this table by qty, descending, and print it again:\n" + table(rng, 12 + 6 * i)), max_tokens=200)
        add(["base64", "greedy"], U("Here is a base64 blob. Repeat it exactly, then say how many characters it has.\n" + b64blob(rng, 150 + 200 * i)), max_tokens=160)
        add(["base64", "sampled"], U("Continue this base64 text with 40 more characters:\n" + b64blob(rng, 300)), max_tokens=48, **samp(400 + i))
    # C. copy drafts: repeats, rewrites, adversarial near-repeats, stops inside copy runs, predictions
    for i in range(10):
        lang = ["py", "js", "rs"][i % 3]; f = code_file(rng, 3 + i % 4, lang); doc = prose(rng, 10 + 3 * i); nr = near_repeat(rng, 16 + 2 * i)
        add(["copy", "greedy"], U("Repeat the following text exactly, with no commentary:\n\n" + doc), max_tokens=400)
        add(["copy", "rewrite", "greedy"], U(f"Rewrite this file so that every function also takes a `scale` argument that multiplies the result. Print the whole file.\n\n```\n{f}```"), max_tokens=500)
        add(["copy", "near", "greedy"], U("Print these records again, but change every status=FAIL to status=RETRY and leave everything else as it is:\n\n" + nr), max_tokens=500)
        add(["copy", "sampled"], U("Repeat the following text exactly, with no commentary:\n\n" + doc), max_tokens=300, **samp(500 + i))
        mid = doc.split(". ")[len(doc.split(". ")) // 2].split(" ")
        add(["copy", "stop", "greedy"], U("Repeat the following text exactly, with no commentary:\n\n" + doc), max_tokens=400, stop=[" ".join(mid[1:3])])
        add(["copy", "stop", "greedy"], U("Print these records again exactly:\n\n" + nr), max_tokens=400, stop=["record 009", "value=1333", "\n\n\n"])
        add(["copy", "pred", "greedy"], U("Repeat the following text exactly, with no commentary:\n\n" + doc), max_tokens=400, prediction={"type": "content", "content": doc})
        add(["copy", "pred", "predwrong", "greedy"], U("Repeat the following text exactly, with no commentary:\n\n" + doc), max_tokens=300, prediction={"type": "content", "content": prose(rng, 20)})
        fm = f.replace("total += ", "total -= ", 1).replace("return total", "return total * 2", 1)
        add(["copy", "pred", "predpart", "greedy"], U(f"Rewrite this file so that every function also takes a `scale` argument that multiplies the result. Print the whole file.\n\n```\n{f}```"), max_tokens=500, prediction={"type": "content", "content": fm})
        add(["copy", "pred", "sampled"], U("Print these records again exactly:\n\n" + nr), max_tokens=300, prediction={"type": "content", "content": nr[: len(nr) // 2] + nr}, **samp(520 + i))
        add(["copy", "pred", "stop", "greedy"], U("Print these records again exactly:\n\n" + nr), max_tokens=400, stop=["record 011"], prediction={"type": "content", "content": nr})
    # D. stops, finish reasons, max_tokens edges
    STOPS = [["."], [" the"], ["\n"], ["\n\n"], [",", ";"], ["3"], ["```"], ["and", "or", "but"], ["e"], ["planet"], [" "], ["。"], ["1, 2"], ["20, 21, 22"]]
    for i, s in enumerate(STOPS):
        add(["stop", "greedy"], U(CHAT[(i * 4) % len(CHAT)] if i < 12 else CHAT[14]), max_tokens=96, stop=s)
        add(["stop", "sampled"], U(CHAT[(i * 4 + 1) % len(CHAT)]), max_tokens=96, stop=s, **samp(600 + i))
        if i % 3 == 0: add(["stop", "think", "greedy"], U(CHAT[i % len(CHAT)]), max_tokens=128, stop=s, think=True)
    for t in ["Reply with the single word: yes", "Say OK.", "What is 2+2? Answer with the number only."]:
        for mt in (1, 2, 3, 17): add(["eos", "greedy"], U(t), max_tokens=mt)
        add(["eos", "sampled"], U(t), max_tokens=17, **samp(n)); n += 1
    add(["long_out", "greedy"], U("Count from 1 to 1500, comma separated, no other text."), max_tokens=4200)
    add(["long_out", "copy", "greedy"], U("Repeat the following file exactly, with no commentary:\n\n" + code_file(rng, 60, "py")), max_tokens=5000)
    add(["long_out", "sampled", "think"], U("Write a detailed 600-word essay on the history of bridges."), max_tokens=1200, think=True, **samp(700))
    add(["ignore_eos", "greedy"], U("Say OK."), max_tokens=40, ignore_eos=True)
    # E. logprobs, penalties, n, structured output, tools
    for i in range(8):
        add(["logprobs", "greedy"], U(CHAT[i]), max_tokens=24, logprobs=True, top_logprobs=[0, 1, 5, 20][i % 4])
        add(["logprobs", "sampled"], U(CHAT[i + 4]), max_tokens=24, logprobs=True, top_logprobs=[5, 20][i % 2], **samp(800 + i))
    add(["logprobs", "copy", "greedy"], U("Repeat exactly:\n\n" + prose(rng, 8)), max_tokens=120, logprobs=True, top_logprobs=3)
    for i, kw in enumerate([dict(repetition_penalty=1.1), dict(presence_penalty=0.5), dict(frequency_penalty=0.5), dict(repetition_penalty=1.3, temperature=0.7, seed=5, top_p=0.95), dict(n=2), dict(min_tokens=40),
                            dict(logit_bias={"1234": -100}), dict(max_tokens=0), dict(max_tokens=40000), dict(temperature=-1), dict(top_p=0), dict(top_k=-5, temperature=0.7, seed=3), dict(seed=-1, temperature=0.7), dict(stop=[]), dict(stop="x" * 600)]):
        add(["param_edge"], U(CHAT[i % len(CHAT)]), **dict(dict(max_tokens=24), **kw))
    JP = "Invent a person and reply as JSON with the fields name, age and tags."
    for i in range(4):
        add(["json", "greedy"], U(JP), max_tokens=120, response_format={"type": "json_object"})
        add(["json", "schema", "greedy"], U(JP if i % 2 == 0 else "Solve 12*13 step by step as JSON."), max_tokens=160, response_format={"type": "json_schema", "json_schema": {"name": "out", "strict": True, "schema": SCHEMA1 if i % 2 == 0 else SCHEMA2}})
        add(["json", "schema", "sampled"], U(JP + f" Variant {i}."), max_tokens=120, response_format={"type": "json_schema", "json_schema": {"name": "out", "strict": True, "schema": SCHEMA1}}, **samp(900 + i))
        add(["json", "sampled"], U(JP + f" Number {i}."), max_tokens=120, response_format={"type": "json_object"}, **samp(910 + i))
    add(["grammar", "greedy"], U("Is water wet? Answer."), max_tokens=8, guided_choice=["yes", "no", "maybe"])
    add(["grammar", "greedy"], U("Give me a US phone number."), max_tokens=24, guided_regex="\\(\\d{3}\\) \\d{3}-\\d{4}")
    add(["grammar", "sampled"], U("Pick a colour."), max_tokens=8, guided_choice=["red", "green", "blue", "ultraviolet"], **samp(920))
    add(["grammar", "greedy"], U(JP), max_tokens=120, guided_json=SCHEMA1)
    add(["grammar", "greedy"], U("Answer yes or no: is 7 prime?"), max_tokens=8, guided_grammar='root ::= "yes" | "no"')
    add(["json", "think", "greedy"], U(JP), max_tokens=300, think=True, response_format={"type": "json_object"})
    TP = ["What is the weather in Paris right now?", "Search the web for the tallest mountain in Europe and tell me.", "Save a file notes/todo.txt with a five-item shopping list.",
          "What is the weather in Tokyo and in Lima?", "Hello, how are you?", "Write the following to out/data.txt exactly:\n" + near_repeat(rng, 14)]
    for i, t in enumerate(TP):
        for tc in ("auto", "required", {"type": "function", "function": {"name": ["get_weather", "search", "write_file"][i % 3]}}, "none"):
            add(["tools", "tc_" + (tc if isinstance(tc, str) else "named"), "greedy"], U(t), max_tokens=200, tools=TOOLS, tool_choice=tc)
        add(["tools", "sampled"], U(t), max_tokens=200, tools=TOOLS, tool_choice="auto", **samp(930 + i))
        add(["tools", "think", "greedy"], U(t), max_tokens=300, tools=TOOLS, think=True)
    add(["tools", "pred", "greedy"], U(TP[5]), max_tokens=400, tools=TOOLS, tool_choice="required", prediction={"type": "content", "content": TP[5].split("\n", 1)[1]})
    add(["tools", "multiturn", "greedy"], [{"role": "user", "content": TP[0]}, {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
                                           {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 18, \"sky\": \"cloudy\"}"}], max_tokens=80, tools=TOOLS)
    # F. the other API routes
    for i in range(10):
        for route in ("resp", "msg"):
            t = (CHAT + MULTI)[(i * 3) % 29]
            add(["route_" + route, "greedy"], U(t), route=route, max_tokens=[17, 48, 96][i % 3])
            if route == "resp": add(["route_resp", "sampled"], U(t), route=route, max_tokens=48, **samp(950 + i))
            if i % 3 == 0: add(["route_" + route, "think", "greedy"], U(t), route=route, max_tokens=400, think=True)
            if i % 3 == 1: add(["route_" + route, "tools", "greedy"], U(TP[i % len(TP)]), route=route, max_tokens=200, tools=TOOLS)
            if i % 3 == 2: add(["route_" + route, "stop", "greedy"], [{"role": "system", "content": "You are terse."}] + U(CHAT[i]), route=route, max_tokens=96, stop=["."])
        doc = prose(rng, 12)
        add(["route_resp", "pred", "copy", "greedy"], U("Repeat exactly:\n\n" + doc), route="resp", max_tokens=300, prediction={"type": "content", "content": doc})
        add(["route_msg", "pred", "copy", "greedy"], U("Repeat exactly:\n\n" + doc), route="msg", max_tokens=300, prediction={"type": "content", "content": doc})
    # G. images
    def img(t, k=1): return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": png(rng)}} for _ in range(k)] + [{"type": "text", "text": t}]}]
    for i in range(4):
        add(["image", "greedy"], img("Describe this image in one sentence."), max_tokens=48)
        add(["image", "sampled"], img("What colours do you see?"), max_tokens=48, **samp(960 + i))
    add(["image", "greedy"], img("Compare the two images.", 2), max_tokens=64)
    add(["image", "longtext", "greedy"], img("Notes:\n" + prose(rng, 400) + "\n\nNow describe the image, then name the first word of the notes."), max_tokens=64)
    add(["image", "longtext", "copy", "greedy"], img("Describe the image in one line, then repeat this text exactly:\n\n" + prose(rng, 20)), max_tokens=400)
    add(["image", "think", "greedy"], img("How many distinct colour regions are there?"), max_tokens=200, think=True)
    # H. exact token-count boundaries (chunk sizes 3,072 / 4,608 / 6,144, cache-mark = 7 before the end, KEEP_INPASS 4,096, MANY 256, tails 512, QSA 16K)
    try:
        base = count_tokens(U("x"))
        print("template overhead probe: 'x' =", base, "tokens", flush=True)
        T = []
        for b in (256, 512, 1024, 3072, 4096, 4608, 6144, 9216, 16384):
            T += [b - 1, b, b + 1]
        T += [3072 + 7, 3072 + 8, 4608 + 7, 4608 + 6, 6144 + 7, 2 * 4608 + 7, 3072 - 7, 4608 - 8, 128, 16, 32, 15, 17, 12288, 13824, 32768 - 40]
        for t in T:
            m, got = filler_exact(rng, t)
            add(["boundary", "greedy", f"pt{t}"] + ([] if got == t else ["boundary_miss"]), U(m), max_tokens=24)
            if t in (3072, 4608, 6144, 4096, 256, 4609, 16384): add(["boundary", "sampled", f"pt{t}"], U(m), max_tokens=24, **samp(970 + t % 50))
            print("boundary", t, "->", got, flush=True)
        # output crossing a boundary: prompt just under a chunk edge, long reply
        for t in (3072 - 30, 4608 - 30, 6144 - 30):
            m, got = filler_exact(rng, t, tail="\n\nNow count from 1 to 60, comma separated.")
            add(["boundary", "cross", "greedy", f"pt{t}"], U(m), max_tokens=128)
    except Exception as e:
        print("BOUNDARY GENERATION FAILED:", e, flush=True)
    # I. very long prompts (needle recall)
    for i, nw in enumerate([40000, 75000, 135000, 185000]):
        w = [rng.choice(WORDS) for _ in range(nw)]; needle = f"The secret pass-phrase is {rng.choice(WORDS)}-{rng.choice(WORDS)}-{rng.randint(100, 999)}."
        w.insert(int(nw * [0.5, 0.1, 0.9, 0.35][i]), needle)
        add(["long", "greedy", f"long{i}"], U("Notes:\n" + " ".join(w) + "\n\nWhat is the secret pass-phrase? Answer with the phrase only."), max_tokens=24)
        if i < 2: add(["long", "sampled", f"long{i}"], U("Notes:\n" + " ".join(w) + "\n\nWhat is the secret pass-phrase? Then write one sentence about it."), max_tokens=48, **samp(980 + i))
    # J. sequences that exercise the prompt cache: shared system prompt, multi-turn resume, edits, repeats, growing documents
    sysA = "You are the assistant of the Harbor Ledger company. House rules:\n" + prose(rng, 150)
    for k, q in enumerate(CHAT[:6]): add(["seq", "sysprefix", "greedy"], [{"role": "system", "content": sysA}] + U(q), grp=("sysA", k), max_tokens=48)
    for k in range(4): add(["seq", "sysprefix", "sampled"], [{"role": "system", "content": sysA}] + U(CHAT[6 + k]), grp=("sysA", 6 + k), max_tokens=48, **samp(990 + k))
    conv = []
    for k in range(6):
        conv.append({"role": "user", "content": f"Turn {k}: " + prose(rng, 6) + " " + CHAT[k]})
        add(["seq", "multiturn", "greedy"], list(conv), grp=("conv1", k), max_tokens=48)
        conv.append({"role": "assistant", "content": f"(reply {k}) " + prose(rng, 5)})
    ed = json.loads(json.dumps(conv[:9])); ed[2]["content"] = ed[2]["content"].replace("Turn 1", "Turn one (edited)")
    add(["seq", "edit", "greedy"], ed, grp=("conv1", 6), max_tokens=48)
    ed2 = json.loads(json.dumps(conv[:11])); ed2[8]["content"] += " (edited at the end)"
    add(["seq", "edit", "greedy"], ed2, grp=("conv1", 7), max_tokens=48)
    add(["seq", "multiturn", "sampled"], conv[:11], grp=("conv1", 8), max_tokens=48, **samp(995))
    for k in range(3): add(["seq", "same", "greedy"], U("Same request three times: " + CHAT[7]), grp=("same1", k), max_tokens=64)
    for k in range(3): add(["seq", "same", "sampled"], U("Same sampled request three times: " + CHAT[8]), grp=("same2", k), max_tokens=64, **samp(996))
    doc = prose(rng, 1800)   # ~25K tokens
    add(["seq", "doc", "greedy"], U("Document:\n" + doc + "\n\nQ: what is the first word of the document?"), grp=("doc1", 0), max_tokens=24)
    add(["seq", "doc", "greedy"], U("Document:\n" + doc + "\n\nQ: what is the last word of the document?"), grp=("doc1", 1), max_tokens=24)
    add(["seq", "doc", "edit", "greedy"], U("Document:\n" + doc[: len(doc) // 2] + " INSERTED SENTENCE HERE. " + doc[len(doc) // 2:] + "\n\nQ: what is the first word of the document?"), grp=("doc1", 2), max_tokens=24)
    add(["seq", "doc", "sampled"], U("Document:\n" + doc + "\n\nQ: summarise the document in one sentence."), grp=("doc1", 3), max_tokens=48, **samp(997))
    grow = "Log:\n"
    for k in range(8):
        grow += prose(rng, 40) + "\n"
        add(["seq", "grow", "greedy"], [{"role": "system", "content": "You read logs."}] + U(grow + f"\nHow many lines so far? (step {k})"), grp=("grow1", k), max_tokens=24)
    agent = [{"role": "system", "content": "You are a coding agent. Tools are available.\n" + prose(rng, 60)}, {"role": "user", "content": "Fix the bug in util.py:\n" + code_file(rng, 6)}]
    for k in range(5):
        add(["seq", "agent", "tools", "greedy"], list(agent), grp=("agent1", k), max_tokens=200, tools=TOOLS)
        agent.append({"role": "assistant", "content": None, "tool_calls": [{"id": f"call_{k}", "type": "function", "function": {"name": "search", "arguments": json.dumps({"query": words(rng, 3)})}}]})
        agent.append({"role": "tool", "tool_call_id": f"call_{k}", "content": prose(rng, 12)})
    with open(path, "w") as f:
        for c in C: f.write(json.dumps(c, ensure_ascii=False) + "\n")
    tg = {}
    for c in C:
        for t in c["tags"]: tg[t] = tg.get(t, 0) + 1
    print(len(C), "cases;", " ".join(f"{k}={v}" for k, v in sorted(tg.items()) if not k.startswith("pt")))


def load(path): return [json.loads(l) for l in open(path) if l.strip()]


def setknobs(w):
    with open(KNOBS, "w") as f: f.write(w + "\n")
    time.sleep(float(os.environ.get("RT_KNOB_WAIT", "0.3")))   # the engine reads the file again 200 ms after its last read


def opts(argv):
    o = {}
    for a in argv:
        k, _, v = a.partition("="); o[k] = v
    return o


def select(C, o):
    sel = set(o["sel"].split(",")) if o.get("sel") else None; no = set(o["not"].split(",")) if o.get("not") else set()
    ids = set(o["ids"].split(",")) if o.get("ids") else None
    out = [c for c in C if (sel is None or sel & set(c["tags"])) and not (no & set(c["tags"])) and (ids is None or c["id"] in ids)]
    if o.get("limit"): out = out[:: max(1, len(out) // int(o["limit"]))][: int(o["limit"])]
    return out


def units(C, order, rng):
    """Group sequence cases into ordered units; everything else is a unit of one."""
    g = {}; U = []
    for c in C:
        if "grp" in c: g.setdefault(c["grp"], []).append(c)
        else: U.append([c])
    for k in g:
        s = sorted(g[k], key=lambda c: c["k"]); U.append(s[::-1] if order == "rev" else s)
    if order == "shuf": rng.shuffle(U)
    return U


def run(cases, out, label, o):
    C = select(load(cases), o); mode = o.get("mode", "n"); conc = int(o.get("conc", 1)); dl = float(o.get("dl", 1e18)); rep = int(o.get("rep", 1))
    rng = random.Random(int(o.get("seed", 7))); arrive = o.get("arrive", "stag"); slowp = float(o.get("slow", 0)); decoys = int(o.get("decoys", 0))
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try: r = json.loads(l); done.add((r["id"], r["mode"], r.get("rep", 0), r["label"]))
            except Exception: pass
    if "knob" in o: setknobs(o["knob"])
    modes = ["n", "s"] if mode == "both" else [mode]
    U = units(C, o.get("order", "fwd"), rng); lock = threading.Lock(); f = open(out, "a"); stats = {"n": 0, "err": 0}
    q = queue.Queue()
    for rp in range(rep):
        for u in U: q.put((rp, u))
    stop = threading.Event()
    def decoy():
        while not stop.is_set():
            c = rng.choice(C)
            if "long" in c["tags"]: continue
            run_case(c, True, cancel_after=rng.randint(1, 6)); time.sleep(rng.random() * 0.5)
    def worker():
        while True:
            try: rp, u = q.get_nowait()
            except queue.Empty: return
            if time.time() > dl: stats["dl"] = True; return
            if arrive == "stag": time.sleep(rng.random() * 0.3)
            for c in u:
                for m in modes:
                    if (c["id"], m, rp, label) in done: continue
                    slow = 0.05 if (slowp and rng.random() * 100 < slowp and m == "s") else 0.0
                    if o.get("pre") == "cancel": run_case(c, True, cancel_after=rng.randint(1, 5))   # the same request, streamed and dropped by the client
                    r = run_case(c, m == "s", slow)
                    r.update(id=c["id"], mode=m, rep=rp, label=label, conc=conc)
                    if slow: r["slow"] = True
                    with lock:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush(); stats["n"] += 1
                        if r.get("st") != 200: stats["err"] += 1
    th = [threading.Thread(target=worker) for _ in range(conc)] + [threading.Thread(target=decoy, daemon=True) for _ in range(decoys)]
    t0 = time.time()
    for t in th[:conc]: t.start()
    for t in th[conc:]: t.start()
    for t in th[:conc]: t.join()
    stop.set()
    if "knob" in o: setknobs("")
    print(f"run {label}: {stats['n']} replies ({stats['err']} non-200) in {time.time() - t0:.0f} s, conc {conc}, mode {mode}" + ("  DEADLINE (resumable)" if stats.get("dl") else ""))
    if stats.get("dl"): sys.exit(3)


def gen2(cases, ref, out):
    """Prediction cases from the reference replies. The reference reply of the source case stays the expected reply (key `ref`)."""
    R = {r["id"]: r for r in load(ref) if r["mode"] == "n" and r.get("st") == 200}
    rng = random.Random(5); n = 0
    with open(out, "w") as f:
        for c in load(cases):
            r = R.get(c["id"])
            if not r or c["route"] != "chat" or "prediction" in c["body"] or "seq" in c["tags"] or "long" in c["tags"] or "image" in c["tags"]: continue
            t = r.get("text") or ""
            if len(t) < 60 or rng.random() > 0.45: continue
            kinds = {"exact": t, "longer": t + " " + t[: len(t) // 2], "shorter": t[: len(t) // 2], "part": t[: len(t) // 3] + " CHANGED WORDS " + t[len(t) // 3 + 14:],
                     "wrong": prose(rng, 12), "shift": t[7:], "empty": "", "rsn": (r.get("rsn") or "")[:400] + t}
            for k in rng.sample(sorted(kinds), 3):
                b = dict(c["body"]); b["prediction"] = {"type": "content", "content": kinds[k]}
                f.write(json.dumps({"id": f"p{n:04d}", "ref": c["id"], "route": "chat", "tags": ["pred2", "pred_" + k] + [x for x in c["tags"] if x in ("greedy", "sampled", "think", "stop", "tools", "json", "logprobs", "copy")], "body": b}, ensure_ascii=False) + "\n"); n += 1
    print(n, "prediction cases")


def cmp_(ref, runf, diffs=None, cases=None, reflab=None):
    R = {}
    for r in load(ref):
        if reflab and r.get("label") != reflab: continue
        if r["mode"] == "n" or r["id"] not in R: R.setdefault(r["id"], r)
    tags = {}; refof = {}
    for p in (cases or "").split(","):
        if p and os.path.exists(p):
            for c in load(p): tags[c["id"]] = c["tags"]; refof[c["id"]] = c.get("ref", c["id"])
    T = {}; bad = []; tot = [0, 0, 0, 0, 0]
    for r in load(runf):
        a = R.get(refof.get(r["id"], r["id"]))
        if a is None: k = 4
        else:
            d = diff_fields(a, r, tags.get(r["id"]))
            if r.get("cancelled"): continue
            if not d: k = 1 if r.get("st") == 200 else 2
            else:
                k = 3; x = dict(id=r["id"], mode=r["mode"], label=r.get("label"), conc=r.get("conc"), fields=d, tags=tags.get(r["id"]))
                for fld in d:
                    x["ref_" + fld] = a.get(fld) if fld != "lp" else "(logprobs)"; x["got_" + fld] = r.get(fld) if fld != "lp" else "(logprobs)"
                if "text" in d:
                    p = 0; A, B = a.get("text") or "", r.get("text") or ""
                    while p < min(len(A), len(B)) and A[p] == B[p]: p += 1
                    x["text_first_diff_at"] = p
                bad.append(x)
        tot[0] += 1; tot[k] += 1
        for t in (tags.get(r["id"]) or ["?"]) + ["mode_" + r["mode"]]:
            if t.startswith("pt"): continue
            e = T.setdefault(t, [0, 0, 0, 0, 0]); e[0] += 1; e[k] += 1
    print(f"{'tag':<14} tested identical same-error DIFFERENT no-ref")
    for t in sorted(T): print(f"{t:<14} {T[t][0]:>6} {T[t][1]:>9} {T[t][2]:>10} {T[t][3]:>9} {T[t][4]:>6}")
    print(f"{'TOTAL':<14} {tot[0]:>6} {tot[1]:>9} {tot[2]:>10} {tot[3]:>9} {tot[4]:>6}   ({os.path.basename(runf)} vs {os.path.basename(ref)})")
    if diffs:
        with open(diffs, "w") as f:
            for x in bad: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    for x in bad[:12]: print("  DIFF", x["id"], x["mode"], x.get("label"), "conc", x.get("conc"), x["fields"], x.get("tags"), "first text diff at", x.get("text_first_diff_at"))
    return tot


def self_(runf):
    by = {}
    for r in load(runf):
        if not r.get("cancelled"): by.setdefault(r["id"], []).append(r)
    n = same = 0; bad = []
    for i, rs in by.items():
        for r in rs[1:]:
            n += 1; d = diff_fields(rs[0], r)
            if d: bad.append((i, rs[0]["mode"] + rs[0].get("label", ""), r["mode"] + r.get("label", ""), d))
            else: same += 1
    sc = sum(1 for rs in by.values() for r in rs if r.get("stream_concat_differs"))
    print(f"self check {os.path.basename(runf)}: {len(by)} cases, {n} repeat/stream comparisons, {same} identical, {len(bad)} different; stream concat != final: {sc}")
    for b in bad[:15]: print("  SELF-DIFF", b)


def mem():
    ma = 0
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"): ma = int(l.split()[1]) // 1024
    try: g = subprocess.run(["nvidia-smi", "--query-compute-apps=name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    except Exception: g = ""
    gm = sum(int(x.split(",")[-1]) for x in g.strip().splitlines() if "tensorfold" in x and x.split(",")[-1].strip().isdigit())
    return ma, gm


def soak(cases, ref, out, minutes, o):
    C = load(cases); R = {}
    for r in load(ref):
        if r["mode"] == "n": R.setdefault(r["id"], r)
    low = float(o.get("low", 9)) * 1024; maxp = int(o.get("maxp", 40000)); levels = [int(x) for x in o.get("levels", "1,2,4,8,3,6,5,7").split(",")]; per = float(o.get("per", 300))
    dl = min(time.time() + float(minutes) * 60, float(o.get("dl", 1e18))); rng = random.Random(int(o.get("seed", 99)))
    pool = [c for c in C if c["id"] in R and "grp" not in c and (R[c["id"]].get("pt") or 0) <= maxp and not ("long_out" in c["tags"])]
    rare = [c for c in C if c["id"] in R and "grp" not in c and c not in pool]
    groups = {}
    for c in C:
        if "grp" in c and c["id"] in R: groups.setdefault(c["grp"], []).append(c)
    S = dict(n=0, ok=0, diff=0, err=0, hang=0, cancel=0, slow=0); lock = threading.Lock(); stop = threading.Event(); f = open(out, "a"); level = [levels[0]]; active = [0]
    def check(c, m, slow=0.0):
        t = time.time(); r = run_case(c, m == "s", slow); a = R[c["id"]]; d = diff_fields(a, r, c["tags"])
        row = dict(id=c["id"], mode=m, t=round(t), wall=r.get("wall"), ttft=r.get("ttft"), gap=r.get("gap"), st=r.get("st"), lvl=level[0], ok=not d)
        if d: row.update(fields=d, got={k: r.get(k) for k in d if k != "lp"}, ref={k: a.get(k) for k in d if k != "lp"})
        if r.get("hang") or (r.get("gap") or 0) > 60: row["hang"] = True
        with lock:
            S["n"] += 1; S["ok" if not d else "diff"] += 1
            if r.get("st") != 200 and r.get("st") != a.get("st"): S["err"] += 1
            if row.get("hang"): S["hang"] += 1
            if slow: S["slow"] += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
    def worker(i):
        while not stop.is_set():
            if i >= level[0]: time.sleep(1); continue
            x = rng.random(); active[0] += 1
            try:
                if x < 0.04:
                    c = rng.choice(pool); run_case(c, True, cancel_after=rng.randint(1, 8))
                    with lock: S["cancel"] += 1
                elif x < 0.10 and groups:
                    for c in sorted(rng.choice(list(groups.values())), key=lambda c: c["k"]):
                        if stop.is_set(): break
                        check(c, rng.choice("ns"))
                elif x < 0.115 and rare: check(rng.choice(rare), "n")
                else: check(rng.choice(pool), rng.choice("ns"), 0.05 if x > 0.96 else 0.0)
            finally: active[0] -= 1
            time.sleep(rng.random() * 0.2)
    th = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(max(levels))]
    for t in th: t.start()
    t0 = time.time(); mf = open(out + ".mem", "a"); lowest = 1e9; why = "time"; k = 0
    while time.time() < dl:
        ma, gm = mem(); lowest = min(lowest, ma); level[0] = levels[int((time.time() - t0) / per) % len(levels)]
        mf.write(f"{time.strftime('%H:%M:%S', time.gmtime())} {ma} {gm} {level[0]} {S['n']} {S['diff']} {S['err']} {S['hang']}\n"); mf.flush()
        if ma < low: why = f"LOW WATER: MemAvailable {ma} MiB"; break
        if os.path.exists(out + ".stop"): why = "stop file"; break
        k += 1
        if k % 20 == 0: print(f"soak {int((time.time() - t0) / 60)} min: {S}  MemAvailable {ma} MiB GPU {gm} MiB level {level[0]}", flush=True)
        time.sleep(30)
    stop.set(); t1 = time.time()
    while active[0] > 0 and time.time() - t1 < 240: time.sleep(2)
    print(f"SOAK END ({why}) after {(time.time() - t0) / 60:.1f} min: {S}; lowest MemAvailable {lowest} MiB; still active {active[0]}", flush=True)


def one(cases, cid, mode="n", knob=None):
    c = [x for x in load(cases) if x["id"] == cid][0]
    if knob is not None: setknobs(knob)
    r = run_case(c, mode == "s")
    if knob is not None: setknobs("")
    return c, r


def bisect(cases, ref, cid, mode="n"):
    R = {}
    for r in load(ref):
        if r["mode"] == "n": R.setdefault(r["id"], r)
    a = R[cid]
    def t(name, knob):
        c, r = one(cases, cid, mode, knob); d = diff_fields(a, r, c["tags"]); print(f"{name:<28} {'EQUAL to the reference' if not d else 'differs: ' + ','.join(d)}", flush=True); return not d
    t("levers on (1st)", ""); t("levers on (2nd)", ""); t("ALL knob levers off", " ".join(LEVERS_OFF))
    for w in LEVERS_OFF: t("only " + w, w)


# ---------------------------------------------------------------- night 11-14: knob pairs in ONE engine


def variant(v):
    """name -> (knob words, streamed, strip the prediction field, cancelled pre-request)"""
    off = v.startswith("off"); core = v.split("_")
    return (" ".join(LEVERS_OFF) if off else "", "s" in core[1:], off, "c" in core[1:])


def strip_pred(c):
    if "prediction" not in c["body"]: return c
    c = dict(c); c["body"] = {k: x for k, x in c["body"].items() if k != "prediction"}; return c


class Knob:
    cur = None
    @classmethod
    def set(cls, w):
        if cls.cur != w: setknobs(w); cls.cur = w


def pair(cases, out, o):
    C = select(load(cases), o); vs0 = o.get("vars", "off,on,off2").split(","); dl = float(o.get("dl", 1e18)); rot = int(o.get("rot", 1))
    rng = random.Random(int(o.get("seed", 7))); done = set()
    if os.path.exists(out):
        for l in open(out):
            try: r = json.loads(l); done.add((r["id"], r["label"]))
            except Exception: pass
    U = units(C, o.get("order", "fwd"), rng); f = open(out, "a"); n = 0; t0 = time.time(); bad = 0
    for ui, u in enumerate(U):
        for c in u:
            k = ui % len(vs0) if rot else 0; vs = vs0[k:] + vs0[:k]; got = {}
            for v in vs:
                if (c["id"], v) in done: continue
                if time.time() > dl:
                    Knob.set(""); print(f"pair: {n} replies, DEADLINE (resumable)"); sys.exit(3)
                w, stream, strip, canc = variant(v); Knob.set(w)
                if canc: run_case(c, True, cancel_after=rng.randint(1, 5))
                r = run_case(strip_pred(c) if strip else c, stream)
                r.update(id=c["id"], mode="s" if stream else "n", label=v, conc=1)
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush(); n += 1; got[v] = r
            # live alarm: on-variants against the off variant of the same form
            for v, r in got.items():
                if v.startswith("on"):
                    b = got.get("off_s" if r["mode"] == "s" else "off")
                    if b is not None and diff_fields(b, r, c["tags"]):
                        bad += 1; print("  LIVE DIFF", c["id"], v, diff_fields(b, r, c["tags"]), c["tags"], flush=True)
    Knob.set("")
    print(f"pair: {n} replies in {time.time() - t0:.0f} s, vars {','.join(vs0)}, live differences {bad}")


def pairsum(out, cases, diffs=None):
    tags = {}
    for p in (cases or "").split(","):
        if p and os.path.exists(p):
            for c in load(p): tags[c["id"]] = c["tags"]
    by = {}
    for r in load(out):
        if r.get("cancelled"): continue
        by.setdefault(r["id"], {}).setdefault(r["label"], r)
    labs = sorted({v for d in by.values() for v in d})
    comps = []
    for v in labs:
        st = "s" in v.split("_")[1:]
        if v.startswith("on"): comps.append((v, "off_s" if st else "off", "LEVERS: " + v + " vs " + ("off_s" if st else "off")))
        elif v.startswith("off") and v not in ("off", "off_s") and not st: comps.append((v, "off", "BASE determinism: " + v + " vs off"))
    if "off_s" in labs: comps.append(("off_s", "off", "BASE stream form: off_s vs off"))
    if "on_s" in labs and "on" in labs: comps.append(("on_s", "on", "stream form with levers: on_s vs on"))
    allbad = []
    for a, b, title in comps:
        T = {}; tot = [0, 0, 0, 0]; bad = []
        for i, d in by.items():
            if a not in d or b not in d: continue
            ra, rb = d[a], d[b]; df = diff_fields(rb, ra, tags.get(i))
            k = 3 if df else (1 if ra.get("st") == 200 else 2)
            tot[0] += 1; tot[k] += 1
            for t in (tags.get(i) or ["?"]):
                if t.startswith("pt"): continue
                e = T.setdefault(t, [0, 0, 0, 0]); e[0] += 1; e[k] += 1
            if df:
                x = dict(id=i, cmp=title, fields=df, tags=tags.get(i))
                for fld in df:
                    if fld != "lp": x["ref_" + fld] = rb.get(fld); x["got_" + fld] = ra.get(fld)
                if "text" in df:
                    p = 0; A, B = rb.get("text") or "", ra.get("text") or ""
                    while p < min(len(A), len(B)) and A[p] == B[p]: p += 1
                    x["text_first_diff_at"] = p
                bad.append(x)
        if not tot[0]: continue
        print(f"== {title}\n{'tag':<14} tested identical same-error DIFFERENT")
        for t in sorted(T): print(f"{t:<14} {T[t][0]:>6} {T[t][1]:>9} {T[t][2]:>10} {T[t][3]:>9}")
        print(f"{'TOTAL':<14} {tot[0]:>6} {tot[1]:>9} {tot[2]:>10} {tot[3]:>9}")
        for x in bad[:10]: print("  DIFF", x["id"], x["fields"], x.get("tags"), "first text diff at", x.get("text_first_diff_at"))
        allbad += bad
    if diffs:
        with open(diffs, "w") as f:
            for x in allbad: f.write(json.dumps(x, ensure_ascii=False) + "\n")


def sess(scripts, out, o):
    """Live sessions. A script: {id, tags, body (parameters), pre: [messages], turns: [{u: text} | {edit: i, u: text} | {regen: 1}]}.
    edit i: the history is cut before its i-th user message, which is replaced (an edited earlier turn); regen: the last reply is dropped and asked again."""
    S = load(scripts); vs0 = o.get("vars", "off,on").split(","); dl = float(o.get("dl", 1e18)); done = set(); rng = random.Random(3)
    if o.get("limit"): S = S[: int(o["limit"])]
    if os.path.exists(out):
        for l in open(out):
            try: r = json.loads(l); done.add((r["sid"], r["label"]))
            except Exception: pass
    f = open(out, "a"); fc = open(out + ".cases", "a"); n = 0; bad = 0
    for si, s in enumerate(S):
        k = si % len(vs0); vs = vs0[k:] + vs0[:k]; tr = {}
        for v in vs:
            if (s["id"], v) in done: continue
            if time.time() > dl:
                Knob.set(""); print(f"sess: {n} replies, DEADLINE (resumable)"); sys.exit(3)
            w, stream, strip, canc = variant(v); Knob.set(w); msgs = [dict(m) for m in s.get("pre", [])]; rows = []
            for ti, t in enumerate(s["turns"]):
                if "edit" in t:
                    ui = [j for j, m in enumerate(msgs) if m["role"] == "user"][t["edit"]]; msgs = msgs[:ui] + [{"role": "user", "content": t["u"]}]
                elif t.get("regen"):
                    while msgs and msgs[-1]["role"] != "user": msgs.pop()
                else: msgs.append({"role": "user", "content": t["u"]})
                body = dict(s["body"]); body["messages"] = [dict(m) for m in msgs]
                c = {"id": f"{s['id']}.t{ti}", "route": "chat", "tags": s["tags"] + (["edit"] if "edit" in t else ["regen"] if t.get("regen") else []) + [f"turn{min(ti, 3)}{'+' if ti >= 3 else ''}"], "body": body, "grp": s["id"], "k": ti}
                if canc and ti: run_case(c, True, cancel_after=rng.randint(1, 5))
                r = run_case(c, stream); r.update(id=c["id"], sid=s["id"], mode="s" if stream else "n", label=v, conc=1, h=hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12])
                rows.append((c, r)); n += 1
                msgs.append({"role": "assistant", "content": r.get("text") or ""})
            for c, r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush(); tr[v] = rows
            if v == "off" or (v == vs[0] and "off" not in vs):
                for c, r in rows: fc.write(json.dumps(c, ensure_ascii=False) + "\n")
                fc.flush()
        for v, rows in tr.items():
            if v.startswith("on") and tr.get("off"):
                for (c, r), (c0, r0) in zip(rows, tr["off"]):
                    if r["h"] != r0["h"] or diff_fields(r0, r, c["tags"]):
                        bad += 1; print("  LIVE DIFF", c["id"], v, "(request differs: an earlier turn diverged)" if r["h"] != r0["h"] else diff_fields(r0, r, c["tags"]), flush=True); break
    Knob.set(""); print(f"sess: {n} replies, {len(S)} sessions, vars {','.join(vs0)}, sessions with a live difference {bad}")


def gen3(prompts, fresh, path, sesspath, seed=31):
    """Extra static cases and session scripts; needs no engine. PROMPTS = kit/prompts.jsonl (320 held), FRESH = kit-D/fresh.jsonl (out of fit)."""
    rng = random.Random(seed); C = []
    P = [json.loads(l)["prompt"] for l in open(prompts) if l.strip()]
    F = []
    for l in open(fresh):
        d = json.loads(l); F.append(d.get("prompt") or d.get("text") or d.get("content") or "")
    F = [x for x in F if isinstance(x, str) and x]
    U = lambda t: [{"role": "user", "content": t}]
    def add(tags, messages, route="chat", **kw):
        b = {"max_tokens": 32, "temperature": 0, "top_p": 1, "chat_template_kwargs": {"enable_thinking": False}}; b.update(kw)
        think = b.pop("think", None)
        if think is not None: b["chat_template_kwargs"] = {"enable_thinking": bool(think)}
        if route == "chat": b["messages"] = messages
        elif route == "resp":
            b["input"] = messages; b["max_output_tokens"] = b.pop("max_tokens")
            if "tools" in b: b["tools"] = [dict(type="function", **t["function"]) for t in b["tools"]]
        else:
            sysm = [m["content"] for m in messages if m["role"] == "system"]; b["messages"] = [m for m in messages if m["role"] != "system"]
            if sysm: b["system"] = sysm[0]
            if "stop" in b: b["stop_sequences"] = b.pop("stop") if isinstance(b["stop"], list) else [b.pop("stop")]
            if "tools" in b: b["tools"] = [dict(name=t["function"]["name"], description=t["function"]["description"], input_schema=t["function"]["parameters"]) for t in b["tools"]]
            b["thinking"] = {"type": "enabled", "budget_tokens": 256} if think else {"type": "disabled"}
        C.append({"id": f"x{len(C):04d}", "route": route, "tags": tags, "body": b}); return C[-1]
    SYS = [("sysS", "You are a careful assistant of the Quartz Meadow library. " + prose(rng, 12)), ("sysM", "Company handbook:\n" + prose(rng, 110)), ("sysL", "Reference manual:\n" + prose(rng, 420))]
    STOPS = [["."], ["\n"], ["\n\n"], [",", ";"], [" the"], ["```"], ["and", "or"], ["e"], [":"], ["1"]]
    def shape(i):
        """parameter shape: (tags, kwargs)"""
        x = rng.random(); mt = rng.choice([1, 2, 17, 17, 48, 64, 96, 128, 256])
        if x < 0.24: return ["greedy"], dict(max_tokens=mt)
        if x < 0.50:
            t = rng.choice([0.2, 0.7, 0.7, 1.0, 1.5]); kw = dict(max_tokens=mt, temperature=t, seed=rng.randrange(1, 10 ** 6), top_p=rng.choice([0.95, 0.95, 1, 0.5, 0.8, 0.1]))
            tk = rng.choice([None, None, 1, 2, 5, 19, 20, 21, 50, 100, 0])
            if tk is not None: kw["top_k"] = tk
            return ["sampled", f"T{t}", "topk_" + ("dflt" if tk is None else str(tk))], kw
        if x < 0.60: return ["greedy", "think"], dict(max_tokens=max(mt, 64), think=True)
        if x < 0.70: return ["sampled", "think"], dict(max_tokens=max(mt, 64), think=True, temperature=rng.choice([0.2, 0.7, 1.0]), seed=rng.randrange(1, 10 ** 6), top_p=0.95)
        if x < 0.78: return ["stop", "greedy"], dict(max_tokens=max(mt, 48), stop=rng.choice(STOPS))
        if x < 0.83: return ["stop", "sampled"], dict(max_tokens=max(mt, 48), stop=rng.choice(STOPS), temperature=0.7, seed=rng.randrange(1, 10 ** 6), top_p=0.95)
        if x < 0.88: return ["penalty", "sampled"], dict(max_tokens=max(mt, 48), temperature=rng.choice([0.7, 1.0]), seed=rng.randrange(1, 10 ** 6), top_p=0.95, **rng.choice([dict(repetition_penalty=1.1), dict(presence_penalty=0.6), dict(frequency_penalty=0.4), dict(repetition_penalty=1.2, presence_penalty=0.3)]))
        if x < 0.91: return ["penalty", "greedy"], dict(max_tokens=max(mt, 48), **rng.choice([dict(repetition_penalty=1.1), dict(presence_penalty=0.6), dict(frequency_penalty=0.4)]))
        if x < 0.94: return ["json", "greedy"], dict(max_tokens=128, response_format={"type": "json_object"})
        if x < 0.96: return ["json", "schema", "sampled"], dict(max_tokens=128, temperature=0.7, seed=rng.randrange(1, 10 ** 6), top_p=0.95, response_format={"type": "json_schema", "json_schema": {"name": "out", "strict": True, "schema": SCHEMA1}})
        if x < 0.98: return ["tools", "tc_auto", "greedy"], dict(max_tokens=160, tools=TOOLS, tool_choice="auto")
        return ["tools", "tc_required", "sampled"], dict(max_tokens=160, tools=TOOLS, tool_choice="required", temperature=0.7, seed=rng.randrange(1, 10 ** 6), top_p=0.95)
    # R. random cross product: prompt x shape x route x (system prompt | earlier turns)
    for i in range(520):
        src = "held" if rng.random() < 0.6 else "fresh"; t = rng.choice(P if src == "held" else F); tg, kw = shape(i); msgs = U(t); extra = []
        y = rng.random()
        if y < 0.18: k, sy = rng.choice(SYS); msgs = [{"role": "system", "content": sy}] + msgs; extra = ["sysprefix", k]
        elif y < 0.30: msgs = U(rng.choice(P)) + [{"role": "assistant", "content": prose(rng, rng.randint(2, 30))}] + msgs; extra = ["multiturn"]
        route = "chat"; z = rng.random()
        if z < 0.10 and "json" not in tg and "penalty" not in tg: route = "resp"
        elif z < 0.20 and "json" not in tg and "penalty" not in tg: route = "msg"
        add(["rand", src] + tg + extra + ([] if route == "chat" else ["route_" + route]), msgs, route=route, **kw)
    # K. top_k x temperature grid on one code and one chat prompt, fixed seed
    for t in (0.2, 0.7, 1.0, 1.5):
        for tk in (1, 20, 21, 50, 0):
            add(["grid", "sampled", f"T{t}", f"topk_{tk}", "code"], U("Write a function that merges two sorted lists in Python, with a docstring and three tests."), max_tokens=96, temperature=t, top_k=tk, seed=77, top_p=0.95)
            add(["grid", "sampled", f"T{t}", f"topk_{tk}", "chat"], U("Tell me a short story about a lighthouse keeper and a storm."), max_tokens=96, temperature=t, top_k=tk, seed=78, top_p=1)
    # N. repeated text and near-repeats that diverge (copy drafts), with stops and sampling
    for i in range(16):
        nr = near_repeat(rng, 20 + i); doc = prose(rng, 14 + i); f = code_file(rng, 4 + i % 3, ["py", "js", "rs"][i % 3])
        add(["copy", "near", "greedy"], U("Print these records again exactly, but swap the first and second word of every third record:\n\n" + nr), max_tokens=420)
        add(["copy", "near", "sampled"], U("Print these records again, but change every status=FAIL to status=RETRY:\n\n" + nr), max_tokens=320, temperature=0.7, seed=4000 + i, top_p=0.95)
        add(["copy", "rewrite", "sampled", "pred"], U(f"Rename the loop variable to `idx` in every function and print the whole file.\n\n```\n{f}```"), max_tokens=420, temperature=0.7, seed=4100 + i, top_p=0.95, prediction={"type": "content", "content": f})
        add(["copy", "stop", "pred", "greedy"], U("Repeat the following text exactly, with no commentary:\n\n" + doc), max_tokens=320, stop=[doc.split(" ")[len(doc.split(" ")) * 2 // 3] + " "], prediction={"type": "content", "content": doc})
        add(["copy", "think", "greedy"], U("Repeat the following text exactly, with no commentary:\n\n" + doc), max_tokens=420, think=True)
        add(["copy", "route_" + ["resp", "msg"][i % 2], "greedy"], U("Repeat the following text exactly, with no commentary:\n\n" + doc), route=["resp", "msg"][i % 2], max_tokens=320)
    # O. long outputs (max_tokens 4,096)
    add(["long_out", "greedy", "mt4096"], U("Count from 1 to 2000, comma separated, no other text."), max_tokens=4096)
    add(["long_out", "sampled", "mt4096"], U("Write a very long, detailed story about a city built on the back of a giant turtle. Do not stop early."), max_tokens=4096, temperature=0.7, seed=4242, top_p=0.95)
    add(["long_out", "copy", "greedy", "mt4096"], U("Repeat the following file exactly twice, with no commentary:\n\n" + code_file(rng, 30, "js")), max_tokens=4096)
    add(["long_out", "think", "greedy", "mt4096"], U("Prove that the square root of 2 is irrational, then list the first 300 prime numbers."), max_tokens=4096, think=True)
    # G. images
    def img(t, k=1): return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": png(rng)}} for _ in range(k)] + [{"type": "text", "text": t}]}]
    for i in range(6):
        add(["image", "greedy"], img("Describe this image in two sentences."), max_tokens=64)
        add(["image", "sampled"], img("Name the shapes and colours."), max_tokens=64, temperature=0.7, seed=4300 + i, top_p=0.95)
    add(["image", "longtext", "greedy"], img("Notes:\n" + prose(rng, 700) + "\n\nDescribe the image, then give the last word of the notes."), max_tokens=64)
    add(["image", "longtext", "sampled"], img("Notes:\n" + prose(rng, 300) + "\n\nDescribe the image in one sentence."), max_tokens=64, temperature=0.7, seed=4310, top_p=0.95)
    add(["image", "greedy"], img("What differs between these three images?", 3), max_tokens=96)
    with open(path, "w") as f:
        for c in C: f.write(json.dumps(c, ensure_ascii=False) + "\n")
    tg = {}
    for c in C:
        for t in c["tags"]: tg[t] = tg.get(t, 0) + 1
    print(len(C), "cases;", " ".join(f"{k}={v}" for k, v in sorted(tg.items())))
    # S. session scripts
    S = []
    for i in range(30):
        kind = ["greedy", "sampled", "greedy", "think", "sampled", "greedy"][i % 6]
        b = {"max_tokens": [48, 96, 64][i % 3], "temperature": 0, "top_p": 1, "chat_template_kwargs": {"enable_thinking": kind == "think"}}
        if kind == "sampled": b.update(temperature=[0.7, 1.0][i % 2], seed=5000 + i, top_p=0.95)
        if kind == "think": b["max_tokens"] = 200
        pre = []
        if i % 3 == 0: pre = [{"role": "system", "content": SYS[(i // 3) % 3][1]}]
        turns = [{"u": rng.choice(P if j % 2 == 0 else F)} for j in range(rng.randint(3, 5))]
        if i % 2 == 0: turns.append({"regen": 1})
        if i % 3 != 2: turns.append({"edit": rng.randint(0, 1), "u": rng.choice(F) + " (edited)"}); turns.append({"u": "Now say that again, shorter."})
        if i % 5 == 4: turns.insert(1, {"u": "Remember this text and repeat it back exactly when asked:\n" + prose(rng, 25)}); turns.append({"u": "Repeat the text I asked you to remember, exactly."})
        S.append({"id": f"s{i:02d}", "tags": ["sess", kind] + (["sysprefix"] if pre else []), "body": b, "pre": pre, "turns": turns})
    with open(sesspath, "w") as f:
        for s in S: f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(len(S), "session scripts,", sum(len(s["turns"]) for s in S), "turns")



def main():
    a = sys.argv[1:]; m = a[0]
    if m == "gen": gen(a[1], int(a[2]) if len(a) > 2 else 12)
    elif m == "run": run(a[1], a[2], a[3], opts(a[4:]))
    elif m == "gen2": gen2(a[1], a[2], a[3])
    elif m == "cmp": cmp_(a[1], a[2], a[3] if len(a) > 3 else None, a[4] if len(a) > 4 else None, a[5] if len(a) > 5 else None)
    elif m == "pair": pair(a[1], a[2], opts(a[3:]))
    elif m == "pairsum": pairsum(a[1], a[2], a[3] if len(a) > 3 else None)
    elif m == "sess": sess(a[1], a[2], opts(a[3:]))
    elif m == "gen3": gen3(a[1], a[2], a[3], a[4], int(a[5]) if len(a) > 5 else 31)
    elif m == "self": self_(a[1])
    elif m == "soak": soak(a[1], a[2], a[3], a[4], opts(a[5:]))
    elif m == "bisect": bisect(a[1], a[2], a[3], a[4] if len(a) > 4 else "n")
    elif m == "one":
        c, r = one(a[1], a[2], a[3] if len(a) > 3 else "n", " ".join(a[4:]) if len(a) > 4 else None); r.pop("lp", None); print(json.dumps(r, ensure_ascii=False, indent=1)[:6000])
    else: print(__doc__)


if __name__ == "__main__": main()
