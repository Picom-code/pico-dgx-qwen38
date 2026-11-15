#!/usr/bin/env python3
"""soak.py: pre-ship soak driver (agent B, night 11-14). Python stdlib only; reuses builder D's rt.py (same directory) for the
request material, the three API routes and the canonical reply.

The claim under test: under mixed load every reply equals the reply the SAME request gets solo from the engine with every lever off
(greedy: identical; sampled: identical per seed), and the engine neither crashes, hangs nor leaks.

  soak.py mk POOL.jsonl [SEED] [k=v]        the request pool, built offline (single requests + agent-session scripts)
  soak.py ref POOL REF.jsonl [k=v]          reference pass on the levers-off engine: every request solo, not streamed; agent sessions
                                            are run turn by turn and their turn requests (history = the reference replies) are written
                                            into REF. Resumable. Options: dl=EPOCH rep=PCT (share repeated at the end: base determinism)
  soak.py load POOL REF OUTDIR MIN [k=v]    load phase: 1-8 (levels=) in flight, cancels, slow readers, bursts, idle gaps, abandoned and
                                            resumed sessions, duplicates, long prompts; every reply checked; monitor every 30 s.
                                            Options: levels=1,2,4,8 per=SEC low=GiB winend=EPOCH cutoff=EPOCH seed=N maxp=TOK longgap=SEC
                                            sess=PCT agents=N (agent mode: N session runners on top of the levels) full=ID
  soak.py inject POOL REF OUTDIR [k=v]      failure injection at the HTTP level, each followed by a health check and canary requests
  soak.py solo POOL REF ID [N] [mode]       one request N times solo; equal to the reference?
  soak.py hammer POOL REF ID N [k=v]        one request N times under background load (conc= bg= knob= mode= cancel=): reproducer / lever bisect
  soak.py bisect POOL REF ID [mode] [N]     one request solo with each lever off through the knob file
  soak.py report OUTDIR [OUTDIR...]         the soak table row(s): requests, mismatches, errors, hangs, memory trend, low-water
Environment: RT_URL (default http://127.0.0.1:18301), RT_MODEL, SOAK_CONTAINER (docker name for the engine log; default q38-run-B).
"""
import sys, os, json, time, random, threading, http.client, urllib.parse, hashlib, subprocess, re, socket
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rt

CONT = os.environ.get("SOAK_CONTAINER", "q38-run-B")
CUTOFF_DEFAULT = 1794744300  # 2026-11-15 12:05:00 UTC: nothing of this driver runs past it
U = lambda t: [{"role": "user", "content": t}]


def now(): return time.time()
def hms(t=None): return time.strftime("%H:%M:%S", time.gmtime(t if t is not None else now()))
def load(p): return [json.loads(l) for l in open(p) if l.strip()]
def opts(argv):
    o = {}
    for a in argv:
        k, _, v = a.partition("="); o[k] = v
    return o

# ---------------------------------------------------------------- progress tracker (hang = no byte for 60 s with a request in flight)

class Track:
    def __init__(self):
        self.lock = threading.Lock(); self.last = now(); self.inflight = {}; self.seq = 0; self.silences = []
    def start(self, cid, action):
        with self.lock:
            self.seq += 1; k = self.seq; self.inflight[k] = [cid, action, now(), None]
            if len(self.inflight) == 1: self.last = now()
            return k
    def tick(self, k):
        t = now(); self.last = t
        e = self.inflight.get(k)
        if e is not None and e[3] is None: e[3] = t
    def end(self, k):
        with self.lock: self.inflight.pop(k, None); self.last = now()
    def snapshot(self):
        with self.lock: return [(v[0], v[1], round(now() - v[2], 1), v[3] is not None) for v in self.inflight.values()]

TR = Track()

# ---------------------------------------------------------------- HTTP (rt.post with a progress hook and more client behaviours)

def post(path, body, stream=False, slow=0.0, cancel_after=None, key=None, stall=None, drop=None, t_first=1700, t_gap=float(os.environ.get('SOAK_TGAP', 120)), raw=None, headers=None):
    """Like rt.post. stall=(n, sec): stop reading for sec seconds after n stream events. drop=sec: send the request, wait sec, close without reading.
    raw=bytes: send these bytes as the body instead of json.dumps(body)."""
    u = urllib.parse.urlparse(rt.URL); t0 = now(); out = {"st": 0}
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=t_first)
    try:
        data = raw if raw is not None else json.dumps(body).encode(); c.connect(); sk = c.sock
        h = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}; h.update(headers or {})
        c.request("POST", path, data, h)
        if drop is not None:
            time.sleep(drop); out["dropped"] = True; out["wall"] = round(now() - t0, 3); return out
        r = c.getresponse(); out["st"] = r.status
        if key: TR.tick(key)
        if not stream or r.status != 200:
            rawb = r.read(); out["wall"] = round(now() - t0, 3)
            try: out["raw"] = json.loads(rawb)
            except ValueError: out["err"] = rawb[:300].decode("utf-8", "replace")
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
            t = now()
            if key: TR.tick(key)
            if last is None:
                out["ttft"] = round(t - t0, 3); sk.settimeout(t_gap)
            else: gap = max(gap, t - last)
            last = t; ev.append(d); n += 1
            if cancel_after is not None and n >= cancel_after:
                out["cancelled"] = True; break
            if stall and n == stall[0]:
                time.sleep(stall[1]); last = now()
            if slow: time.sleep(slow); last = now()
        out["events"] = ev; out["gap"] = round(gap, 2); out["wall"] = round(now() - t0, 3)
    except Exception as e:
        out["err"] = type(e).__name__ + ": " + str(e)[:200]; out["wall"] = round(now() - t0, 3)
        if "timed out" in str(e): out["hang"] = True
    finally:
        try: c.close()
        except Exception: pass
    return out


def get(path, timeout=5):
    u = urllib.parse.urlparse(rt.URL)
    try:
        c = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout); c.request("GET", path); r = c.getresponse(); b = r.read(); c.close()
        return r.status, b
    except Exception as e:
        return 0, (type(e).__name__ + ": " + str(e)[:100]).encode()


def send(case, stream, **kw):
    body = dict(case["body"]); body["model"] = rt.MODEL
    if stream:
        body["stream"] = True
        if case["route"] == "chat": body["stream_options"] = {"include_usage": True}
    res = post(rt.PATHS[case["route"]], body, stream, **kw)
    o = rt.canon(case["route"], res, stream) if not res.get("dropped") else {"st": 0, "dropped": True, "wall": res.get("wall")}
    return o

# ---------------------------------------------------------------- the pool

def mkbody(route, messages, **kw):
    """One request body on a route from OpenAI-style messages (the conversion rt.py's matrix uses)."""
    b = {"max_tokens": 64, "temperature": 0, "top_p": 1, "chat_template_kwargs": {"enable_thinking": False}}
    b.update(kw); think = b.pop("think", None)
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
    return b


def repo_text(rng, n_units):
    """Agent context: a made-up repository dump (code files in three languages, notes, tables); ~450 tokens a unit."""
    o = []
    for i in range(n_units):
        k = rng.random()
        if k < 0.5: o.append(f"### file src/{rng.choice(rt.WORDS)}_{i}.{['py', 'js', 'rs'][i % 3]}\n```\n{rt.code_file(rng, 4, ['py', 'js', 'rs'][i % 3])}```")
        elif k < 0.8: o.append(f"### notes/{rng.choice(rt.WORDS)}_{i}.md\n{rt.prose(rng, 28)}")
        elif k < 0.9: o.append(f"### data/{rng.choice(rt.WORDS)}_{i}.md\n{rt.table(rng, 14)}")
        else: o.append(f"### logs/{rng.choice(rt.WORDS)}_{i}.log\n{rt.near_repeat(rng, 12)}")
    return "\n\n".join(o)


def mk(path, seed=1114, o=None):
    o = o or {}; rng = random.Random(seed); C = []; scale = float(o.get("scale", 1.0))
    def add(tags, messages, route="chat", **kw):
        refof = kw.pop("ref_of", None)
        c = {"id": f"q{len(C):04d}", "kind": "single", "route": route, "tags": tags, "body": mkbody(route, messages, **kw)}
        if refof: c["ref_of"] = refof
        C.append(c); return c
    n = [0]
    def samp(**kw):
        n[0] += 1; d = dict(temperature=0.7, top_p=0.95, seed=5000 + n[0]); d.update(kw); return d
    CH = rt.CHAT; ML = rt.MULTI
    # chat: greedy, sampled, thinking, other languages, odd unicode
    for i, t in enumerate(CH):
        add(["chat", "greedy"], U(t), max_tokens=[96, 160, 256][i % 3])
        add(["chat", "sampled"], U(t), max_tokens=[96, 128][i % 2], **samp(**[{}, dict(top_k=20), dict(temperature=1.0), dict(top_k=5, top_p=0.8)][i % 4]))
        if i % 2 == 0: add(["chat", "greedy", "think"], U(t), max_tokens=220, think=True)
        if i % 3 == 0: add(["chat", "sampled", "think"], U(t), max_tokens=220, think=True, **samp())
    for i, t in enumerate(ML[:10]): add(["chat", "nonenglish", "greedy" if i % 2 else "sampled"], U(t), max_tokens=96, **({} if i % 2 else samp()))
    for i in (0, 2, 4, 7, 13): add(["chat", "unicode", "greedy"], U(rt.ODD[i]), max_tokens=64)
    add(["chat", "system", "greedy"], [{"role": "system", "content": "You are terse. Answer in one sentence."}] + U(CH[3]), max_tokens=64)
    for i in range(4):
        conv = []
        for k in range(3 + i):
            conv.append({"role": "user", "content": f"Turn {k}: " + rt.prose(rng, 4) + " " + CH[(i + k) % len(CH)]}); conv.append({"role": "assistant", "content": f"(reply {k}) " + rt.prose(rng, 5)})
        add(["chat", "multiturn", "greedy"], conv[:-1], max_tokens=96)
    # code
    for i, l in enumerate(rt.LANGS[:14]):
        add(["code", "greedy"], U(f"Write {rt.TASKS[i % len(rt.TASKS)]} in {l}. Code only."), max_tokens=[160, 256, 400][i % 3])
        if i % 3 == 0: add(["code", "sampled"], U(f"Write {rt.TASKS[(i + 1) % len(rt.TASKS)]} in {l}, with comments."), max_tokens=256, **samp(top_k=20))
    # rewrites / copies, without and with `prediction` (a wrong or right prediction never changes the reply: its reference is the request without it)
    for i in range(10):
        lang = ["py", "js", "rs"][i % 3]; f = rt.code_file(rng, 4 + i % 5, lang); doc = rt.prose(rng, 12 + 4 * i); nr = rt.near_repeat(rng, 16 + 2 * i)
        m1 = U(f"Rewrite this file so that every function also takes a `scale` argument that multiplies the result. Print the whole file.\n\n```\n{f}```")
        m2 = U("Print these records again, but change every status=FAIL to status=RETRY and leave everything else as it is:\n\n" + nr)
        m3 = U("Repeat the following text exactly, with no commentary:\n\n" + doc)
        a = add(["rewrite", "copy", "greedy"], m1, max_tokens=700)
        add(["rewrite", "copy", "pred", "greedy"], m1, max_tokens=700, prediction={"type": "content", "content": f}, ref_of=a["id"])
        b = add(["rewrite", "copy", "near", "greedy"], m2, max_tokens=600)
        add(["rewrite", "copy", "near", "pred", "greedy"], m2, max_tokens=600, prediction={"type": "content", "content": nr.replace("FAIL", "RETRY")}, ref_of=b["id"])
        c = add(["copy", "greedy"], m3, max_tokens=500)
        if i % 2 == 0: add(["copy", "pred", "predwrong", "greedy"], m3, max_tokens=500, prediction={"type": "content", "content": rt.prose(rng, 20)}, ref_of=c["id"])
        if i % 3 == 0:
            d = add(["copy", "sampled"], m3, max_tokens=300, **samp())
            add(["copy", "pred", "sampled"], m3, max_tokens=300, prediction={"type": "content", "content": doc}, ref_of=d["id"], **{k: v for k, v in d["body"].items() if k in ("temperature", "top_p", "seed")})
        if i % 4 == 1:
            for route in ("resp", "msg"):
                e = add(["route_" + route, "copy", "greedy"], m3, route=route, max_tokens=400)
                add(["route_" + route, "copy", "pred", "greedy"], m3, route=route, max_tokens=400, prediction={"type": "content", "content": doc}, ref_of=e["id"])
    # structured output, tools, stops, logprobs
    JP = "Invent a person and reply as JSON with the fields name, age and tags."
    for i in range(4):
        add(["json", "greedy"], U(JP + f" Variant {i}."), max_tokens=140, response_format={"type": "json_object"})
        add(["json", "schema", "greedy"], U(JP if i % 2 == 0 else "Solve 12*13 step by step as JSON."), max_tokens=200, response_format={"type": "json_schema", "json_schema": {"name": "out", "strict": True, "schema": rt.SCHEMA1 if i % 2 == 0 else rt.SCHEMA2}})
        add(["json", "sampled"], U(JP + f" Number {i}."), max_tokens=140, response_format={"type": "json_object"}, **samp())
    TP = ["What is the weather in Paris right now?", "Search the web for the tallest mountain in Europe and tell me.", "Save a file notes/todo.txt with a five-item shopping list.",
          "What is the weather in Tokyo and in Lima?", "Hello, how are you?", "Write the following to out/data.txt exactly:\n" + rt.near_repeat(rng, 14)]
    for i, t in enumerate(TP):
        add(["tools", "greedy"], U(t), max_tokens=260, tools=rt.TOOLS, tool_choice="auto")
        add(["tools", "greedy", "tc_required"], U(t), max_tokens=260, tools=rt.TOOLS, tool_choice="required")
        if i % 2 == 0: add(["tools", "sampled"], U(t), max_tokens=260, tools=rt.TOOLS, tool_choice="auto", **samp())
        if i % 3 == 0: add(["tools", "think", "greedy"], U(t), max_tokens=320, tools=rt.TOOLS, think=True)
    add(["tools", "multiturn", "greedy"], [{"role": "user", "content": TP[0]}, {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
                                           {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 18, \"sky\": \"cloudy\"}"}], max_tokens=100, tools=rt.TOOLS)
    for i, s in enumerate([["."], ["\n\n"], ["and", "or", "but"], ["```"], ["3"], ["planet"]]): add(["stop", "greedy" if i % 2 == 0 else "sampled"], U(CH[(i * 2 + 1) % len(CH)]), max_tokens=120, stop=s, **({} if i % 2 == 0 else samp()))
    for i in range(4): add(["logprobs", "greedy" if i % 2 == 0 else "sampled"], U(CH[i + 5]), max_tokens=24, logprobs=True, top_logprobs=[1, 5][i % 2], **({} if i % 2 == 0 else samp()))
    # the other routes
    for i in range(8):
        for route in ("resp", "msg"):
            t = (CH + ML)[(i * 3 + 1) % 29]
            add(["route_" + route, "greedy"], U(t), route=route, max_tokens=[48, 96, 160][i % 3])
            if i % 2 == 0: add(["route_" + route, "sampled"], U(t), route=route, max_tokens=64, **samp())
            if i % 4 == 1: add(["route_" + route, "tools", "greedy"], U(TP[i % len(TP)]), route=route, max_tokens=220, tools=rt.TOOLS)
            if i % 4 == 2: add(["route_" + route, "think", "greedy"], U(t), route=route, max_tokens=300, think=True)
            if i % 4 == 3: add(["route_" + route, "system", "stop", "greedy"], [{"role": "system", "content": "You are terse."}] + U(CH[i]), route=route, max_tokens=96, stop=["."])
    # images
    def img(t, k=1): return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": rt.png(rng)}} for _ in range(k)] + [{"type": "text", "text": t}]}]
    add(["image", "greedy"], img("Describe this image in one sentence."), max_tokens=64)
    add(["image", "sampled"], img("What colours do you see?"), max_tokens=64, **samp())
    add(["image", "greedy"], img("Compare the two images.", 2), max_tokens=80)
    add(["image", "copy", "greedy"], img("Describe the image in one line, then repeat this text exactly:\n\n" + rt.prose(rng, 16)), max_tokens=300)
    # long outputs
    add(["long_out", "greedy"], U("Count from 1 to 700, comma separated, no other text."), max_tokens=2200)
    add(["long_out", "copy", "greedy"], U("Repeat the following file exactly, with no commentary:\n\n" + rt.code_file(rng, 30, "py")), max_tokens=2600)
    add(["long_out", "sampled", "think"], U("Write a detailed 500-word essay on the history of bridges."), max_tokens=1000, think=True, **samp())
    # documents of 5K-40K tokens; two questions share a document (prompt-cache resume)
    for i, ns in enumerate([350, 700, 1400, 2100, 2800]):
        doc = rt.prose(rng, int(ns * scale))
        add(["doc", "greedy", f"doc{i}"], U("Document:\n" + doc + "\n\nQ: what is the first word of the document? Then summarise the document in one sentence."), max_tokens=80)
        add(["doc", "greedy" if i % 2 else "sampled", f"doc{i}"], U("Document:\n" + doc + "\n\nQ: what is the last word of the document? Then name its three most frequent words."), max_tokens=80, **({} if i % 2 else samp()))
        if i == 2: add(["doc", "edit", "greedy", f"doc{i}"], U("Document:\n" + doc[: len(doc) // 2] + " INSERTED SENTENCE HERE. " + doc[len(doc) // 2:] + "\n\nQ: what is the first word of the document?"), max_tokens=40)
    # long prompts, ~25K to ~200K tokens (needle recall); words x ~1.35 = tokens. Sent a few times an hour by the load phase.
    for i, nw in enumerate([18000, 37000, 74000, 110000, 146000]):
        nw = int(nw * scale); w = [rng.choice(rt.WORDS) for _ in range(nw)]; needle = f"The secret pass-phrase is {rng.choice(rt.WORDS)}-{rng.choice(rt.WORDS)}-{rng.randint(100, 999)}."
        w.insert(int(nw * [0.5, 0.1, 0.9, 0.35, 0.6][i]), needle); notes = " ".join(w)
        add(["long", "greedy", f"long{i}"], U("Notes:\n" + notes + "\n\nWhat is the secret pass-phrase? Answer with the phrase only."), max_tokens=32)
        if i in (1, 3): add(["long", "sampled", f"long{i}"], U("Notes:\n" + notes + "\n\nWhat is the secret pass-phrase? Then write one sentence about it."), max_tokens=64, **samp())
        if i == 2: add(["long", "greedy", "think", f"long{i}"], U("Notes:\n" + notes + "\n\nWhich word comes right after the secret pass-phrase sentence?"), max_tokens=200, think=True)
    # agent sessions: a long system prompt of the agent's own (three of them share the first part), turns that append the reply and a tool result
    S = []
    shared = "Company handbook (shared by every agent of this team):\n" + repo_text(rng, int(20 * scale))
    spec = [(120, 6)] * 6 + [(88, 6)] * 4 + [(44, 6)] * 6 + [(9, 8)] * 8      # (context units, turns): ~54K x 6, ~40K x 4, ~20K x 6, ~4K x 8 tokens
    for i, (units, turns) in enumerate(spec):
        route = "chat"; tools = False; sampled = (i % 5 == 3); sh = (i % 6 == 1 and units >= 44)
        if units == 9 and i % 4 == 0: route = "msg"
        if units == 9 and i % 4 == 1: route = "resp"
        if i % 3 == 2 and route == "chat": tools = True
        sysm = "You are a coding agent working in the repository below. Be brief and concrete.\n\n" + (shared + "\n\n" if sh else "") + repo_text(rng, max(2, int((units - (20 if sh else 0)) * scale)))
        first = f"Task {i}: find the function that is most likely to overflow for large x, say why in two sentences, then propose the smallest fix as a diff."
        tr = []
        for k in range(turns):
            kind = rng.random()
            if kind < 0.45: body = rt.prose(rng, rng.randint(20, 130))
            elif kind < 0.75: body = rt.code_file(rng, rng.randint(3, 14), rng.choice(["py", "js", "rs"]))
            elif kind < 0.9: body = rt.near_repeat(rng, rng.randint(10, 40))
            else: body = rt.table(rng, rng.randint(8, 30))
            ask = rng.choice(["Continue with the next step.", "Now apply the fix and print the changed function only.", "What should be tested next? One line.",
                              "Summarise what we know so far in three bullet points.", "Write the commit message for this change.", "Print the patched file section again, unchanged apart from the fix."])
            tr.append({"result": body, "ask": ask})
        kw = dict(max_tokens=[120, 160, 200, 96][i % 4])
        if sampled: kw.update(temperature=0.7, top_p=0.95, seed=7000 + i)
        if tools: kw.update(tools=rt.TOOLS, tool_choice="auto")
        S.append({"id": f"s{i:02d}", "kind": "session", "route": route, "tags": ["session", "sampled" if sampled else "greedy", "big" if units >= 88 else "mid" if units >= 44 else "small"] + (["tools"] if tools else []) + (["sharedsys"] if sh else []) + (["route_" + route] if route != "chat" else []),
                  "sys": sysm, "first": first, "turns": tr, "kw": kw, "tools": tools})
    with open(path, "w") as f:
        for c in C + S: f.write(json.dumps(c, ensure_ascii=False) + "\n")
    tg = {}
    for c in C + S:
        for t in c["tags"]: tg[t] = tg.get(t, 0) + 1
    print(f"{len(C)} single requests ({sum(1 for c in C if 'ref_of' in c)} take another request's reference), {len(S)} sessions with {sum(len(s['turns']) for s in S)} turns; bytes {os.path.getsize(path)}")
    print(" ".join(f"{k}={v}" for k, v in sorted(tg.items()) if not re.match(r"(long|doc)\d", k)))

# ---------------------------------------------------------------- sessions: turn requests built from the reference replies

def turn_case(s, k, msgs):
    return {"id": f"{s['id']}.t{k}", "kind": "turn", "route": s["route"], "tags": s["tags"] + [f"turn{k}"], "sess": s["id"], "k": k, "body": mkbody(s["route"], msgs, **s["kw"])}


def extend(s, k, msgs, reply):
    """History after turn k: the reply (text and / or tool calls) and the next tool result + ask."""
    t = s["turns"][k]; text = reply.get("text") or ""; tools = reply.get("tools") or []
    if tools and s["route"] == "chat":
        msgs.append({"role": "assistant", "content": text or None, "tool_calls": [{"id": f"call_{k}_{j}", "type": "function", "function": {"name": a, "arguments": b}} for j, (a, b) in enumerate(tools)]})
        for j in range(len(tools)): msgs.append({"role": "tool", "tool_call_id": f"call_{k}_{j}", "content": t["result"] if j == 0 else "ok"})
        msgs.append({"role": "user", "content": t["ask"]})
    else:
        msgs.append({"role": "assistant", "content": text if text.strip() else "(no reply)"})
        msgs.append({"role": "user", "content": "Tool output:\n" + t["result"] + "\n\n" + t["ask"]})

# ---------------------------------------------------------------- reference pass

KEEP = ("st", "err", "text", "rsn", "tools", "fin", "pt", "ct", "lp", "wall")


def ref(poolp, refp, o):
    P = load(poolp); dl = min(float(o.get("dl", 1e18)), CUTOFF_DEFAULT); rep = float(o.get("rep", 8)); sel = set(o["sel"].split(",")) if o.get("sel") else None
    done = {}
    if os.path.exists(refp):
        for r in load(refp): done[r["id"]] = r
    f = open(refp, "a"); t0 = now(); nrun = 0; bad = 0
    def one(c, extra=None):
        nonlocal nrun, bad
        k = TR.start(c["id"], "ref"); r = send(c, False, key=k); TR.end(k); nrun += 1
        row = {"id": c["id"], "ref": {x: r[x] for x in KEEP if x in r}}
        if "tools" in c["body"] and r.get("st") == 200:
            # the server's streamed and non-streamed replies differ in FORMAT when a tool call is cut by max_tokens (not streamed: raw text; streamed: a partial
            # tool call), with the levers off too: such requests get a streamed reference of their own
            k2 = TR.start(c["id"], "ref_s"); r2 = send(c, True, key=k2); TR.end(k2); nrun += 1
            row["ref_s"] = {x: r2[x] for x in KEEP if x in r2}
        if extra: row.update(extra)
        if r.get("st") != 200: bad += 1; print(f"  ref {c['id']}: status {r.get('st')} {str(r.get('err'))[:160]}", flush=True)
        f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush(); done[c["id"]] = row; return r
    singles = [c for c in P if c["kind"] == "single" and "ref_of" not in c and (sel is None or sel & set(c["tags"]))]
    # short first, long prompts last, so that a cut reference pass still covers most of the pool
    singles.sort(key=lambda c: len(json.dumps(c["body"])))
    for c in singles:
        if c["id"] in done: continue
        if now() > dl: print("REF DEADLINE (resumable)"); sys.exit(3)
        one(c)
        if nrun % 40 == 0: print(f"  ref {nrun} requests, {now() - t0:.0f} s", flush=True)
    for s in [c for c in P if c["kind"] == "session" and (sel is None or sel & set(c["tags"]))]:
        msgs = [{"role": "system", "content": s["sys"]}, {"role": "user", "content": s["first"]}]
        for k in range(len(s["turns"])):
            c = turn_case(s, k, msgs)
            if c["id"] in done: reply = done[c["id"]]["ref"]
            else:
                if now() > dl: print("REF DEADLINE (resumable)"); sys.exit(3)
                reply = one(c, {"body": c["body"], "route": c["route"], "tags": c["tags"], "sess": s["id"], "k": k})
            if reply.get("st") != 200: break
            extend(s, k, msgs, reply)
        print(f"  ref session {s['id']} done ({s['tags'][2]}, prompt tokens of the last turn {done.get(c['id'], {}).get('ref', {}).get('pt')}), {now() - t0:.0f} s", flush=True)
    # base determinism: a share of the requests again, solo; must equal the first pass
    rng = random.Random(3); again = [i for i in sorted(done) if not i.endswith(".again") and rng.random() * 100 < rep and (done[i]["ref"].get("pt") or 0) < 60000]
    C = cases(poolp, refp); same = diff = 0
    for i in again:
        if i + ".again" in done or now() > dl: continue
        k = TR.start(i, "ref2"); r = send(C[i], False, key=k); TR.end(k); d = rdiff(C[i]["ref"], r, C[i]["tags"])
        same += not d; diff += bool(d)
        f.write(json.dumps({"id": i + ".again", "of": i, "same": not d, "fields": d, "got": {x: r.get(x) for x in d if x != "lp"}}, ensure_ascii=False) + "\n"); f.flush()
        if d: print(f"  BASE NOT DETERMINISTIC? {i}: {d}", flush=True)
    print(f"REF DONE: {len([i for i in done if not i.endswith('.again')])} references ({bad} non-200 in this run), repeat check {same} identical / {diff} different, {now() - t0:.0f} s")


def cases(poolp, refp):
    """id -> case with its reference reply (singles, prediction variants, session turns)."""
    P = load(poolp); R = {}; C = {}
    for r in load(refp):
        if "ref" in r: R[r["id"]] = r
    for c in P:
        if c["kind"] != "single": continue
        rid = c.get("ref_of", c["id"])
        if rid in R:
            c = dict(c); c["ref"] = R[rid]["ref"]
            if "ref_s" in R[rid]: c["ref_s"] = R[rid]["ref_s"]
            C[c["id"]] = c
    for i, r in R.items():
        if "body" in r:
            C[i] = {"id": i, "kind": "turn", "route": r["route"], "tags": r["tags"], "sess": r["sess"], "k": r["k"], "body": r["body"], "ref": r["ref"]}
            if "ref_s" in r: C[i]["ref_s"] = r["ref_s"]
    return C

# ---------------------------------------------------------------- checking

def refof(c, stream): return c["ref_s"] if stream and "ref_s" in c else c["ref"]


def norm(k, v):
    # reasoning text: the streamed form ends with one more newline than the non-streamed form (levers off too): not a reply difference
    return (v or "").rstrip("\n") if k == "rsn" else v


def rdiff(a, r, tags):
    skip = ("pt",) if tags and any(t.startswith("route_msg") for t in tags) else ()
    return [k for k in rt.CMP_FIELDS if k not in skip and norm(k, a.get(k)) != norm(k, r.get(k))]


def verdict(c, r, stream=False):
    """(ok, fields, klass). klass: ok | diff | admission | http5xx | error | hang | cancel_ok | cancel_bad"""
    a = refof(c, stream)
    if r.get("cancelled"):
        bad = [k for k in ("text", "rsn") if not (a.get(k) or "").startswith((r.get(k) or "").rstrip("\n") if k == "rsn" else (r.get(k) or ""))]
        return (not bad, bad, "cancel_ok" if not bad else "cancel_bad")
    if r.get("hang"): return (False, ["hang"], "hang")
    d = rdiff(a, r, c["tags"])
    if r.get("stream_concat_differs"): d.append("stream_concat")
    if not d: return (True, [], "ok")
    st = r.get("st") or 0
    if st != 200 and a.get("st") == 200:
        msg = str(r.get("err") or "").lower()
        if st in (429, 503) or (st == 400 and any(w in msg for w in ("queue", "capacity", "busy", "does not fit", "too many", "memory", "pool"))): return (False, d, "admission")
        if st >= 500: return (False, d, "http5xx")
        return (False, d, "error")
    return (False, d, "diff")

# ---------------------------------------------------------------- monitor

PC = re.compile(r"prompt cache: (\d+) tokens, resumed at (\d+), kept (\d+); (\d+) states, (\d+) MiB and (\d+) MiB spare of (\d+) MiB \(hits (\d+), misses (\d+), evicted (\d+), refused (\d+), failed (\d+)\)")
CRASH = re.compile(r"panic|segfault|segmentation|SIGSEGV|SIGABRT|core dumped|CUDA error|CUDA_ERROR|out of memory|OutOfMemory|illegal memory|unreachable|assertion|stack trace|thread \d+ panic", re.I)
NOTE = re.compile(r"q38 snapdelta|q38 pool|q38 elastic|q38 lane|check_diff|mismatch|warning|refus|evict", re.I)


def meminfo():
    ma = 0
    try:
        for l in open("/proc/meminfo"):
            if l.startswith("MemAvailable"): ma = int(l.split()[1]) // 1024
    except OSError: ma = -1
    try: g = subprocess.run(["nvidia-smi", "--query-compute-apps=name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    except Exception: g = ""
    gm = sum(int(x.split(",")[-1]) for x in g.strip().splitlines() if x.split(",")[-1].strip().isdigit())
    return ma, gm


class Monitor(threading.Thread):
    def __init__(self, out, S, low_mib, stopfn):
        super().__init__(daemon=True); self.out = out; self.S = S; self.low = low_mib; self.stopfn = stopfn; self.halt = threading.Event()
        self.since = int(now()); self.crash = 0; self.lowest = 1e12; self.docker = True; self.cache = {}; self.notes = {}; self.pc_lines = 0; self.down = 0
    def logs(self):
        if not self.docker: return
        t = int(now())
        try: p = subprocess.run(["docker", "logs", "--since", str(self.since), CONT], capture_output=True, text=True, timeout=25, errors="replace")
        except Exception: self.docker = False; return
        if p.returncode != 0 and "No such container" in (p.stderr or ""): self.cache["container"] = "gone"; return
        self.since = t
        with open(self.out + "/engine_tail.log", "a") as f:
            for l in (p.stdout + p.stderr).splitlines():
                m = PC.search(l)
                if m:
                    v = [int(x) for x in m.groups()]; self.pc_lines += 1
                    self.cache.update(states=v[3], mib=v[4], spare=v[5], budget=v[6], hits=v[7], misses=v[8], evicted=v[9], refused=v[10], failed=v[11]); continue
                if CRASH.search(l):
                    self.crash += 1; f.write(hms() + " CRASH? " + l[:600] + "\n")
                    with open(self.out + "/crash_lines.txt", "a") as g: g.write(hms() + " " + l[:1000] + "\n")
                elif NOTE.search(l):
                    k = l[:28]; self.notes[k] = self.notes.get(k, 0) + 1
                    if self.notes[k] <= 30 or self.notes[k] % 200 == 0: f.write(hms() + " " + l[:500] + "\n")
    def run(self):
        mf = open(self.out + "/mon.jsonl", "a")
        while not self.halt.is_set():
            ma, gm = meminfo(); self.logs()
            if ma >= 0: self.lowest = min(self.lowest, ma)
            st, hb = get("/health"); h = {}
            try: h = json.loads(hb)
            except Exception: h = {"raw": hb[:120].decode("utf-8", "replace")}
            self.down = self.down + 1 if st != 200 else 0
            fl = TR.snapshot(); sil = now() - TR.last if fl else 0.0
            row = dict(t=round(now()), hms=hms(), ma=ma, gpu=gm, inflight=len(fl), silence=round(sil, 1), health_st=st, health=h, cache=dict(self.cache), crash=self.crash, pc_lines=self.pc_lines, S=dict(self.S))
            mf.write(json.dumps(row) + "\n"); mf.flush()
            if fl and sil > 60:
                with open(self.out + "/silence.jsonl", "a") as g: g.write(json.dumps(dict(t=hms(), silence=round(sil, 1), inflight=fl, health=h)) + "\n")
            if 0 <= ma < self.low: self.stopfn(f"LOW WATER: MemAvailable {ma} MiB < {int(self.low)} MiB"); break
            if self.down >= 4 and not os.path.exists(self.out + "/OUTAGE_OK"): self.stopfn("ENGINE DOWN: /health failed 4 times in a row (2 min)"); break
            self.halt.wait(30)

# ---------------------------------------------------------------- load phase

def loadphase(poolp, refp, out, minutes, o):
    os.makedirs(out, exist_ok=True); C = cases(poolp, refp); rng = random.Random(int(o.get("seed", 99)))
    levels = [int(x) for x in o.get("levels", "1,2,4,8,3,6,5,7").split(",")]; per = float(o.get("per", 240)); maxp = int(o.get("maxp", 45000))
    low = float(o.get("low", 9)) * 1024; winend = float(o.get("winend", 1e18)); cutoff = float(o.get("cutoff", CUTOFF_DEFAULT)); longgap = float(o.get("longgap", 600))
    sess_pct = float(o.get("sess", 22)); agents = int(o.get("agents", 0)); burst_every = float(o.get("burst", 420)); idle_every = float(o.get("idle", 660))
    dl = min(now() + float(minutes) * 60, winend, cutoff)
    ok200 = lambda c: c["ref"].get("st") == 200
    singles = [c for c in C.values() if c["kind"] == "single" and ok200(c)]
    pool = [c for c in singles if (c["ref"].get("pt") or 0) <= maxp and "long_out" not in c["tags"]]
    rare = [c for c in singles if "long_out" in c["tags"]]
    longs = sorted([c for c in singles if (c["ref"].get("pt") or 0) > maxp], key=lambda c: c["ref"].get("pt") or 0)
    if o.get("maxlong"): longs = [c for c in longs if (c["ref"].get("pt") or 0) <= int(o["maxlong"])]
    sess = {}
    for c in C.values():
        if c["kind"] == "turn" and ok200(c): sess.setdefault(c["sess"], {})[c["k"]] = c
    SS = {}
    for sid, d in sess.items():
        n = 0
        while n in d: n += 1
        if n: SS[sid] = {"turns": [d[k] for k in range(n)], "next": 0, "busy": False, "laps": 0}
    S = dict(n=0, ok=0, diff=0, err=0, http5xx=0, admission=0, hang=0, cancel=0, cancel_bad=0, drop=0, slow=0, dup=0, turns=0, long=0, burst=0, idle=0, tok=0)
    lock = threading.Lock(); stop = threading.Event(); why = ["time"]; level = [levels[0]]; f = open(out + "/rows.jsonl", "a"); ff = open(out + "/fails.jsonl", "a")
    def stopfn(w):
        why[0] = w; stop.set()
    print(f"load: {len(pool)} pool requests, {len(rare)} long-output, {len(longs)} long prompts ({[c['ref'].get('pt') for c in longs]}), {len(SS)} sessions / {sum(len(s['turns']) for s in SS.values())} turns; "
          f"levels {levels} x {per:.0f} s; until {hms(dl)} UTC; low-water stop {low / 1024:.1f} GiB", flush=True)

    def do(c, action="plain", stream=None, **kw):
        """One checked request. Returns the class."""
        if stream is None: stream = rng.random() < 0.6
        cancelled = "cancel_after" in kw and kw["cancel_after"] is not None
        if cancelled or kw.get("stall") or kw.get("slow"): stream = True
        k = TR.start(c["id"], action); t = now()
        try: r = send(c, stream, key=k, **kw)
        finally: TR.end(k)
        if r.get("dropped"):
            with lock: S["drop"] += 1
            return "drop"
        okk, d, kl = verdict(c, r, stream)
        if (r.get("gap") or 0) > 60 and not kw.get("stall") and kl in ("ok", "diff"): kl = "hang"; okk = False; d = d + ["gap"]
        row = dict(t=round(t), hms=hms(t), id=c["id"], a=action, m="s" if stream else "n", lvl=level[0], st=r.get("st"), wall=r.get("wall"), ttft=r.get("ttft"), gap=r.get("gap"), ct=r.get("ct"), k=kl)
        with lock:
            S["n"] += 1; S["tok"] += r.get("ct") or 0
            if kl == "ok": S["ok"] += 1
            elif kl == "cancel_ok": S["cancel"] += 1
            elif kl == "cancel_bad": S["cancel"] += 1; S["cancel_bad"] += 1
            elif kl == "diff": S["diff"] += 1
            elif kl == "admission": S["admission"] += 1
            elif kl == "http5xx": S["http5xx"] += 1
            elif kl == "hang": S["hang"] += 1
            else: S["err"] += 1
            if kw.get("slow") or kw.get("stall"): S["slow"] += 1
            f.write(json.dumps(row) + "\n"); f.flush()
            if not okk:
                a = refof(c, stream); x = dict(row, fields=d, inflight=TR.snapshot(), got={q: r.get(q) for q in ("st", "err", "text", "rsn", "tools", "fin", "pt", "ct")}, ref={q: a.get(q) for q in ("st", "err", "text", "rsn", "tools", "fin", "pt", "ct")})
                if "text" in d:
                    p = 0; A, B = a.get("text") or "", r.get("text") or ""
                    while p < min(len(A), len(B)) and A[p] == B[p]: p += 1
                    x["text_first_diff_at"] = p
                ff.write(json.dumps(x, ensure_ascii=False) + "\n"); ff.flush()
                print(f"  !! {hms()} {kl.upper()} {c['id']} action {action} mode {row['m']} level {level[0]} fields {d} status {r.get('st')} {str(r.get('err') or '')[:140]}", flush=True)
        return kl

    def session_step(maxturns=None, think=True):
        with lock:
            free = [s for s in SS.values() if not s["busy"]]
            if not free: return False
            s = rng.choice(free); s["busy"] = True
        try:
            n = maxturns or rng.randint(1, 3)
            for _ in range(n):
                if stop.is_set(): break
                c = s["turns"][s["next"]]; x = rng.random()
                if x < 0.08:      # the client gives up in the middle of the turn, then asks again
                    do(c, "sess_cancel", cancel_after=rng.randint(1, 30)); do(c, "sess_retry")
                elif x < 0.12: do(c, "sess_slow", slow=0.05)
                else: do(c, "sess")
                with lock: S["turns"] += 1
                s["next"] += 1
                if s["next"] >= len(s["turns"]):
                    s["laps"] += 1; s["next"] = 0 if rng.random() < 0.6 else rng.randrange(len(s["turns"]))   # start again, or branch from an earlier turn
                    break
                if think: time.sleep(0.3 + rng.random() * 1.7)   # the tool's run time
        finally: s["busy"] = False     # abandoned here; someone resumes it later
        return True

    def action():
        x = rng.random() * 100
        if x < sess_pct and SS: session_step(); return
        x = rng.random() * 100; c = rng.choice(pool)
        if x < 5: do(c, "cancel", cancel_after=rng.randint(1, 40))
        elif x < 8: do(c, "cancel_retry", cancel_after=rng.randint(1, 12)); do(c, "after_cancel")
        elif x < 10: do(c, "drop", drop=rng.random() * 2.0)
        elif x < 14: do(c, "slow", slow=0.05)
        elif x < 15.5: do(c, "stall", stall=(rng.randint(1, 6), rng.choice([5, 12, 25])))
        elif x < 20:      # the same request two to four times at once
            k = rng.randint(2, 4); c2 = rng.choice(pool + [t for s in SS.values() for t in s["turns"][:2]]) if rng.random() < 0.3 else c
            th = [threading.Thread(target=do, args=(c2, "dup"), daemon=True) for _ in range(k - 1)]
            for t in th: t.start()
            do(c2, "dup")
            for t in th: t.join()
            with lock: S["dup"] += 1
        elif x < 21.5 and rare: do(rng.choice(rare), "long_out")
        else: do(c, "plain")

    active = [0]
    def worker(i):
        while not stop.is_set():
            if i >= level[0]: time.sleep(0.5); continue
            active[0] += 1
            try: action()
            except Exception as e: print("worker exception", type(e).__name__, str(e)[:200], flush=True)
            finally: active[0] -= 1
            time.sleep(rng.random() * 0.2)

    def agent_runner(i):
        # agent mode: one runner an agent slot; each keeps taking a free session and runs it turn by turn (growing history), with tool pauses
        time.sleep(rng.random() * 3)
        while not stop.is_set():
            if not session_step(maxturns=rng.randint(2, 6)): time.sleep(1)
            time.sleep(rng.random() * 2)

    def longer():
        i = 0; time.sleep(min(90, longgap / 4))
        while not stop.is_set() and longs:
            c = longs[i % len(longs)] if rng.random() < 0.7 else rng.choice(longs); i += 1
            if dl - now() < 240: break
            if o.get("full15"):     # a full-window request arriving while 15 short ones run
                th15 = [threading.Thread(target=do, args=(rng.choice(pool), "with_full"), daemon=True) for _ in range(int(o["full15"]))]
                for t in th15: t.start()
                time.sleep(0.4)
            do(c, "long", stream=rng.random() < 0.5)
            with lock: S["long"] += 1
            if o.get("full") == "pair" and rng.random() < 0.3: do(c, "long_again")    # the same long prompt again: resumes from the cache
            stop.wait(longgap * (0.6 + 0.8 * rng.random()))

    def bursts():
        while not stop.wait(burst_every * (0.7 + 0.6 * rng.random())):
            k = rng.randint(8, 14); pick = [rng.choice(pool) for _ in range(k)]; kind = rng.random()
            with lock: S["burst"] += 1
            if kind < 0.3:    # a burst of clients that all hang up at once
                th = [threading.Thread(target=do, args=(c, "burst_drop"), kwargs=dict(drop=0.05 + rng.random() * 0.5), daemon=True) for c in pick]
            elif kind < 0.5: th = [threading.Thread(target=do, args=(c, "burst_cancel"), kwargs=dict(cancel_after=rng.randint(1, 5)), daemon=True) for c in pick]
            else: th = [threading.Thread(target=do, args=(c, "burst"), daemon=True) for c in pick]
            for t in th: t.start()
            for t in th: t.join(600)

    def idles():
        while not stop.wait(idle_every * (0.7 + 0.6 * rng.random())):
            keep = level[0]; level[0] = 0; S["idle"] += 1; stop.wait(rng.choice([15, 30, 45, 75])); level[0] = keep

    mon = Monitor(out, S, low, stopfn); mon.start()
    th = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(max(levels))]
    th += [threading.Thread(target=agent_runner, args=(i,), daemon=True) for i in range(agents)]
    th += [threading.Thread(target=x, daemon=True) for x in ([longer] if longs and longgap > 0 else []) + ([bursts] if burst_every > 0 else []) + ([idles] if idle_every > 0 else [])]
    for t in th: t.start()
    t0 = now(); k = 0; idle_flag = False
    while now() < dl and not stop.is_set():
        if not idle_flag or level[0] != 0: pass
        if level[0] != 0 or True:
            lv = levels[int((now() - t0) / per) % len(levels)]
            if level[0] != 0: level[0] = lv
        if os.path.exists(out + "/STOP"): stopfn("stop file"); break
        k += 1
        if k % 30 == 0: print(f"load {int((now() - t0) / 60)} min {hms()}: {S} level {level[0]} in flight {len(TR.snapshot())} MemAvailable low {int(mon.lowest)} MiB cache {mon.cache} crash lines {mon.crash}", flush=True)
        time.sleep(10)
    if now() >= dl and why[0] == "time": why[0] = "time" if dl < min(winend, cutoff) else ("WINDOW CAP" if winend <= cutoff else "CUTOFF")
    stop.set(); t1 = now()
    while TR.snapshot() and now() - t1 < float(o.get("drain", 300)): time.sleep(2)
    left = TR.snapshot(); mon.halt.set(); time.sleep(1)
    end = dict(why=why[0], minutes=round((now() - t0) / 60, 1), S=S, lowest_mib=int(mon.lowest), crash_lines=mon.crash, still_in_flight=left, cache=mon.cache, t0=round(t0), t1=round(now()), opts=o)
    json.dump(end, open(out + "/end.json", "w"))
    print(f"LOAD END ({why[0]}) after {end['minutes']} min: {S}; lowest MemAvailable {int(mon.lowest)} MiB; crash lines {mon.crash}; still in flight {len(left)}", flush=True)

# ---------------------------------------------------------------- failure injection (HTTP level)

def rawsock(data, wait=0.0, read=True, timeout=20):
    u = urllib.parse.urlparse(rt.URL); s = socket.create_connection((u.hostname, u.port), timeout=timeout); got = b""
    try:
        s.sendall(data)
        if wait: time.sleep(wait)
        if read:
            try:
                while len(got) < 4096:
                    b = s.recv(4096)
                    if not b: break
                    got += b
                    if b"\r\n\r\n" in got: break
            except Exception as e: got += b" [" + type(e).__name__.encode() + b"]"
    finally: s.close()
    return got[:200].decode("utf-8", "replace")


def inject(poolp, refp, out, o):
    os.makedirs(out, exist_ok=True); C = cases(poolp, refp); rng = random.Random(7); f = open(out + "/inject.jsonl", "a"); res = []
    canary = [c for c in C.values() if c["kind"] == "single" and c["ref"].get("st") == 200 and (c["ref"].get("pt") or 0) < 3000 and "long_out" not in c["tags"]]
    turns = [c for c in C.values() if c["kind"] == "turn" and c["ref"].get("st") == 200 and (c["ref"].get("pt") or 0) < 30000]
    def canaries(n=6):
        bad = 0
        for c in rng.sample(canary, n) + (rng.sample(turns, 1) if turns else []):
            sm = rng.random() < 0.5; r = send(c, sm); okk, d, kl = verdict(c, r, sm)
            if not okk: bad += 1; f.write(json.dumps(dict(canary=c["id"], k=kl, fields=d, got={q: r.get(q) for q in ("st", "err", "text", "fin", "ct")}, ref={q: c["ref"].get(q) for q in ("text", "fin", "ct")}), ensure_ascii=False) + "\n"); f.flush()
        return bad
    def step(name, fn, n_can=6):
        t = now()
        try: got = fn()
        except Exception as e: got = "client exception " + type(e).__name__ + ": " + str(e)[:160]
        st, hb = get("/health", 10); bad = canaries(n_can) if st == 200 else -1
        row = dict(step=name, result=str(got)[:400], health=st, canaries_bad=bad, s=round(now() - t, 1)); res.append(row)
        f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
        print(f"inject {name:<34} -> {str(got)[:150]} | health {st} | canaries wrong {bad}", flush=True)
    P = rt.PATHS; msg = U("Say OK."); base = {"model": rt.MODEL, "messages": msg, "max_tokens": 8, "temperature": 0}
    def st_of(r): return f"HTTP {r.get('st')} {str(r.get('err') or (json.dumps(r.get('raw'))[:160] if r.get('raw') is not None else ''))[:200]}"
    for route in ("chat", "resp", "msg"):
        step(f"malformed json {route}", lambda: st_of(post(P[route], None, raw=b'{"model": "x", "messages": [{"role": "user", "content": "hi"')))
        step(f"not json {route}", lambda: st_of(post(P[route], None, raw=b"\xff\xfe\x00garbage \x00\x01")))
        step(f"empty body {route}", lambda: st_of(post(P[route], None, raw=b"")))
        step(f"wrong types {route}", lambda: st_of(post(P[route], {"model": rt.MODEL, "messages": "hello", "input": 5, "max_tokens": "many"})))
    step("json array body", lambda: st_of(post(P["chat"], [1, 2, 3])))
    step("deeply nested json", lambda: st_of(post(P["chat"], None, raw=b'{"messages":' + b"[" * 20000 + b"]" * 20000 + b"}")))
    step("unknown route", lambda: st_of(post("/v1/nothing", base)))
    step("max_tokens beyond the window", lambda: st_of(post(P["chat"], dict(base, max_tokens=400000))))
    step("max_tokens 2^40", lambda: st_of(post(P["chat"], dict(base, max_tokens=1 << 40))))
    step("max_tokens negative", lambda: st_of(post(P["chat"], dict(base, max_tokens=-5))))
    step("max_tokens = window exactly", lambda: st_of(post(P["chat"], dict(base, max_tokens=262144))))
    words300k = " ".join(rng.choice(rt.WORDS) for _ in range(int(float(o.get("overwords", 230000)))))
    step("prompt over the context limit", lambda: st_of(post(P["chat"], dict(base, messages=U("Notes:\n" + words300k + "\nSay OK.")), t_first=900)), 4)
    step("prompt over the limit, streamed", lambda: st_of(post(P["chat"], dict(base, stream=True, messages=U("Notes:\n" + words300k + "\nSay OK.")), stream=True, t_first=900)), 4)
    step("prompt + max_tokens over the limit", lambda: st_of(post(P["chat"], dict(base, max_tokens=200000, messages=U("Notes:\n" + words300k[: len(words300k) // 3] + "\nSay OK.")), t_first=900)), 4)
    for mb in (8, 64, 200):
        step(f"oversized body {mb} MB", lambda: st_of(post(P["chat"], None, raw=b'{"model":"x","messages":[{"role":"user","content":"' + b"a" * (mb << 20) + b'"}]}', t_first=600)), 4)
    step("content-length larger than body", lambda: rawsock(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: 5000\r\n\r\n{\"model\":", wait=1.0, read=False))
    step("content-length 10 GB, no body", lambda: rawsock(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: 10737418240\r\n\r\n", wait=1.0))
    step("header without end", lambda: rawsock(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Junk: " + b"j" * 200000, wait=0.5, read=False))
    step("binary garbage on the socket", lambda: rawsock(bytes(rng.getrandbits(8) for _ in range(4096)), wait=0.5))
    step("chunked body cut in the middle", lambda: rawsock(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n10\r\n{\"model\": \"x\", \"", wait=0.5, read=False))
    def dropburst(n, stream, after):
        th = []
        for i in range(n):
            c = rng.choice(canary)
            kw = dict(cancel_after=after) if after else dict(drop=rng.random() * 0.3)
            th.append(threading.Thread(target=send, args=(c, True if after else stream), kwargs=kw, daemon=True))
        for t in th: t.start()
        for t in th: t.join(300)
        return f"{n} connections dropped"
    step("40 connections dropped before a reply", lambda: dropburst(40, False, None))
    step("40 streams dropped after 1 event", lambda: dropburst(40, True, 1))
    step("100 streams dropped after 3 events", lambda: dropburst(100, True, 3))
    def idleconns():
        u = urllib.parse.urlparse(rt.URL); ss = []
        for i in range(200):
            try: ss.append(socket.create_connection((u.hostname, u.port), timeout=5))
            except Exception as e: return f"connection {i} refused: {type(e).__name__}"
        r = st_of(post(P["chat"], base)); [s.close() for s in ss]; return "200 idle sockets open, then a request: " + r
    step("200 idle connections", idleconns)
    bad = sum(1 for r in res if r["canaries_bad"] != 0); down = sum(1 for r in res if r["health"] != 200)
    s5 = [r["step"] for r in res if re.match(r"HTTP 5\d\d", r["result"])]
    json.dump(dict(steps=len(res), canary_failures=bad, health_failures=down, http5xx_steps=s5), open(out + "/inject_end.json", "w"))
    print(f"INJECT END: {len(res)} steps; steps after which a canary was wrong or the engine was down: {bad}; health not 200: {down}; steps answered with HTTP 5xx: {s5}")

# ---------------------------------------------------------------- solo re-run, report

def solo(poolp, refp, cid, n=3, mode="n"):
    C = cases(poolp, refp); c = C[cid]; same = 0
    for i in range(int(n)):
        r = send(c, mode == "s"); okk, d, kl = verdict(c, r, mode == "s"); same += okk
        print(f"solo {cid} run {i + 1}: {kl} {d} ct {r.get('ct')} fin {r.get('fin')} wall {r.get('wall')}", flush=True)
        if not okk and "text" in d:
            A, B = refof(c, mode == "s").get("text") or "", r.get("text") or ""; p = 0
            while p < min(len(A), len(B)) and A[p] == B[p]: p += 1
            print(f"   first difference at char {p}: ref {A[max(0, p - 30):p + 50]!r} | got {B[max(0, p - 30):p + 50]!r}")
    print(f"SOLO {cid}: {same}/{n} equal to the reference"); return same


def bisect(poolp, refp, cid, mode="n", n=2):
    """One request with each lever switched off through the knob file (engine started with LEV=knob; RT_KNOBS = the file)."""
    C = cases(poolp, refp); c = C[cid]
    def t(name, knob):
        rt.setknobs(knob); res = []
        for i in range(int(n)):
            r = send(c, mode == "s"); okk, d, kl = verdict(c, r, mode == "s"); res.append("EQUAL" if okk else "differs:" + ",".join(d))
        print(f"{name:<30} {' | '.join(res)}", flush=True)
    t("levers on", ""); t("ALL knob levers off", " ".join(rt.LEVERS_OFF))
    for w in rt.LEVERS_OFF: t("only " + w, w)
    rt.setknobs("")


def hammer(poolp, refp, cid, n, o):
    """The request CID N times (streamed and not, alternating unless mode=n|s) while conc other requests are in flight (bg=ID,ID,... or random short ones).
    knob="words" writes the knob file first (engine started with LEV=knob). Prints how many target replies and background replies were wrong."""
    C = cases(poolp, refp); c = C[cid]; conc = int(o.get("conc", 4)); rng = random.Random(int(o.get("seed", 5))); mode = o.get("mode", "mix")
    bg = [C[i] for i in o["bg"].split(",")] if o.get("bg") else [x for x in C.values() if x["kind"] == "single" and x["ref"].get("st") == 200 and (x["ref"].get("pt") or 0) < 6000 and "long_out" not in x["tags"]]
    if "knob" in o: rt.setknobs(o["knob"])
    stop = threading.Event(); B = [0, 0]; lock = threading.Lock(); badids = {}
    def bgw():
        while not stop.is_set():
            x = rng.choice(bg); sm = rng.random() < 0.6; ca = rng.randint(1, 20) if rng.random() < float(o.get("cancel", 0.1)) else None
            r = send(x, True if ca else sm, cancel_after=ca); okk, d, kl = verdict(x, r, True if ca else sm)
            with lock:
                B[0] += 1; B[1] += not okk
                if not okk: badids[x["id"]] = badids.get(x["id"], 0) + 1
    th = [threading.Thread(target=bgw, daemon=True) for _ in range(conc)]
    for t in th: t.start()
    bad = 0; first = None
    for i in range(int(n)):
        sm = (i % 2 == 1) if mode == "mix" else mode == "s"; r = send(c, sm); okk, d, kl = verdict(c, r, sm); bad += not okk
        if not okk and first is None:
            A, Bt = refof(c, sm).get("text") or "", r.get("text") or ""; q = 0
            while q < min(len(A), len(Bt)) and A[q] == Bt[q]: q += 1
            first = f"run {i + 1} {kl} {d} first text difference at char {q}: ref {A[max(0, q - 30):q + 40]!r} | got {Bt[max(0, q - 30):q + 40]!r}"
    stop.set()
    for t in th: t.join(120)
    if "knob" in o: rt.setknobs("")
    print(f"HAMMER {cid} conc {conc} knob [{o.get('knob', '')}]: target wrong {bad}/{n}; background wrong {B[1]}/{B[0]} {badids if badids else ''}" + (f"\n   {first}" if first else ""), flush=True)
    return bad


def fit(xs, ys):
    n = len(xs)
    if n < 3: return 0.0
    mx = sum(xs) / n; my = sum(ys) / n; d = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / d if d else 0.0


def report(dirs):
    print("| run | min | checked | identical | MISMATCH | cancelled (prefix wrong) | HTTP 5xx | admission | other errors | hangs | crash lines | silences>60s | MemAvail trend GiB/h (after 30 min) | GPU trend GiB/h | low-water GiB | cache states / MiB max | evicted |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in dirs:
        try: e = json.load(open(d + "/end.json"))
        except Exception: e = {}
        rows = load(d + "/rows.jsonl") if os.path.exists(d + "/rows.jsonl") else []; M = load(d + "/mon.jsonl") if os.path.exists(d + "/mon.jsonl") else []
        k = {}
        if os.path.exists(d + "/OUTAGE"):     # "t_stop t_ready": requests cut by a deliberate engine restart are counted apart
            a0, a1 = [float(x) for x in open(d + "/OUTAGE").read().split()[:2]]
            for r in rows:
                if r["k"] not in ("ok", "cancel_ok") and r["t"] + (r.get("wall") or 0) >= a0 - 2 and r["t"] <= a1 + 2: r["k"] = "outage"
            print(f"   ({os.path.basename(d.rstrip('/'))}: deliberate restart {hms(a0)} -> {hms(a1)}: {sum(1 for r in rows if r['k'] == 'outage')} requests cut or refused by it, not counted as errors; after it: {sum(1 for r in rows if r['t'] > a1 + 2)} checked, {sum(1 for r in rows if r['t'] > a1 + 2 and r['k'] not in ('ok', 'cancel_ok'))} bad)")
        for r in rows: k[r["k"]] = k.get(r["k"], 0) + 1
        t0 = M[0]["t"] if M else 0; late = [m for m in M if m["t"] - t0 >= 1800 and m["ma"] > 0] or [m for m in M if m["ma"] > 0]
        tr = fit([(m["t"] - t0) / 3600 for m in late], [m["ma"] / 1024 for m in late]); gtr = fit([(m["t"] - t0) / 3600 for m in late], [m["gpu"] / 1024 for m in late])
        lowm = min([m["ma"] for m in M if m["ma"] > 0] or [0]); sil = len(load(d + "/silence.jsonl")) if os.path.exists(d + "/silence.jsonl") else 0
        cs = max([m["cache"].get("states", 0) for m in M] or [0]); cm = max([m["cache"].get("mib", 0) for m in M] or [0]); ev = max([m["cache"].get("evicted", 0) for m in M] or [0])
        mins = e.get("minutes") or (round((M[-1]["t"] - t0) / 60, 1) if M else 0)
        print(f"| {os.path.basename(d.rstrip('/'))} | {mins} | {len(rows)} | {k.get('ok', 0)} | {k.get('diff', 0)} | {k.get('cancel_ok', 0) + k.get('cancel_bad', 0)} ({k.get('cancel_bad', 0)}) | {k.get('http5xx', 0)} | {k.get('admission', 0)} | {k.get('error', 0)} | {k.get('hang', 0)} | "
              f"{max([m['crash'] for m in M] or [0])} | {sil} | {tr:+.2f} | {gtr:+.2f} | {lowm / 1024:.1f} | {cs} / {cm} | {ev} |  end: {e.get('why', '?')}")
        by = {}
        for r in rows: by.setdefault(r["a"], [0, 0]); by[r["a"]][0] += 1; by[r["a"]][1] += r["k"] not in ("ok", "cancel_ok")
        print("   by action: " + " ".join(f"{a}={v[0]}" + (f"(bad {v[1]})" if v[1] else "") for a, v in sorted(by.items())))
        lv = {}
        for r in rows: lv[r["lvl"]] = lv.get(r["lvl"], 0) + 1
        print("   by level: " + " ".join(f"{a}:{v}" for a, v in sorted(lv.items())) + f"; streamed {sum(1 for r in rows if r['m'] == 's')} / not {sum(1 for r in rows if r['m'] == 'n')}; max in flight seen by the monitor {max([m['inflight'] for m in M] or [0])}")


def main():
    a = sys.argv[1:]
    if not a: print(__doc__); return
    m = a[0]
    if m == "mk": mk(a[1], int(a[2]) if len(a) > 2 and a[2].isdigit() else 1114, opts(a[2:]))
    elif m == "ref": ref(a[1], a[2], opts(a[3:]))
    elif m == "load": loadphase(a[1], a[2], a[3], a[4], opts(a[5:]))
    elif m == "inject": inject(a[1], a[2], a[3], opts(a[4:]))
    elif m == "solo": solo(a[1], a[2], a[3], a[4] if len(a) > 4 else 3, a[5] if len(a) > 5 else "n")
    elif m == "bisect": bisect(a[1], a[2], a[3], a[4] if len(a) > 4 else "n", a[5] if len(a) > 5 else 2)
    elif m == "hammer": hammer(a[1], a[2], a[3], a[4], opts(a[5:]))
    elif m == "report": report(a[1:])
    else: print(__doc__)


if __name__ == "__main__": main()
