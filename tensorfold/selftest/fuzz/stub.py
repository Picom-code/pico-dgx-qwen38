#!/usr/bin/env python3
"""stub.py PORT : a stand-in engine for dry-checking rt.py on the Mac (no GPU). Deterministic replies from a hash of the request,
the three routes, streamed and not. STUB_BUG=1: when the knob file (RT_KNOBS) holds no words (= levers on) a request with top_k 21
gets another reply and a streamed request with max_tokens 17 loses its last word: the fuzzer must report both."""
import sys, os, json, hashlib, random, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KN = os.environ.get("RT_KNOBS", "/tmp/stub_knobs"); BUG = os.environ.get("STUB_BUG") == "1"
W = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()


def levers_on():
    try: return not open(KN).read().strip()
    except OSError: return True


def reply(b, stream):
    key = {k: v for k, v in b.items() if k not in ("stream", "stream_options", "prediction", "model")}
    msgs = b.get("messages") or b.get("input") or []
    pt = max(1, len(json.dumps(msgs, ensure_ascii=False)) // 4)
    mt = b.get("max_tokens", b.get("max_output_tokens", 32))
    if b.get("logprobs"): return None, "logprobs are not supported by this model or backend", pt
    if b.get("n", 1) != 1: return None, "n must be 1; multiple choices are not supported", pt
    rng = random.Random(hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest())
    if BUG and levers_on() and b.get("top_k") == 21: rng = random.Random(1)
    n = rng.randint(1, 60); words = [rng.choice(W) for _ in range(n)]; fin = "stop"
    if n > mt: words = words[:mt]; fin = "length"
    if BUG and levers_on() and stream and mt == 17: words = words[:-1]
    text = " ".join(words)
    for s in ([b["stop"]] if isinstance(b.get("stop"), str) else b.get("stop") or b.get("stop_sequences") or []):
        if s and s in text: text = text[: text.index(s)]; fin = "stop"
    return (text, fin, len(text.split())), None, pt


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"data":[]}')
    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"]))); stream = bool(b.get("stream")); r, err, pt = reply(b, stream)
        if err:
            d = json.dumps({"error": {"message": err}}).encode(); self.send_response(400); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(d); return
        text, fin, ct = r; route = "resp" if "responses" in self.path else "msg" if "messages" in self.path else "chat"
        self.send_response(200); self.send_header("Content-Type", "text/event-stream" if stream else "application/json"); self.end_headers()
        def ev(d): self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n"); self.wfile.flush()
        parts = [w + " " for w in text.split(" ")]; parts[-1] = parts[-1][:-1]
        try:
            if route == "chat":
                if not stream: self.wfile.write(json.dumps({"choices": [{"message": {"content": text}, "finish_reason": fin}], "usage": {"prompt_tokens": pt, "completion_tokens": ct}}).encode()); return
                for p in parts: ev({"choices": [{"delta": {"content": p}}]}); time.sleep(0.002)
                ev({"choices": [{"delta": {}, "finish_reason": fin}]}); ev({"choices": [], "usage": {"prompt_tokens": pt, "completion_tokens": ct}}); self.wfile.write(b"data: [DONE]\n\n")
            elif route == "resp":
                final = {"status": "completed" if fin == "stop" else "incomplete", "incomplete_details": None if fin == "stop" else {"reason": "max_output_tokens"},
                         "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}], "usage": {"input_tokens": pt, "output_tokens": ct}}
                if not stream: self.wfile.write(json.dumps(final).encode()); return
                for p in parts: ev({"type": "response.output_text.delta", "delta": p})
                ev({"type": "response.completed", "response": final})
            else:
                sr = "end_turn" if fin == "stop" else "max_tokens"
                if not stream: self.wfile.write(json.dumps({"content": [{"type": "text", "text": text}], "stop_reason": sr, "usage": {"input_tokens": pt, "output_tokens": ct}}).encode()); return
                ev({"type": "message_start", "message": {"usage": {"input_tokens": pt}}}); ev({"type": "content_block_start", "content_block": {"type": "text"}})
                for p in parts: ev({"type": "content_block_delta", "delta": {"type": "text_delta", "text": p}})
                ev({"type": "message_delta", "delta": {"stop_reason": sr}, "usage": {"output_tokens": ct}})
        except (BrokenPipeError, ConnectionResetError): pass


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
