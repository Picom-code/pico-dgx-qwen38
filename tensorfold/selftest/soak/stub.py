#!/usr/bin/env python3
"""stub.py PORT : a fake engine for dry-checking soak.py on the Mac (no GPU). Three routes, streamed and not, deterministic replies
(a hash of the request), /health. Faults by environment: STUB_FLIP=p (a reply differs with probability p when >1 request is in flight),
STUB_500=p, STUB_HANG=p (stops sending in the middle of a stream), STUB_MS=ms a token, STUB_PAR=n (requests served at once)."""
import sys, os, json, time, hashlib, random, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FLIP = float(os.environ.get("STUB_FLIP", 0)); E500 = float(os.environ.get("STUB_500", 0)); HANG = float(os.environ.get("STUB_HANG", 0)); MS = float(os.environ.get("STUB_MS", 2)) / 1000
SEM = threading.Semaphore(int(os.environ.get("STUB_PAR", 3))); RUN = [0]; LOCK = threading.Lock(); N = [0]
W = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def js(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health": self.js(200, {"status": "ok", "running": RUN[0], "served": N[0]})
        elif self.path == "/v1/models": self.js(200, {"data": [{"id": "stub"}]})
        else: self.js(404, {"error": {"message": "no such route"}})
    def do_POST(self):
        try: n = int(self.headers.get("Content-Length") or 0)
        except ValueError: n = 0
        if n > 32 << 20:
            self.js(413, {"error": {"message": "body too large"}}); self.close_connection = True; return
        raw = self.rfile.read(n)
        route = {"/v1/chat/completions": "chat", "/v1/responses": "resp", "/v1/messages": "msg"}.get(self.path)
        if route is None: return self.js(404, {"error": {"message": "no such route"}})
        try: b = json.loads(raw); assert isinstance(b, dict)
        except Exception: return self.js(400, {"error": {"message": "invalid JSON"}})
        msgs = b.get("messages") if route != "resp" else b.get("input")
        if not isinstance(msgs, list): return self.js(400, {"error": {"message": "messages must be a list"}})
        mt = b.get("max_tokens", b.get("max_output_tokens", 64))
        if not isinstance(mt, int) or mt < 0: return self.js(400, {"error": {"message": "max_tokens"}})
        key0 = {k: v for k, v in b.items() if k not in ("stream", "stream_options", "prediction", "model")}; pt = len(json.dumps(key0)) // 5
        if pt + mt > 262144: return self.js(400, {"error": {"message": "prompt + max_tokens exceed the context window"}})
        key = {k: v for k, v in b.items() if k not in ("stream", "stream_options", "prediction", "model")}
        h = int(hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest(), 16); rng = random.Random(h)
        want = 8 + h % 90; ntok = min(mt, want); fin = "length" if mt < want else "stop"
        toks = [rng.choice(W) + " " for _ in range(ntok)]
        with LOCK: RUN[0] += 1; N[0] += 1; busy = RUN[0]
        try:
            with SEM:
                if random.random() < E500: return self.js(500, {"error": {"message": "stub internal error"}})
                if busy > 1 and random.random() < FLIP and toks: toks[len(toks) // 2] = "WRONG "
                time.sleep(min(2.0, pt / 200000.0))
                tool = ("tools" in b and h % 2 == 0 and route == "chat")
                if not b.get("stream"):
                    time.sleep(MS * ntok); text = "".join(toks)
                    if route == "chat":
                        m = {"role": "assistant", "content": text}
                        if tool: m = {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "search", "arguments": json.dumps({"query": text[:20]})}}]}; fin = "tool_calls"
                        return self.js(200, {"choices": [{"index": 0, "message": m, "finish_reason": fin}], "usage": {"prompt_tokens": pt, "completion_tokens": ntok}})
                    if route == "resp": return self.js(200, {"status": "completed" if fin == "stop" else "incomplete", "incomplete_details": None if fin == "stop" else {"reason": "max_output_tokens"}, "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}], "usage": {"input_tokens": pt, "output_tokens": ntok}})
                    return self.js(200, {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn" if fin == "stop" else "max_tokens", "usage": {"input_tokens": pt, "output_tokens": ntok}})
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Connection", "close"); self.end_headers(); self.close_connection = True
                def ev(o): self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n"); self.wfile.flush()
                hang_at = ntok // 2 if random.random() < HANG else -1
                if route == "msg": ev({"type": "message_start", "message": {"usage": {"input_tokens": pt}}}); ev({"type": "content_block_start", "content_block": {"type": "text"}})
                for i, t in enumerate(toks):
                    if i == hang_at: time.sleep(float(os.environ.get('STUB_HANG_S', 200)))
                    time.sleep(MS)
                    if route == "chat":
                        if tool:
                            ev({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"name": "search" if i == 0 else None, "arguments": json.dumps({"query": "".join(toks)[:20]}) if i == 0 else ""}}]}}]})
                        else: ev({"choices": [{"index": 0, "delta": {"content": t}}]})
                    elif route == "resp": ev({"type": "response.output_text.delta", "delta": t})
                    else: ev({"type": "content_block_delta", "delta": {"type": "text_delta", "text": t}})
                text = "".join(toks)
                if route == "chat":
                    ev({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tool else fin}]}); ev({"choices": [], "usage": {"prompt_tokens": pt, "completion_tokens": ntok}})
                elif route == "resp": ev({"type": "response.completed" if fin == "stop" else "response.incomplete", "response": {"status": "completed" if fin == "stop" else "incomplete", "incomplete_details": None if fin == "stop" else {"reason": "max_output_tokens"}, "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}], "usage": {"input_tokens": pt, "output_tokens": ntok}}})
                else: ev({"type": "message_delta", "delta": {"stop_reason": "end_turn" if fin == "stop" else "max_tokens"}, "usage": {"output_tokens": ntok}})
                self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError): self.close_connection = True
        finally:
            with LOCK: RUN[0] -= 1


if __name__ == "__main__":
    s = ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H); s.daemon_threads = True; s.request_queue_size = 256; s.serve_forever()
