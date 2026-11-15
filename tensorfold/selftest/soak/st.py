#!/usr/bin/env python3
# st.py DIR [N] : compact status of a soak run dir (box side)
import json, sys, os, time
d = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
P = {}
for l in open(os.path.expanduser("~/q38/kit-B/soak/pool.jsonl")):
    c = json.loads(l); P[c["id"]] = c
M = [json.loads(l) for l in open(d + "/mon.jsonl")] if os.path.exists(d + "/mon.jsonl") else []
print(time.strftime("%H:%M:%S", time.gmtime()), d, "monitor lines", len(M))
step = max(1, len(M) // n)
for m in M[::step][-n:] + M[-1:]:
    c = m["cache"]; S = m["S"]
    print(f"{m['hms']} ma {m['ma']} gpu {m['gpu']} infl {m['inflight']} sil {m['silence']} cache {c.get('states')}/{c.get('mib')}/{c.get('budget')} ev {c.get('evicted')} ref {c.get('refused')} fail {c.get('failed')} crash {m['crash']} | n {S['n']} ok {S['ok']} diff {S['diff']} err {S['err']} 5xx {S['http5xx']} adm {S['admission']} hang {S['hang']} cancel {S['cancel']}/{S['cancel_bad']} turns {S['turns']} long {S['long']} tok {S['tok']}")
if M: print("low-water MiB", min(m["ma"] for m in M), "max gpu", max(m["gpu"] for m in M))
F = [json.loads(l) for l in open(d + "/fails.jsonl")] if os.path.exists(d + "/fails.jsonl") else []
by = {}
for x in F:
    t = P.get(x["id"], {}).get("tags") or ["turn:" + x["id"]]
    k = (x["id"], x["k"], ",".join(x["fields"]), " ".join(t[:4])); by.setdefault(k, []).append(x)
for k, v in sorted(by.items()): print("  FAIL", k, "x", len(v), "levels", sorted({x["lvl"] for x in v}), "modes", "".join(sorted({x["m"] for x in v})), "actions", sorted({x["a"] for x in v}), "first", v[0]["hms"], "st", v[0]["got"].get("st"), str(v[0]["got"].get("err") or "")[:100])
if os.path.exists(d + "/silence.jsonl"):
    L = [json.loads(l) for l in open(d + "/silence.jsonl")]
    print("silences > 60 s:", len(L), [(x["t"], x["silence"], [(i[0], i[1], i[2], i[3]) for i in x["inflight"]][:4]) for x in L[-3:]])
if os.path.exists(d + "/crash_lines.txt"): print("CRASH LINES:", open(d + "/crash_lines.txt").read()[:600])
