#!/usr/bin/env python3
"""overlap.py : proof that the round-9 fitting corpus and the fresh set share no prompt with any bench set (and none with each other).
Bench side: the 320 held prompts, qk.py's 3-prompt bench / 16 distinct code prompts / identity set / sparkDash-method prompts, the edit
and agent sets, the qeval tasks. Measure: exact match, and the share of a prompt's word 4-grams that also occur in the other prompt."""
import json, re, sys, os
Q = os.path.expanduser("~/.hermes/cache/scratch/q38")
def sh(t):
    w = re.findall(r"\w+", t.lower()); return set(zip(w, w[1:], w[2:], w[3:])) if len(w) >= 4 else {tuple(w)}
bench = []
for l in open(Q + "/kit/prompts.jsonl"): bench.append(("held", json.loads(l)["prompt"]))
for n in ("edit", "agent"):
    for l in open(Q + f"/A9/bench/{n}.jsonl"): bench.append((n, json.loads(l)["prompt"]))
NS = {"__file__": Q + "/kit/qk.py", "__name__": "qk"}; exec(compile(open(NS["__file__"]).read().split("\ndef main():")[0], "qk.py", "exec"), NS)
for k in ("CHAT", "CODE", "SD_CODE"): bench += [(k, p) for p in NS[k]]
bench += [("SD", NS["SD_PROSE"]), ("SD", NS["SD_STRUCT"])]
for c, L in NS["IDENT"].items(): bench += [("ident", p) for p in L]
bench += [("ident_long", m if isinstance(m, str) else json.dumps(m)) for _, m in NS["IDENT_LONG"]]
src = open(Q + "/kit/qeval_tasks.py").read(); bench += [("qeval", s) for s in re.findall(r'"((?:[^"\\]|\\.){25,})"', src)]
print("bench prompts", len(bench), {k: sum(1 for b in bench if b[0] == k) for k in dict(bench)})
B = [(k, p, sh(p)) for k, p in bench]
def load(n): return [json.loads(l) for l in open(Q + f"/A9/corpus/{n}.jsonl")]
def worst(S, T, name):
    ex = 0; top = []
    for x in S:
        a = sh(x["prompt"])
        for k, p, b in T:
            if x["prompt"].strip() == p.strip(): ex += 1
            c = len(a & b) / max(1, min(len(a), len(b))); top.append((c, x["id"], k))
    top.sort(reverse=True)
    if os.environ.get("SHOW"):
        for c, i, k in top[:12]: print("   ", round(c, 3), i, k, "|", S[i]["prompt"][:110].replace("\n", " "))
    print(f"{name}: exact matches {ex}; largest shared 4-gram share {top[0][0]:.3f} (id {top[0][1]} vs {top[0][2]}); pairs above 0.10: {sum(1 for t in top if t[0] > 0.10)}; top5 {[(round(c, 3), i, k) for c, i, k in top[:5]]}")
fit, fresh = load("fit"), load("fresh")
worst(fit, B, "fit vs bench"); worst(fresh, B, "fresh vs bench"); worst(fresh, [("fit", x["prompt"], sh(x["prompt"])) for x in fit], "fresh vs fit")
