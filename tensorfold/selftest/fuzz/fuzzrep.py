#!/usr/bin/env python3
"""fuzzrep.py RUNS_DIR MATRIX_DIR [OUT_DIFFS.jsonl] : the night's exactness matrix from whatever result files exist.

RUNS_DIR holds the engine directories D_RT_N_on, D_RT_N_off, D_RT_N_on2, D_RT_N_chk, D_RT_N_u16, D_RT_N_u16d, D_RT_N_ag (head: ~/q38/runs).
The reference of every comparison is the request with every knob lever off, solo, in the recipe-default engine (label off / off_s),
except the BASE rows (levers-off against levers-off). Counts: tested / identical / errored (both sides refused the same way) / DIFFERENT.
"""
import sys, os, json, importlib.util

H = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location("rt", os.path.join(H, "rt.py")); rt = importlib.util.module_from_spec(sp); sp.loader.exec_module(rt)
RUNS, M = sys.argv[1], sys.argv[2]; OUT = sys.argv[3] if len(sys.argv) > 3 else None
P = lambda e, f: os.path.join(RUNS, "D_RT_N_" + e, f)


def load(p):
    o = []
    if os.path.exists(p):
        for l in open(p):
            try: o.append(json.loads(l))
            except ValueError: pass
    return o


TAGS = {}
for f in ("cases1.jsonl", "cases2.jsonl", "cases3.jsonl", "cases4.jsonl"):
    for c in load(os.path.join(M, f)): TAGS[c["id"]] = c["tags"]
for c in load(P("on", "sess.jsonl.cases")) + load(P("on3", "sess4.jsonl.cases")): TAGS[c["id"]] = c["tags"]

GROUP = [("PROMPT", ["chat", "nonenglish", "unicode", "code", "table", "base64", "copy", "near", "rewrite", "boundary", "cross", "long", "image", "longtext", "rand", "held", "fresh"]),
         ("OUTPUT", ["eos", "long_out", "mt4096", "stop", "ignore_eos"]),
         ("PARAMETERS", ["greedy", "sampled", "sampgrid", "grid", "T0.2", "T0.7", "T1.0", "T1.5", "topk_dflt", "topk_0", "topk_1", "topk_2", "topk_5", "topk_19", "topk_20", "topk_21", "topk_50", "topk_100", "penalty", "param_edge", "logprobs",
                         "think", "budget", "effort", "json", "schema", "grammar", "tools", "tc_auto", "tc_required", "tc_named", "tc_none"]),
         ("PREDICTION", ["pred", "predwrong", "predpart", "pred2", "pred_exact", "pred_wrong", "pred_part", "pred_longer", "pred_shorter", "pred_shift", "pred_empty", "pred_rsn"]),
         ("ROUTE", ["route_chat", "route_resp", "route_msg"]),
         ("CACHE / SESSION", ["seq", "sysprefix", "sysS", "sysM", "sysL", "multiturn", "edit", "same", "doc", "grow", "agent", "sess", "turn0", "turn1", "turn2", "turn3+", "regen"])]
ORDER = {t: (gi, ti) for gi, (g, ts) in enumerate(GROUP) for ti, t in enumerate(ts)}


def tagsof(i):
    t = [x for x in (TAGS.get(i) or ["?"]) if not x.startswith("pt")]
    if not any(x.startswith("route_") for x in t): t.append("route_chat")
    return t


class Tab:
    def __init__(s): s.T = {}; s.tot = [0, 0, 0, 0]; s.bad = []
    def add(s, i, ref, got, what):
        d = rt.diff_fields(ref, got, TAGS.get(i)); k = 3 if d else (1 if got.get("st") == 200 else 2)
        s.tot[0] += 1; s.tot[k] += 1
        for t in tagsof(i):
            e = s.T.setdefault(t, [0, 0, 0, 0]); e[0] += 1; e[k] += 1
        if d:
            x = dict(id=i, run=what, fields=d, tags=TAGS.get(i))
            for f in d:
                if f != "lp": x["ref_" + f] = ref.get(f); x["got_" + f] = got.get(f)
            if "text" in d:
                p = 0; A, B = ref.get("text") or "", got.get("text") or ""
                while p < min(len(A), len(B)) and A[p] == B[p]: p += 1
                x["text_first_diff_at"] = p; x["text_lens"] = [len(A), len(B)]
            s.bad.append(x)


def byid(rows, lab=None):
    o = {}
    for r in rows:
        if r.get("cancelled") or (lab and r.get("label") != lab): continue
        o.setdefault(r["id"], r)
    return o


ALL = Tab(); BASE = Tab(); RUNSUM = []; BASEFORM = Tab(); NONDET = {}
REF = {}; REFS = {}    # solo knob-off replies (non-stream / stream) by id


def pairfile(e, f, title):
    rows = load(P(e, f))
    if not rows: return
    labs = sorted({r["label"] for r in rows}); d = {l: byid(rows, l) for l in labs}
    for l in labs:
        if l == "off": REF.update(d[l])
        if l == "off_s": REFS.update(d[l])
    # requests whose levers-off reply is not the same twice: the base is not deterministic there, so no lever claim can be tested on them
    nd = set()
    for l in labs:
        if l.startswith("off") and l not in ("off", "off_s") and "s" not in l.split("_")[1:] and "off" in d:
            nd |= {i for i, r in d[l].items() if i in d["off"] and rt.diff_fields(d["off"][i], r, TAGS.get(i))}
    # shape rule: /v1/messages drops `seed`, so EVERY sampled request on that route is unseeded (two equal levers-off replies are luck)
    for i in {i for l in labs for i in d[l]}:
        tg = TAGS.get(i) or []
        if "route_msg" in tg and "sampled" in tg: nd.add(i)
    nd |= set(NONDET)
    for i in nd: NONDET.setdefault(i, f"{e}/{f}")
    for l in labs:
        st = "s" in l.split("_")[1:]; b = d.get("off_s" if st else "off")
        if l.startswith("on") and b:
            t = Tab()
            for i, r in d[l].items():
                if i in nd: continue
                if i in b: t.add(i, b[i], r, f"{e}/{f} {l}"); ALL.add(i, b[i], r, f"{e}/{f} {l}")
            RUNSUM.append((f"{title}: {l} vs {'off_s' if st else 'off'} (same engine, back to back)", e + "/" + f, t))
        elif l.startswith("off") and l not in ("off", "off_s") and not st and "off" in d:
            t = Tab()
            for i, r in d[l].items():
                if i in d["off"]: t.add(i, d["off"][i], r, f"{e}/{f} {l}"); BASE.add(i, d["off"][i], r, f"{e}/{f} {l}")
            RUNSUM.append((f"BASE determinism, {title}: {l} vs off (levers off twice)", e + "/" + f, t))
    if "off_s" in d and "off" in d:
        for i, r in d["off_s"].items():
            if i in d["off"]: BASEFORM.add(i, d["off"][i], r, f"{e}/{f} off_s vs off")


def runfile(e, f, title, stream_ok=False):
    rows = load(P(e, f))
    if not rows: return
    t = Tab(); n = 0
    for r in rows:
        if r.get("cancelled"): continue
        ref = (REFS if r.get("mode") == "s" and r["id"] in REFS else REF).get(r["id"])
        if ref is None or r["id"] in NONDET: n += 1; continue
        if r.get("mode") == "s" and r["id"] not in REFS:
            # a streamed reply against a non-streamed reference: the two known base stream forms are not lever differences
            a = dict(ref); b = dict(r)
            # base stream form 1: the non-streamed reasoning text is right-trimmed, the streamed one keeps its trailing newlines
            if (b.get("rsn") or "") != (a.get("rsn") or "") and (b.get("rsn") or "").rstrip() == (a.get("rsn") or "").rstrip(): b["rsn"] = a.get("rsn")
            if a.get("tools") == [] and b.get("tools") and "<tool_call>" in (a.get("text") or ""): continue
            r = b
        t.add(r["id"], ref, r, f"{e}/{f}"); ALL.add(r["id"], ref, r, f"{e}/{f}")
    RUNSUM.append((title + (f" [{n} replies without a usable solo reference skipped]" if n else ""), e + "/" + f, t))


def envfile(e, f, title):
    """flags-off engine against the knob-off replies: levers-off against levers-off (BASE; covers knob-off == flags-off and the no-knob levers)"""
    rows = load(P(e, f))
    if not rows: return
    t = Tab()
    for r in rows:
        if r["id"] in REF and r["id"] not in NONDET: t.add(r["id"], REF[r["id"]], r, f"{e}/{f}")
    RUNSUM.append((title, e + "/" + f, t)); return t


pairfile("on", "pair1.jsonl", "matrix 1 (625 fixed shapes)"); pairfile("on", "pair3.jsonl", "matrix 3 (random cross product, grids, copies, images)")
pairfile("on", "sess.jsonl", "live multi-turn sessions (resume, regenerate, edited turn; on_c = after a cancelled stream)")
pairfile("on", "pairL.jsonl", "long prompts (40K-185K words)"); pairfile("on", "pairO.jsonl", "long outputs (max_tokens 4,096)")
pairfile("on2", "pair2.jsonl", "prediction field built from the real reply (exact / longer / shorter / partly / wrong / shifted / empty)")
pairfile("on2", "pairS3.jsonl", "streamed, matrix 3"); pairfile("on2", "pairS1.jsonl", "streamed, matrix 1"); pairfile("on2", "pairC.jsonl", "request after the same request was cancelled mid-stream")
pairfile("on3", "pairX.jsonl", "re-check of x0606 (streamed reasoning, levers off and on)"); pairfile("on3", "pair4.jsonl", "matrix 4 (second random cross product, new seed)"); pairfile("on3", "sess4.jsonl", "live multi-turn sessions, STREAMED"); pairfile("on3", "pairS4.jsonl", "streamed, matrix 4")
runfile("on2", "c2.jsonl", "2 in flight + 1 client that cancels streams (default profile)"); runfile("on2", "c3.jsonl", "3 clients on 2 lanes, each request after a cancelled copy of itself, streamed and not")
runfile("chk", "k1.jsonl", "device check modes ON, matrix 1"); runfile("chk", "ks.jsonl", "device check modes ON, session turns replayed"); runfile("chk", "k3.jsonl", "device check modes ON, matrix 3"); runfile("chk", "kL.jsonl", "device check modes ON, long prompts")
runfile("u16b", "u16.jsonl", "USERS16 profile, 16 in flight (burst) + 2 cancelling clients, matrix 1"); runfile("u16b", "u8s.jsonl", "USERS16 profile, 8 in flight, streamed, slow readers, matrix 3")
runfile("u16b", "u4.jsonl", "USERS16 profile, 4 in flight, each after a cancelled copy"); runfile("u16b", "u3.jsonl", "USERS16 profile, 3 sessions in flight (turns replayed)")
runfile("u16b", "u16y.jsonl", "USERS16 profile, 16 in flight (burst), matrix 4"); runfile("u16d", "u16d.jsonl", "USERS16 profile with the recipe's own pool (16 x 40,960), 16 in flight"); runfile("ag4", "ag.jsonl", "AGENTS profile (4 lanes on the head), 4 sessions in flight"); runfile("ag3", "ag.jsonl", "AGENTS profile (3 lanes on the head), 3 sessions in flight"); runfile("ag3", "agy.jsonl", "AGENTS profile, 3 in flight, matrix 4, each after a cancelled copy, streamed and not")
E = [envfile("off", "cold.jsonl", "FLAGS-OFF ENGINE, session turns COLD (reverse order, fresh engine) vs resumed knob-off"), envfile("off", "env1.jsonl", "FLAGS-OFF ENGINE vs knob-off, matrix 1"),
     envfile("off", "env3.jsonl", "FLAGS-OFF ENGINE vs knob-off, matrix 3"), envfile("off", "envL.jsonl", "FLAGS-OFF ENGINE vs knob-off, long prompts"), envfile("off", "envO.jsonl", "FLAGS-OFF ENGINE vs knob-off, long outputs")]


def line(name, v): return f"| {name:<22} | {v[0]:>6} | {v[1]:>9} | {v[3]:>9} | {v[2]:>7} |"


HDR = f"| {'shape':<22} | tested | identical | DIFFERENT | errored |"; SEP = "+" + "-" * 24 + "+--------+-----------+-----------+---------+"
print("A. LEVERS ON against LEVERS OFF, every comparison of the night added up, by request shape (a request counts once per run it was in)")
print(SEP); print(HDR)
seen = set()
for g, ts in GROUP:
    rows = [t for t in ts if t in ALL.T]
    if rows: print(SEP); print(f"| {g:<22} |        |           |           |         |")
    for t in rows: print(line(t, ALL.T[t])); seen.add(t)
rest = sorted(t for t in ALL.T if t not in seen)
if rest:
    print(SEP); print(f"| {'OTHER':<22} |        |           |           |         |")
    for t in rest: print(line(t, ALL.T[t]))
print(SEP); print(line("ALL COMPARISONS", ALL.tot)); print(SEP)
print("\nB. Run by run")
W = max([len(x[0]) for x in RUNSUM] + [10])
print(f"| {'run':<{W}} | tested | identical | DIFFERENT | errored |")
for title, f, t in RUNSUM: print(f"| {title:<{W}} | {t.tot[0]:>6} | {t.tot[1]:>9} | {t.tot[3]:>9} | {t.tot[2]:>7} |")
print(f"\nC. BASE (no lever involved). Levers off twice: tested {BASE.tot[0]}, identical {BASE.tot[1]}, different {BASE.tot[3]}, errored {BASE.tot[2]}. Streamed vs non-streamed with levers off: tested {BASEFORM.tot[0]}, identical {BASEFORM.tot[1]}, different {BASEFORM.tot[3]}")
nt = {}
for i in NONDET:
    for t in tagsof(i): nt[t] = nt.get(t, 0) + 1
print(f"   Requests whose levers-off reply differs between two runs (base NOT deterministic; left out of table A): {len(NONDET)}" + (": by tag " + ", ".join(f"{k} {v}" for k, v in sorted(nt.items(), key=lambda x: -x[1])[:14]) if NONDET else ""))
for i in sorted(NONDET)[:40]: print("     ", i, NONDET[i], TAGS.get(i))
allbad = ALL.bad + [dict(x, base=True) for x in BASE.bad] + [dict(x, base=True) for t in E if t for x in t.bad]
print(f"\nD. Differences: {len(ALL.bad)} lever, {len(BASE.bad)} base (off twice), {sum(len(t.bad) for t in E if t)} flags-off engine vs knob-off")
for x in allbad[:60]: print("  ", "BASE" if x.get("base") else "LEVER", x["id"], x["run"], x["fields"], x.get("tags"), "text first differs at", x.get("text_first_diff_at"), x.get("text_lens"))
if OUT:
    with open(OUT, "w") as f:
        for x in allbad + [dict(x, baseform=True) for x in BASEFORM.bad]: f.write(json.dumps(x, ensure_ascii=False) + "\n")
