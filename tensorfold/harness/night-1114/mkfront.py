#!/usr/bin/env python3
"""mkfront.py VALS.json REPO [--write] : the front page's numbers table (README.md block between the first box-drawing table's code fences)
and assets/numbers.json from the night's measured values (tableN.py's vals.json). Without --write it only prints. Nothing is typed in
except the custom engine R column (measured in an earlier window, 2026-11-14, labelled so) and the labels."""
import json, sys, os, re
V = json.load(open(sys.argv[1])); REPO = sys.argv[2]; WRITE = "--write" in sys.argv
DATE = "2026-11-15"
def f(key, col):
    x = (V.get(key) or {}).get(col)
    try: return float(x)
    except (TypeError, ValueError): return None
def fmt(x, nd=1): return "-" if x is None else (f"{x:,.0f}" if nd == 0 else f"{x:.{nd}f}")
def offs(key, nd=1):
    a, b = f(key, "off"), f(key, "off2")
    if a is None and b is None: return "-"
    if b is None: return fmt(a, nd) + " (one run)"
    return f"{fmt(a, nd)} / {fmt(b, nd)}"
def offmean(key):
    v = [x for x in (f(key, "off"), f(key, "off2")) if x is not None]; return sum(v) / len(v) if v else None
def on(key, nd=1): return fmt(f(key, "on"), nd)
def tri(keys, fn): return " / ".join(fmt(fn(k), 0) for k in keys)
U = V.get("users", {}); AG = V.get("agents", {})
def u(lab, n, fld="tps"):
    x = U.get(lab, {}).get(str(n)); return None if not x else x[fld]
def umia(n, fld="tps"):
    v = [x for x in (u("miafp", n, fld), u("miafp2", n, fld)) if x is not None]; return sum(v) / len(v) if v else None
def urange(lab, n):
    runs = []
    for l in ([lab] if lab != "mia" else ["miafp", "miafp2"]):
        x = U.get(l, {}).get(str(n)); runs += x["runs"] if x else []
    return f"{min(runs):.0f}-{max(runs):.0f}" if runs else "-"
def lastrange(lab, n):
    runs = []
    for l in ([lab] if lab != "mia" else ["miafp", "miafp2"]):
        x = U.get(l, {}).get(str(n)); runs += x["lasts"] if x else []
    return (f"{min(runs):.1f}-{max(runs):.1f} s" if f"{min(runs):.1f}" != f"{max(runs):.1f}" else f"{max(runs):.1f} s") if runs else "-"
def ag(k):
    a = AG.get(k); return "-" if not a else f"{a['fu_med']:.1f} s"
SD = ("sd_prose", "sd_code", "sd_structured")
def pct(key, lower=False):
    o, x = offmean(key), f(key, "on")
    if o is None or x is None: return ""
    return f"{(x / o - 1) * 100:+.0f} %"
rows = [
 ("chat, 1 user, 3 prompts", offs("chat3"), on("chat3"), pct("chat3")),
 ("code, 1 user, 3 prompts", offs("code3"), on("code3"), pct("code3")),
 ("320 held prompts, 1 user", offs("held"), on("held"), pct("held")),
 ("sparkDash prose / code /", tri(SD, offmean), tri(SD, lambda k: f(k, "on")), "+19 / +26 / +52 %"),
 ("  structured", "", "", ""),
 ("seeded sampled chat, T 0.7", offs("samp_chat"), on("samp_chat"), pct("samp_chat")),
 ("first token, short prompt", offs("ttft_ms", 0) + " ms", on("ttft_ms", 0) + " ms", pct("ttft_ms")),
 ("prefill, 116K-token prompt", offs("prefill_128000", 0), on("prefill_128000", 0), pct("prefill_128000")),
 ("227K-token prompt, first token", offs("ctx_ttft", 0) + " s", on("ctx_ttft", 0) + " s", pct("ctx_ttft")),
 ("file-rewrite set", offs("edit", 0), on("edit", 0), pct("edit")),
 ("  with \"prediction\" (a)", offs("pred1_num", 0), on("pred1_num", 0), pct("pred1_num")),
 ("agent loop, 46 turns", offs("loop0", 0), on("loop0", 0), pct("loop0")),
 ("4 / 8 users, distinct (b)", f"{fmt(umia(4), 0)} / {fmt(umia(8), 0)}", f"{fmt(u('mu', 4), 0)} / {fmt(u('mu', 8), 0)}", f"{(u('mu',4)/umia(4)-1)*100:+.0f} / {(u('mu',8)/umia(8)-1)*100:+.0f} %"),
 ("12 / 16 users, distinct (b)", f"{fmt(umia(12), 0)} / {fmt(umia(16), 0)}", f"{fmt(u('mu', 12), 0)} / {fmt(u('mu', 16), 0)}", f"{(u('mu',12)/umia(12)-1)*100:+.0f} / {(u('mu',16)/umia(16)-1)*100:+.0f} %"),
 ("last first token at 16 users", lastrange("mia", 16), lastrange("mu", 16), ""),
 ("5 agents x 56K, follow-up", ag("agmia5_at5"), ag("agA_at5"), "25 of 25 cached"),
 ("  turn's first token (c)", "", "", ""),
 ("10 agents x 56K, same (c)", ag("agmia_at10"), ag("agA_at10"), "13 of 20 cached"),
 ("KV cache", "bf16; (b) (c) fp8", "bf16 (exact)", ""),
 ("quality (qeval /35)", (V.get("qeval", {}).get("off") or "-").split("/")[0], (V.get("qeval", {}).get("on") or "-").split("/")[0], ""),
]
W0, W = 32, 18
def line(l, m, r): return l + "─" * (W0 + 1) + (m + "─" * (W + 2)) * 3 + r
def cells(a, b, c, d): return f"│ {a:<{W0}}│ {b:<{W}} │ {c:<{W}} │ {d:<{W}} │"
T = [line("┌", "┬", "┐"), cells("spark-923b, tok/s", "levers off", "pico-dgx-qwen38", "change"), cells("", "(first / last run)", "(this repo)", ""), line("├", "┼", "┤")]
T += [cells(*r) for r in rows] + [line("└", "┴", "┘")]
for r in rows:
    for c in r[1:]: assert len(c) <= W, (r[0], c)
    assert len(r[0]) <= W0, r[0]
win = lambda k: (V.get("window", {}).get(k) or "?")
notes = f"""  spark-923b · temp 0 · thinking off · decode timed from the first token · {DATE} (UTC)
  One-user rows: the final image (patches 0101-0129), official NVIDIA weights, bf16 KV cache,
    3 streams at 262K, ONE test window, the same image three times: levers off (04:46-05:57),
    this repo (05:57-07:03), levers off again (07:03-08:12). "Levers off" = MiaAI-Lab's patches
    and launcher with every lever of ours switched off. Same texts both ways: counts below.
  (a) 16 numbered-line rewrites sent with the request's "prediction" field; the field also
      works with the levers off (it is part of the same image)
  (b) many users: a second window the same night (09:02-09:55), same box, same engine build.
      Left column = MiaAI-Lab's shipped shape for 16 users: 16 fixed slots x 64K, fp8 KV, our
      levers off, measured before AND after ours (mean shown). Ours: USERS16 profile, bf16 KV,
      16 lanes x 32,768 tokens. Median of 3 runs of 512 tokens each
  (c) same window. Left: MiaAI-Lab's default (5 x 262K, fp8 KV, levers off). Ours: AGENTS
      profile, bf16 KV. With 10 agents the bf16 prompt cache is too small: 7 of 20 follow-ups
      were read again from scratch. Our 5-agent run followed the 10-agent run on the same
      engine (its opening turn was already cached; follow-up turns are what is compared)
  sparkDash = sparkDash 1.8.8's own c1 tests, median of 5 runs each."""
C1, C2 = 26, 12
def l2(l, m, r): return l + "─" * (C1 + 1) + (m + "─" * (C2 + 2)) * 5 + r
def c2(*a): return f"│ {a[0]:<{C1}}│ " + " │ ".join(f"{x:<{C2}}" for x in a[1:]) + " │"
R2 = [("weights", "official", "re-quantised", "official", "official", "official (e)"),
      ("KV cache", "bf16 exact", "its own", "fp8", "8-bit", "fp8"),
      ("chat, 3 prompts", "52.2-54.9", "54.4-54.8", "46.0", "51.0", "49.4-50.3"),
      ("code, 3 prompts", "88.1-91.3", "90.6-91.4", "78.9", "82.5", "82.0-82.6"),
      ("104 held prompts", "67.1-68.6", "65.4-65.6", "58.7", "63.1", "60.4-60.5"),
      ("first token, short prompt", "110-123 ms", "119-120 ms", "176-178 ms", "200-202 ms", "160-161 ms"),
      ("4 / 8 users, distinct", "156-163/221-225", "138 / 165", "149 / 155", "81 / 82", "108-109 / 111"),
      ("cold prefill, 29K prompt", "2,478-2,625", "1,814-2,130", "2,474-2,481", "1,833", "3,936-3,992"),
      ("runs that night", "5", "2", "1", "1", "2")]
for r in R2:
    for c in r[1:]: assert len(c) <= C2 + 3, r
C2 = 15
T2 = [l2("┌", "┬", "┐"), c2("spark-4e5c, same windows", "this repo", "custom engine R", "MiaAI-Lab", "custom engine V", "a vLLM recipe"), c2("(d)", "fresh install", "v0.3.13", "as shipped", "v0.7.9", "fe8e336b"), l2("├", "┼", "┤")] + [c2(*r) for r in R2] + [l2("└", "┴", "┘")]
notes2 = """  (d) measured by us with our kit on spark-4e5c, 2026-11-15, two test windows, one engine at a
      time, every project with its own shipped defaults; ranges over all runs of the night.
      Ours here is the from-scratch install of this repo (16-user rows with the pool fix).
  (e) NVIDIA's weights with one shard replaced by a fine-tuned draft head (drafts only)
  What it says: on generation speed this repo leads every official-weights setup we could
    install. custom engine R (re-quantised weights, so a different model numerically) TIES on the 3-prompt
    chat / code bench and is 2-5 % behind on the held prompts. This repo is NOT the fastest at
    reading long prompts: a vLLM recipe prefills a cold prompt at 3.5-4.0K tok/s
    against our 2.4-2.6K. Nobody's output identity but ours was tested. Details:
    """
block = "```\n" + "\n".join(T) + "\n" + notes + "\n```\n\n```\n" + "\n".join(T2) + "\n" + notes2 + "\n```"
print(block)
# numbers.json
def r1(x): return None if x is None else round(x, 1)
c16 = u("mu", 16); stats = [[f"{on('chat3', 0)}/{on('code3', 0)}", "chat/code, 1 user, tok/s"], [on("held"), "320 held prompts, tok/s"],
         [fmt(c16, 0), "16 users, distinct, tok/s"], [f"{on('ttft_ms', 0)} ms", f"first token (was {fmt(offmean('ttft_ms'), 0)})"]]
single = []
for key, lab, note in (("code3", "code", "3 prompts, median of 9"), ("held", "held", "320 held prompts"), ("chat3", "chat", "3 prompts, median of 9")):
    single += [[f"pico-dgx-qwen38  {lab}", r1(f(key, "on")), "bf16 KV · " + note, True],
               [f"levers off       {lab}", r1(offmean(key)), "same image, our levers off (mean of 2 runs)", False]]
users = []
for n in (16, 8, 4):
    if u("mu", n) is not None: users.append([f"pico-dgx-qwen38  c{n}", r1(u("mu", n)), "bf16 KV, 16-lane pool", True])
    if umia(n) is not None: users.append([f"MiaAI config     c{n}", r1(umia(n)), "16 x 64K slots, fp8 KV, levers off", False])
    if n == 16 and u("mu", 12) is not None: users += [["pico-dgx-qwen38  c12", r1(u("mu", 12)), "bf16 KV, 16-lane pool", True], ["MiaAI config     c12", r1(umia(12)), "16 x 64K slots, fp8 KV, levers off", False]]
J = {"banner": {"subtitle": "Qwen3.8-Flash-Next on one DGX Spark  ·  official FP4 weights  ·  exact output  ·  262K  ·  16 users", "stats": stats},
     "charts": [{"file": "single.png", "title": "one user: decode tok/s",
                 "sub": f"spark-923b · temp 0 · thinking off · decode timed from the first token · ours and levers off: one window, same image, {DATE}",
                 "rows": single, "note": "levers off = MiaAI-Lab's patches and launcher, every lever of ours off, measured before and after. Other projects: see the second table in the README."},
                {"file": "users.png", "title": "many users: total decode tok/s, a different prompt per user",
                 "sub": f"512 tok · temp 0 · thinking off · spark-923b (GB10) · distinct coding prompts · one window, {DATE}",
                 "rows": users, "note": "c4 ... c16 = users at once, median of 3 runs. Same window, same engine build; the MiaAI config was measured before and after ours (mean)."}]}
if any(r[1] is None for c in J["charts"] for r in c["rows"]): print("WARNING: a chart row has no value:", [r for c in J["charts"] for r in c["rows"] if r[1] is None])
if WRITE:
    p = os.path.join(REPO, "README.md"); s = open(p).read()
    a = s.index("```\n┌"); b = s.index("## ✧ what \"nothing given up\" means"); block = block + "\n\n"
    open(p, "w").write(s[:a] + block + s[b:]); json.dump(J, open(os.path.join(REPO, "assets/numbers.json"), "w"), indent=1, ensure_ascii=False)
    print("written README.md table + assets/numbers.json")
else: print(json.dumps(J["banner"], ensure_ascii=False))
