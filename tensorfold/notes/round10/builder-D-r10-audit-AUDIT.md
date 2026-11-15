# Round 10 leave-one-out audit of the recipe's default levers (builder D, head spark-3caa)

Base: integ9 7061396 (patches 0101-0118), official NVIDIA FP4-Local checkpoint, recipe round-9 defaults.
Audit binary: D10a sha256 528ae962f5987f4d = integ9 + knob-file words for the levers (measuring aid) + CERT_TOPK_ROWS.
Everything below is MEASURED on the head on 2026-11-14 unless a line says otherwise.

## Method (and why it is not the usual A/B)

- ONE running engine; a lever is switched off through TF_FLASHNEXT_Q38_KNOBFILE (re-read every 200 ms). Every prompt is sent once per
  variant, back to back, order rotated by prompt, with the reference (no knob word) twice. "Lever value" = time with ONLY that lever
  off / reference time - 1, so + means the lever pays inside the full stack. The second reference against the first is the ruler's own
  noise: +-0.05..0.16 % (95 %) on the per-prompt mean.
- Why: two engine STARTS of the same binary and flags differ by up to ~2 % here (references first / last of the env-engine series:
  held 67.51 / 66.27, T0.7 48.93 / 48.30; texts identical). Checked on DEEP: env engines say +4.3 %, the knob says +2.6 %, builder B's
  round-9 env pairs said +2.2 %; the per-category differences agree, the env engine is ~1.5 points slow in every category. Env-flag
  pairs cannot judge a lever under ~2 %; rows marked (env) below carry +-1 %.
- Sets: held = every 3rd of the 320 held prompts (107; 104 with more than 16 tokens), 256 tokens, greedy, thinking off. Sampled = the
  first 40 held chat prompts, 256 tokens, temperature 0.7, seed 9000 + id, server top_k 20 / top_p 0.95, thinking off and on.
  First token = a new short prompt every request, 30 rounds. 16 users = 16 x 32K lanes on the head (pool + 128-row passes), 2 cycles
  of: reference, variants in rotated order, reference; a variant = 2 x distinct-16 (code prompts) + the 107 held prompts with 16 in
  flight. Fresh = builder A's out-of-fit set (every 4th prompt), fitted levers only.
- Identity: every variant returned the reference's text for every prompt (greedy) or per seed (sampled, reasoning included), also
  with 16 in flight. Counts are in the raw tables (results/rD10, PROGRESS-D.md round 10).

## The table (lever value in %, + = the lever pays; n/a = cannot act in that mode by construction)

```
+----------------------------+----------+---------+---------+----------+-----------+-----------+-------------------------------+
| lever (recipe flag)        | 1 user   | 1 user  | 1 user  | first    | 16 users  | 16 users  | verdict                       |
|                            | greedy   | T0.7    | T0.7    | token    | held, 16  | distinct  |                               |
|                            | held-104 | think   | think   | (ms)     | in flight | (noise    |                               |
|                            |          | off     | on      |          |           | +-2.5)    |                               |
+----------------------------+----------+---------+---------+----------+-----------+-----------+-------------------------------+
| ruler noise (95 %)         | +-0.05   | +-0.08  | +-0.08  | +-1.5 ms | +-0.15    | +-2.5     |                               |
| CERT (certified head)      | +5.18    | +3.16   | +3.42   | 0        | -0.06     | +0.1      | keep; ZERO at 16 users        |
|  of which CERT_TOPK=20     | n/a      | +3.16   | +3.47   | 0        | 0 (rows>8)| n/a       | keep; all of CERT when sampled|
|  of which sm8 copy (env)   | +0.8..2.7| +2.9..4.4 (with top-k)       | | not run   |           | keep                          |
| FUSED=3 (+ST, read-out)    | +1.69    | +1.64   | +1.72   | 0        | +0.16     | -1.4      | keep; ZERO at 16 users        |
|  of which FUSED_HALF       | +1.09    | not run | not run | 0        | not run   |           | keep                          |
| HALF=96 (#1171 half tiles) | -0.21    | -0.23   | not run | 0        | +0.06     | -1.4      | NEGATIVE at 1 user: turn OFF  |
| MTP_LAST (#1925)           | +0.33    | +0.15   | not run | 0        | (in lone) |           | keep (small, real greedy)     |
| LATE_SKIP=16               | +0.33    | +0.31   | not run | 0        | +1.62     | -0.3      | keep                          |
| KEEP_INPASS=4096           | -0.02    | not run | not run | -43 ms   | +6.8 with |  +1.3     | keep (first token lever)      |
| LANE_REUSE                 | +2.41    | not run | not run | -21 ms   | LANE_REUSE|           | keep; also DECODE speed       |
| HEAD2 (rotation r9)        | +3.20    | +0.02   | -0.01   | 0        | (in lone) |           | keep; zero when sampled       |
|  HEAD2_FAST=3 (B, env r9)  | +0.94 (builder B, round 9; not re-measured)             |           | keep                          |
|  HEAD2_SAMPLED=1 (ADD-ONE) | n/a      | +1.5    | +0.9    | 0        | n/a       |           | NOT a default today: turn ON  |
| DEEP=0.9, DEEP_MAX=12      | +2.59    | +0.28   | +0.25   | 0        | (in lone) |           | keep; chat +0.75, code +6.8   |
| STOP2=0.5                  | +1.24    | +0.69   | +0.68   | 0        | (in lone) |           | keep                          |
| lone-chain group at 16     |          |         |         |          | +0.57     | -1.8      | (HEAD2 + DEEP + STOP2 +       |
|                            |          |         |         |          |           |           |  MTP_LAST together: tails)    |
| GDN_MANY=2                 | +0.05 as a group (with    | 0        | +3.24     | +1.9      | keep (16 users)               |
| ATTN_LANES                 |  BEST_LANES, QSA_CERT):   | 0        | +1.69     | +1.3      | keep (16 users)               |
| BEST_LANES                 |  zero at one user         | 0        | +0.64     | -0.2      | keep (small)                  |
| QSA_CERT + QSA_HALF        |                           | 0        | not run   |           | long prompts only (builder C) |
| COPY_ROWS=32 (env)         | 0 on held; edit set +18.5..20 %       | | +2.4      | -1.7      | keep; not negative at 16      |
| ELASTIC                    | not a speed lever (prompt cache size); see the memory note below    | keep, but mind the reserve    |
| MANY=256                   | not re-measured (first token of 16 arrivals; rounds 6-8)            | keep                          |
+----------------------------+----------+---------+---------+----------+-----------+-----------+-------------------------------+
```

Per category, one user greedy (chat / code / math): CERT +5.57 / +4.39 / +4.05; FUSED +1.77 / +1.57 / +1.43; HALF -0.22 / -0.13 /
-0.28; LANE_REUSE +1.16 / +5.74 / +4.89; HEAD2 +2.92 / +3.70 / +4.11; DEEP +0.75 / +6.77 / +7.22; STOP2 +1.38 / +0.86 / +0.98;
MTP_LAST +0.26 / +0.50 / +0.50; LATE_SKIP +0.36 / +0.39 / +0.11.

Fresh set (builder A's out-of-fit prompts, every 4th = 31 prompts; fitted levers; value %; texts 31/31 identical; ruler +-0.22):
CERT +5.34, FUSED +2.10, HEAD2 +2.32 (chat +2.0, code +3.1, math +4.8, struct +3.2, agent +5.1, non-English +0.9), DEEP +2.16
(chat +0.2, code +4.3, math +8.9, struct +7.5, agent +9.2, non-English -0.4 = noise), STOP2 +1.38. The fitted levers hold out of fit.

Whole stack against levers off (env engines, +-1 %): held +13.9..16.0 %, T0.7 +5.9..7.2 %; sparkDash 1.8.8 itself (greedy, its own
runner): c1 prose 56.3 -> 61.3 (+8.9 %), c1 code 91.8 -> 121.0 (+32 %), c1 structured 105.3 -> 166.9 (+58 %), catalog prose
43.5 -> 48.0, structural 66.0 -> 77.0, code 81.0 -> 98.0; first token ~95 -> ~70 ms.

## Blunt reading

1. HALF=96 is a negative default at one user (-0.21 +- 0.07 greedy, -0.23 +- 0.12 sampled) and zero at 16 users. It was +1 % when it
   was ported in round 6; FUSED has since taken the read-out projections it helped, and what is left loses. Recommend HALF=0.
2. CERT and FUSED do nothing at 16 users (-0.06 and +0.16 on a +-0.15 ruler). They are not harmful there, so no per-mode switch is
   needed; they are simply one-user levers. Do not quote them as many-user gains.
3. HEAD2, DEEP, STOP2 are close to zero for sampled text (HEAD2 exactly zero: it is switched off for sampled chains). The sampled
   request is the under-served mode: default vs levers-off is +6..7 % sampled against +14..16 % greedy. HEAD2_SAMPLED=1 (builder B's
   lever, off in the recipe) is +1.5 % (thinking off) / +0.9 % (thinking on) with seeded texts identical: make it a default.
4. LANE_REUSE is worth +2.4 % of DECODE speed (code +5.7 %), not only its 21 ms of first token. Nobody had measured that.
5. KEEP_INPASS, MANY, QSA_CERT, ELASTIC are first-token / prompt levers: zero for decode speed by design, fine.
6. The distinct-16 bench cannot see anything under ~3 % (two-run means swing 270-293). Every 16-user claim under 3 % must come from
   the held-prompts-16-in-flight run (four references 199.3-200.0).
7. integ9 took D9b, not D9c: without CERT_TOPK_ROWS, 16 SAMPLED users lose 1.6 % (187.7 against 190.7; top-k off 190.6). The row
   limit (default 8) is in this round's patch.
8. Memory: with ELASTIC the prompt cache grows into everything the reserve leaves (110-130 MiB a kept state; 17.7 GiB after ~140
   distinct prompts). On the head MemAvailable went 30 -> 10.6 GiB with the default 9 GiB reserve. Not a hang risk by itself (the
   budget is capped), but any box that runs something beside the engine should raise TENSORFOLD_MEMORY_RESERVE_GIB (13 on the head).

## Recommended default changes (recipe.sh)

```
+----------------------------------+-----------------------------+--------------------------------------------------------------+
| change                           | modes it moves              | evidence                                                     |
+----------------------------------+-----------------------------+--------------------------------------------------------------+
| HALF=0 (was 96)                  | 1 user +0.2 %, 16 users 0   | knob pairs, greedy and sampled                               |
| HEAD2_SAMPLED=1 (was off)        | sampled +0.9..1.5 %         | add-one knob pairs, 40 + 40 seeded prompts identical         |
| CERT_TOPK_ROWS=8 (new, patch)    | 16 sampled users +1.6 %     | knob run 16 in flight; round 9 D9c                           |
| CERT_GATHER=1 (new, patch)       | sampled +1.6..1.7 %         | see handoff/D/r10-levers/NOTE.md                             |
| the three sampled changes as one | T0.7 +3.6 %, thinking +2.8 %| one engine, D10e, seeded texts identical                     |
| EXP_HALF (new, patch): stay OFF  | zero drafted; no-draft +1.0 | see NOTE.md (turn on only for --no-drafts deployments)       |
+----------------------------------+-----------------------------+--------------------------------------------------------------+
```
The recommended sampled stack was then run as ONE configuration against today's default inside one engine (D10e; seeded; texts
identical in every variant): T0.7 thinking off 49.10 -> 50.85 (+3.56 %; gather alone 49.93), thinking on 51.97 -> 53.44 (+2.83 %;
gather alone 52.80); greedy (36 held prompts) 64.50 -> 64.68 (+0.27 %, from HALF=0). NOT run on it: edit / agent / needle / image /
qeval, 16 users with the three changes together (each was run alone at 16 users or cannot act there).

No lever needs a per-request switch: none of the defaults is negative in one mode and positive in another by more than noise,
except HALF (negative at one user, zero elsewhere), which simply goes off.

## Raw data

Head ~/q38/runs/D_A_k1 (one-user knob audit), D_A_u16 (+ /s sampled), D_A_e* (env engines), D_A_b2 / D_A_e1 / D_A_e2 / D_A_e16
(new levers), D_A_k1f (fresh). Mac copies: q38/results/rD10, summaries q38/results_D10_*.txt. Tools: this directory (aud.py,
aud16.sh, aud16rep.py, tri.py, cmpcat.py, run10.sh, wq.sh).
