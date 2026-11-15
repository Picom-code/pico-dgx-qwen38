# Builder D, round 11 handoff: no-draft first-request fix, sampled / chat draft study, calibrated gate (measured ~zero), measuring aids

Patch: `D11c-cumulative.patch` = ONE cumulative patch against integ9 7061396. It REPLACES round 10's `D10e-cumulative.patch` (it contains
D10e: CERT_GATHER, EXP_HALF, CERT_TOPK_ROWS, knob words). `git apply --check` passes on Mac q38/wt/I9. Source: Mac q38/wt/D11, branch r11-D,
commit 99f2bbe. Built on the head as ~/q38/tfbin/D11c, binary sha256 5bccf01950764374 (tensorfold-native). Patch and table sha256: `SHA256`.
Everything below is MEASURED on the head (spark-3caa), 2026-11-14, official FP4-Local checkpoint, one user 3 x 262K unless a line says otherwise;
recipe round-9 defaults + round 10's recommended stack (CERT_GATHER=1, HEAD2_SAMPLED=1, HALF_BELOW=0). SIMULATED lines are labelled.

## What is in the patch (new in round 11)

```
+------------------------------------+---------+----------------------------------------------------------------------------------+
| flag                               | default | what                                                                             |
+------------------------------------+---------+----------------------------------------------------------------------------------+
| TF_FLASHNEXT_Q38_NGRAM_SPLIT=N     | 0 (off) | a gather of <= N n-gram table rows (N <= 256): rows in the row cache are copied  |
|  (recommend: recipe default 256)   |         | on the calling thread; each missing row's codes and scales are read by two pool  |
|                                    |         | threads woken together (Pool.runWide); falls back to the stock gather on failure |
| TF_FLASHNEXT_Q38_GCAL=FILE         | unset   | calibrated stop rule for lone chains (tables g / st / sn + 'set' lines; file     |
|  (recommend: leave OFF)            |         | gcal_r11.txt). Measured ~zero out of fit: see below                              |
| TF_FLASHNEXT_Q38_SCHAIN=S,P,B,MAX  | unset   | a sampled chain's own early-stop line / deep line / deep base / depth cap        |
|  (aid; leave unset)                |         | (simulated re-tunes of the raw signal: +0.7..1.2 %, not pursued)                 |
| TF_STEP_LOG lines                  | aid     | QSTEP now ends with a monotonic time (us); new line QCT <us> <head steps> at the |
|                                    |         | chain's end: a log gives chain and verify time of every round                    |
| knob-file words                    | aid     | ngram_split, gcal, gcal_c, gcal_stop2, gcal_deep_p, gcal_deep_base, s_stop2,     |
|                                    |         | s_deep_p, s_deep_base, s_max, deep_p, deep_base, chain_max                       |
+------------------------------------+---------+----------------------------------------------------------------------------------+
```

## 1. The no-draft stream's slow first request: cause and fix (NGRAM_SPLIT)

Cause (engine's own GATHER_LOG lines + four controlled groups, kit `fr.py`): the n-gram table (28.7 GiB) is read row by row with O_DIRECT
positional reads into a row cache. A drafted stream reads its tokens' rows AHEAD while the draft head runs (DRAFT_AHEAD, pending token
included). A stream without drafts has no read-ahead: on text the row cache has not seen, each one-token pass reads 16 rows from the drive
inside the gather (4 rows a thread one after the other, 2 reads a row, workers woken one by one): 3.9-4.7 ms, while only layer 0
(~0.75 ms) is queued on the GPU. A repeat of the text finds the rows in the cache (0.3-0.4 ms). It is not the prompt cache and not buffers:
a no-draft request right after a DRAFTED run of the same prompt has no penalty; one after a 4-token request (prompt cache hit, rows cold) has it.

```
+----------------------------------------------+--------------------------------+--------------------------------+
| 8 chat prompts x 256 greedy, 1st vs 3rd run   | stock gather                   | NGRAM_SPLIT=256                |
+----------------------------------------------+--------------------------------+--------------------------------+
| no-draft stream, first request               | -11.15 % +- 1.01 (25.73 tok/s) | -3.92 % +- 0.37 (27.76 tok/s)  |
| no-draft stream, repeat (2nd vs 3rd)         | +0.03 +- 0.16                  | +0.00 +- 0.11                  |
| drafted stream, first request                | -0.40 +- 0.78 (57.4)           | +0.33 +- 0.47 (57.5)           |
| cold gather a pass / wait at the n-gram layer| 3.9-4.0 ms / 4.5 ms            | 1.35-1.47 ms / 1.9 ms          |
+----------------------------------------------+--------------------------------+--------------------------------+
```
=> a no-draft stream's first pass over a text is +7.9 % faster (25.73 -> 27.76 tok/s). Drafted streams never had the penalty (also checked
on round 10's pair files: first request against later ones -0.2..-0.9 % time, i.e. none) and do not change:
knob pairs, lever value: drafted greedy held-107 -0.02 % (+-0.68), drafted T0.7 chat24 +0.31 % (+-0.29). What is left of the penalty (3.9 %)
is one round of 32 concurrent drive reads (~1.4 ms) against ~0.75 ms hidden behind layer 0.
Identity (flag on, D11c): identity set 51/51 and held 107/107 identical to D10e's texts; knob variants identical (104/104 held, 23/23
sampled seeded, 23/23 no-draft); 16 in flight (16 x 32K): held 107/107 identical to D10a's 16-user reference; 0 crash lines.

## 2. Sampled requests: why the draft levers looked weak, and what a sampled chain does

- They are not weak. Knob pairs on 53 MIXED prompts at T0.7 (seeded, texts identical in every variant):
```
+----------------+--------+-------+-------+-------+----------------------------------------+
| lever (T0.7)   | all    | chat  | code  | math  | greedy audit round 10 (all/chat/code/math) |
+----------------+--------+-------+-------+-------+----------------------------------------+
| DEEP 0.9,6     | +1.87  | +0.94 | +3.41 | +5.35 | +2.59 / +0.75 / +6.77 / +7.22          |
| STOP2 0.5      | +1.33  | +1.72 | +0.25 | +0.50 | +1.24 / +1.38 / +0.86 / +0.98          |
| HEAD2_SAMPLED  | +2.91  | +3.05 | +2.08 | +3.30 | HEAD2 +3.20 / +2.92 / +3.70 / +4.11    |
+----------------+--------+-------+-------+-------+----------------------------------------+
```
  Round 10's sampled audit read ~0 because its ruler is 40 CHAT prompts (DEEP is +0.75 % on chat for greedy too) and HEAD2 was off for
  sampled chains in that reference. On the same 54 mixed prompts the stack runs 67.64 tok/s greedy and 64.92 at T0.7 (-4.0 %).
- A sampled chain draws with the SAME key (seed, position, token id) the main model will use (Gumbel-max over the tempered top-k / top-p
  set). Its draw is the head's best token in 91.3 % of drafts (kept 0.785; when it is not the best: 0.309). Accept rates equal greedy's by
  category (chat 0.60 / 0.61, code 0.755 / 0.746, math 0.738 / 0.739); sampled chains are ~8 % shorter (3.54 against 3.91 drafts a round).
- 'The right confidence': with shared noise P(match) = sum_i 1 / sum_j max(p_j/p_i, q_j/q_i) (p main model, q draft head), but the draft
  side knows only q and the noise. Tested as predictors of 'kept' (held half, AUC): drawn probability 0.867, top probability 0.867,
  tempered-set probability 0.761, keyed-race margin 0.762, logistic on all of them 0.875. No signal beyond the drawn probability.
- The gate's input is over-confident as a 'will be kept' probability, for greedy and sampled alike and equally in chat, code and math:
  0.5-0.6 -> kept 0.38-0.41; 0.7-0.8 -> 0.50-0.52; 0.9-0.97 -> 0.71-0.73; 0.97-0.995 -> 0.82-0.83; 0.9995+ -> 0.97-0.99.

## 3. The calibrated gate (GCAL): built, measured, ~zero. Default OFF, not recommended

Rule: multiply CALIBRATED probabilities (tables fitted on always-16 logs of 107 held prompts, 19k drafts a mode), propose a draft while the
product >= line c, compute the next head step while it is >= line s.
```
+-----------------------------------------------+------------------+------------------+
|                                               | greedy           | T0.7 seeded      |
+-----------------------------------------------+------------------+------------------+
| SIMULATED, out of sample (0.15 / 0.30, d16)   | +1.6 / +1.9 %    | +2.5 / +2.2 %    |
| measured, 36 FIT prompts (lines picked here)  | +0.7 .. +1.1 %   | +1.2 .. +2.0 %   |
| MEASURED, 106 UNTOUCHED prompts (final lines) | +0.12 % (+-0.16) | +0.40 % (+-0.17) |
|   by category chat / code / math              | +0.19/+0.34/-0.69| +0.52/+0.52/-0.62|
+-----------------------------------------------+------------------+------------------+
```
(final lines: greedy 0.20 / 0.40, sampled 0.20 / 0.30, depth 12; value = reference time / lever time - 1; texts identical 105/105 and
106/106. T0.7 thinking, chat30: +0.61 % (+-0.3), reasoning + answer 30/30 identical.)
Identity with GCAL + NGRAM_SPLIT=256 both ON (engine gp / g16, D11c): identity set 51/51, held 107/107 and seeded T0.7 chat40 40/40
identical to D10e's texts; every knob variant identical to its reference; 16 in flight (16 x 32K): held 107/107 identical to D10a's
16-user reference, 198.8 tok/s (references 198.5-200.1), distinct-16 274.9 (cold); 0 crash lines in every engine. The simulation and the small fit set both over-read. Why it nets to zero: the stock lines were tuned on the raw probabilities
already; a monotone re-mapping of the same signal re-labels mostly the same decisions. Kept in the patch behind its flag for whoever
wants to re-fit it on other text; `gcal_r11.txt` is the measured table; `tools/gcalfit.py` fits and simulates.

## 4. Chat: where rounds end, and the sibling (tree) row costed with real row prices: not built

```
+------+--------------------------------------------------+---------------------------------------+
| cat  | first wrong draft at position 0 / 1 / 2 / 3-5 / 6+| true token = head's rank 2 / 3-8 / not in top 8 |
+------+--------------------------------------------------+---------------------------------------+
| chat | 0.28 / 0.24 / 0.16 / 0.22 / 0.10                 | 0.32 / 0.33 / 0.34                    |
| code | 0.14 / 0.11 / 0.10 / 0.24 / 0.42                 | 0.52 / 0.28 / 0.20                    |
| math | 0.10 / 0.08 / 0.10 / 0.30 / 0.43                 | 0.48 / 0.30 / 0.22                    |
+------+--------------------------------------------------+---------------------------------------+
```
Real costs, one user, today's stack (step log with times): verify pass = 31.9 + 3.11 ms x rows (rows 2..13 where the real policy goes),
chain = 1.32 ms + 1.17 a further head step (sampled 1.37 + 1.23). Unbiased depth sweep on chat text (gate off): a row is 3.3 ms for rows
2-5, 2.8 for 5-7, 2.1-2.5 for 7-13. So a draft row is ~3.1 ms, not the 2.4 ms the round-8/9 simulators used.
Sibling row (rank-2 token), SIMULATED on the greedy always-16 log with these costs: oracle (exactly at the miss) +3.3 % (chat +3.4);
priced by the head at any position (needs the in-kernel fork) +0.68 % (chat +0.73); first position only (the alias-lane shape, +0.4 ms
estimate for the repeated pending row) +0.53 % (chat +0.58). Below the 1 % bar, and this simulator over-read the gate by 1.5-2 points:
NOT BUILT. Chat-specific lines: the calibration is the same in every category and the calibrated gate measured +0.2..0.5 % on chat.

## Not done / for the integrator
- recipe.sh: add TF_FLASHNEXT_Q38_NGRAM_SPLIT=256 (and round 10's CERT_GATHER=1, HALF=0, HEAD2_SAMPLED=1 if not merged yet).
- Edit / agent / needle / image / qeval suites were not run on D11c.
- The remaining 3.9 % of a no-draft stream's first pass (start the reads from the lane thread before the gather thread is spawned: est. 1-1.5 points).
