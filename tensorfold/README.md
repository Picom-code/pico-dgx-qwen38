# Qwen3.8-Flash-Next on one DGX Spark: official NVIDIA checkpoint, lossless, measured

This is the lab notebook behind the top-level [README](../README.md): every round's tables, corrections and dead ends, newest first. The short version and the current numbers are there; the state when work stopped is in [notes/wrapup-2026-11-14/WRAPUP.md](notes/wrapup-2026-11-14/WRAPUP.md). The recipe: NVIDIA's own quantised
checkpoint (`nvidia/Qwen3.8-Flash-Next-FP4-Local`) served by TensorFold with MiaAI-Lab's patch set, plus what we measured
and built on top. Rule for everything here: **lossless**. A speed change counts only if greedy output is token-identical
to the same engine and checkpoint with the change off (51-prompt identity set and the held prompts, texts compared).

**Current numbers:** the top-level [README](../README.md) and [notes/final-2026-11-15/FINAL.md](notes/final-2026-11-15/FINAL.md): the final image
(patches 0101-0129), bf16 KV cache by default, levers off / recipe default / levers off in one window on spark-923b, the many-user table,
the identity counts and the check-mode counters, with the names of the raw files.

Everything below is the history, newest first. Read it with two things in mind: sections dated round 10 or earlier were measured with MiaAI-Lab's
**fp8 KV cache** (then the default; lossy against bf16, see "KV cache"), and their "recipe default" is that round's default, not today's.
Today `./recipe.sh start` is the exact profile: bf16 KV, 3 streams at 262,144 tokens; `KV=fp8` is an explicit, labelled, lossy opt-in.

## Credits first

| What | Who | Licence |
|---|---|---|
| Qwen3.8-Flash-Next | Alibaba Qwen team | model licence on its card |
| FP4-Local checkpoint (quantisation-aware distilled) | NVIDIA | on the model card |
| TensorFold engine (Zig + CUDA, every kernel that matters here, the 4-bit shortlisted draft head, the draft allocator) | ashhart/TensorFold contributors | Apache-2.0 |
| Patches 0001-0017, prepare.sh, start.sh (fp8 KV, prefill speed, draft gate, n-gram reader, vision) | MiaAI-Lab | MIT |
| custom engine R (the faster engine on a re-quantised checkpoint, used as the comparison) | custom engine R | MIT |
| a third engine | NVIDIA | Apache-2.0 |
| Ours | measurements, the kit, patches 0101-0123: a certified exact LM head, wider drafted verify passes, batched prompt passes, pool admission, adaptive copy windows, a fused hyper-connection read-out, lane batching for 9-16 streams (DeltaNet launches, best-draft read-back, attention launches), lazy DeltaNet end-state writes, first-token work (prompt-cache mark inside the prompt pass, reused stream buffers), ports of two upstream TensorFold changes (#1171 half tiles, #1925 MTP last row), profilers, probes and kernel microbenches | this repo |

Honest summary of our part: the engine and almost all of its speed are TensorFold's and MiaAI-Lab's work. Our changes that measure
positive and keep the output identical (tables below) add up to +13.7 % for one user on the 320 held prompts, +32 % on file rewrites,
+7 % on seeded sampled requests, a short prompt's first token 182 -> 116 ms and +28 % at 16 users, against MiaAI-Lab's configuration on
the same image in the same window (round 9, 2026-11-14). Round 9 added a certified indexer select and an elastic prompt cache (builder C),
the certified head for sampled passes on a 7-bit coarse copy (builder D), deep chains, an early stop and a two-stage draft head (builder B;
SVD-softmax, Shim et al. 2017, is the idea behind the preview). Everything else
we built or tried measured zero or negative and is listed as such. It still trails custom engine R's engine on a re-quantised checkpoint for one
user on chat (50.7 against 54.6); code and the held prompts are within 1 % of it.
The certified head is the coarse-read / rigorous-bound / exact-recompute scheme we first built for DeepSeek on four
Sparks; bounding a dot product by a quantisation residual (Cauchy-Schwarz) and re-scoring the survivors exactly is a
long-known idea in maximum-inner-product search. No "first" claims.

## Changelog (corrections first)

- **2026-11-14, patch 0120: bug fix in round 9's patch 0116 (certified indexer select). Found and fixed by builder C.**
  The helper that reads the seven indexer knobs (`q38EnvInt` in `cuda_knobs.zig`) kept ONE cached value for all of them, so each knob
  returned whatever the first one read. With the recipe's `QSA_CERT=1` the engine therefore ran the select's float slack at 1 millionth
  (designed: 100), `QSA_CERT_MIN` at 1 (designed: 64, so decode rows took the certified select too) and `QSA_STATS` on.
  What this means for the round-9 numbers below: every run with the round-9 levers on (tables "recipe default", 16 users, the check-mode
  run) was measured with those unintended values. No wrong output was seen: the device check mode compared 3.56 M row selections on
  spark-923b and 4.5 M on spark-4e5c with 0 differing lists (at slack 2 millionths), and all text identity checks passed. But the
  designed safety margin of the bound was not in force, so "certified" rested on measurement for that lever in round 9.
  After the fix (same box, one window, binary before / after / before): held prompts 64.32 / 64.06 / 64.00 tok/s, prefill 116K
  2450 / 2430 / 2435 tok/s, texts identical: the fix is speed-neutral within noise. Audit: this helper is the only generic knob reader
  in our patches (26 other cached knobs are one per function); no other knob was affected.

## KV cache: bf16 by default since round 11 (this section was written in round 10, when fp8 was the default), and other places where the arithmetic is not a plain bf16 reference

Every number above and below was measured with MiaAI-Lab's default **fp8 KV cache**. Their README says it plainly: fp8 KV is lossy against
a bf16 KV cache (about 98.8 % top-1 agreement, about 1.8x the cache capacity). Our "lossless" claim is narrower than it may sound:
**each of OUR levers returns the same text as the same engine without it, with the KV format held fixed.** It does not say the fp8 KV
cache equals bf16 KV; it does not.
`KV_DTYPE=bf16 PARALLEL=3 ./recipe.sh start` serves with the exact KV cache (3 streams at 262K instead of 5). Round 10 measured it once
(same window as the tables, see "bf16 KV check" below); making bf16 the default, with its own full tables, 16-user pool sizes and check
modes, is the open item at the top of "What is left".

Known by-design differences from a plain bf16 computation, and whether each can change the output:

| where | what | can it change the reply? |
|---|---|---|
| the checkpoint | NVIDIA's FP4-Local: NVFP4 experts, MXFP8 projections, BF16 head and hyper-connections | it IS the official checkpoint; another engine serving it (NVIDIA's card: vLLM + b12x with an MXFP8 LM head and A16/A4 activations) need not produce the same tokens |
| KV cache fp8 (MiaAI default) | attention keys / values stored as fp8 | YES (lossy against bf16 KV, ~98.8 % top-1). `KV_DTYPE=bf16` is exact |
| MX128 prompt tile, `PROMPT_GEMM2` (MiaAI 0015) | two pipeline stages for the prompt tile | no: MiaAI documents "same bits" (not re-verified by us) |
| 4-bit draft head, two-stage draft head, copy drafts, prediction field, deep chains | lower-precision or guessed DRAFTS | no: a draft is kept only where it equals the model's own token |
| certified LM head (ours) | 7-bit coarse copy + rigorous bound + exact BF16 recompute of every candidate | no: device check mode, 0 mismatches (greedy best + value; sampled top-20 values and columns) |
| certified indexer select (ours) | coarse scores as halves + bound + exact sums in between | no: device check mode, 6.08 M row lists, 0 differ |
| sampled requests | keyed Gumbel-max draw over the row's top_k | same token per seed as without our levers; not comparable token-for-token with another engine's sampler |
| many users | which rows share a pass | TensorFold's contract: each stream equals its solo run; we check texts with 16 in flight against one-user texts |
| anything else in TensorFold / MiaAI patches 0001-0017 | not audited line by line by us | unknown beyond their own statements |

## Numbers, current recipe (spark-923b, 2026-11-14, patches 0101-0123, round 10)

```
Round 10 final. spark-923b, DOCUMENTED PATH: recipe.sh prepare (patches 0101-0123 + the two-stage head's file) then recipe.sh start, vision on.
ONE shared window (S141010, 2026-11-14 14:10-18:50 UTC), one image. Order: levers off (14:14-15:10), custom engine R (15:13-15:46), levers off again (15:46-16:36),
ours with 16-user profile (16:37-16:51), ours recipe default (16:52-17:43). The default run has two levers-off runs BEFORE it and none after (the planned third
was given up for a bf16-KV check); the two levers-off runs, 1.5 h apart, agree to 0.4 % on the 320 held prompts and 0.1 % on seeded sampled.
KV cache format of every number in this file: fp8 (MiaAI-Lab's default; their note: lossy against a bf16 KV cache, ~98.8 % top-1). See the bf16 lines at the end.
All MEASURED. "ours vs mean off" = recipe default against the mean of the two levers-off runs.
custom engine R column: custom engine R v0.3.13 on turboderp's EXL3 4.05 bpw checkpoint = RE-QUANTISED weights, NOT the official checkpoint: not a lossless comparison.
Ours = NVIDIA's official FP4-Local checkpoint; every text of the default run is identical to the levers-off run's (below).

+-------------------------------------------------------+--------------------+-------------------+-----------------------+----------------------------------+------------------+
| bench                                                 | levers off (first) | levers off (last) | recipe default (ours) | custom engine R v0.3.13, EXL3 re-quantised | ours vs mean off |
+-------------------------------------------------------+--------------------+-------------------+-----------------------+----------------------------------+------------------+
| first token, short prompt (ms)                        | 208                | 208               | 119                   | 119                              | -42.8 %          |
| 3-prompt chat, 9 runs (median)                        | 47.0               | 46.7              | 51.5                  | 55.3                             | +9.9 %           |
| 3-prompt code, 9 runs (median)                        | 76.8               | 77.9              | 86.9                  | 90.4                             | +12.3 %          |
| sparkDash-method prose T0                             | 56.8               | 56.5              | 61.8                  | 66.8                             | +9.1 %           |
| sparkDash-method code (task 0)                        | 92.0               | 91.5              | 122.6                 | 110.2                            | +33.6 %          |
| sparkDash-method code, 8 tasks                        | 87.7               | 87.1              | 106.1                 | 108.4                            | +21.4 %          |
| sparkDash-method structured                           | 105.7              | 105.0             | 169.2                 | 153.3                            | +60.6 %          |
| sparkDash 1.8.8 itself, prose (median)                | 56.3               | 56.2              | 61.0                  | 66.8                             | +8.4 %           |
| sparkDash 1.8.8 itself, code (median)                 | 91.6               | 91.6              | 121.4                 | 110.7                            | +32.5 %          |
| sparkDash 1.8.8 itself, structured (median)           | 105.0              | 104.5             | 167.5                 | 154.1                            | +59.9 %          |
| ALL 320 held prompts, all                             | 57.34              | 57.12             | 65.51                 | 64.36                            | +14.5 %          |
|   held 320: chat                                      | 49.97              | 49.84             | 56.20                 | 56.68                            | +12.6 %          |
|   held 320: code                                      | 74.51              | 74.26             | 87.66                 | 82.01                            | +17.8 %          |
|   held 320: math                                      | 72.68              | 71.86             | 86.98                 | 81.68                            | +20.4 %          |
| edit set (24 file rewrites), wall tok/s               | 136.5              | 136.6             | 219.0                 | -                                | +60.4 %          |
| agent set (16 turns), wall tok/s                      | 53.7               | 53.6              | 62.9                  | -                                | +17.2 %          |
| seeded sampled T0.7, 24 held chat prompts             | 46.33              | 46.29             | 51.72                 | -                                | +11.7 %          |
| seeded sampled T0.7 with thinking, 24 prompts         | 47.51              | -                 | 52.58                 | -                                | +10.7 %          |
| seeded sparkDash-method prose T0.7 thinking, 6 rounds | 51.54              | -                 | 57.12                 | -                                | +10.8 %          |
| prefill 7.4K / 29K / 117K prompt (tok/s)              | 2384 / 2437 / 2292 | -                 | 2508 / 2544 / 2466    | -                                |                  |
| needle 182K-token prompt: first token s               | 82.7               | -                 | 75.4                  | -                                | -8.8 %           |
|   recall                                              | 3/3                | -                 | 3/3                   | -                                |                  |
| 227,621-token prompt: first token s                   | 107.7              | -                 | 94.3                  | -                                | -12.4 %          |
|   recall                                              | 3/3                | -                 | 3/3                   | -                                |                  |
| image                                                 | PASS ttft 0.47 s   | -                 | PASS ttft 0.52 s      | -                                |                  |
| qeval (4 in flight)                                   | 33/35              | -                 | 33/35                 | -                                |                  |
| 4 users distinct, default profile (tok/s)             | 151.9              | -                 | 162.0                 | 137.3 (its 8-slot default)       | +6.6 %           |
| min MemAvailable MiB during the suite                 | 30096              | 34556             | 8394                  | 27786                            |                  |
| max GPU MiB during the suite                          | 85437              | 81572             | 107914                | 86344                            |                  |
+-------------------------------------------------------+--------------------+-------------------+-----------------------+----------------------------------+------------------+
sparkDash 1.8.8 catalogs prose / structural / code: levers off 43.6 / 66.1 / 81.0; ours 48.1 / 76.8 / 97.6; custom engine R 49.4 / 74.8 / 95.1.
IDENTITY, recipe default against levers off (same engine, same checkpoint, same window): identity set 51/51, ALL 320 held texts 320/320, edit set 24/24,
agent set 16/16; seeded sampled (T 0.7) 24/24, with thinking 24/24, sparkDash-method 6/6 identical per seed. 0 crash lines.
MEMORY of the default run: lowest MemAvailable 8.4 GiB, GPU max 107.9 GiB (levers off 30.1 GiB / 85.4 GiB): the elastic prompt cache + 128-row window buffers.
That is BELOW the engine's 9 GiB reserve figure (round 9: 10.5 GiB): a dedicated box only; ELASTIC=0 otherwise.

SAME-WINDOW COMPARISON with custom engine R (re-quantised checkpoint; ours is output-identical to the official checkpoint served without the levers):
+--------------------------------------------+------------------------+---------------------------+
| bench                                      | ours, recipe default   | custom engine R v0.3.13 (EXL3 4.05) |
+--------------------------------------------+------------------------+---------------------------+
| 3-prompt chat / code, 9 runs               | 51.5 / 86.9            | 55.3 / 90.4               |
| ALL 320 held prompts (chat / code / math)  | 65.51 (56.20 / 87.66 / 86.98) | 64.36 (56.68 / 82.01 / 81.68) |
| sparkDash 1.8.8 prose / code / structured  | 61.0 / 121.4 / 167.5   | 66.8 / 110.7 / 154.1      |
| sparkDash catalogs prose / struct / code   | 48.1 / 76.8 / 97.6     | 49.4 / 74.8 / 95.1        |
| first token, short prompt                  | 119 ms                 | 119 ms                    |
| 1 / 4 / 8 users, distinct prompts          | 90.1 / 153.3 / 213.2 (USERS16 profile; 4 users on the default profile 162.0) | 87.2 / 137.3 / 166.6 |
|   last first token at 1 / 4 / 8 users      | 0.17 / 0.63 / 0.80 s   | 0.16 / 0.85 / 2.17 s      |
+--------------------------------------------+------------------------+---------------------------+
Plain reading: custom engine R is still faster on prose-like single-user text (3-prompt chat +7 %, code +4 %, sparkDash prose +10 %); ours is faster on the 320 held
prompts as a whole (+1.8 %; code +6.9 %, math +6.5 %, chat -0.8 %), on sparkDash code and structured (+10 %, +9 %), and with 4-8 users (+12..+28 %).
Not ahead everywhere; and the two columns do not serve the same weights.

MANY USERS, final image, USERS16=1 (pool of 16 x 48K tokens, see README: was 16 x 64K before patch 0122), same window, MEASURED:
 1 / 4 / 8 / 12 users 90.1 / 153.3 / 213.2 / 257.1 (last first token 0.17 / 0.63 / 0.80 / 1.07 s); 16 users 274.4 / 288.7 / 284.6 (1.16 / 0.81 / 0.81 s);
 16 copies 353.2; waves of 16 162.3 (last first token 1.24 s), waves of 4 102.2 (0.57 s); 136,607-token prompt + 15 short: recall 3/3, shorts' last first
 token 1.11 s; held 107 with 16 in flight 197.2 greedy / 187.4 seeded sampled. MiaAI's 16 x 64K configuration was NOT re-run in this window: round 9's same-window
 figure was 215-224 (last first token 6.8 s).
```

### bf16 KV check (round 10, same window)

Measured on spark-923b in the same window as the tables above (image 0101-0123; `KV_DTYPE=bf16 PARALLEL=3`; fp8 column = the default run 30 min earlier):

| | fp8 KV, levers on | bf16 KV, levers on | bf16 KV, levers off |
|---|---|---|---|
| streams at a 262K window | 5 (5.21 GiB each) | 3 (8.36 GiB each) | 3 |
| first token, short prompt | 119 ms | 119 ms | - |
| 3-prompt chat / code | 51.5 / 86.9 (9 runs) | 52.3 / 88.7 (5 runs) | - |
| 104 held prompts (chat / code / math) | 66.3-67.1 in other windows today | 67.11 (56.96 / 94.07 / 87.95) | 58.02 |
| seeded sampled chat | 51.72 | 50.24 | 44.66 |
| edit / agent set | 219.0 / 62.9 | 218.8 / 63.4 | 135.3 / 54.3 |
| prefill 7.3K / 29K / 116K tok/s | 2508 / 2544 / 2466 | 2334 / 2410 / 2345 | - |

- Decode speed is the same with bf16 KV; prefill is 5-7 % slower, sampled -3 % (inside 0.7 % start noise only partly); 3 streams instead of 5.
- Our levers under bf16 KV, on against off: identity set 51/51, held 107/107, edit 24/24, agent 16/16 texts identical (seeded sampled: not compared, a script error).
- fp8 KV against bf16 KV, both with the levers on: only 14 of the 51 identity-set replies are identical (mean shared prefix 0.51). fp8 KV does change replies.
- DECISION: `recipe.sh start` now defaults to **bf16 KV, 3 streams** for the default profile (`KV=fp8` gives the round-10 tables' 5-stream configuration: lossy,
  labelled). `USERS16=1` and `AGENTS=1` still use fp8 KV: their pools have not been sized or measured with bf16. NOT yet done with bf16: the full tables
  (320 held, sparkDash), 16 users, texts with 16 in flight, the device check modes, sampled identity.
- Device check modes on the final image (fp8 KV): certified indexer select 6,077,176 row selections against the stock kernels, 0 lists differ, 0 exact-sum bit
  differences, 0 bound violations (largest deviation 0.64 of the bound at the designed slack 100); identity set 51/51.

## Numbers, round 9 (spark-923b, 2026-11-14 early, patches 0101-0118; superseded; measured with the knob-cache bug of the Changelog)

```
Round 9 final, one user. spark-923b, DOCUMENTED PATH: recipe.sh prepare (patches 0101-0118 + the two-stage head's file) then recipe.sh start, vision on.
ONE window (2026-11-14 04:02-06:50 UTC), both columns the same image. MEASURED; every number is read from the logs by harness/round9/table9.py.
"levers off" = CERT=0 MANY=0 COPY_ROWS=16 FUSED=0 HALF=0 MTP_LAST=0 GDN_MANY=0 BEST_LANES=0 ATTN_LANES=0 LATE_SKIP=0 KEEP_INPASS=0 LANE_REUSE=0 HEAD2=0 R9=0
(= MiaAI-Lab's configuration on our build), run 04:13-05:11. "recipe default" = `./recipe.sh start` with nothing set, run 06:01-06:49 (the many-user runs sat
between them: the two columns are 1.8 h apart in one window, not reference-first-and-last; the same-hour pairs for each lever are in final_r9_levers.txt).
+-------------------------------------------------------+--------------------+-----------------------+---------+
| bench                                                 | levers off         | recipe default (ours) | change  |
+-------------------------------------------------------+--------------------+-----------------------+---------+
| first token, short prompt (ms)                        | 182                | 116                   | -36.3 % |
| 3-prompt chat, 9 runs (median)                        | 46.6               | 50.7                  | +8.8 %  |
| 3-prompt code, 9 runs (median)                        | 76.3               | 88.2                  | +15.6 % |
| sparkDash-method prose T0                             | 56.9               | 61.5                  | +8.1 %  |
| sparkDash-method prose T0.7 thinking (unseeded)       | 52.2               | 51.1                  | -2.1 %  |
| sparkDash-method code (task 0)                        | 92.3               | 121.7                 | +31.9 % |
| sparkDash-method code, 8 tasks                        | 87.8               | 105.4                 | +20.0 % |
| sparkDash-method structured                           | 106.2              | 168.9                 | +59.0 % |
| sparkDash 1.8.8 itself, prose (median)                | 56.5               | 61.3                  | +8.5 %  |
| sparkDash 1.8.8 itself, code (median)                 | 91.9               | 121.3                 | +32.0 % |
| sparkDash 1.8.8 itself, structured (median)           | 105.2              | 166.3                 | +58.1 % |
| sparkDash 1.8.8 catalogs prose / structural / code    | 43.6 / 66.1 / 81.2 | 48.0 / 76.6 / 97.7    |         |
| ALL 320 held prompts, all                             | 57.34              | 65.22                 | +13.7 % |
|   held 320: chat                                      | 49.96              | 55.83                 | +11.7 % |
|   held 320: code                                      | 74.58              | 87.83                 | +17.8 % |
|   held 320: math                                      | 72.70              | 86.90                 | +19.5 % |
| edit set (24 file rewrites), wall tok/s               | 137.3              | 180.8                 | +31.7 % |
| agent set (16 turns), wall tok/s                      | 54.5               | 62.9                  | +15.4 % |
| seeded sampled T0.7, 24 held chat prompts             | 46.52              | 49.57                 | +6.6 %  |
| seeded sampled T0.7 with thinking, 24 prompts         | 47.65              | 50.87                 | +6.8 %  |
| seeded sparkDash-method prose T0.7 thinking, 6 rounds | 51.59              | 55.27                 | +7.1 %  |
| prefill 7.4K / 29K / 117K prompt (tok/s)              | 2382 / 2449 / 2288 | 2502 / 2517 / 2403    |         |
| needle 182K-token prompt: recall, first token s       | 3/3, 83.3          | 3/3, 77.2             | -7.3 %  |
| 227,621-token prompt: recall, first token s           | 3/3, 108.2         | 3/3, 98.1             | -9.3 %  |
| image                                                 | PASS ttft 0.51 s   | PASS ttft 0.42 s      |         |
| qeval (4 in flight)                                   | 33/35              | 33/35                 |         |
| idle MemAvailable GiB / GPU MiB                       | 37 / 78438         | 35 / 79675            |         |
+-------------------------------------------------------+--------------------+-----------------------+---------+
IDENTITY default vs off (texts): identity set 51/51, ALL 320 held prompts 320/320, edit set 24/24, agent set 16/16 identical.
SAMPLED (temperature 0.7, same seeds, server top_k 20 / top_p 0.95): 24/24, 24/24 (thinking on: reasoning + answer) and 6/6 texts identical between the columns;
  speed +6.6 %, +6.8 %, +7.1 % (per-prompt +6.4 +- 0.9, +6.4 +- 1.2, +7.2 +- 1.0 %). The unseeded sparkDash-method T0.7 row draws different tokens in each column,
  so it is not a pair (round 8 read 55.1 against 49.5 the same way: noise of the draw, not a slowdown).
DEVICE CHECK MODES on this image before the two columns (CERT=2 QSA_CERT=2): indexer select 3,559,920 row lists against the stock kernels: 0 differ, 0 bit
  differences in exact sums, 0 bound violations, 0 fallbacks; certified head 20,135 greedy rows 0 mismatches; sampled top-k 3,510 rows 0 mismatches, 0 bound
  violations of 4.5 billion entries. 0 panic / segfault lines in every engine log of the window.
MEMORY: with the elastic prompt cache (ELASTIC=1, default) the engine's GPU use grew from 79.7 to 108.1 GiB during the suite and MemAvailable fell to 10.5 GiB
  (levers off: 86.6 GiB, 31.9 GiB). That is the engine's own 9 GiB reserve, the level five full 262K streams would reach anyway; the cache gives the room back when
  a stream opens. On a box that runs other things beside the engine set ELASTIC=0 or raise TENSORFOLD_MEMORY_RESERVE_GIB.
sparkDash 1.8.8 itself, earlier rounds on this path: round 6 (patches 0101-0107) 59.8 / 97.2 / 110.4 prose / code / structured; round 4: 58.8 / 95.6.
NOT RUN in the one-user columns this round (window time): 4 and 5 users on the default profile, sparkDash-method waves (round 8: 160.4 / 175.0; waves 135.2 / 225.6).
Not lossless-comparable, for honesty: custom engine R v0.3.13 on an EXL3 re-quantised checkpoint measured 54.6 / 88.6 chat / code and 65.66 on the 320 held prompts on this
box (first bake-off, 2026-11-12, another day). Ours on the official checkpoint: 50.7 / 88.2 (3-prompt medians; single runs spread +-4) and 65.22. For one user we
still trail it on chat (-7 %); code and the held prompts are within 1 % of it, not ahead.
```

### Many users, round 9 (patches 0101-0118)

```
Round 9 final, many users. spark-923b, DOCUMENTED PATH (recipe.sh prepare with patches 0101-0118, recipe.sh start, vision on), ONE window
(2026-11-14 05:12-05:57 UTC), same image in every column. Distinct coding prompts, 512 tokens, greedy, total tok/s (last first token). MEASURED.
+--------------------+----------------------------------------+------------------------------------------+------------------------------------------+
| users              | MiaAI config, 16 x 64K slots           | USERS16=1, recipe default                | USERS16=1 PREFIX_WAIT=0                  |
|                    | (every lever of ours off)              | (PREFIX_WAIT=60000)                      | (reference, run before / after)          |
+--------------------+----------------------------------------+------------------------------------------+------------------------------------------+
| 1                  | 75.8 (0.24 s)                          | 88.1 (0.18 s)                            | 87.6 / 86.8                              |
| 4                  | 142.7 (1.17 s)                         | 158.7 (0.52 s)                           | 157.3 / 159.1                            |
| 8                  | 199.3 (2.83 s)                         | 215.9 (0.68 s)                           | 212.7 / 219.9                            |
| 12                 | 223.4 (4.83 s)                         | 258.6 (0.73 s)                           | 264.3 / 259.3                            |
| 16                 | 215.6 / 224.3 / 214.7 (6.81/6.74/6.78) | 280.7 / 287.6 / 272.8 (0.88/0.67/0.69)   | 272.0 282.0 275.7 / 286.1 286.7 281.0    |
| 16, mean of three  | 218.2                                  | 280.4                                    | 276.6 / 284.6                            |
| 16 copies          | 244.1 (6.55 s)                         | 369.4 (0.63 s)                           | 364.5 / 364.6                            |
| waves of 16 held   | 127.6, last first token 7.22 s         | 163.6, 1.10 s                            | 163.7, 1.09 s                            |
| waves of 4 held    | 87.4, 1.35 s                           | 102.7, 0.54 s                            | 102.1, 0.55 s                            |
| held 107, 16 in flight |                                    | 196.9                                    | 196.9                                    |
+--------------------+----------------------------------------+------------------------------------------+------------------------------------------+
16 users: 280.4 against 218.2 = +28.5 %; 12 users +16 %; 8 users +8 %; 4 users +11 %; 1 user +16 %; last first token at 16 users 6.8 s -> 0.7-0.9 s.
PREFIX_WAIT at 16 distinct users: 280.4 with it against 276.6 before and 284.6 after without it (mean 280.6): no cost; first-token times equal.
  It is a recipe default from this round (measured gain, builder C, round 8: 8 requests on a NEW 25.5K-token system prompt, last first token 86.7 -> 12.9 s).
Identity (texts): waves of 16 128/128 and waves of 4 64/64 identical between the MiaAI column and USERS16; waves of 16 with PREFIX_WAIT 128/128 identical
to without; held 107 prompts with 16 in flight 107/107 identical to the round-7 build's texts, with and without PREFIX_WAIT.
USERS16 keeps the 262,144-token window: 136,607-token prompt + 15 short: recall 3/3, shorts' last first token 0.97 s; 227,621-token prompt 3/3,
first token 98.1 s (2320 tok/s; round 8: 2141).
Against round 8 on this path (patches 0101-0113: 81.6 / 153.7 / 219.9 / 260.0 / 276.9 283.0 278.6, copies 347.9, waves 160.1): 16 users unchanged
(this round's levers act on lone chains and long prompts), 1 user 81.6 -> 88, copies 348 -> 365-369.
NOT MEASURED HERE: sampled requests at 16 users. Builder D measured on the head that CERT_TOPK=20 costs 16 SAMPLED users 3-4 % without a row limit that
is not in patches 0101-0118, so USERS16 sets CERT_TOPK=0 (recipe.sh); the greedy runs above had CERT_TOPK=20 set, which greedy passes do not read.
```

### Round 10 levers (four builders in parallel, one Spark each; patches 0119-0123)

Paired on spark-923b on the documented path unless a line names another box. Two starts of the same configuration differ by up to 1.2 %
on the 104 held prompts and 0.7 % on the seeded sampled ruler in one window (measured: 66.27 / 67.09 and 50.14 / 50.48), so levers under
~1 % are judged by the builders' in-engine pairs (knob file, one running engine), not by two engine starts.

| lever (recipe setting) | from | measured | status |
|---|---|---|---|
| 128-row verify windows for copied text (`COPY_ROWS=128`, patch 0122) | builder B | edit set (24 file rewrites) 177.6 / 181.5 -> 215.2 / 218.3 / 219.1 = +21 %; held, sampled, agent set unchanged; B on spark-3d24: rewrites +20.5 %, write-file tool calls +21 % | default (default profile) |
| `prediction` request field (OpenAI's predicted-outputs shape, patch 0122) | builder B | B, spark-3d24: numbered-file rewrite +149 %, real agent loop +18 %, follow-up sessions +12 %; a wrong prediction costs -0.2 % and cannot change the reply | on when a request sends it |
| two-stage draft head for sampled chains (`HEAD2_SAMPLED=1`, in 0118) | builder B | seeded sampled chat +2.3 %, with thinking +1.8 %, sampled code +3.4 % (923b, with the long-prompt set on); B +1.8 %, D +1.5 / +0.9 % in-engine; texts identical per seed | default |
| certified top-k row limit (`CERT_TOPK_ROWS=8`, patch 0119) | builder D | 16 sampled users 188.0 / 189.6 with `CERT_TOPK=20` against 187.9 / 189.6 without (round 9 without the limit: -1.6..-4 %) | default; USERS16 now keeps `CERT_TOPK=20` |
| certified head's exact tiles in one launch (`CERT_GATHER=1`, patch 0123) | builder D | D, head box, inside one engine: sampled +1.6 % (+-0.2), greedy +0.2 % (zero) | default |
| long prompts: list select, row-cut scores, draft-head prompt rows stop after cache writes, warm-up pass (`QSA_LIST`, `QSA_ROWS`, `HEAD_KV=512`, `WARM=3072`, patch 0121) | builder C | 923b: prefill 29K +1.2 %, 116K +2.8 %, 7.3K +5..9 %, short-prompt first token 115-118 -> 110 ms; C on spark-4e5c: 27K +2.6 %, 109K +3.6 %, 214K +5.4 %, 7K +8 % | default |
| knob-cache bug fix (patch 0120) | builder C | see Changelog; speed-neutral | fix |
| half-unit decode experts (`TF_FLASHNEXT_Q38_EXP_HALF=1`, port of upstream TensorFold #1351, patch 0123) | builder D | +1.0 % for no-draft streams only; zero with drafts | off (set it for `--no-drafts` deployments) |
| start-up arena (`ARENA=1`, patch 0121) | builder C | faster lane opens and kept-state writes on long prompts; idle engine leaves 7-8 GiB MemAvailable | option, off |
| `DEEP_MAX=15` | builder B | 66.91 against 66.89 at 12 on the held prompts: no gain here | stays 12 |
| `QSA_SLACK=32` | builder C | 116K prefill 2534 against 2533 at the designed 100: no gain here | stays 100 |
| `HALF=0` (builder D's audit: -0.21 +- 0.07 % greedy, -0.23 % sampled inside one engine) | builder D | my two-start pair: sampled +0.2 %, greedy -0.2 %: inside start noise, not confirmed | `HALF=96` kept; worth nothing at one user |

Measured dead in round 10 (not shipped):
- **Cascade for the certified head** (a cheaper first stage before the 8-bit stage) and the **exact row-mean shift**. Offline study on
  260,000 dumped head-input rows (`notes/round10/casc10_raw.txt`, tool `harness/round10/casc.py`): a low-rank first stage (rotation fitted
  on other text) bounds to 30-50 logits and removes nothing (a head input keeps 49 % of its norm outside the first 512 fitted
  directions); a 4-bit first stage keeps 65-95 % of the tiles; the row-mean shift lowers the bound from 0.99 to 0.88 logits and the
  exact tiles from 9 to 9. What is left is fewer bits in the one coarse copy: a 6-bit copy reads 457 MiB instead of 606 with 18-33
  surviving tiles, a 6/5-bit copy on reordered inputs 412 MiB. That is at most +1.0 to +1.4 % (arithmetic, if a 5/6-bit unpack kernel
  kept the 8-bit kernel's read rate); not built.
- Builder D: certified coarse router (bounded under 1 %); builder C: sampled cut for the select, 13 prompt expert tile shapes;
  builder B: other sessions' replies as a draft source (the engine's `--recall` already does it; opt-in because it shares text between
  users), 256/512-row windows, per-round copy-or-chain choice.

Regression to know about: patch 0122 sizes every stream's window buffers for 128 rows at compile time (+0.45 GiB a lane), so the
`USERS16` profile no longer fits a pool of 16 x 64K tokens; it now uses 16 x 48K (the largest single request is still 262,144 tokens).
Sizing those buffers at run time is the follow-up.

What is a port and what is ours (round 10): `EXP_HALF` is a port of upstream TensorFold #1351; the `prediction` field follows OpenAI's
predicted-outputs API and the copy drafting it feeds is TensorFold's own suffix-lookup proposer (ideas: SuffixDecoding, arXiv 2411.04975);
`HEAD_KV` applies upstream's "stop a prompt chunk's last layer at the keys and values" to the draft head; the certified head / select
family is the long-known bound-and-rescore idea from maximum-inner-product search. The list select, the gather launch, the row limit and
the 128-row copy windows are our builds on those. No "first" claims.

### Round 9 levers (four builders in parallel, one Spark each)

```
Round 9 levers. spark-923b, DOCUMENTED PATH (recipe.sh prepare, recipe.sh start, vision on), 2026-11-14 00:08-03:41 UTC, ONE shared window.
Every pair: same image, reference first and last, one user, greedy, 384 tokens a prompt. MEASURED. tok/s = tokens / decode time.
"held-104" = every third of the 320 held prompts. "fresh-123" = head2/corpus/fresh.jsonl: 123 prompts written for this round and never used to
fit or tune anything (40 chat, 26 code, 16 math, 9 structured, 8 agent / tool transcripts, 24 non-English in 21 languages).

A. Two-stage draft head (HEAD2) with the rotation RE-FITTED on a corpus that shares no prompt with any bench set (image 0101-0115)
+---------------------------------------------+-------------+---------------------------+----------------+----------+-----------+
| run                                         | first token | held-104 (chat/code/math) | fresh-123      | edit set | agent set |
+---------------------------------------------+-------------+---------------------------+----------------+----------+-----------+
| P_ref1  HEAD2 off                           | 117 ms      | 62.43 (53.49/85.72/79.33) | 54.24          | 176.1    | 58.9      |
| P_new   HEAD2 on, rotation r9 (new)         | 112 ms      | 63.75 (54.58/87.66/81.17) | 55.17          | 176.8    | 59.5      |
| P_ref2  HEAD2 off                           | 121 ms      | 62.15 (53.19/85.63/79.13) | 54.16          | 175.7    | -         |
| P_old   HEAD2 on, rotation r512 (round 8)   | -           | 63.75 (54.56/87.73/81.24) | 54.93          | 176.2    | 59.1      |
+---------------------------------------------+-------------+---------------------------+----------------+----------+-----------+
New rotation against the mean reference: held (now OUTSIDE the fit) +2.3 %, fresh +1.8 % (chat +2.3, code +2.4, math +2.6, structured +2.0,
agent +2.5, non-English +0.4), edit set +0.5 %, agent set +1.0 %. 3-prompt bench x9: 50.7 / 84.6 against 49.5 / 82.7 and 51.7 / 83.5 (inside its noise).
Texts against P_ref1: identity set 51/51, held 107/107, fresh 123/123, edit 24/24, agent 16/16 identical, for the new and the old rotation.
Offline replay on real draft-head inputs (pick of the two-stage head == pick of the full 4-bit draft head, block form):
+------------------------------------------+---------------------------+---------------------------+
| head inputs from                         | rotation r512 (round 8)   | rotation r9 (this round)  |
+------------------------------------------+---------------------------+---------------------------+
| fitting corpus, 260 prompts, 110,919 rows| 97.05 %                   | 98.43 % (in its fit)      |
| fresh-123, 59,290 rows                   | 96.33 %                   | 97.63 %                   |
|   of which non-English / agent           | 90.25 % / 93.60 %         | 95.12 % / 96.99 %         |
| held-104, 30,970 rows                    | 98.88 % (in its fit)      | 98.47 % (outside its fit) |
+------------------------------------------+---------------------------+---------------------------+
=> HEAD2 is a recipe default with rotation r9. The file is built by `recipe.sh prepare` from the checkpoint and the shipped 5.2 MB rotation
(sha256 of the file on a GB10 GPU: fce7c913060d45f2...; rebuilt from the rotation alone: byte-identical).

B. The round-9 levers of builders B, C and D together (image 0101-0118; R9=0 = round-8 defaults + HEAD2 r9, R9=1 = the new defaults)
+---------------------------------------------+---------------------------+-----------------------------------------------------------+
| run                                         | held-104 (chat/code/math) | fresh-123 all (chat/code/math/struct/agent/non-English)   |
+---------------------------------------------+---------------------------+-----------------------------------------------------------+
| Y_ref   R9=0                                | 63.44 (54.35/87.11/80.73) | 55.05 (47.84/75.73/81.86/80.93/82.78/37.54)               |
| Y_all   R9=1                                | 66.56 (56.14/94.24/88.64) | 57.31 (49.06/79.45/87.24/86.53/89.97/39.06)               |
| Y_ref2  R9=0                                | 63.66 (54.41/87.87/81.36) | 55.41 (48.16/76.24/82.40/81.56/83.23/37.76)               |
+---------------------------------------------+---------------------------+-----------------------------------------------------------+
R9=1 against the mean reference: held +4.7 % (chat +3.2, code +7.7, math +9.4); fresh +3.8 % (chat +2.2, code +4.6, math +6.2, structured +6.5,
agent +8.4, non-English +3.7). Texts against P_ref1 (HEAD2 off, image 0101-0115): held 107/107, fresh 123/123 identical in all three runs.
What R9=1 turns on, with each builder's own paired result on its box (see notes/round9/builder-*-NOTE.md):
+--------------------------------------+---------+--------------------------------------------------------------------------------------+
| setting                              | builder | builder's own pair (texts identical in every row)                                    |
+--------------------------------------+---------+--------------------------------------------------------------------------------------+
| DEEP=0.9 (chains up to 12 drafts)    | B       | held-104 +2.0 % (chat +0.3, code +5.6, math +6.9)                                    |
| STOP2=0.5 (skip a doomed draft step) | B       | +0.6 % without HEAD2, +0.9 % on top of HEAD2 + FAST                                  |
| HEAD2_FAST=3 (int8 preview, bf16 rot)| B       | +0.9 % over HEAD2                                                                    |
| CERT_CODE=sm8 + CERT_TOPK=20         | D       | sampled T0.7 +4.0..4.8 %, with thinking +2.9..3.9 %, greedy +1.7 % (seeded, 40/40)   |
| QSA_CERT=1 + QSA_HALF=1              | C       | cold prompt first token: 110K -4.0 %, 214K -8.7 %; nothing below ~30K                |
| ELASTIC=1 (no pool profiles)         | C       | 5 agents x 56K-token prompts: follow-up first token 64.8 -> 0.99 s (default profile) |
+--------------------------------------+---------+--------------------------------------------------------------------------------------+
The three stages on held-104 in this window: no HEAD2 62.3 -> HEAD2 r9 63.6..63.8 -> all round-9 levers 66.56.
```

What did not make it in round 9: builder D's row limit for the sampled certified head at 16 users (arrived after the final image was built; `USERS16` therefore keeps `CERT_TOPK=0`), the two-stage head for sampled chains (`TF_FLASHNEXT_Q38_HEAD2_SAMPLED=1`, in patch 0118, not measured, off), tree verify (builder B: +1.0 to +1.6 % simulated, not built: the verify window of this backend is prefix-shaped; design note in `notes/round9/`). Measured zero and left off inside the patches: `EXP_WAVES`, `RMS_TILES`, `SGATE` (builder D), `QSA_LOOP` (builder C).

## Numbers, round 8 (spark-923b, 2026-11-13 evening, patches 0101-0113)

```
Round 8 final, one user. spark-923b, DOCUMENTED PATH: recipe.sh prepare (patches 0101-0113) then recipe.sh start, vision on. ONE window (2026-11-13 20:34-21:27 UTC),
both columns the same image. MEASURED. "levers off" = CERT=0 MANY=0 COPY_ROWS=16 FUSED=0 HALF=0 MTP_LAST=0 GDN_MANY=0 BEST_LANES=0 ATTN_LANES=0 LATE_SKIP=0 KEEP_INPASS=0
LANE_REUSE=0 (= MiaAI-Lab's configuration on our build).
+-----------------------------------------------+------------------------+--------------------------+----------------------+
| bench                                         | levers off             | recipe default (ours)    | change               |
+-----------------------------------------------+------------------------+--------------------------+----------------------+
| first token, short prompt (ms)                | 184                    | 111                      | -40 %                |
| 3-prompt chat, 9 runs (median)                | 47.6                   | 50.1                     | +5.3 %               |
| 3-prompt code, 9 runs (median)                | 77.0                   | 84.7                     | +10.0 %              |
| sparkDash-method prose T0                     | 56.6                   | 60.9                     | +7.6 %               |
| sparkDash-method prose T0.7 thinking          | 55.1                   | 49.5                     | (sampled)            |
| sparkDash-method code (task 0)                | 91.8                   | 99.1                     | +8.0 %               |
| sparkDash-method code, 8 tasks                | 87.4                   | 94.0                     | +7.6 %               |
| sparkDash-method structured                   | 105.7                  | 113.6                    | +7.5 %               |
| held prompts, same 104, all                   | 58.01                  | 62.38                    | +7.5 %               |
| held same 104: chat / code / math             | 49.79 / 78.92 / 73.98  | 53.44 / 85.72 / 79.28    | +7.3 / +8.6 / +7.2 % |
| edit set (24 file rewrites), wall tok/s       | 137.2                  | 176.1                    | +28.4 %              |
| edit set, decode tok/s; tokens a round        | 152.4; 12.59           | 198.1; 18.38             | +30.0 %              |
| agent set (16 turns), wall; decode            | 54.2; 71.0             | 59.1; 76.0               | +9.0 %; +7.0 %       |
| 4 users distinct / copies                     | 146.9 / 150.0          | 160.4 / 158.1            | +9.2 / +5.4 %        |
| 4 users last first token (s)                  | 1.25                   | 0.29                     |                      |
| 5 users distinct / copies                     | 162.9 / 167.6          | 175.0 / 192.3            | +7.4 / +14.7 %       |
| 5 users last first token (s)                  | 1.66                   | 0.43                     |                      |
| 16 users queued on 5 slots                    | 153.0 (37.7 s)         | 171.4 (33.6 s)           | +12.0 %              |
| sparkDash-method wave 5 prose / code          | 132.7 / 214.0          | 135.2 / 225.6            |                      |
| prefill 7.4K / 29K / 117K prompt (tok/s)      |                        | 2503 / 2519 / 2340       |                      |
| needle 182K-token prompt                      |                        | 3/3 (2198 tok/s)         |                      |
| 227,621-token prompt                          |                        | 3/3 (2134 tok/s)         |                      |
| image                                         |                        | PASS (0.42 s)            |                      |
| qeval (4 in flight)                           |                        | 33/35 (math 12/14, prose 5/5, reason 16/16) |   |
| idle MemAvailable GiB / GPU MiB               | 38 / 78,438            | 37 / 79,527              |                      |
+-----------------------------------------------+------------------------+--------------------------+----------------------+
IDENTITY default vs off: ident 51/51, held 107/107, edit 24/24, agent 16/16 texts identical.
Not run this round (window time): the 320 held prompts in full and sparkDash 1.8.8 itself (round 6, patches 0101-0107: held-320 61.33; sparkDash 59.8 / 97.2 / 110.4).
Round 6 on this path (window Q, patches 0101-0107): first token 182 ms, chat 51.1 / code 82.7, held same 107 62.26, edit set 173.4, agent set 57.3.
Not lossless-comparable, for honesty: custom engine R v0.3.13 on an EXL3 re-quantised checkpoint measured 54.6 / 88.6 chat / code and held-320 65.66 on this box
(first bake-off, 2026-11-12). Ours on the official checkpoint: 50.1 / 84.7 (3-prompt medians; single runs spread +-4), held-104 62.38. We still trail it for one user.
```

### Many users, round 8 (patches 0101-0113)

```
Round 8 final, many users. spark-923b, DOCUMENTED PATH (recipe.sh prepare with patches 0101-0113, recipe.sh start, vision on), ONE window
(2026-11-13 20:04-20:31 UTC). Distinct coding prompts, 512 tokens, total tok/s (last first token). MEASURED.
+--------------------+----------------------------------------+------------------------------------------+
| users              | MiaAI config, 16 x 64K slots           | USERS16=1, recipe default                |
|                    | (every lever of ours off)              | (patches 0101-0113)                      |
+--------------------+----------------------------------------+------------------------------------------+
| 1                  | 75.4 (0.24 s)                          | 81.6 (0.19 s)                            |
| 4                  | 140.1 (1.22 s)                         | 153.7 (0.53 s)                           |
| 8                  | 189.9 (2.88 s)                         | 219.9 (0.70 s)                           |
| 12                 | 209.1 (4.91 s)                         | 260.0 (0.99 s)                           |
| 16                 | 218.4 / 219.2 / 217.3 (6.80/6.73/6.76) | 276.9 / 283.0 / 278.6 (0.95/0.70/0.71)   |
| 16 copies          | 245.4 (6.52 s)                         | 347.9 (0.65 s)                           |
| waves of 16 held   | 127.7, last first token 7.20 s         | 160.1, 1.12 s                            |
| waves of 4 held    | 87.2, 1.36 s                           | 101.3, 0.55 s                            |
+--------------------+----------------------------------------+------------------------------------------+
16 users: 279.5 (mean of three) against 218.3 = +28 %; 12 users +24 %; 8 users +16 %; last first token at 16 users 6.8 s -> 0.8 s.
Identity (texts): waves of 16 128/128 and waves of 4 64/64 identical between the two columns; held 107 prompts with 16 in flight 107/107
identical to the round-7 build's texts (194.1 tok/s total).
USERS16 keeps the 262,144-token window: 136,607-token prompt + 15 short: recall 3/3, shorts' last first token 1.01 s;
227,621-token prompt 3/3 (2141 tok/s).
Round 7 on the same path (patches 0101-0109, windows T / U): 81.2 / 151.5 / 212.3 / 247.9 / 269.4 + 266.1 + 272.7 (1.2 s), copies 336.8,
waves of 16 151.5 (1.63 s), waves of 4 97.2 (0.71 s).
```

### Round 8 levers (four builders in parallel, one Spark each)

```
Round 8 levers (four builders in parallel, one Spark each). MEASURED unless a line says otherwise; every pair in one window, same binary, flag on / off.
+-----------------------------------------------+---------------------------------------------------------------+----------------------------------------------+
| lever (flag, patch, builder)                  | measured                                                      | identity (texts)                             |
+-----------------------------------------------+---------------------------------------------------------------+----------------------------------------------+
| attention lanes in one launch                 | 923b, 16 users distinct x3: ref 270.2 / on 275.2 / ref 268.0  | held 107 with 16 in flight 107/107, waves of |
| ATTN_LANES=1 (TF_FLASHNEXT_Q38_ATTN_LANES),   | (+2.3 %); held 16 in flight 178.0 / 180.7 / 177.6 (+1.6 %);   | 16 96/96, one-user ident 51/51; flag off vs  |
| patch 0110, A. Idea: upstream TensorFold      | waves of 16 149.7 / 154.1 / 150.9; 9 users 214.1 -> 218.8;    | the round-7 build: waves 96/96               |
| #1739; own form (by-value lane table, fp8     | 12 users 250.4 -> 252.0. GPU time: QSA group 8.47 -> 5.68 ms  |                                              |
| and bf16 caches)                              | of a 197.5 ms pass (pass 197.5 -> 193.9)                      |                                              |
+-----------------------------------------------+---------------------------------------------------------------+----------------------------------------------+
| drafted windows skip the DeltaNet end-state   | 923b, 16 users (ATTN_LANES on in all): refs 273.9 / 268.6,    | held 16 in flight 107/107 at T = 8, 16, 32;  |
| write, LATE_SKIP=16                           | T=16 275.1, T=32 274.2, T=8 276.6 (+1.5 %); held 16 in flight | waves of 16 96/96; one user held 107/107,    |
| (TF_FLASHNEXT_Q38_LATE_SKIP), patch 0111, A   | 180.9 / 179.9 -> 182.8 / 182.6 / 183.0 (+1.3 %). GPU time:    | ident 51/51, edit set 4 in flight 24/24      |
|                                               | DeltaNet group 34.08 -> 31.21 ms a pass. One user: 107 held x |                                              |
|                                               | 256 tokens 346.2 / 345.4 s -> 344.8 s (+0.3 %)                |                                              |
+-----------------------------------------------+---------------------------------------------------------------+----------------------------------------------+
| prompt-cache mark kept inside the prompt pass | 4e5c (builder C), one user, one engine switched by knob file: | ident 51/51, needle 109K 3/3, short 24/24,   |
| KEEP_INPASS=4096 + a finished stream's fixed  | new short prompt first token 175.3 -> 112.6 ms (-36 %; KEEP   | agent 4/4, prefill 2/2, held 40/40; score    |
| buffers reused, LANE_REUSE=N, patch 0112, C.  | alone 133, REUSE alone 155); agent follow-up turn -13 %; cold | hashes 71/71; 16 in flight 107/107           |
| Ideas: upstream #1866 and #1763; own forms    | 2.7K prompt -8 %. 16 lanes: last first token 8 users 0.71 ->  |                                              |
|                                               | 0.49 s, 4 users 0.52 -> 0.38 s; 16 users +2.9 % throughput    |                                              |
+-----------------------------------------------+---------------------------------------------------------------+----------------------------------------------+
| prefix wait, PREFIX_WAIT=60000 (port of       | 4e5c (C), 16 lanes: 8 requests on a NEW 25.5K system prompt,  | texts 8/8, 16/16, 24/24                      |
| upstream #1321), patch 0112, OPTION (not a    | last first token 86.7 -> 12.9 s; 16 on a new 11K prompt 76.9  |                                              |
| default: final tables ran without it)         | -> 9.0 s; 8 agents x 3 turns wall 92.2 -> 35.9 s              |                                              |
+-----------------------------------------------+---------------------------------------------------------------+----------------------------------------------+
| agents profile AGENTS=1 (pool 5 x 131K), C    | 5 agents, 56K system prompts: follow-up first token 85.1 ->   | texts 15/15                                  |
| setting only, OPTION                          | 1.0 s (cache budget 4.1 -> 13.4 GiB), wall 365.6 -> 138.0 s   |                                              |
+-----------------------------------------------+---------------------------------------------------------------+----------------------------------------------+
| two-stage draft head, HEAD2=1 after            | 923b documented path, image 0101-0115, one window: held-104   | ident 51/51, held 107/107 (923b, documented  |
| `recipe.sh head2` (TF_FLASHNEXT_Q38_HEAD2),   | 62.46 / 62.42 -> 63.81 (+2.2 %; every category +2.1..2.4); | path); B: 51/51 + 107/107 in four tests;     |
| patch 0115, builder B, OPTION (the rotation   | 3-prompt chat 50.7 -> 52.1, code 85.2 -> 85.6. 3d24 (B),      | image PASS                                   |
| was fitted on the held prompts' head inputs). | paired: 61.85 / 62.15 -> 63.02 (+1.6 %); chain step 1.69 ->   |                                              |
| Idea: SVD-softmax (Shim et al. 2017)          | 1.37 ms. Lone greedy chains only; +151 MiB                    |                                              |
+-----------------------------------------------+---------------------------------------------------------------+----------------------------------------------+
Both of A's levers together, GPU time of a 16-user pass (GPROF, ~79 rows): 197.5 -> 191.5 ms (QSA 8.47 -> 5.65, DeltaNet 34.45 -> 31.21; MoE 130 unchanged).

DEAD / ZERO this round (do not retry):
- expert items of up to 64 pairs (D, TF_FLASHNEXT_Q38_EXP_PAIRS): 16 users -2.5 %, copies -6.7 %. Not integrated.
- row-trimmed 64-row lane tiles (D, TF_FLASHNEXT_Q38_ROW_TRIM): zero; every dense group unchanged to 0.05 ms. Not integrated.
- draft depth 8 / 10, gate 0.3 at 16 users (D): inside the reference spread or negative. MiaAI's 0.4 / 6 stays.
- expert storage order by co-activation, exact expert skips, a certified coarse router (D, measured statistics): nothing to gain
  (adjacent-id use 0.447 against 0.446 chance; zero routing weights 0 of 25.7 million; router 1.4 % of GPU time).
- per-lane PLE conv + window shift (A): 32 launches a pass, ~0.04 % (ESTIMATE from KPROF: 0.03 ms of 28.8). Not built.
- prompt cache sibling-first eviction (C, TF_FLASHNEXT_Q38_CACHE_SIBLINGS): zero. The flag is still in patch 0112 (off) because the measured
  image was built from that version of C's patch.
- exact prefill levers (C): none built; the one measured gap is the indexer's block scores on long prompts (design note only).
Measured facts worth keeping (D): at 16 users a pass reads 228 distinct experts a layer (44.6 % of 512), 30.1 GiB of experts a pass, at >= 254 GB/s
(93 % of the memory bus): the expert path is at the bus rate. (A): the DeltaNet chain kernel is the state read + write (2 x 3.1 MB a stream and layer)
at ~215 GB/s; a build running on the box (3 CPUs) costs the engine ~1.5 % at 16 users (shared memory bus): never measure beside a build.
The image `recipe.sh prepare` builds now holds patches 0101-0115 (0114 draft-study logging, 0115 two-stage head, both off by default). With HEAD2 off it gives
the texts of the 0101-0113 image the tables above were measured on (ident 51/51, held 107/107) at the same speed (held-104 62.46 against 62.38).
```

Round 8 in one paragraph: at 9-16 users the per-stream launch loops that were left (attention: five or six launches a stream a
layer) now run as one launch a kernel, and a drafted window no longer writes DeltaNet end states it may not need; a new prompt's
first token lost a 7-row second pass and 22 ms of buffer allocation. The expert path was measured at the memory bus (228 distinct
experts a layer at 16 users, >= 254 GB/s): nothing exact is left there. Options measured but not defaults: `PREFIX_WAIT=60000`
(requests that share a new long prefix wait for one prefill), `AGENTS=1` (pool of 5 x 131K windows so every agent's prompt
states stay cached) and `HEAD2=1` (builder B's two-stage draft head: one user +1.6 to +2.2 % with identical texts; an option
because its 5 MB rotation was fitted on the head inputs of our own held prompts, so the gain on other text is not proven).

## Numbers, round 6 one-user table (spark-923b, 2026-11-13 morning, patches 0101-0107)

```
Window Q (2026-11-13 10:11-12:12 UTC), spark-923b, DOCUMENTED PATH: recipe.sh prepare (patches 0101-0107 = build kp21) then recipe.sh start, vision on.
Both columns: same image, same window. MEASURED. "levers off" = CERT=0 MANY=0 COPY_ROWS=16 FUSED=0 HALF=0 MTP_LAST=0 (= MiaAI-Lab's configuration on our build).
+-----------------------------------------------+------------------------+--------------------------+----------+
| bench                                         | levers off             | recipe default (ours)    | change   |
+-----------------------------------------------+------------------------+--------------------------+----------+
| first token, short prompt (ms)                | 182                    | 182                      | 0        |
| 3-prompt chat, 9 runs (median)                | 47.1                   | 51.1                     | +8.5 %   |
| 3-prompt code, 9 runs (median)                | 78.4                   | 82.7                     | +5.5 %   |
| sparkDash-method prose T0                     | 56.2                   | 60.8                     | +8.2 %   |
| sparkDash-method prose T0.7 thinking          | 46.8                   | 52.9                     | (sampled)|
| sparkDash-method code (task 0)                | 90.8                   | 98.8                     | +8.8 %   |
| sparkDash-method code, 8 tasks                | 86.2                   | 93.6                     | +8.6 %   |
| sparkDash-method structured                   | 106.4                  | 113.3                    | +6.5 %   |
| held prompts, same 107, all                   | 58.40                  | 62.26                    | +6.6 %   |
| held same 107: chat / code / math             | 50.13 / 79.39 / 74.45  | 53.36 / 85.39 / 79.01    | +6.4 / +7.6 / +6.1 % |
| held, all 320 (315 answered): all             | (not run)              | 61.33                    |          |
| held 320: chat / code / math                  |                        | 53.45 / 79.91 / 77.40    |          |
| edit set (24 file rewrites), wall tok/s       | 137.9                  | 173.4                    | +25.7 %  |
| edit set, decode tok/s; tokens a round        | 153.1; 12.59           | 197.9; 18.38             | +29.3 %  |
| agent set (16 turns), wall; decode            | 54.5; 71.4             | 57.3; 75.8               | +5.1 %; +6.2 % |
| 4 users distinct / copies                     | 149.5 / 151.3          | 149.9 / 168.6            | 0 / +11 %|
| 4 users last first token (s)                  | 1.22                   | 0.58                     |          |
| 5 users distinct / copies                     | 167.1 / 166.9          | 171.0 / 181.4            | +2.3 / +8.7 % |
| 5 users last first token (s)                  | 1.61                   | 0.61                     |          |
| 16 users queued on 5 slots                    | 157.3 (35.6 s)         | 164.7 (32.8 s)           | +4.7 %   |
| sparkDash-method wave 5 prose / code          | 126.7 / 216.7          | 136.4 / 222.4            |          |
| sparkDash 1.8.8 real prose / code / structured|                        | 59.8 / 97.2 / 110.4 (5-run mean) |  |
| sparkDash 1.8.8 catalogs prose / struct / code|                        | 46.5 / 70.7 / 86.7       |          |
| prefill 7.3K / 29K / 117K prompt (tok/s)      |                        | 2450 / 2521 / 2354       |          |
| needle 182K-token prompt                      |                        | 3/3 (2253 tok/s)         |          |
| 227,621-token prompt                          |                        | 3/3 (2160 tok/s)         |          |
| image                                         |                        | PASS (0.53 s)            |          |
| qeval (4 in flight)                           |                        | 33/35 (math 12/14, prose 5/5, reason 16/16) | |
| idle MemAvailable GiB / GPU MiB               | 37 / 78,438            | 37 / 79,124              |          |
+-----------------------------------------------+------------------------+--------------------------+----------+
IDENTITY default vs off: ident 51/51, held 107/107, edit 24/24, agent 16/16 texts identical.
Certified head during the suite: 40,000 calls, 217,676 rows, 3 whole-head fallbacks.
Not lossless-comparable, for honesty: custom engine R v0.3.13 on an EXL3 re-quantised checkpoint measured 54.6 / 88.6 chat / code and held-320 65.66
on this box in the first bake-off (2026-11-12). Ours on the official checkpoint: 51.1 / 82.7, held-320 61.33. We still trail it.
```

### Many users, round 7 (patches 0101-0109, 2026-11-13 afternoon)

```
Round 7 final, spark-923b, DOCUMENTED PATH (recipe.sh prepare with patches 0101-0109, recipe.sh start, vision on). Distinct coding prompts, 512 tokens, total tok/s (last first token). MEASURED.
+--------------------+--------------------------------+----------------------------------+--------------------------------------------+
| users              | MiaAI config, 16 x 64K slots   | USERS16 as of round 6            | USERS16 now (recipe default: + GDN_MANY=2, |
|                    | (window T)                     | (GDN_MANY=0, window T)           | BEST_LANES=1; window U, one hour later)    |
+--------------------+--------------------------------+----------------------------------+--------------------------------------------+
| 1                  | 74.3 (0.25 s)                  | 82.1 (0.24 s)                    | 81.2 (0.20 s)                              |
| 4                  | 148.4 (1.15 s)                 | 154.6 (0.58 s)                   | 151.5 (0.57 s)                             |
| 8                  | 194.0 (2.84 s)                 | 213.8 (0.80 s)                   | 212.3 (0.75 s)                             |
| 12                 | 221.7 (4.82 s)                 | 243.3 (0.93 s)                   | 247.9 (0.92 s)                             |
| 16                 | 218.3 / 215.0 (6.85 / 6.76 s)  | 254.9 / 259.7 (1.29 / 1.14 s)    | 269.4 / 266.1 / 272.7 (1.30 / 1.13 / 1.12) |
| 16 copies          | 244.8 (6.53 s)                 | 321.7 (1.09 s)                   | 336.8 (1.07 s)                             |
| waves of 16 held   | 127.4, last first token 7.20 s | 147.9, 1.60 s                    | 151.5, 1.63 s                              |
| waves of 4 held    | 87.5, 1.32 s                   | 96.2, 0.74 s                     | 97.2, 0.71 s                               |
+--------------------+--------------------------------+----------------------------------+--------------------------------------------+
16 users: 269 against MiaAI's configuration 217 (+24 %; windows T and U, the same afternoon) ; same-window pairs: window T 261.5 (GDN lanes on) vs 216.7 = +20.7 %;
window U BEST_LANES on vs off: 269.4 vs 267.2 (12 users 247.9 vs 242.2, copies 336.8 vs 331.3).
Identity (texts): USERS16 vs MiaAI 16 x 64K waves-16 128/128 and waves-4 64/64; held 107 with 16 in flight 107/107 in every lever run on 4e5c; edit set 4 in flight 24/24; one user 54/54.
USERS16 keeps the 262,144-token window: 136,607-token prompt + 15 short: recall 3/3, shorts' last first token 1.56 s; 227,621-token prompt 3/3 (2145 tok/s) (window T).

Levers of round 7 at 16 users, test launcher on spark-4e5c (three runs each; held 107 with 16 in flight):
+-----------------------------------------+------------------------------------+---------------------+
| run                                     | 16 users distinct                  | held 16 in flight   |
+-----------------------------------------+------------------------------------+---------------------+
| window S ref / GDN groups of 8 / ref    | 260.1 / 269.2 / 256.6 (means)      | 174.2 / 177.3 / -   |
| window T ref first, last                | 256.8, 262.2                       | 174.5, 172.9        |
| window T GDN table of 16                | 263.1                              | 177.4               |
| window T GDN groups of 8                | 263.3                              | 175.9               |
| window U ref (GDN 16) first, last       | 264.0, 263.0                       | 178.1, 176.3        |
| window U + BEST_LANES=1                 | 269.5                              | 178.2               |
+-----------------------------------------+------------------------------------+---------------------+
GPROF 16 users (ms a pass at ~78 rows): stock 200.7 (GDN 39.4) -> GDN lanes 196.0-197.9 (GDN 34.3-34.8). KPROF: fn_row_best 0.487 -> 0.028 + 0.225 ms a profiled pass with BEST_LANES.

One user is unchanged by round 7 (the levers act only above 8 lanes / in shared rounds). Window T on the 0101-0108 image: chat 51.1 / code 81.6 (3-prompt, 9 runs), first token 182 ms,
held-107 63.1 tok/s decode (54.80 ms a round), edit set 173.3, agent set 57.1; texts equal to window Q's (edit 24/24, agent 16/16, held 107/107 equal or a prefix: 256 vs 384 tokens).
The full one-user table is results/final_r6_923b.txt (window Q): levers off 47.1 / 78.4, held-107 58.40 -> recipe 51.1 / 82.7, 62.26 (+6.6 %), edit set 137.9 -> 173.4.
Not lossless-comparable: custom engine R on a re-quantised EXL3 checkpoint measured 54.6 / 88.6 chat / code and held-320 65.66 on this box (first bake-off). We still trail it for one user.
```

The round-6 many-user table (window Q, before the lane-batching levers; MiaAI 16 x 64K 214.0 / 219.5, USERS16 260.6 / 262.7) is in `notes/final_r6_manyuser.txt`.

| lever (default in `recipe.sh`) | measured, paired in one window | identity |
|---|---|---|
| certified exact LM head (`CERT=1`) | one user +3.7 % (held-320) | 51/51, 320/320; 1,350,612 rows checked against the whole head, 0 mismatches |
| fused hyper-connection read-out + half tiles (`FUSED=3 FUSED_HALF=1 HALF=96`) | one user +2.5 % (56.33 -> 54.96 ms a round) | 51/51, 107/107; 19.5 billion output words compared on the device, 0 differ |
| MTP last row (`MTP_LAST=1`, port of upstream #1925) | first draft step 1.95 -> 1.69 ms: about +0.5 % | 107/107 |
| adaptive 32-row copy windows (`COPY_ROWS=32`) | file rewrites +24 to +26 %; chat and agent turns unchanged | 24/24, 16/16, 40/40 |
| batched prompt passes (`MANY=256`) | 16 users +14 to +17 %, last first token 6.8 s -> 1.2 s | waves 192/192, 160/160, 64/64; first-score hashes 270/270 |
| 128-row drafted passes (`USERS16=1`) | 16 users +4 to +8 % | 107/107 with 16 in flight |
| DeltaNet lane batching for 9-16 streams (`GDN_MANY=2`) | 16 users +1.4 to +4.2 % by window (GPU time of the DeltaNet group 39.4 -> 34.3 ms of a 200 ms pass) | 107/107 with 16 in flight, waves 96/96, edit set 24/24, one user 54/54; checked run 22.9 billion words, 0 differ |
| attention lanes in one launch (`ATTN_LANES=1`; idea: upstream #1739) | 16 users +1.6 to +2.3 % (GPU time of the attention group 8.47 -> 5.68 ms of a 197.5 ms pass) | 107/107 with 16 in flight, waves 96/96, one user 51/51 |
| lazy DeltaNet end-state writes (`LATE_SKIP=16`) | 16 users +1.3 to +1.5 % (DeltaNet group 34.1 -> 31.2 ms a pass); one user +0.3 % | 107/107 with 16 in flight at T = 8 / 16 / 32, one user 107/107, 51/51, edit set 24/24 |
| prompt-cache mark inside the prompt pass (`KEEP_INPASS=4096`; idea: upstream #1866) + reused stream buffers (`LANE_REUSE`; idea: upstream #1763) | short prompt first token 175 -> 113 ms (4e5c), 184 -> 111 ms (923b final); agent follow-up turn -13 %; 8 users last first token 0.71 -> 0.49 s | ident 51/51, needle 3/3, agent 4/4, held 40/40, score hashes 71/71; 107/107 with 16 in flight |
| one launch for every stream's best draft (`BEST_LANES=1`) | 16 users +0.8 to +2.3 % | 107/107 with 16 in flight, one user 54/54, edit set 24/24 |
| pool admission (`USERS16=1`) | 16 lanes keep the 262,144-token window; no speed cost | mixed 137K / 91K / 182K prompts with shorts: recall 3/3, 0 errors |

## Bake-off against other engines (spark-923b, 2026-11-12; our column here is the round-4 stack, certified head only)

```
+-----------------------------+-----------------------+------------------------+--------------------------+-----------------------+-----------------------+
| engine                      | pico stack (cert on)  | levers off, same image | MiaAI start.sh (morning) | custom engine R EXL3 re-quant   | a third engine gb10       |
+-----------------------------+-----------------------+------------------------+--------------------------+-----------------------+-----------------------+
| ttft ms                     | 180                   | 178                    | 178                      | 110                   | 181                   |
| chat c1                     | 48.9                  | 47.4                   | 47.2                     | 54.6                  | 39.3                  |
| code c1                     | 82.0                  | 79.0                   | 77.8                     | 88.6                  | 64.0                  |
| sdm prose T0                | 59.1                  | 56.9                   | 57.4                     | 67.9                  | 53.0                  |
| sdm prose T.7 think         | 51.6                  | 44.5                   | 45.8                     | 51.0                  | 39.2                  |
| sdm code                    | 95.9                  | 92.2                   | 93.1                     | 111.6                 | 63.3                  |
| sdm struct                  | 110.1                 | 106.1                  | 107.5                    | 155.4                 | 54.4                  |
| sparkDash prose/code        | 58.8 / 95.6           | - / -                  | 57.0 / 92.7              | 67.7 / 111.5          | 47.7 / 63.3           |
| sD catalogs p/s/c           | 45.2 / 68.7 / 84.2    | - / - / -              | 44.2 / 66.9 / 82.1       | 49.8 / 75.4 / 95.8    | 33.7 / 42.7 / 52.3    |
| held chat/code/math         | 51.91 / 77.05 / 75.36 | 50.01 / 74.65 / 72.70  | 50.45 / 79.96 / 74.97    | 56.67 / 86.93 / 84.96 | 38.03 / 52.76 / 52.06 |
| held all                    | 59.50                 | 57.39                  | 58.78                    | 65.66                 | 42.83                 |
| distinct 4                  | 150.0                 | -                      | 142.2                    | 138.8                 | 125.2                 |
| copies 4                    | 154.9                 | -                      | 149.4                    | 143.3                 | 104.0                 |
| max slots distinct/copies   | 5: 162.4 / 167.3      | -: - / -               | 5: 162.9 / 166.7         | 8: 173.1 / 195.5      | 8: 158.2 / 166.9      |
| 16 queued                   | 158.9                 | -                      | 158.4                    | 164.7                 | 128.3                 |
| sd wave5 prose/code         | 127.4 / 218.2         | - / -                  | 125.9 / 215.0            | 124.9 / 185.5         | 92.8 / 120.9          |
| prefill 8K/32K/128K         | 2388 / 2464 / 2305    | -                      | 2426 / 2467 / 2315       | 2019 / 2113 / 2037    | 2588 / 2777 / 2848    |
| needle 200K                 | 3/3                   | -                      | 3/3                      | 3/3                   | 3/3                   |
| ctx                         | 227621 3/3            | - -                    | 227621 3/3               | 227621 3/3            | 227621 3/3            |
| img                         | PASS                  | -                      | PASS                     | PASS                  | PASS                  |
| qeval                       | 33/35                 | -                      | 33/35                    | 32/35                 | 32/35                 |
| idle MemAvail GiB / GPU MiB | 38 / 79124            | 36 / 78438             | 37 / 78215               | 29 / 85114            | 6 / 106446            |
+-----------------------------+-----------------------+------------------------+--------------------------+-----------------------+-----------------------+
```

- `pico stack` and `levers off` are the same image in one test window, started the documented way (`recipe.sh start` on the image
  `recipe.sh prepare` built, vision on): the only difference is `CERT=1` against `CERT=0`. `levers off` is MiaAI-Lab's configuration.
- `MiaAI start.sh (morning)` is MiaAI's own image and launcher measured 15 hours earlier on the same box; the same configuration read
  about 2 % lower on the 320 held prompts in the evening window, so compare the stack with `levers off`, not with the morning column.
- chat / code: our 3-prompt bench, 512 tokens, temperature 0, thinking off, median of 9 (single runs spread +-4 tok/s).
  held: our 320 held prompts, tokens over decode time. sparkDash-method: 32-token warm-up, 256-token stream, as sparkDash 1.8.8 measures.
- custom engine R runs a different checkpoint (turboderp EXL3 4.05 bpw: attention, shared expert and LM head at 4 / 4 / 6 bits
  against NVIDIA's 8 / 8 / 16; 24 % fewer bytes per step). It is still faster for one user, and it is not the official weights.
  **The lossless stack does not catch it: 50.1 / 84.7 (round 8; held-320 61.33 in round 6) against 54.6 / 88.6 (65.66).**

### Paired proof for the round-4 stack (same box, same window, same image)

| bench | levers off (MiaAI configuration) | stack (`CERT=1`) | change |
|---|---|---|---|
| chat / code, 3-prompt bench, 9 runs | 47.4 / 79.0 | 48.9 / 82.0 | +3.2 % / +3.8 % |
| sparkDash-method prose / code / structured | 56.9 / 92.2 / 106.1 | 59.1 / 95.9 / 110.1 | +3.9 % / +4.0 % / +3.8 % |
| 320 held prompts: all / chat / code / math | 57.39 / 50.01 / 74.65 / 72.70 | 59.50 / 51.91 / 77.05 / 75.36 | +3.7 % / +3.8 % / +3.2 % / +3.7 % |
| identity set, greedy | | 51/51 identical | |
| held prompt texts | | 320/320 identical | |
| every head call checked against the whole head | | 1,350,612 rows, 0 mismatches | |

### Many users, round-4 state (distinct coding prompts, 512 tokens each, all arriving at once; total tok/s)

```
+---------------+------------------------------------------+----------------------------------------------+--------------------------------------------------------+
| users at once | MiaAI default: 5 slots x 262K            | MiaAI config, 16 slots x 64K (64-row passes) | pico USERS16: 16 x 64K, 128-row passes, certified head |
+---------------+------------------------------------------+----------------------------------------------+--------------------------------------------------------+
| 1             | 76.5                                     | 74.1                                         | 78.7                                                   |
| 4             | 147.7                                    | 145.9                                        | 140.5                                                  |
| 5             | 163.2                                    | -                                            | -                                                      |
| 8             | 151.2 (3 wait; last first token 15.7 s)  | 194.9                                        | 199.1                                                  |
| 16            | 155.4 (11 wait; last first token 38.4 s) | 219.0 / 221.4                                | 229.9 / 228.3                                          |
| 4 copies      | 148.6                                    | 153.2                                        | 150.5                                                  |
| 8 copies      | 156.5                                    | 213.4                                        | 220.2                                                  |
| 16 copies     | 159.8                                    | 243.7                                        | 269.9                                                  |
+---------------+------------------------------------------+----------------------------------------------+--------------------------------------------------------+
spark-923b, one window, all three started by recipe.sh (vision on). Single runs spread about +-3 % (4 users read 140-150 across boxes).
Same change on the other two boxes, 16 users distinct: spark-3d24 64 rows 216.4 / 221.0 -> 128 rows 236.2 / 234.2;  spark-4e5c 128 rows 232.4 / 230.9, with the certified head 234.8 / 234.0.
```

- MiaAI's default serves 5 users at a 262K window; more wait in the queue. `USERS16=1 ./recipe.sh start` serves 16 at a 64K window
  with drafted verify passes of up to 128 rows (TensorFold stops at 64) and the certified head.
- Why wider passes help: with 16 streams a pass of 64 rows leaves each stream about 3 drafts a round; an extra draft row costs about
  1.1 ms there because the rows share experts, so 97 rows a round give 4.7 tokens a stream instead of 3.7.
  Texts are the same: 107 held prompts with 16 in flight, 64 against 128 rows, 107/107 identical.
- `copies` repeats 3 prompts (shared experts and drafts make it easier); quote the distinct number.
- In this round-4 table 16 users arriving together were admitted one prompt pass at a time (last one starts after ~6.7 s). Batched
  prompt passes (patch 0106) removed that: see the current table above.
- Other slot shapes that start: 12 x 128K (12 users 216 distinct), 16 x 80K (16 users 231.9 / 228.7 distinct). 16 x 96K does not fit (31.6 GiB needed, 30.8 available).

## Memory

**Read this first: the default profile trades free memory for prompt-cache hits.** With `ELASTIC=1` (the default of the one-user /
5-stream profile) the prompt cache grows into the room the five streams' windows are not using. During our full suite the engine's
GPU use went from 79.7 to 108.1 GiB and the box's MemAvailable fell to about 10.5 GiB (with the levers off: 86.6 GiB and 31.9 GiB).
The floor is the engine's own reserve (`TENSORFOLD_MEMORY_RESERVE_GIB`, 9 GiB in MiaAI-Lab's launcher), and a stream that opens takes
its room back from the cache first. What it buys: agents with long prompts of their own resume from kept states instead of
prefilling again (5 agents with 56K-token system prompts: follow-up first token 65 s -> 1 s, builder C, spark-4e5c).

- This is the right default for a box that runs only this engine. It is the WRONG default on a box that runs anything else
  (a gateway, a vision encoder, a build): use `ELASTIC=0 ./recipe.sh start` (the prompt cache keeps its start-up size, about 4 GiB,
  and ~20 GiB stay free), or keep it and raise the reserve (`TENSORFOLD_MEMORY_RESERVE_GIB=13` left 15 GiB free in builder D's run).
- `USERS16=1` and `AGENTS=1` use a fixed pool and are not elastic: their memory use does not grow.
- `ARENA=1` (option, off by default, builder C, round 10) takes the whole planned room once at start-up: faster lane opens and
  kept-state writes on long prompts, but the IDLE engine then leaves only 7-8 GiB MemAvailable. Not tested with pool profiles.
- Fixed costs of our levers: certified head copy +0.6 GiB, two-stage draft head file 30.7 MB, 16-user profile: pool of 16 x 64K windows.

## Reproduce

Current commands (final image, patches 0101-0129; checked end to end on spark-923b in the night of 2026-11-15, see
[notes/final-2026-11-15/FINAL.md](notes/final-2026-11-15/FINAL.md)):

```
./recipe.sh prepare    # clones MiaAI-Lab's recipe at 20e15e4, adds patches/01*.patch, builds the image, fetches the checkpoint, builds the two-stage head's file
./recipe.sh start      # serves on 127.0.0.1:8888: bf16 KV cache (exact), 3 streams at 262,144 tokens, every lever of ours on
USERS16=1 ./recipe.sh start   # 16 lanes sharing a memory pool of 16 x 32,768 tokens (bf16 KV); one request can still take the full 262,144
AGENTS=1 ./recipe.sh start    # 5 lanes, pool of 5 x 131,072 tokens, elastic prompt cache with delta states: agents with long histories
# every lever of ours off, same image, bf16 KV, 3 streams (the "levers off" column of the final tables):
CERT=0 MANY=0 COPY_ROWS=16 FUSED=0 HALF=0 MTP_LAST=0 GDN_MANY=0 BEST_LANES=0 ATTN_LANES=0 LATE_SKIP=0 KEEP_INPASS=0 LANE_REUSE=0 HEAD2=0 R9=0 R10=0 R11=0 ./recipe.sh start
# MiaAI-Lab's shipped shape for 16 users (the many-user reference): the line above plus
#   KV=fp8 PARALLEL=16 CONTEXT=65536 TF_FLASHNEXT_PRODUCT_STREAMS=16
KV=fp8 ./recipe.sh start                         # MiaAI-Lab's fp8 KV cache (LOSSY against bf16): 5 streams at 262K; with USERS16=1 a pool of 16 x 65,536
R11=0 ./recipe.sh start                          # the round-10 defaults (round 11's settings off)
HEAD2_ROT=r512 ./recipe.sh head2                 # the round-8 rotation for the two-stage head (comparison only)
ELASTIC=0 ./recipe.sh start                      # keep the prompt cache at its start-up size
./recipe.sh bench      # the same benches as the table
```

Pinned: TensorFold v1.1.4 `eab8582`, MiaAI-Lab recipe `20e15e4`, base image `nvcr.io/nvidia/pytorch:26.07-py3`
`@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c`, Zig 0.17.0, checkpoint revision `b9ce18a6`.
Measured on DGX OS 7.2.3, kernel 6.14.0-1015-nvidia, driver 580.173.02. No host changes are needed or made.

This path is the measured one: `recipe.sh prepare` ran end to end on spark-923b (about 5 minutes with the base image and the
checkpoint already on the box) and the `pico stack`, `levers off` and many-user rows above were served by `recipe.sh start` on the
image it built. Patches 0101-0118 applied to MiaAI's tree reproduce our build tree exactly (same git tree hash 09bb8b51 for 0101-0118; 7628b4db for 0101-0123); `recipe.sh prepare` was re-run end to end with each patch set (0101-0107, -0108, -0109, -0113, -0115, -0118; the head2 file it builds has sha256 fce7c913060d45f2 on a GB10 GPU); the round-8 tables were measured on the 0101-0113 image, and the 0101-0115 image with `HEAD2` off gives the same texts and speed.

### The configuration

| Setting | Value | Why |
|---|---|---|
| launcher | MiaAI's `start.sh` (keeps the image entrypoint) | the entrypoint switches to the container's CUDA forward-compat userland (libcuda 610.43 on a 580.173 host driver): +4.5 % against a launcher that overrides it; same texts. NVIDIA's newer cuda-compat 13.4 / 13.5 libraries add nothing more (measured) |
| certified LM head | on (`TF_FLASHNEXT_Q38_CERT=1`) | +3.2 to +4.0 % for one user, same tokens; +0.7 GiB GPU memory; sampled, grammar and logprobs requests keep the whole head |
| draft gate | MiaAI's default (`TF_FLASHNEXT_CONFIDENCE=-0.4`, `TF_FLASHNEXT_DRAFTS=6`) | the option 0.5 / 8 (`GATE=structured`) is a trade, same texts: structured output +16 %, prose -2.8 % |
| KV cache | bf16 (exact) since round 11: 3 streams at 262K, USERS16 pool 16 x 32,768, AGENTS pool 5 x 131,072 | MiaAI's default is fp8: their note says lossy against bf16 KV (~98.8 % top-1), measured here: 37 of 51 identity-set replies differ. `KV=fp8` is the opt-in. Our identity tests hold the KV format fixed |
| fused read-out, half tiles, MTP last row | on (`FUSED=3 FUSED_HALF=1 HALF=96 MTP_LAST=1`) | +2.5 % one user, same output words; flat at 16 users (passes are expert-bound there) |
| copy windows | `COPY_ROWS=32`: 15 copy tokens, 31 after a copy landed whole | file rewrites +24 to +26 %, nothing lost on chat |
| prompt passes | `MANY=256`: waiting prompts of up to 256 tokens share one pass | first token for the last of 16 users 6.8 s -> 1.2 s |
| 9-16 streams | `GDN_MANY=2`, `BEST_LANES=1`: lane-batched DeltaNet launches and best-draft read-back above 8 lanes | 16 users +2 to +5 % together; nothing changes at 1-8 streams |
| shared rounds | `ATTN_LANES=1`: a shared round's attention in one launch a kernel; `LATE_SKIP=16`: a drafted window writes DeltaNet end states only when more than 16 rows would wait in its log | 16 users +1.6 to +2.3 % and +1.3 to +1.5 %; one user +0.3 % |
| first token | `KEEP_INPASS=4096`, `LANE_REUSE` = number of streams | a short prompt's first token 184 -> 111 ms; agent follow-up turns -13 % |
| two-stage draft head | `HEAD2=1`, rotation `r9` (fitted on 260 prompts outside every bench set), `HEAD2_FAST=3` | held prompts (outside the fit) +2.3 %, fresh prompts +1.8 %; +0.9 % more from the integer preview |
| draft depth | `DEEP=0.9` (up to 12 drafts while the chain's product is >= 0.9), `STOP2=0.5` | code, math and structured output +5 to +7 %, chat +0.3 %; sparkDash structured 105 -> 166 |
| sampled requests | `CERT_CODE=sm8`, `CERT_TOPK=20` (not with `USERS16`) | seeded sampled T 0.7 +6.6 to +7.1 % against levers off, same text |
| long prompts | `QSA_CERT=1` (certified indexer select) | 182K prompt first token 83.3 -> 77.2 s, 227K 108.2 -> 98.1 s |
| prompt cache | `ELASTIC=1` (default profile), `PREFIX_WAIT=60000` | agents with long prompts of their own resume instead of re-prefilling (builder C: 65 s -> 1 s); costs up to ~20 GiB of idle headroom |
| 16 users | `USERS16=1`: 16 lanes, pool admission at 32,768 tokens a lane with bf16 KV (65,536 with `KV=fp8`; the round-8 figures on the right are fp8), `TF_FLASHNEXT_Q38_WIDE_ROWS=128` | 16 users 280 tok/s against 218 (round 8); a full 262,144-token stream takes 4.9 GiB of the 25.7 GiB pool |

## How the certified head works

The LM head is 248,320 x 2,560 BF16 = 1.18 GiB, read once per verify pass (5.3 ms of a 58 ms round).

1. At load we make an MXFP8 copy of it (E4M3 codes, a power-of-two scale per 32 inputs: 0.61 GiB) and, for every vocabulary
   row, the norm of the copy's error. Mean error is 2.66 % of a row's norm.
2. A greedy verify pass multiplies the rows by the copy with TensorFold's own MXFP8 kernel. Each score is then known to within
   `c[v] * ||x||` (Cauchy-Schwarz on the measured error, plus an allowance for float rounding in both kernels).
3. Any entry whose upper bound is below the best lower bound, less two BF16 steps, cannot be the best and cannot tie with it after
   the head's BF16 rounding (1.4 % of real rows have a top-1 / top-2 tie in BF16, so ties matter). The 64-entry tiles holding
   the remaining entries (about 47 of 3,880 for a one-user window) are computed by the head's own BF16 kernel on the original
   weights: the same blocks on the same bytes, so the same bits. Everything else in the row is -inf.
4. The engine's unchanged argmax kernel then returns the same token as with the whole head.
5. If too many tiles remain, a score is not finite, the pass has more than 256 rows, or any stream in the pass samples, uses a
   grammar or asks for log-probabilities, the whole head runs as before.

Checked: `TF_FLASHNEXT_Q38_CERT=2` runs the whole head beside the certified one on every call and compares each row's best value
and id and every recomputed entry bit for bit: 1,350,612 rows from one-user windows, 5 and 16 streams in flight and wide
passes: 0 mismatches. Offline study on 124,579 real head rows before building (notes/raw/head_pruning_study.txt): MXFP8, int8 and
int6 codes prune; 4-bit and 5-bit codes do not.

## Where the time goes (one user, measured)

```
a 1-row step reads 6.47 GiB:  dense 4.13 (attention, hyper-connection, shared expert)  +  LM head 1.18  +  experts 1.15
stock dense kernel alone:     head 240 GB/s · q_proj 232 · hyper-connection 215 · o_proj 205   (the box's ceiling is about there)
drafted round, levers off:    58.2 ms = 42.4 + 4.14 x drafts verified  (4.8 rows, 3.4 tokens)
drafted round, certified:     56.0 ms = 40.7 + 4.01 x drafts verified
one draft:                    1.70 ms MTP chain step (4-bit shortlisted draft head ~0.55 ms of it) + ~2.4 ms verify row
host:                         one thread; waits for the GPU 84 % of the time; 1520 launches a round
```

A 2,000 ns spin before every launch (3 ms of extra host time a round) slows the round by 0.22 ms: the GPU does not wait for the
host. So launch cost, CUDA-graph capture and stream tricks have at most ~1 % to give, and the pace is set by bytes read.

## What we built and what it measured

| Lever (flag) | Result | Identity |
|---|---|---|
| certified exact LM head (`TF_FLASHNEXT_Q38_CERT=1`) | one user +3.2 to +4.0 %; 16 users +1 % | 51/51, held texts identical, 1,350,612 rows checked, 0 mismatches |
| wider drafted verify passes (`TF_FLASHNEXT_Q38_WIDE_ROWS=128`) | 16 users +6 to +8 %; 1 to 8 users unchanged; 192 adds nothing | 107/107 with 16 in flight |
| image entrypoint kept (launcher) | +4.5 % (MiaAI's launcher already does this) | 51/51, 107/107 |
| newer CUDA userland (NVIDIA cuda-compat 13.4 / 13.5 mounted into the container) | -0.4 / -0.5 % (noise): nothing beyond the image's 610.43 | 54/54 |
| host wait policy (`TF_FLASHNEXT_Q38_SCHED=spin`) | +0.5 % (noise): the driver default already spins | 54/54 |
| shared-expert side stream off (`TF_FLASHNEXT_SHARED_SIDE=0`) | -1.5 %: MiaAI's setting stays | 54/54 |
| host draw threads 0 | 0 % for greedy | 54/54 |
| priced draft allocation (`TF_FLASHNEXT_ROUND_MS=46+2.45` or `40+3.2`) | 16 users -3 to -5 % | - |
| gate 0.5 / 8 | held +1.15 %, structured +16 %, 3-prompt code -3.8 %, prose -2.8 %: a trade | 51/51, 107/107 |
| qmmf pipeline depth, expert prefetch depth, 1-warp expert blocks | -0.3 to -2.6 % | bit-identical kernel outputs |
| programmatic dependent launch (`TF_FLASHNEXT_Q38_PDL=1`) | +0.65 % without drafts, 0 % with | 51/51, 107/107 |
| n-gram rows through the page cache / mmap | 0 % / -6 % with drafts | 6/6 |
| looser gates 0.25 / 0.20 / 0.12 | -1.8 / -3.6 / -6.3 % | identical |
| deeper or looser drafting at 16 users (160 / 192 rows with 8 / 10 drafts, gates 0.3 / 0.5) | 223-236: nothing over 128 rows with MiaAI's gate | - |
| launch-delay probe (`TF_FLASHNEXT_Q38_LAUNCH_DELAY_NS=2000 / 4000`) | round +0.22 / +0.39 ms for +3.0 / +6.1 ms of host time: launches are 93 % hidden | 54/54 |
| batched prompt passes, pool admission, adaptive copy windows, fused read-out, half tiles, MTP last row (rounds 5-6) | see the lever table at the top | see there |
| short context copies (`TF_FLASHNEXT_Q38_SHORT_COPY=3,7 / 5,7 / 3,15`) | edit +0.1 / -0.2 / +1.4 %, agent +0.8 / +0.4 / +0.9 %, chat 0: inside noise, off by default | 24/24, 16/16, 80/80 |
| expert tile rows 64 / 128 at 16 users (`TF_FLASHNEXT_PROMPT_EXPERT_TILE_ROWS`) | +0.8 / -1 % (noise) | 107/107 |
| prompt chunk 4096 / 8192 (`TF_FLASHNEXT_CHUNK`) | prefill 0 to +2 % for +3.7 GiB | needle 3/3 |
| 128-row lane tiles for passes of more than 64 rows (`TF_FLASHNEXT_Q38_ROWS128=2048`, patch 0108, off) | zero in the engine: the stock kernel already runs the two 64-row tiles of a weight tile together, so the second finds it in the GPU cache (our microbench's +24 to +83 % used the wrong band argument) | 107/107 |
| routed-expert kernels: decode form vs shared-memory tile vs prefetch depth vs hardware FP4 / FP8 decode (microbench `kit/mb/mbx.cu`) | the stock kernels already read 220-230 GB/s from 5 rows up; tile at low rows 0; hardware decode 0 | every variant's bits equal stock |
| draft gate re-fit after the stack (0.4/8, 0.35/8, 0.3/6) | 63.4 / 62.9 / 62.3 against 63.2 tok/s: MiaAI's 0.4/6 stays | 107/107 |
| copy rows 32 against 16 at 16 users | 261.9 against 258.1 (three runs each, spread +-9): free | 107/107 |
| fused read-out with 2 / 6 stages; fused read-out for wide passes (`HCF_MAX=128`) | 2: -0.4 ms against 3 stages' -0.9; 6: 0; wide: flat | 107/107 |

Not built, with the reason:

| Lever | Why not |
|---|---|
| fix "4.8 us per launch with drafts" | it is not a per-launch price: ~8,500 launch calls block ~0.5 ms each while the GPU works through its queue; the spin test above shows it costs ~0.4 ms a round at most |
| CUDA-graph capture of the drafted round | removes host submission time only: bound ~0.9 % from the same test (not the 12 % we estimated before measuring) |
| fusing glue kernels | bound ~1.5 % (about 430 mergeable launches x 2 us of GPU time) for a large exactness risk |
| lower-precision or shortlisted draft head | TensorFold already ships it: 79,591 rows re-coded to 4 bits |
| expert kernel toward 225 GB/s | no concrete cause in this round's profiles; two geometry variants were already slower |
| prompt-lookup / suffix drafts | not simulated: our step logs have no token ids |

## What is left (estimates, not measured)

| Lever | Estimate | Needs |
|---|---|---|
| builder D's row limit for the sampled certified head (`CERT_TOPK_ROWS=8`, patch D9c, measured on the head: 16 sampled users back to par) | lets `USERS16` use `CERT_TOPK=20` | integrate + measure 2-5 and 16 sampled users on 923b |
| two-stage head for sampled chains (`TF_FLASHNEXT_Q38_HEAD2_SAMPLED=1`, in patch 0118) | +1.6 to 2 % on sampled requests (builder D's estimate) | a seeded pair (`kit/sq.py`) |
| tree verify: one sibling row at a draft position | +1.0 to +1.6 % simulated (builder B) | verify windows that are not prefix-shaped: attention cache slots, DeltaNet fork, rollback of a path |
| non-English text (37-39 tok/s against 48-56 for English chat) | unknown | the MTP chain's drafts are rarely kept there: a study of the draft head's shortlist (79,591 rows) on those languages |
| the indexer on long prompts after the certified select | a chunk still grows 1.2 ms per 1K tokens of context (stock 2.3) | per-kernel profile of the remaining growth |
| a public parallel drafter (DFlash2 / DSpark for this model) with an 8-bit body in place of the MTP chain | chat +3 to +8 %, agent work negative (estimate from the model cards) | a drafter runner for Flash Next; measure its accepted length at temperature 0 first |
| prefill 2.3-2.5K tok/s | none found: a prompt row costs 0.389 ms = ~30 TFLOP/s of bf16 tensor-core work in every group | - |
| 6-bit coarse copy for the certified head | +1 % | a bit-unpacking GEMV kernel that still reads at ~230 GB/s |
| a re-quantised checkpoint (custom engine R's route) | +15 % and more | not lossless against the official weights; Pico's call |

## Files

- `recipe.sh`: prepare / start / stop / bench / head2.
- `patches/0101` profilers (`TF_KPROF`, `TF_STEP_LOG`), `0102` buffered n-gram reader, `0103` PDL + `TF_HOSTPROF`, `0104` kernel knobs,
  `0105` certified LM head, wide rows, per-kernel launch timer, sync call sites, wait-policy knob, head-row dump, launch-delay probe,
  `0106` batched prompt passes, pool admission, adaptive copy windows, certified head on plain rounds, GPU time by group,
  `0107` fused decode read-out (+ on-device check), half tiles (upstream #1171), MTP last row (upstream #1925), short context copies,
  `0108` DeltaNet lane batching for 9-16 streams, 128-row lane tiles (off: zero), `0109` one launch for every lane's best draft,
  `0110` attention lanes in one launch, `0111` drafted windows skip DeltaNet end-state writes, `0112` builder C's first-token levers
  (KEEP_INPASS, LANE_REUSE, prefix wait; PLOG / KNOBFILE aids), `0113` builder D's expert statistics log (`TF_FLASHNEXT_Q38_XSTAT`), `0114` builder B's draft-study logging, `0115` builder B's two-stage
  draft head (default since round 9; `head2/` holds its rotations r9 and r512, `mkhead2.py`, the draft vocabulary list, the fitting corpus and the fresh set,
  `mkcorpus.py`, `overlap.py`, `evalrot.py`), `0116` builder C's certified indexer select + elastic prompt cache, `0117` builder D's certified head for sampled
  passes (SM8 coarse copy, top-k), `0118` builder B's deep chains, early stop, fast two-stage head.
  Apply after MiaAI's 0001-0017 (`recipe.sh prepare` does).
- `kit/`: `qk.py` (every bench), `suite.sh`, `heldpar.py` (held prompts with N in flight), `eng.sh` (test launcher), `rs.sh` (documented-path launcher
  for the test boxes), `tfbuild.sh`, `table.py`, `tfsteps.py` (round cost fit), `hstudy.py` (head pruning study), `hpk.py`, `syncsites.py`,
  `kprof.py`, `gatesim.py`, `mustep.py`, `mb/` (kernel microbenches).
- `notes/PROGRESS.md`: the full measurement log, including mistakes and corrections. `notes/raw/`: suite logs, the pruning study, profiles.
  `harness/`: the test-window scripts as run.
