# Final numbers, night of 2026-11-15 (UTC)

Box: spark-923b (one DGX Spark, GB10). Engine: MiaAI-Lab's TensorFold recipe (v1.1.4 + their 17 patches) + patches 0101-0129 of this repo,
NVIDIA's FP4-Local checkpoint at revision b9ce18a6, bf16 KV cache. Every engine start went through `recipe.sh start` on the image `recipe.sh prepare`
built. Two test windows; nothing else ran on the box's GPU in them.

## Which image each number is from

| window (UTC) | what ran | image |
|---|---|---|
| S150430, 04:30-09:00 | `recipe.sh prepare` check, USERS16 / AGENTS start smokes, the one-user bracket (levers off 04:46-05:57, recipe default 05:57-07:03, levers off 07:03-08:12), check modes 08:12-08:27 | id b052a3f158ad (built 2026-11-14 18:51 UTC by `recipe.sh prepare`, patches 0101-0129) |
| S150902, 09:02-11:00 | many users (MiaAI shape 09:02-09:18, USERS16 09:18-09:45, MiaAI shape 09:45-09:55), agent traces (AGENTS 09:55-10:06, MiaAI default 10 agents 10:06-10:21, 5 agents 10:50-11:00), the seeded 16-in-flight reference 10:21-10:31, `recipe.sh bench` 10:31-10:50 | id 14204217aee6 |

Image 14204217aee6 is what `recipe.sh prepare` gave after the recipe.sh changes merged this night (own image name, base image by digest): the build took
6 seconds, every step came from the build cache, and the image has the SAME 74 filesystem layers as b052a3f158ad (same layer list, same creation time);
only the label `tf.patches` differs. No engine code was rebuilt this night: both ids hold the same engine binary (tensorfold-native sha256
af27a1e7b029e243e6c94da92515826929c57b62fbbfa62586def16d39a5a2e4, read from both images on spark-923b at 11:02 UTC; agents B and D tested the same file).

recipe.sh in the windows: 04:35-08:12 sha256 66fe0314 / fc1c8e02 (the repo's file + agent D's option validation, then + the 16-user pool size; the engine
environment of every option set used is byte-identical to the final file's, checked by evaluating both on the Mac); from the check-mode run at 08:12 on:
sha256 a422375e = the file in this commit.

## What `recipe.sh prepare / start` did (the last two recipe.sh edits, never started before this night)

- `recipe.sh prepare` on the box that holds the image: rc 0 in 4 s, image unchanged, checkpoint verified, two-stage head file present.
- `./recipe.sh start` (no options): ready in 202-220 s; engine arguments `--context 262144 --parallel 3 --kv-cache bf16`; "3 streams at once, 8.39 GiB each";
  prompt passes of 4,608 rows; the container's environment against the list evaluated from recipe.sh on the Mac (harness/night-1114/intended_default.txt): 41 of 41 compared
  knob variables equal, plus the three HEAD2 variables (raw/N_on/env.txt); USERS16: 42 of 42.
- `USERS16=1 ./recipe.sh start`: the committed pool size 16 x 40,960 started on spark-923b (28.0 GiB of budget; 16 users 293.5 / 301.1 tok/s) but did NOT
  start on spark-4e5c's fresh install ("needs 27.38 GiB ... the budget leaves 27.18 GiB", agent C). Fixed: 16 x 32,768 (C measured it on 4e5c; the many-user
  table below was measured with it on 923b: "16 streams at once, 1.48 GiB each", 4.4 GiB of prompt cache left).
- `AGENTS=1 ./recipe.sh start` (5 lanes, pool 5 x 131,072 under bf16): ready in 199 s, "5 streams at once, 4.66 GiB each"; worked as committed.
- `./recipe.sh bench` end to end: rc 0 on a no-option start through the shipped recipe.sh (10:31-10:50 UTC, window 2): first token 120 ms, 3-prompt chat / code 54.2 / 88.6,
  sparkDash-method prose / code / structured 70.0 / 117.3 / 162.8, held prompts (104) 67.32, prefill 2558 / 2515 / 2472 tok/s, 3 users 142.3 (raw/N_bench/bench.log).
  (A first try at 08:31 failed for a harness reason: the box's copy of the repo directory had no kit/ folder.)
- From-scratch install (agent C, spark-4e5c, fresh clone of main 42291bf, nothing reused but the base image): INSTALL.md beside this note.

## Claims that failed or changed this night (said plainly)

1. **10 agents x 56K histories.** The front page said "142 s -> 5.5 s". That was measured with the fp8 KV cache on spark-4e5c. On the bf16 default tonight:
   follow-up first token median 64.5 s, only 13 of 20 follow-ups resumed from the cache (MiaAI default in the same window: 199.8 s, 0 of 20). Ten 56K
   histories do not fit the bf16 prompt cache beside 5 busy lanes (a kept token costs 30.1 KiB in bf16, 17.5 in fp8). Five agents fit: 25 of 25 resumed,
   follow-up median 1.29 s (MiaAI default, same window: 11.6 s, 15 of 25 resumed, wall 373 s against 76 s; note that our 5-agent trace ran on the engine that had just
   served the 10-agent trace, so its opening turn was already cached; the follow-up turns are the comparison). Not bisected further. The front page now shows both rows.
2. **3-prompt code bench.** 88.2 tok/s tonight (90.8 at the wrap-up). custom engine R's 90.4 (earlier window, re-quantised weights) is above it, so the front page says
   custom engine R is faster on the 3-prompt chat and code bench; this repo is ahead on the 320 held prompts and on sparkDash's tests.
3. **16-user pool size** (above): the committed default did not start on a second box.
4. **"Sampled output identical per seed"** holds on /v1/chat/completions and /v1/responses. /v1/messages does not forward `seed` in the base engine
   (TensorFold's translate code), so sampled replies there differ run to run, levers on or off (agents B and D; confirmed with every lever off: 12 of 12
   differ). A one-line engine patch exists (agent D, handoff) but was NOT built or tested, so it is NOT in this commit.
5. **Memory.** With the default elastic prompt cache the box's free memory falls to about 9 GiB under load (lowest in tonight's one-user suite 8.9 GiB;
   7 GiB in the 16-user run with a 136K-token prompt; agent B's soak: 7.5-9.2 GiB). "Keeps 9 GiB free" would be wrong.
6. Tonight's box read about 3 % faster than at the wrap-up on identical work (levers off, same texts: held 59.1 against 57.4). Only same-window pairs count.

## Identity (texts compared as exact strings)

Recipe default against every lever off, same window, same image, bf16 KV, against the first AND the last levers-off run:
identity set 51/51; all 320 held prompts 320/320; file rewrites 24/24; agent turns 16/16; seeded sampled T 0.7: chat 24/24, thinking 24/24, sparkDash-method 6/6;
56 rewrite requests 56/56 without and 56/56 with `prediction`; agent loop 46/46 without and 46/46 with `prediction`. The two levers-off runs give the same
texts as each other and as the wrap-up run of 2026-11-14 on the same image.
With 16 users in flight (USERS16 profile): all 320 held prompts equal the ONE-user levers-off texts 320/320; identity set on that profile 51/51.
Seeded sampled with 16 in flight against the one-user levers-off sampled texts: 107/107 identical (held prompts, every third, 256 tokens, T 0.7, seed 4242; N_mu/helds16.jsonl against N_offs/helds1.jsonl).
`prediction` on the three routes (4 items each): texts 4/4 equal with and without it on /v1/chat/completions, /v1/responses, /v1/messages; 13 of 13 malformed
predictions answered 200 with the reference text on each route.

## Check modes on (N_chk, 08:12-08:27, default profile; engine counters)

Sample: prompts of 29K, 117K and 227,621 tokens, sparkDash-method x1, 12 seeded sampled requests, the identity set, 2 agents x 4 turns of 20K, 28 rewrite
requests with `prediction`. No counter in the log is nonzero; 0 crash lines.

| check | rows compared | differences |
|---|---|---|
| certified head, greedy (last counter line; it prints every 1,000 calls) | 24,753 rows in 5,000 calls | best-mismatch 0, value-mismatch 0 |
| certified head, sampled top-k | 3,457 rows | topk-mismatch 0, bound violations 0 of 5.54 billion entries |
| certified indexer select | 6,657,316 row selections (5,040,698 through the listed form) | list_diff 0, exact_bit_diff 0, bound_viol 0, fallback 0, list_miss 0; largest ratio to the bound 0.647 |
| delta prompt states (SNAP_DELTA=2) | 124 keeps, 119 shared, 25.25 GB compared | check_diff 0 |
| scratch guard | 5,760 passes | bad 0 |
| prompt DeltaNet producer groups | 6,120 launches, 74.2 billion words | diff 0 |
| router bands | 5,808 passes, 13.4 billion words | diff 0 |

Texts in check mode: identity set 51/51 equal to the default and to levers off; seeded sampled 12/12; rewrite requests 28/28; 227,621-token prompt recall 3/3.
Agent B ran the same check modes under 30 minutes of mixed load on spark-3d24 (67,569 head rows, 22.5 million selections, 280 GB of delta state): all zero.

## Base-engine facts the exactness claim does not cover (agents B and D, levers off)

- `logprobs` and `n` > 1 are refused (400); repetition / presence / frequency penalties, `logit_bias` and `min_tokens` are accepted and ignored.
- A streamed and a non-streamed reply carry the same tokens but not the same bytes in two cases: with thinking on, the streamed reasoning ends with one extra
  newline; a tool call cut off by `max_tokens` is returned parsed when streamed and as raw text when not.
- /v1/messages ignores `seed` (above).

## Not measured this night

- custom engine R on this box in the same window (its front-page column is from 2026-11-14, spark-923b). Agent C's same-window competitor run on spark-4e5c: see the same-window comparison note.
- The 182K needle, image and qeval rows for the SECOND levers-off run (skipped for time; the first levers-off run and the default have them).
- An agent trace on the AGENTS profile with every lever off (text identity of the traces themselves); the traces' identity rests on the agent set, the agent loop
  and the check modes' delta-state compare.
- KV=fp8 profiles, RECALL=1, GATE=structured, ARENA=1: not started this night.
- The saved env.txt files lack TF_FLASHNEXT_Q38_KEEP_INPASS because the log filter dropped names containing "pass"; the variable was read directly from a running
  default-profile container (=4096) and agent C's install compared it.

## Merged this night (no engine change, no rebuild)

| from | what | proof |
|---|---|---|
| agent D (fuzz) | recipe.sh option validation: only the literal `KV=fp8` selects the lossy cache (a typo used to fall through to fp8 silently); nonsense option values stop; QSA_SLACK below 32 refused; USERS16 + AGENTS together refused | engine environment byte-identical for every valid option set used (D: 24 sets; A: 11 sets); all of tonight's starts ran with it |
| agent C (fresh install) | USERS16 bf16 pool 16 x 40,960 -> 16 x 32,768 | failed / started on spark-4e5c; started and measured on spark-923b (table) |
| agent C (fresh install) | recipe.sh: HF_HOME rule, base image by digest, no pull of a tag that cannot exist, own image / container / state names, `verify`, stale patch removal, message for start before prepare, head2 file owned by the user, `bench` rows | C: prepare + start + 51/51 texts on spark-4e5c; A: environment byte-identical, check modes + window 2 started with it |
| agent C | README "run it": prerequisites, times, test request, stop | from the measured install |
| agents B, D | README wording: /v1/messages seed, memory low-water, unsupported request fields | measured, levers off |
| NOT merged | agent D's one-line /v1/messages seed patch (engine change; not built, not tested) | - |

## Fuzz and soak on the other Sparks (same engine binary; their notes are beside this file)

- Agent D (head Spark), exactness fuzz: 7,210 levers-on against levers-off comparisons, 7,135 identical, 75 refused the same way, 0 different; a second engine started with the
  flags off against knob-off: 1,138, 0 different; 713 requests with the device check modes on: 0 mismatches (72,876 greedy head rows, 18,222 top-k rows, 4.19 million selections).
  Not covered there: this recipe's full profile sizes (the head ran 2 lanes, 16 x 16,384 and 4 / 3 agent lanes), 50K-200K-token prompts with many in flight, video / JPEG / WebP,
  RECALL / COPY_FIRST / ARENA / GATE=structured; its delta-state and tall read-out check modes printed no counter (the delta-state check of spark-923b above did). The fuzzer is in
  tensorfold/selftest/.
- Agent B (spark-3d24), soak (final): 334 minutes of mixed load on the default, USERS16 and AGENTS profiles; 6,909 replies checked; 6,834 of 6,834 equal their one-user levers-off text once the 4 sampled /v1/messages requests are set aside (they differ with the levers off too); 0 crash lines, 0 hangs, 0 HTTP 5xx apart from 13 clean 503s during a deliberate restart; MemAvailable low-water 7.9-9.2 GiB. Full note: SOAK.md.
  their solo levers-off text, 0 crashes, 0 HTTP 5xx; the only mismatches are the sampled /v1/messages requests. Its final file was not complete when this was committed.

## Tables (printed by harness/night-1114/tableN.py from the raw files; nothing typed in)

```
ONE USER, spark-923b, final image, bf16 KV, 3 x 262K, one window
+-----------------------------------------------------+--------------------+----------------+-------------------+---------------------+
| bench                                               | levers off (first) | recipe default | levers off (last) | default vs mean off |
+-----------------------------------------------------+--------------------+----------------+-------------------+---------------------+
| first token, short prompt (ms)                      | 180                | 120            | 183               | -33.9 %             |
| 3-prompt chat, 9 runs (median)                      | 48.2               | 53.5           | 47.0              | +12.4 %             |
| 3-prompt code, 9 runs (median)                      | 78.4               | 88.2           | 76.9              | +13.6 %             |
| ALL 320 held prompts                                | 59.10              | 66.87          | 58.35             | +13.9 %             |
|   prompts answered with > 16 tokens                 | 315                | 315            | 315               |                     |
|   held: chat                                        | 51.63              | 57.52          | 50.94             | +12.2 %             |
|   held: code                                        | 76.25              | 89.40          | 75.37             | +17.9 %             |
|   held: math                                        | 73.99              | 86.49          | 73.11             | +17.6 %             |
| sparkDash-method prose T0                           | 60.0               | 70.7           | 59.1              | +18.7 %             |
| sparkDash-method prose T0.7 thinking                | 53.7               | 53.4           | 47.7              | +5.3 %              |
| sparkDash-method code (task 0)                      | 94.5               | 118.6          | 93.0              | +26.5 %             |
| sparkDash-method code, 8 tasks                      | 89.2               | 108.5          | 88.2              | +22.3 %             |
| sparkDash-method structured                         | 109.0              | 164.1          | 107.7             | +51.5 %             |
| sparkDash 1.8.8 itself, c1 prose (median of 5)      | 59.4               | 70.2           | 59.0              | +18.6 %             |
| sparkDash 1.8.8 itself, c1 code (median of 5)       | 93.6               | 117.1          | 92.6              | +25.8 %             |
| sparkDash 1.8.8 itself, c1 structured (median of 5) | 107.8              | 162.9          | 106.4             | +52.1 %             |
| sparkDash 1.8.8 catalog prose (mean)                | 44.5               | 49.1           | 44.0              | +11.0 %             |
| sparkDash 1.8.8 catalog structural (mean)           | 67.0               | 77.3           | 66.2              | +16.1 %             |
| sparkDash 1.8.8 catalog code (mean)                 | 81.0               | 97.6           | 80.7              | +20.7 %             |
| prefill, 7.4K-token prompt (tok/s)                  | 2325               | 2561           | 2306              | +10.6 %             |
| prefill, 29K-token prompt (tok/s)                   | 2366               | 2564           | 2359              | +8.5 %              |
| prefill, 116K-token prompt (tok/s)                  | 2192               | 2455           | 2186              | +12.2 %             |
| rewrite-a-file set (24 rewrites), wall tok/s        | 140.0              | 227.3          | 138.2             | +63.4 %             |
| agent set (16 turns), wall tok/s                    | 56.4               | 64.3           | 55.7              | +14.7 %             |
| 56 rewrite requests, no `prediction` (wall tok/s)   | 105.06             | 152.26         | 103.81            | +45.8 %             |
| 56 rewrite requests, with `prediction`              | 134.09             | 265.11         | 132.57            | +98.8 %             |
|   numbered-lines rewrites (16), no prediction       | 77.62              | 95.20          | 76.77             | +23.3 %             |
|   numbered-lines rewrites (16), with prediction     | 140.62             | 295.94         | 139.05            | +111.6 %            |
| agent loop, 46 turns, no prediction (wall tok/s)    | 64.89              | 82.53          | 64.08             | +28.0 %             |
| agent loop, 46 turns, with prediction               | 73.58              | 96.80          | 72.62             | +32.4 %             |
| seeded sampled T0.7, 24 held chat prompts           | 46.32              | 51.09          | 45.85             | +10.9 %             |
| seeded sampled T0.7 with thinking, 24 prompts       | 49.80              | 54.41          | 49.26             | +9.9 %              |
| seeded sparkDash-method prose T0.7 thinking, 6      | 49.22              | 53.75          | 48.60             | +9.9 %              |
| 182K-token needle: first token (s)                  | 86.9               | 75.4           | -                 | -13.2 %             |
|   recall                                            | 3/3                | 3/3            | -                 |                     |
| 227K-token prompt: first token (s)                  | 112.9              | 94.5           | 113.9             | -16.7 %             |
|   prompt tokens                                     | 227621             | 227621         | 227621            |                     |
|   recall                                            | 3/3                | 3/3            | 3/3               |                     |
| image                                               | PASS               | PASS           | -                 |                     |
| qeval (4 in flight)                                 | 32/35              | 32/35          | -                 |                     |
| 3 users distinct, default profile (tok/s)           | 131.7              | 140.2          | -                 | +6.5 %              |
| lowest MemAvailable in the suite (MiB)              | 26000              | 8941           | 31108             |                     |
| highest GPU memory (MiB)                            | 91662              | 108911         | 87116             |                     |
| run start - end (UTC)                               | 04:49-05:36        | 06:01-06:42    | 07:07-07:51       |                     |
+-----------------------------------------------------+--------------------+----------------+-------------------+---------------------+

MANY USERS, distinct prompts, 512 tokens, total decode tok/s (first / last user's first token), median of the runs
+-------+------------------------------------------------------------+--------------------------------+----------------------------+
| users | MiaAI shipped config, first (16 x 64K, fp8 KV, levers off) | USERS16 profile (bf16 KV pool) | MiaAI shipped config, last |
+-------+------------------------------------------------------------+--------------------------------+----------------------------+
| 1     | 75.3 (0.21 / 0.21 s)                                       | 86.3 (0.14 / 0.14 s)           | 75.1 (0.24 / 0.24 s)       |
| 4     | 143.9 (0.19 / 1.24 s)                                      | 160.0 (0.14 / 0.39 s)          | 145.1 (0.35 / 1.14 s)      |
| 8     | 195.7 (0.21 / 2.92 s)                                      | 218.9 (0.13 / 0.64 s)          | 194.6 (0.35 / 2.77 s)      |
| 12    | 222.6 (0.21 / 4.80 s)                                      | 260.7 (0.13 / 0.72 s)          | 222.4 (0.20 / 4.82 s)      |
| 16    | 219.9 (0.20 / 6.75 s)                                      | 288.1 (0.14 / 0.92 s)          | 218.6 (0.21 / 6.79 s)      |
+-------+------------------------------------------------------------+--------------------------------+----------------------------+

AGENT TRACES (at.py): follow-up turn's first token
+------------+-------------+-----------------------------------+----------------------------------------+----------+------------------------+
| run        | requests ok | follow-ups resumed from the cache | follow-up first token median / max (s) | wall (s) | prompt tokens (median) |
+------------+-------------+-----------------------------------+----------------------------------------+----------+------------------------+
| agA_at10   | 30/30       | 13/20                             | 64.52 / 138.19                         | 376.8    | 53189                  |
| agA_at5    | 30/30       | 25/25                             | 1.29 / 3.76                            | 76.2     | 54704                  |
| agmia_at10 | 30/30       | 0/20                              | 199.76 / 326.28                        | 701.3    | 53204                  |
| agmia5_at5 | 30/30       | 15/25                             | 11.59 / 126.08                         | 373.0    | 54710                  |
+------------+-------------+-----------------------------------+----------------------------------------+----------+------------------------+

IDENTITY / CHECK LINES (chain logs, as printed on the box)
### step prep 04:35:17 left 7080 s
recipe.sh prepare rc 0 in 4 s; image before b052a3f158ad after b052a3f158ad UNCHANGED; recipe.sh sha 66fe0314fab8c18b; patches 29 sha 5249399d0d777385
### step u16s 04:35:22 left 7075 s
READY recipe [USERS16=1] in 211 s; MemAvailable 35 GiB; GPU MiB 80464; image tensorfold-qwen38:v1.1.4-eab8582 entrypoint /opt/nvidia/nvidia_entrypoint.sh
env knobs: 68 -> TF_FLASHNEXT_CHUNK=6144 TF_FLASHNEXT_Q38_CACHE_ELASTIC=2 TF_FLASHNEXT_Q38_CERT=1 TF_FLASHNEXT_Q38_HEAD2=/root/.cache/huggingface/q38/head2_r9.bin TF_FLASHNEXT_Q38_LANE_REUSE=5 TF_FLASHNEXT_Q38_POOL_CTX=40960 TF_FLASHNEXT_Q38_SNAP_DELTA=1 TF_FLASHNEXT_Q38_WINDOW_ROWS=32 
q38 qsa knobs: cert=1 cert_min=1 slack=100.0e-6 half=1 list=4096 rows=1 loop=0 stats=0
[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 77.53 GiB; 16 streams at once, 1.71 GiB each at a 262144-token window, of 28.0 GiB left after a 9.0 GiB reserve; prompts in passes of up to 6144 rows, a 0.7 GiB prompt cache; own glue kernels
[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 77.53 GiB; 16 streams at once, 1.71 GiB each at a 262144-token window, of 28.0 GiB left after a 9.0 GiB reserve; prompts in passes of up to 6144 rows, a 0.7 GiB prompt cache; own glue kernels
USERS16 STARTS with [USERS16=1 ]
### step ags 04:39:52 left 6805 s
READY recipe [AGENTS=1] in 199 s; MemAvailable 35 GiB; GPU MiB 80838; image tensorfold-qwen38:v1.1.4-eab8582 entrypoint /opt/nvidia/nvidia_entrypoint.sh
env knobs: 69 -> TF_FLASHNEXT_CHUNK=6144 TF_FLASHNEXT_Q38_CACHE_ELASTIC=2 TF_FLASHNEXT_Q38_CERT=1 TF_FLASHNEXT_Q38_HEAD2=/root/.cache/huggingface/q38/head2_r9.bin TF_FLASHNEXT_Q38_LANE_REUSE=5 TF_FLASHNEXT_Q38_POOL_CTX=131072 TF_FLASHNEXT_Q38_SNAP_DELTA=1 TF_FLASHNEXT_Q38_WINDOW_ROWS=128 
q38 qsa knobs: cert=1 cert_min=1 slack=100.0e-6 half=1 list=4096 rows=1 loop=0 stats=0
[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 77.53 GiB; 5 streams at once, 4.66 GiB each at a 262144-token window, of 27.4 GiB left after a 9.0 GiB reserve; prompts in passes of up to 6144 rows, a 4.1 GiB prompt cache; own glue kernels
[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 77.53 GiB; 5 streams at once, 4.66 GiB each at a 262144-token window, of 27.4 GiB left after a 9.0 GiB reserve; prompts in passes of up to 6144 rows, a 4.1 GiB prompt cache; own glue kernels
AGENTS STARTS with [AGENTS=1 ]
### step off 04:43:52 left 0 s
N_off SKIPPED: 0 s left, need 4800
off DID NOT RUN: chain stops 04:43:52
### step off 04:46:28 left 7079 s
READY recipe [CERT=0 MANY=0 COPY_ROWS=16 FUSED=0 HALF=0 MTP_LAST=0 GDN_MANY=0 BEST_LANES=0 ATTN_LANES=0 LATE_SKIP=0 KEEP_INPASS=0 LANE_REUSE=0 HEAD2=0 R9=0 R10=0 R11=0] in 178 s; MemAvailable 39 GiB; GPU MiB 78217; image tensorfold-qwen38:v1.1.4-eab8582 entrypoint /opt/nvidia/nvidia_entrypoint.sh
env knobs: 44 -> TF_FLASHNEXT_Q38_CACHE_ELASTIC=0 TF_FLASHNEXT_Q38_CERT=0 TF_FLASHNEXT_Q38_LANE_REUSE=0 TF_FLASHNEXT_Q38_WINDOW_ROWS=16 
[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 75.92 GiB; 3 streams at once, 7.99 GiB each at a 262144-token window, of 30.5 GiB left after a 9.0 GiB reserve; prompts in passes of up to 3072 rows, a 6.5 GiB prompt cache; own glue kernels
pred1 vs pred0: identical 56/56 (only in A 0, only in B 0)
loop1 vs loop0: identical 46/46 (only in A 0, only in B 0)
-- off vs the wrap-up run of the same image (F11_off)
ident_on.json F11_off vs N_off: identical 51/51  mean shared-prefix fraction 1.000  differ: []
held.jsonl F11_off vs N_off: identical 320/320  mean shared-prefix fraction 1.000  differ: []
edit.jsonl F11_off vs N_off: identical 24/24  mean shared-prefix fraction 1.000  differ: []
agent.jsonl F11_off vs N_off: identical 16/16  mean shared-prefix fraction 1.000  differ: []
sq07off: cmp: identical 24/24  A 45.13 -> B 46.32 tok/s (+2.64 %)  per-prompt ratio +2.68 % +- 0.12 (95 %)  identical-only +2.68 % n 23
sq07on: cmp: identical 24/24  A 48.32 -> B 49.80 tok/s (+3.06 %)  per-prompt ratio +3.05 % +- 0.10 (95 %)  identical-only +3.05 % n 24
sqsd: cmp: identical 6/6  A 47.78 -> B 49.22 tok/s (+3.02 %)  per-prompt ratio +3.03 % +- 0.15 (95 %)  identical-only +3.03 % n 6
pred0: identical 56/56 (only in A 0, only in B 0)
pred1: identical 56/56 (only in A 0, only in B 0)
loop0: identical 46/46 (only in A 0, only in B 0)
loop1: identical 46/46 (only in A 0, only in B 0)
### step on 05:57:41 left 6466 s
READY recipe [] in 202 s; MemAvailable 37 GiB; GPU MiB 80117; image tensorfold-qwen38:v1.1.4-eab8582 entrypoint /opt/nvidia/nvidia_entrypoint.sh
env knobs: 68 -> TF_FLASHNEXT_CHUNK=4608 TF_FLASHNEXT_Q38_CACHE_ELASTIC=1 TF_FLASHNEXT_Q38_CERT=1 TF_FLASHNEXT_Q38_HEAD2=/root/.cache/huggingface/q38/head2_r9.bin TF_FLASHNEXT_Q38_LANE_REUSE=3 TF_FLASHNEXT_Q38_SNAP_DELTA=1 TF_FLASHNEXT_Q38_WINDOW_ROWS=128 
q38 qsa knobs: cert=1 cert_min=1 slack=100.0e-6 half=1 list=4096 rows=1 loop=0 stats=0
[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 76.85 GiB; 3 streams at once, 8.39 GiB each at a 262144-token window, of 30.0 GiB left after a 9.0 GiB reserve; prompts in passes of up to 4608 rows, a 4.8 GiB prompt cache; own glue kernels
pred1 vs pred0: identical 56/56 (only in A 0, only in B 0)
loop1 vs loop0: identical 46/46 (only in A 0, only in B 0)
chat      no prediction   106.8 tok/s  with prediction   423.2 tok/s  texts identical 4/4  non-200 0
responses no prediction   105.8 tok/s  with prediction   424.5 tok/s  texts identical 4/4  non-200 0
messages  no prediction   105.7 tok/s  with prediction   423.9 tok/s  texts identical 4/4  non-200 0
q38 cert gather: calls 20000 tiles/call 12.6
q38 cert: calls 20000 rows 87562 whole-head 0 tiles/call 16.6 runs/call 8.8 checked-rows 0 best-mismatch 0 value-mismatch 0 held 0
q38 cert gather: calls 20000 tiles/call 12.6
q38 cert topk: wide passes not tried 25
q38 cert topk: calls 516 whole-head 0 tiles/call 288.8 runs/call 49.6 topk-rows-checked 0 topk-mismatch 0 bound-violations 0 of 0 K entries
q38 cert gather: calls 25000 tiles/call 11.8
q38 cert gather: calls 30000 tiles/call 11.2
q38 cert gather: calls 35000 tiles/call 10.8
q38 cert gather: calls 40000 tiles/call 10.8
q38 cert: calls 40000 rows 236713 whole-head 0 tiles/call 13.0 runs/call 8.1 checked-rows 0 best-mismatch 0 value-mismatch 0 held 0
q38 cert gather: calls 40000 tiles/call 10.8
q38 cert topk: calls 516 whole-head 0 tiles/call 288.8 runs/call 49.6 topk-rows-checked 0 topk-mismatch 0 bound-violations 0 of 0 K entries
q38 cert gather: calls 45000 tiles/call 26.8
q38 cert gather: calls 50000 tiles/call 26.1
-- on vs off (same window)
ident_on.json N_off vs N_on: identical 51/51  mean shared-prefix fraction 1.000  differ: []
held.jsonl N_off vs N_on: identical 320/320  mean shared-prefix fraction 1.000  differ: []
edit.jsonl N_off vs N_on: identical 24/24  mean shared-prefix fraction 1.000  differ: []
agent.jsonl N_off vs N_on: identical 16/16  mean shared-prefix fraction 1.000  differ: []
sq07off: cmp: identical 24/24  A 46.32 -> B 51.09 tok/s (+10.29 %)  per-prompt ratio +9.57 % +- 1.22 (95 %)  identical-only +9.57 % n 23
sq07on: cmp: identical 24/24  A 49.80 -> B 54.41 tok/s (+9.26 %)  per-prompt ratio +9.32 % +- 0.87 (95 %)  identical-only +9.32 % n 24
sqsd: cmp: identical 6/6  A 49.22 -> B 53.75 tok/s (+9.21 %)  per-prompt ratio +9.29 % +- 0.94 (95 %)  identical-only +9.29 % n 6
pred0: identical 56/56 (only in A 0, only in B 0)
pred1: identical 56/56 (only in A 0, only in B 0)
loop0: identical 46/46 (only in A 0, only in B 0)
loop1: identical 46/46 (only in A 0, only in B 0)
### step off2 07:03:46 left 6975 s
READY recipe [CERT=0 MANY=0 COPY_ROWS=16 FUSED=0 HALF=0 MTP_LAST=0 GDN_MANY=0 BEST_LANES=0 ATTN_LANES=0 LATE_SKIP=0 KEEP_INPASS=0 LANE_REUSE=0 HEAD2=0 R9=0 R10=0 R11=0] in 215 s; MemAvailable 39 GiB; GPU MiB 78217; image tensorfold-qwen38:v1.1.4-eab8582 entrypoint /opt/nvidia/nvidia_entrypoint.sh
env knobs: 44 -> TF_FLASHNEXT_Q38_CACHE_ELASTIC=0 TF_FLASHNEXT_Q38_CERT=0 TF_FLASHNEXT_Q38_LANE_REUSE=0 TF_FLASHNEXT_Q38_WINDOW_ROWS=16 
[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 75.92 GiB; 3 streams at once, 7.99 GiB each at a 262144-token window, of 31.0 GiB left after a 9.0 GiB reserve; prompts in passes of up to 3072 rows, a 7.1 GiB prompt cache; own glue kernels
pred1 vs pred0: identical 56/56 (only in A 0, only in B 0)
loop1 vs loop0: identical 46/46 (only in A 0, only in B 0)
-- off2 vs off
ident_on.json N_off vs N_off2: identical 51/51  mean shared-prefix fraction 1.000  differ: []
held.jsonl N_off vs N_off2: identical 320/320  mean shared-prefix fraction 1.000  differ: []
edit.jsonl N_off vs N_off2: identical 24/24  mean shared-prefix fraction 1.000  differ: []
agent.jsonl N_off vs N_off2: identical 16/16  mean shared-prefix fraction 1.000  differ: []
sq07off: cmp: identical 24/24  A 46.32 -> B 45.85 tok/s (-1.01 %)  per-prompt ratio -1.02 % +- 0.12 (95 %)  identical-only -1.02 % n 23
sq07on: cmp: identical 24/24  A 49.80 -> B 49.26 tok/s (-1.09 %)  per-prompt ratio -1.08 % +- 0.11 (95 %)  identical-only -1.08 % n 24
sqsd: cmp: identical 6/6  A 49.22 -> B 48.60 tok/s (-1.25 %)  per-prompt ratio -1.24 % +- 0.20 (95 %)  identical-only -1.24 % n 6
pred0: identical 56/56 (only in A 0, only in B 0)
pred1: identical 56/56 (only in A 0, only in B 0)
loop0: identical 46/46 (only in A 0, only in B 0)
loop1: identical 46/46 (only in A 0, only in B 0)
-- on vs off2
ident_on.json N_off2 vs N_on: identical 51/51  mean shared-prefix fraction 1.000  differ: []
held.jsonl N_off2 vs N_on: identical 320/320  mean shared-prefix fraction 1.000  differ: []
edit.jsonl N_off2 vs N_on: identical 24/24  mean shared-prefix fraction 1.000  differ: []
agent.jsonl N_off2 vs N_on: identical 16/16  mean shared-prefix fraction 1.000  differ: []
sq07off: cmp: identical 24/24  A 45.85 -> B 51.09 tok/s (+11.42 %)  per-prompt ratio +10.70 % +- 1.24 (95 %)  identical-only +10.70 % n 23
sq07on: cmp: identical 24/24  A 49.26 -> B 54.41 tok/s (+10.47 %)  per-prompt ratio +10.52 % +- 0.90 (95 %)  identical-only +10.52 % n 24
sqsd: cmp: identical 6/6  A 48.60 -> B 53.75 tok/s (+10.59 %)  per-prompt ratio +10.66 % +- 0.89 (95 %)  identical-only +10.66 % n 6
pred0: identical 56/56 (only in A 0, only in B 0)
pred1: identical 56/56 (only in A 0, only in B 0)
loop0: identical 46/46 (only in A 0, only in B 0)
loop1: identical 46/46 (only in A 0, only in B 0)
### step chk 08:12:45 left 2836 s
READY recipe [CERT=2 QSA_CERT=2 TF_FLASHNEXT_Q38_QSA_STATS=1 GDN_PROD=12 TF_FLASHNEXT_Q38_ROUTE_BAND_CHECK=1 TF_FLASHNEXT_Q38_SCRATCH_CHECK=1 TF_FLASHNEXT_Q38_HC_DN_TALL_CHECK=1 SNAP_DELTA=2 TF_FLASHNEXT_Q38_PLOG=1] in 226 s; MemAvailable 36 GiB; GPU MiB 80419; image pico-dgx-qwen38-tf:v1.1.4-eab858
env knobs: 73 -> TF_FLASHNEXT_CHUNK=4608 TF_FLASHNEXT_Q38_CACHE_ELASTIC=1 TF_FLASHNEXT_Q38_CERT=2 TF_FLASHNEXT_Q38_HEAD2=/root/.cache/huggingface/q38/head2_r9.bin TF_FLASHNEXT_Q38_LANE_REUSE=3 TF_FLASHNEXT_Q38_SNAP_DELTA=2 TF_FLASHNEXT_Q38_WINDOW_ROWS=128 
q38 qsa knobs: cert=2 cert_min=1 slack=100.0e-6 half=1 list=4096 rows=1 loop=0 stats=1
[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 76.91 GiB; 3 streams at once, 8.39 GiB each at a 262144-token window, of 29.3 GiB left after a 9.0 GiB reserve; prompts in passes of up to 4608 rows, a 4.1 GiB prompt cache; own glue kernels
q38 cert: calls 1000 rows 5698 whole-head 0 tiles/call 186.1 runs/call 31.7 checked-rows 5698 best-mismatch 0 value-mismatch 0 held 27573120
q38 cert: calls 2000 rows 9418 whole-head 0 tiles/call 186.9 runs/call 32.7 checked-rows 9418 best-mismatch 0 value-mismatch 0 held 52718272
q38 cert: calls 3000 rows 13876 whole-head 0 tiles/call 127.5 runs/call 24.2 checked-rows 13876 best-mismatch 0 value-mismatch 0 held 55271616
q38 cert: calls 4000 rows 20043 whole-head 0 tiles/call 97.6 runs/call 19.8 checked-rows 20043 best-mismatch 0 value-mismatch 0 held 58569536
q38 cert: calls 5000 rows 24753 whole-head 0 tiles/call 79.6 runs/call 17.1 checked-rows 24753 best-mismatch 0 value-mismatch 0 held 60923072
q38 cert gather: calls 5000 tiles/call 45.1
q38 cert: SM8 copy of the head made (625 MiB), mean residual 0.0087 of a row's norm, check 1, top-k up to 20
q38 cert topk: calls 515 whole-head 0 tiles/call 354.3 runs/call 56.3 topk-rows-checked 1816 topk-mismatch 0 bound-violations 0 of 1276352 K entries
q38 cert topk: calls 994 whole-head 0 tiles/call 367.6 runs/call 59.1 topk-rows-checked 3457 topk-mismatch 0 bound-violations 0 of 5544672 K entries
q38 cert topk: wide passes not tried 10
q38 gdnprod check launches=6120 rows_last=972 words=74198370816 diff=0
q38 qsacert first rows=6657316 between=170899152 fallback=0 max_between=93 redone=0 checked=6657316 list_diff=0 exact_bit_diff=0 bound_viol=0 max_ratio=0.64728 listed=5040698 list_miss=0
q38 qsa knobs: cert=2 cert_min=1 slack=100.0e-6 half=1 list=4096 rows=1 loop=0 stats=1
q38 routeband check passes=5808 rows_last=2235 words=13447117152 diff=0
q38 scratch check passes=5760 cap=4608 full_rows=3072 bad=0
q38 scratch trim: slices kept for 3072 rows, head_local for 256 (check mode: guard words after every trimmed buffer)
q38 snapdelta keep at=974 from=2 own_kib=142152 whole_kib=142211 chain=2 orphan_mib=255 keeps=124 shared=119 kept_mib=30969 whole_mib=37757 check_bytes=25252078592 check_diff=0
mismatch-like lines: 10
chk ident vs N_on: identical 51/51  mean shared-prefix fraction 1.000  differ: []
chk ident vs N_off: identical 51/51  mean shared-prefix fraction 1.000  differ: []
chk seeded sampled (12) vs on: cmp: identical 12/12  A 52.44 -> B 42.61 tok/s (-18.76 %)  per-prompt ratio -18.85 % +- 0.29 (95 %)  identical-only -18.85 % n 11
chk rewrite(held) vs on: identical 28/28 (only in A 28, only in B 0)
### step bench 08:31:27 left 1714 s
READY recipe [] in 220 s; MemAvailable 36 GiB; GPU MiB 80117; image pico-dgx-qwen38-tf:v1.1.4-eab8582 entrypoint /opt/nvidia/nvidia_entrypoint.sh
env knobs: 68 -> TF_FLASHNEXT_CHUNK=4608 TF_FLASHNEXT_Q38_CACHE_ELASTIC=1 TF_FLASHNEXT_Q38_CERT=1 TF_FLASHNEXT_Q38_HEAD2=/root/.cache/huggingface/q38/head2_r9.bin TF_FLASHNEXT_Q38_LANE_REUSE=3 TF_FLASHNEXT_Q38_SNAP_DELTA=1 TF_FLASHNEXT_Q38_WINDOW_ROWS=128 
q38 qsa knobs: cert=1 cert_min=1 slack=100.0e-6 half=1 list=4096 rows=1 loop=0 stats=0
[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 76.85 GiB; 3 streams at once, 8.39 GiB each at a 262144-token window, of 28.8 GiB left after a 9.0 GiB reserve; prompts in passes of up to 4608 rows, a 3.6 GiB prompt cache; own glue kernels
### step prep 09:02:02 left 7080 s
recipe.sh prepare rc 0 in 6 s; image before b052a3f158ad after 14204217aee6 CHANGED; recipe.sh sha a422375ed1611ba6; patches 29 sha 5249399d0d777385
### step miafp 09:02:09 left 7065 s
READY recipe [CERT=0 MANY=0 COPY_ROWS=16 FUSED=0 HALF=0 MTP_LAST=0 GDN_MANY=0 BEST_LANES=0 ATTN_LANES=0 LATE_SKIP=0 KEEP_INPASS=0 LANE_REUSE=0 HEAD2=0 R9=0 R10=0 R11=0 KV=fp8 PARALLEL=16 CONTEXT=65536 TF_FLASHNEXT_PRODUCT_STREAMS=16] in 214 s; MemAvailable 38 GiB; GPU MiB 78169; image pico-dgx-qwen3
env knobs: 44 -> TF_FLASHNEXT_Q38_CACHE_ELASTIC=0 TF_FLASHNEXT_Q38_CERT=0 TF_FLASHNEXT_Q38_LANE_REUSE=0 TF_FLASHNEXT_Q38_WINDOW_ROWS=16 
[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 75.88 GiB; 16 streams at once, 1.57 GiB each at a 65536-token window (fp8 caches), of 29.4 GiB left after a 9.0 GiB reserve; prompts in passes of up to 3072 rows, a 4.3 GiB prompt cache; own glue kernels
[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 75.88 GiB; 16 streams at once, 1.57 GiB each at a 65536-token window (fp8 caches), of 29.4 GiB left after a 9.0 GiB reserve; prompts in passes of up to 3072 rows, a 4.3 GiB prompt cache; own glue kernels
### step mu 09:18:06 left 7031 s
READY recipe [USERS16=1] in 214 s; MemAvailable 35 GiB; GPU MiB 80464; image pico-dgx-qwen38-tf:v1.1.4-eab8582 entrypoint /opt/nvidia/nvidia_entrypoint.sh
env knobs: 68 -> TF_FLASHNEXT_CHUNK=6144 TF_FLASHNEXT_Q38_CACHE_ELASTIC=2 TF_FLASHNEXT_Q38_CERT=1 TF_FLASHNEXT_Q38_HEAD2=/root/.cache/huggingface/q38/head2_r9.bin TF_FLASHNEXT_Q38_LANE_REUSE=5 TF_FLASHNEXT_Q38_POOL_CTX=32768 TF_FLASHNEXT_Q38_SNAP_DELTA=1 TF_FLASHNEXT_Q38_WINDOW_ROWS=32 
q38 qsa knobs: cert=1 cert_min=1 slack=100.0e-6 half=1 list=4096 rows=1 loop=0 stats=0
[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 77.53 GiB; 16 streams at once, 1.48 GiB each at a 262144-token window, of 28.0 GiB left after a 9.0 GiB reserve; prompts in passes of up to 6144 rows, a 4.4 GiB prompt cache; own glue kernels
q38 cert: SM8 copy of the head made (625 MiB), mean residual 0.0087 of a row's norm, check 0, top-k up to 20
q38 scratch trim: slices kept for 3072 rows, head_local for 256
q38 qsa knobs: cert=1 cert_min=1 slack=100.0e-6 half=1 list=4096 rows=1 loop=0 stats=0
[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 77.53 GiB; 16 streams at once, 1.48 GiB each at a 262144-token window, of 28.0 GiB left after a 9.0 GiB reserve; prompts in passes of up to 6144 rows, a 4.4 GiB prompt cache; o
q38 cert gather: calls 5000 tiles/call 61.6
[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 77.53 GiB; 16 streams at once, 1.48 GiB each at a 262144-token window, of 28.0 GiB left after a 9.0 GiB reserve; prompts in passes of up to 6144 rows, a 4.4 GiB prompt cache; own glue kernels
ALL 320 held, 16 in flight (USERS16) vs SOLO levers-off (N_off, window 1): identical 320/320  mean shared-prefix fraction 1.000  differ: []
ALL 320 held, 16 in flight (USERS16) vs solo recipe default (N_on): identical 320/320  mean shared-prefix fraction 1.000  differ: []
identity set on USERS16 (one at a time) vs solo levers-off (N_off): identical 51/51  mean shared-prefix fraction 1.000  differ: []
### step miafp2 09:45:54 left 6279 s
READY recipe [CERT=0 MANY=0 COPY_ROWS=16 FUSED=0 HALF=0 MTP_LAST=0 GDN_MANY=0 BEST_LANES=0 ATTN_LANES=0 LATE_SKIP=0 KEEP_INPASS=0 LANE_REUSE=0 HEAD2=0 R9=0 R10=0 R11=0 KV=fp8 PARALLEL=16 CONTEXT=65536 TF_FLASHNEXT_PRODUCT_STREAMS=16] in 214 s; MemAvailable 39 GiB; GPU MiB 78169; image pico-dgx-qwen3
env knobs: 44 -> TF_FLASHNEXT_Q38_CACHE_ELASTIC=0 TF_FLASHNEXT_Q38_CERT=0 TF_FLASHNEXT_Q38_LANE_REUSE=0 TF_FLASHNEXT_Q38_WINDOW_ROWS=16 
[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 75.88 GiB; 16 streams at once, 1.57 GiB each at a 65536-token window (fp8 caches), of 30.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 3072 rows, a 5.8 GiB prompt cache; own glue kernels
[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 75.88 GiB; 16 streams at once, 1.57 GiB each at a 65536-token window (fp8 caches), of 30.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 3072 rows, a 5.8 GiB prompt cache; own glue kernels
### step agA 09:54:59 left 6648 s
READY recipe [AGENTS=1] in 220 s; MemAvailable 36 GiB; GPU MiB 80838; image pico-dgx-qwen38-tf:v1.1.4-eab8582 entrypoint /opt/nvidia/nvidia_entrypoint.sh
env knobs: 69 -> TF_FLASHNEXT_CHUNK=6144 TF_FLASHNEXT_Q38_CACHE_ELASTIC=2 TF_FLASHNEXT_Q38_CERT=1 TF_FLASHNEXT_Q38_HEAD2=/root/.cache/huggingface/q38/head2_r9.bin TF_FLASHNEXT_Q38_LANE_REUSE=5 TF_FLASHNEXT_Q38_POOL_CTX=131072 TF_FLASHNEXT_Q38_SNAP_DELTA=1 TF_FLASHNEXT_Q38_WINDOW_ROWS=128 
q38 qsa knobs: cert=1 cert_min=1 slack=100.0e-6 half=1 list=4096 rows=1 loop=0 stats=0
[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 77.53 GiB; 5 streams at once, 4.66 GiB each at a 262144-token window, of 29.2 GiB left after a 9.0 GiB reserve; prompts in passes of up to 6144 rows, a 5.9 GiB prompt cache; own glue kernels
[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 77.53 GiB; 5 streams at once, 4.66 GiB each at a 262144-token window, of 29.2 GiB left after a 9.0 GiB reserve; prompts in passes of up to 6144 rows, a 5.9 GiB prompt cache; own glue kernels
### step agmia 10:06:15 left 6887 s
READY recipe [CERT=0 MANY=0 COPY_ROWS=16 FUSED=0 HALF=0 MTP_LAST=0 GDN_MANY=0 BEST_LANES=0 ATTN_LANES=0 LATE_SKIP=0 KEEP_INPASS=0 LANE_REUSE=0 HEAD2=0 R9=0 R10=0 R11=0 KV=fp8] in 218 s; MemAvailable 39 GiB; GPU MiB 78217; image pico-dgx-qwen38-tf:v1.1.4-eab8582 entrypoint /opt/nvidia/nvidia_entrypoi
env knobs: 44 -> TF_FLASHNEXT_Q38_CACHE_ELASTIC=0 TF_FLASHNEXT_Q38_CERT=0 TF_FLASHNEXT_Q38_LANE_REUSE=0 TF_FLASHNEXT_Q38_WINDOW_ROWS=16 
[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 75.92 GiB; 5 streams at once, 4.85 GiB each at a 262144-token window (fp8 caches), of 30.6 GiB left after a 9.0 GiB reserve; prompts in passes of up to 3072 rows, a 6.4 GiB prompt cache; own glue kernels
[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 75.92 GiB; 5 streams at once, 4.85 GiB each at a 262144-token window (fp8 caches), of 30.6 GiB left after a 9.0 GiB reserve; prompts in passes of up to 3072 rows, a 6.4 GiB prompt cache; own glue kernels
### step offs 10:21:36 left 6024 s
READY recipe [CERT=0 MANY=0 COPY_ROWS=16 FUSED=0 HALF=0 MTP_LAST=0 GDN_MANY=0 BEST_LANES=0 ATTN_LANES=0 LATE_SKIP=0 KEEP_INPASS=0 LANE_REUSE=0 HEAD2=0 R9=0 R10=0 R11=0] in 206 s; MemAvailable 39 GiB; GPU MiB 78217; image pico-dgx-qwen38-tf:v1.1.4-eab8582 entrypoint /opt/nvidia/nvidia_entrypoint.sh
env knobs: 44 -> TF_FLASHNEXT_Q38_CACHE_ELASTIC=0 TF_FLASHNEXT_Q38_CERT=0 TF_FLASHNEXT_Q38_LANE_REUSE=0 TF_FLASHNEXT_Q38_WINDOW_ROWS=16 
[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 75.92 GiB; 3 streams at once, 7.99 GiB each at a 262144-token window, of 30.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 3072 rows, a 6.9 GiB prompt cache; own glue kernels
seeded sampled held/3: 16 in flight (USERS16, N_mu) vs SOLO levers-off: identical 107/107  mean shared-prefix fraction 1.000  differ: []
### step bench 10:31:30 left 5430 s
READY recipe [] in 193 s; MemAvailable 37 GiB; GPU MiB 80117; image pico-dgx-qwen38-tf:v1.1.4-eab8582 entrypoint /opt/nvidia/nvidia_entrypoint.sh
env knobs: 68 -> TF_FLASHNEXT_CHUNK=4608 TF_FLASHNEXT_Q38_CACHE_ELASTIC=1 TF_FLASHNEXT_Q38_CERT=1 TF_FLASHNEXT_Q38_HEAD2=/root/.cache/huggingface/q38/head2_r9.bin TF_FLASHNEXT_Q38_LANE_REUSE=3 TF_FLASHNEXT_Q38_SNAP_DELTA=1 TF_FLASHNEXT_Q38_WINDOW_ROWS=128 
q38 qsa knobs: cert=1 cert_min=1 slack=100.0e-6 half=1 list=4096 rows=1 loop=0 stats=0
[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 76.85 GiB; 3 streams at once, 8.39 GiB each at a 262144-token window, of 30.0 GiB left after a 9.0 GiB reserve; prompts in passes of up to 4608 rows, a 4.8 GiB prompt cache; own glue kernels
### step agmia5 10:50:40 left 4280 s
READY recipe [CERT=0 MANY=0 COPY_ROWS=16 FUSED=0 HALF=0 MTP_LAST=0 GDN_MANY=0 BEST_LANES=0 ATTN_LANES=0 LATE_SKIP=0 KEEP_INPASS=0 LANE_REUSE=0 HEAD2=0 R9=0 R10=0 R11=0 KV=fp8] in 208 s; MemAvailable 39 GiB; GPU MiB 78217; image pico-dgx-qwen38-tf:v1.1.4-eab8582 entrypoint /opt/nvidia/nvidia_entrypoi
env knobs: 44 -> TF_FLASHNEXT_Q38_CACHE_ELASTIC=0 TF_FLASHNEXT_Q38_CERT=0 TF_FLASHNEXT_Q38_LANE_REUSE=0 TF_FLASHNEXT_Q38_WINDOW_ROWS=16 
[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 75.92 GiB; 5 streams at once, 4.85 GiB each at a 262144-token window (fp8 caches), of 30.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 3072 rows, a 6.6 GiB prompt cache; own glue kernels
[tensorfold] CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 75.92 GiB; 5 streams at once, 4.85 GiB each at a 262144-token window (fp8 caches), of 30.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 3072 rows, a 6.6 GiB prompt cache; own glue kernels
```

## Raw result files

On spark-923b under `~/q38/runs/`, copied to the Mac `q38/results/n14/`. Logs, counters and per-run JSON are in `raw/` beside this note;
the reply texts of the big sets (held.jsonl, ident_on.json, edit / agent / pred / loop / sq*.jsonl, at*.json) are not in the repo, their names are listed here.

```
./N_agA/at10.json                                2775665 bytes
./N_agA/at5.json                                 1462620 bytes
./N_agA/engine.log                                 34322 bytes
./N_agA/env.txt                                     2151 bytes
./N_ag_args                                           10 bytes
./N_agmia/at10.json                              2777308 bytes
./N_agmia/engine.log                               19137 bytes
./N_agmia/env.txt                                   1348 bytes
./N_agmia5/at5.json                              1464451 bytes
./N_agmia5/engine.log                              19134 bytes
./N_agmia5/env.txt                                  1348 bytes
./N_ags/at2.json                                  209029 bytes
./N_ags/engine.log                                 14350 bytes
./N_ags/env.txt                                     2151 bytes
./N_ags/ok                                             0 bytes
./N_ags/users.jsonl                                  488 bytes
./N_bench/bench.log                                 2271 bytes
./N_bench/engine.log                              122892 bytes
./N_bench/env.txt                                   2118 bytes
./N_chk/at2.json                                  240926 bytes
./N_chk/engine.log                                212561 bytes
./N_chk/engine_chk.log                            212561 bytes
./N_chk/env.txt                                     2277 bytes
./N_chk/ident_on.json                              55428 bytes
./N_chk/pred1.jsonl                               228346 bytes
./N_chk/sq07off.jsonl                              11445 bytes
./N_miafp/engine.log                              201462 bytes
./N_miafp/env.txt                                   1349 bytes
./N_miafp/held16.jsonl                             96851 bytes
./N_miafp/users.jsonl                              10840 bytes
./N_miafp/waves16.jsonl                           131998 bytes
./N_miafp/waves4.jsonl                             40783 bytes
./N_miafp2/engine.log                              64046 bytes
./N_miafp2/env.txt                                  1349 bytes
./N_miafp2/users.jsonl                             10834 bytes
./N_mu/engine.log                                 435854 bytes
./N_mu/env.txt                                      2117 bytes
./N_mu/held16.jsonl                                97210 bytes
./N_mu/held16f.jsonl                              387113 bytes
./N_mu/helds16.jsonl                               97741 bytes
./N_mu/ident_on.json                               55426 bytes
./N_mu/users.jsonl                                 10844 bytes
./N_mu/waves16.jsonl                              131653 bytes
./N_mu/waves4.jsonl                                40788 bytes
./N_off/agent.jsonl                                12299 bytes
./N_off/edit.jsonl                                 81024 bytes
./N_off/engine.log                                493622 bytes
./N_off/env.txt                                     1348 bytes
./N_off/held.jsonl                                386929 bytes
./N_off/ident_on.json                              55417 bytes
./N_off/loop0.jsonl                               456742 bytes
./N_off/loop1.jsonl                               456714 bytes
./N_off/mem.log                                    20757 bytes
./N_off/pred0.jsonl                               469523 bytes
./N_off/pred1.jsonl                               469461 bytes
./N_off/qeval-N_off.json                           53539 bytes
./N_off/sdrun.log                                   1751 bytes
./N_off/sq07off.jsonl                              24519 bytes
./N_off/sq07on.jsonl                               28295 bytes
./N_off/sqsd.jsonl                                  7892 bytes
./N_off/suite.log                                   3845 bytes
./N_off2/agent.jsonl                               12301 bytes
./N_off2/edit.jsonl                                81025 bytes
./N_off2/engine.log                               456337 bytes
./N_off2/env.txt                                    1348 bytes
./N_off2/held.jsonl                               386936 bytes
./N_off2/ident_on.json                             55424 bytes
./N_off2/loop0.jsonl                              456747 bytes
./N_off2/loop1.jsonl                              456715 bytes
./N_off2/mem.log                                   19499 bytes
./N_off2/pred0.jsonl                              469527 bytes
./N_off2/pred1.jsonl                              469459 bytes
./N_off2/sdrun.log                                  1751 bytes
./N_off2/sq07off.jsonl                             24524 bytes
./N_off2/sq07on.jsonl                              28294 bytes
./N_off2/sqsd.jsonl                                 7890 bytes
./N_off2/suite.log                                  3428 bytes
./N_offs/engine.log                                55023 bytes
./N_offs/env.txt                                    1348 bytes
./N_offs/helds1.jsonl                              97680 bytes
./N_on/agent.jsonl                                 12304 bytes
./N_on/edit.jsonl                                  81021 bytes
./N_on/engine.log                                 530643 bytes
./N_on/env.txt                                      2118 bytes
./N_on/held.jsonl                                 386948 bytes
./N_on/ident_on.json                               55419 bytes
./N_on/loop0.jsonl                                456743 bytes
./N_on/loop1.jsonl                                456712 bytes
./N_on/mem.log                                     18442 bytes
./N_on/pred0.jsonl                                469567 bytes
./N_on/pred1.jsonl                                469586 bytes
./N_on/qeval-N_on.json                             53546 bytes
./N_on/rt.json                                     74199 bytes
./N_on/sdrun.log                                    1742 bytes
./N_on/sq07off.jsonl                               24515 bytes
./N_on/sq07on.jsonl                                28292 bytes
./N_on/sqsd.jsonl                                   7892 bytes
./N_on/suite.log                                    3855 bytes
./N_prepare.log                                     6689 bytes
./N_u16_args                                          10 bytes
./N_u16s/engine.log                                24957 bytes
./N_u16s/env.txt                                    2117 bytes
./N_u16s/ok                                            0 bytes
./N_u16s/users.jsonl                                2598 bytes
./tables.txt                                       33079 bytes
./vals.json                                        12692 bytes
./wN1.log                                           5206 bytes
./wN1b.log                                         22202 bytes
./wN1c.log                                          1264 bytes
./wN2.log                                          19219 bytes
./wN3.log                                           2184 bytes
```
