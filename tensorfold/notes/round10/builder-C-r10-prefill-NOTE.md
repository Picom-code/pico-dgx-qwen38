# Builder C, round 10: long-prompt prefill and first token (one cumulative patch against integ9 7061396)

Patch: `C-r10.patch` = `git diff 7061396 HEAD` of Mac `q38/wt/C10` (branch r10-C, HEAD 793d4ee), sha256 bbc97f5555304f31, 1,174 lines,
12 files; `q38/mkhandoffC10.sh` printed TREE_EQUAL (base + patch = tree 3af61b4e). Binary BUILT AND MEASURED from exactly this tree:
C24, sha 6d4d3460d89d453e (spark-4e5c `~/q38/tfbin/C24`). Build: `./shipC10.sh NAME 16 24g`.
Everything below is MEASURED on spark-4e5c unless it says ESTIMATE; one shared window (2026-11-14 04:37-08:15 UTC), fresh engines, env
flags, recipe one-user profile (5 x 262,144, fp8 KV, vision on, recipe R9=1 environment; this box has no HEAD2 file, so the draft
side differs from 923b; prompt passes do not). Full log: `q38/PROGRESS-C.md` (Round 10). Raw logs: Mac `q38/results/rC10.tgz`.

## 0. FIRST: a bug fix in round 9's patch 0116 (not a flag; changes what the shipped recipe runs)
`cuda_knobs.zig q38EnvInt` kept its value in a struct that is ONE type for every call, so QSA_CERT, QSA_CERT_MIN, QSA_SLACK, QSA_HALF,
QSA_STATS and QSA_LOOP shared one cache and each returned the first one's value. With the recipe's QSA_CERT=1 the engine ran slack 1
millionth (designed: 100), CERT_MIN 1 (designed: 64; decode rows took the certified path), QSA_STATS on. Details and proof:
`handoff/C/URGENT-r9-knob-cache.md`. The fix: the struct names its variable. The engine now prints what it reads:
`q38 qsa knobs: cert=1 cert_min=64 slack=100.0e-6 half=1 list=4096 rows=1 loop=0 stats=1`.
No output changes (C22/C23/C24 with every new flag off against the base binary: ident 51/51, all texts, score hashes 74/74 + 74/74).
Cost of the designed margin: more blocks take the exact sum (27 a row instead of 12.6): 109K 44.28 -> 44.3-44.6 s, 214K 91.03 ->
91.2-91.8 s (+0.1..+0.8 %). `TF_FLASHNEXT_Q38_QSA_SLACK=32` gets most of that back (18 a row; -0.5 % of a long prompt against 100) and
is still 2x over the pessimistic worst case of the two fp32 sums (~1.6e-5 x sum|q||k|) and 160x over the largest deviation measured.
PROPOSED recipe: QSA_SLACK=32. (Code default stays 100.)

## 1. Flags (all default OFF in the patch; proposed recipe defaults in brackets)
| flag | what | proposed |
| `TF_FLASHNEXT_Q38_QSA_LIST=1` | certified select without barrier passes: 12-bit histogram pass, list pass (a warp reserves list slots with one atomic add), cut / classification / exact sums on the ~900 listed blocks, placement through a bitmap prefix count. Rows of >= 4,096 blocks (16K tokens); any miss takes the four-pass body | on |
| `TF_FLASHNEXT_Q38_QSA_ROWS=1` | coarse scores' launch cut by rows: a block takes 64 rows x 256 key blocks (queries loaded once) instead of 256 rows x 64 key blocks; the same mma in the same order | on |
| `TF_FLASHNEXT_Q38_HEAD_KV=512` (needs MTP_LAST) | the MTP draft head's prompt rows, except each call's last 512, stop after their cache writes (no block scores / selection / attention / gate / out projection / write-back: nothing read them) | on |
| `TF_FLASHNEXT_Q38_ARENA=1` (needs CACHE_ELASTIC; `_ARENA_MIB=N` to size it) | the lanes' + prompt cache's planned room taken ONCE at start-up (29.3 GiB, 4.3 s) and served through memory.zig's sub-allocator hook: best fit, merge on free, zero fill on hand-out (not for kept states). A kept state is never placed outside it: the cache evicts until it has a gap (`placeable` hook in core/prompt_cache.zig) | on for the one-user profile; SEE CAVEAT |
| `TF_FLASHNEXT_Q38_WARM=3072` | one made-up 3,072-row prompt pass on a throw-away sequence at start-up (1.2-1.4 s) | on |
Measuring aids, off: `TF_FLASHNEXT_Q38_HHASH=1` (hash of the head's state after each prompt pass), knob-file words `tile_pairs`,
`tile_gu`, `tile_down` (prompt expert tile), `qsa_list`, `qsa_rows`.

## 2. Speed: first-token time of a cold prompt, fresh engines, same prompts, reference first and last
| prompt tokens | off first (C22) | off last (C23) | ARENA+LIST+ROWS+WARM (C23) | + HEAD_KV=512 (C24, the handoff binary) | gain of the full set vs mean off |
| 7,276   | 3.10 s 2347 tok/s | 3.10 s 2345 | 2.92 s 2494 | 2.87 s 2538 | +8.0 % |
| 27,502  | 10.67 s 2576 | 10.64 s 2584 | 10.56 s 2604 | 10.39 s 2647 | +2.6 % |
| 109,142 | 44.62 s 2446 | 44.31 s 2463 | 43.64 s 2501 | 42.90 s 2544 | +3.6 % |
| 214,471 | 91.78 s 2337 | 91.15 s 2353 | 88.23 s 2431 | 86.79 s 2471 | +5.4 % |
| short new prompt (19 tokens), first token | 99.7 ms | 99.9 ms | 96.7 ms | 96.9 ms | -3 % |
"off" = this patch with every new flag off (knob fix in, slack 100). Against the base binary as shipped (C20, slack 1 through the bug:
3.14 / 10.67 / 44.28 / 91.03 s): +9.4 / +2.7 / +3.2 / +4.9 %. With QSA_SLACK=32 on top (measured on the C23 set: 2.89 / 10.50 / 43.39 /
87.82 s): about another +0.5 % at 109K-214K (ESTIMATE for the full set).
By lever (full 3,072-row chunks, PLOG walls):
| lever | 109K | 214K | where |
| QSA_LIST | -0.9 % | -2.4 % | a chunk at the end of 214K: 1,406 -> 1,329 ms |
| QSA_ROWS | -0.4 % | -0.8 % | 1,329 -> 1,318 ms |
| HEAD_KV=512 | -2.4 % | -2.0 % (and -2.2 % at 27.5K) | every chunk: 1,154 -> 1,128 ms at 7K context |
| ARENA | lane open 114 -> 28 ms (109K), 209 -> 50 ms (214K), 29 -> 9 ms (2.8K); last chunk with its kept state 986 -> 798 ms (1,620 rows @109K), 1,006 -> 485 (640 @182K), 1,542 -> 1,167 (2,501 @214K) | | |
| WARM | the engine's first 3,072-row chunk 1,360-1,386 -> 1,173-1,198 ms (once per start) | | |
Decode is unchanged: the same 300-token task after 1.9K / 54.6K / 109K / 182K tokens: 44.4 / 48.5 / 44.0 / 42.0 tok/s off, 44.4 / 48.5 /
43.8 / 42.2 on.

## 3. Identity
- Certified select, device check mode (`QSA_CERT=2`: the stock kernels run too and every row's list is compared on the device), C22 and
  C23 with LIST (+ROWS), slack 100: 4,490,634 row selections each (27K prompt, 109K needle 3/3, 214K prompt): 0 lists differ, 0
  exact-sum bit differences, 0 bound violations, 0 fallbacks; the listed path served 3,919,434 rows and gave up on 0. The in-between
  counts with ROWS equal those without to the block (132,584,847): the coarse scores are the same values.
- C24 with the full set against the BASE binary's run (C20, same requests, same order, both fresh): ident set 51/51 texts identical;
  warm 1/1, prefill 4/4 (2.8K-214K), decode-after-context 4/4, needle 1/1 (3/3), short 24/24; score hashes after the prompt pass
  74/74 and at release 74/74. The same for C22 (ARENA+LIST+WARM) and C23 (+ROWS).
- HEAD_KV: the MTP head's state after every prompt pass (its last row's output, ALL its keys, ALL its values; prompts of 19 to
  214,471 tokens) hashed with `TF_FLASHNEXT_Q38_HHASH=1` in the off and the on engine (C24): 86 head states, 86 identical.
- Agents (5 x 56K own system prompt x 3 turns), flags off vs the full set (C24, and C23 before it): texts 15/15, short 24/24, score
  hashes after the prompt pass 27/27 (release-time hashes 13-20 of 27: the concurrent requests' schedule noise, round 8).

## 4. CAVEAT on ARENA (read before making it a default)
- The idle engine holds the planned room: MemAvailable 7-8 GiB (36-37 without). That is the plan's own reserve, but it is always
  held, not only under load.
- C23 (first form): in the 5-agent run five 1.1 GiB states found no gap and fell back to cuMemAlloc on top of the arena; the box was
  left with 2 GiB. C24 (this patch) forbids that for kept states: the cache evicts until the state has a gap; lane buffers may still
  fall back (short-lived). C24 re-run, 5 agents x 56K x 3 turns: fallback 0, MemAvailable 7 GiB at the end, wall 125.9 -> 120.2 s,
  follow-up first token median 0.96 -> 0.82 s, sum of first-token waits 426.5 -> 407.5 s; the cache evicted 15 states instead of 8
  (15 kept instead of 22) and lost no hit (10 of 10).
- NOT TESTED: 16 agents x 20K on the default profile with the arena (heavy state churn: more evictions could cost hits there). Run it
  before ARENA becomes a default; until then ARENA is safe to ship as an option and the other four flags as defaults.
- Not with TF_FLASHNEXT_Q38_POOL_CTX profiles (USERS16): the arena is installed with the elastic cache only. 16 users untested.

## 5. Dropped / zero (measured)
- Sampled cut (a 1/8 sample predicts the threshold): exact, but it bought only -0.3 % at 109K (the select's cost was its barrier
  passes, not the radix passes) and depends on the text. Replaced by QSA_LIST.
- Prompt expert tile config (`TF_FLASHNEXT_PROMPT_EXPERT_TILE_CFG`, never swept before): 13 configs, the default 64,1,2 is the best
  (MoE 405-412 ms a chunk; others 417-551). Zero.
- Partial chunks: MoE = ~127 ms a chunk + 0.092 ms a row (every expert is read once whatever the row count). No exact lever.
- Decode against context: one token a step: 40.2 ms at 2K -> 41.7 ms at 182K (+3.7 %). Nothing built.

## 6. For the next round
- Tensor-core ceiling of this box (kitC/mb/mb_mma.cu): 122 TFLOP/s bf16 mma from registers, 61.5 with one operand from shared memory
  per mma pair, 3.8 when the operand streams from memory (233-240 GB/s). The prompt pass runs ~30: a quarter of the ceiling.
- a third engine's prompt pass quantises ACTIVATIONS (W4A4 experts, optional A8 dense): not portable under the lossless rule (see
  PROGRESS-C.md 05:06). Whether the checkpoint's own W4A4 is "the official arithmetic" is Pico's call.
- The draft head's prompt pass still costs ~1-2 % of a long prompt after HEAD_KV (projections, hyper-connection read-out).
