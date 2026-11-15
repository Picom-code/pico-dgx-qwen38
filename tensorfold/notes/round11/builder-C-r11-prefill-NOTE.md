# Builder C, round 11: why the prompt pass runs at a quarter of the tensor-core ceiling, and three exact levers (one cumulative patch against r10-C 793d4ee)

Patch: `C-r11.patch` = `git diff 793d4ee HEAD` of Mac `q38/wt/C11` (branch r11-C, HEAD b37b4e0), sha256 ff7a98e55d633133, 522 lines, 6 files;
`q38/mkhandoffC11.sh` printed TREE_EQUAL (base + patch = tree 8202b0a2). Base 793d4ee = integ9 7061396 + my round-10 patch (C-r10.patch), so on
integ10 it applies after patch 0121. Binary BUILT AND MEASURED from exactly this tree: C32, sha 69638727a9799a29 (spark-4e5c `~/q38/tfbin/C32`);
C31 (sha ec07be368c7be43f) = the same tree without the router bands (commit ce1f180). Build: `./shipC11.sh NAME 16 24g`.
Everything below is MEASURED on spark-4e5c unless it says ESTIMATE; two shared windows (2026-11-14 08:20-08:58 and 09:26-12:00 UTC), fresh engines, env
flags, recipe one-user profile (5 x 262,144, fp8 KV, vision on; no HEAD2 file on this box: the draft side differs from 923b, prompt passes do not).
Full log: `q38/PROGRESS-C.md` (Round 11). Raw logs: Mac `q38/results/rC11.tgz`. Microbenches (the engine's own kernels, engine launch arguments): `q38/kitC/mb/`.

## 1. Flags (all default OFF in the patch)
| flag | what | proposed |
| `TF_FLASHNEXT_Q38_GDN_PROD=2` | the prompt DeltaNet kernel with TWO producer groups (`fn_gdn_fused_p2`, kernels/cuda/q38_gdn_prod.cuh): group A forms q and k (conv, SiLU, norms), group B forms v, g, beta and the gated norm; the chain reads k again from shared memory for its second loop (106 registers a thread instead of 166, so 512 threads fit a CTA). Every value is fn_gdn_fused's expression in its order | ON (no cost) |
| `TF_FLASHNEXT_Q38_GDN_PROD=3` | three groups (640 threads): measured worse than two | off |
| `TF_FLASHNEXT_Q38_GDN_PROD=12` / `13` | check mode: fn_gdn_fused runs first into scratch, then the new kernel; out, xs and the end state compared word by word on the device ("q38 gdnprod check ... diff=" every 36 launches) | off |
| `TF_FLASHNEXT_Q38_ROUTE_BAND=2048` | the router logits of a prompt pass launched in bands of N rows (N a multiple of 128) instead of one launch that walks all rows once per expert tile; the same kernel on row ranges | ON (no cost; matters with CHUNK > 3072) |
| `TF_FLASHNEXT_Q38_ROUTE_BAND_CHECK=1` | check mode: the single launch runs first into scratch, logits compared on the device ("q38 routeband check ... diff=" every 48 passes) | off |
| `TF_FLASHNEXT_CHUNK=4608` (or 6144) | NO CODE: the engine's existing knob (rows a prompt pass; default 3,072). Expert weights are read once a chunk, so more rows mean fewer weight bytes a row | 4608 ON (kept every cache hit of the 5-agent trace); 6144 only as an OPTION for long single prompts: SEE SECTION 4 |
| `TF_FLASHNEXT_Q38_XPLAN=FILE` (+ `_XPLAN_MAX=N`) | measuring aid: dumps the expert plan (items: expert, first, count) of prompt MoE blocks of >= 1,024 rows; `kitC/xp.py` reads it | off |

## 2. Speed: first-token time of a cold prompt, fresh engines, same prompts, references first and last
Reference R10 = the round-10 recipe set (ARENA=1, QSA_LIST=1, QSA_ROWS=1, WARM=3072, HEAD_KV=512, QSA_SLACK=32).
| prompt tokens | R10 first / last (C31) | + GDN_PROD=2 + ROUTE_BAND=2048, 3,072 rows (C32) | + GDN_PROD=2 + CHUNK=6144 (C31; C32 first / last) | + all three (C32) |
| 7,276   | 2.87 / 2.87 s = 2538 tok/s | 2.82 s 2577 (+1.5 %) | 2.76 s 2640 (2.76 / 2.75) | 2.72 s 2679 (+5.6 %) |
| 27,502  | 10.37 / 10.37 s = 2652 | 10.19 s 2699 (+1.8 %) | 9.97 s 2758 (9.97 / 9.99) | 9.80 s 2805 (+5.8 %) |
| 109,142 | 42.64 / 42.72 s = 2558 | 42.02 s 2597 (+1.5 %) | 40.67 s 2684 (40.68 / 40.64) | 39.92 s 2734 (+6.9 %) |
| 214,471 | 86.46 / 86.53 s = 2480 | 85.23 s 2516 (+1.5 %) | 82.12 s 2612 (82.11 / 82.09) | 80.59 s 2661 (+7.3 %) |
Full request list, C31, R10 first / "+ GDN_PROD=2 + CHUNK=6144" / R10 last: needle 109,126 tokens 42.7 s -> 40.8 s (3/3, same answer); short new prompt first
token (median of 12) 95.7 -> 95.9 ms; decode after 1.9K / 54.6K / 109K / 182K 44.4 / 48.5 / 43.9 / 42.1 -> 44.3 / 48.3 / 43.9 / 42.1 tok/s (same texts); ident
set median 65.8 -> 65.6 tok/s. The full list on C32 with all three: section 8.
By lever:
| lever | 7.3K | 27.5K | 109K | 214K | where |
| GDN_PROD=2 alone (C31; reference R10 without SLACK, first and last 2.86 / 10.34 / 42.68 / 86.52 and 2.87 / 10.41 / 42.82 / 86.77 s) | -1.2 % | -1.6 % | -1.3 % | -1.3 % | fn_gdn_fused 3.16 -> 2.57 ms a call (KPROF): -21 ms a chunk |
| GDN_PROD=3 alone | | | | -0.6 % | |
| CHUNK=6144 on top of GDN_PROD=2 | -2.8 % | -2.4 % | -3.6 % | -3.9 % | KPROF per 3,072 rows: gate/up 209.5 -> 170.2 ms, down 158.4 -> 137.0 |
| ROUTE_BAND=2048 on top of both (C32, first and last 2.76 / 9.97 / 40.68 / 82.11 and 2.75 / 9.99 / 40.64 / 82.09) | -1.3 % | -1.8 % | -1.8 % | -1.8 % | tf_route_logits_wide at 6,144 rows 1.70 ms a call against 0.28 at 3,072 |
| ROUTE_BAND=2048 at 3,072 rows | ~-0.2 % | | | | the rows stay cached there |
CHUNK=8192 and 12288 do not start with 5 x 262,144 lanes (memory budget).
THE MIDDLE POINT, PROPOSED: GDN_PROD=2 + ROUTE_BAND=2048 + CHUNK=4608 (C32, one fresh run on the R10 set, same window): 2.71 / 9.89 / 40.36 / 81.96 s =
2690 / 2780 / 2704 / 2617 tok/s (-5.6 / -4.6 / -5.4 / -5.2 % against R10); arena 27.9 GiB (R10: 29.3; 6144: 26.4-26.7). Texts 5/5, score hashes 5/5 + 5/5.

## 3. Identity
- GDN_PROD check mode (`=12`): at 3,072 rows (C31): 4,320 launches, 46,190,462,976 words of out + xs + end state compared on the device against fn_gdn_fused on
  the same inputs, 0 differ; at 6,144 rows (C32): 2,232 launches, 44,548,392,960 words, 0 differ. Prompts of 2.8K / 7.3K / 27.5K / 109K / 214K tokens.
  Microbench (`kitC/mb/mb_gdnf.cu`, random inputs): 43,253,760 output bytes, 0 differ, for every split tried.
- ROUTE_BAND check mode (C32, CHUNK=6144): 2,928 router passes, 8,940,235,680 logit words against the single launch, 0 differ.
- C31 "GDN_PROD=2 + CHUNK=6144" against the BASE binary's run at 3,072 rows (C_y_ref_C20, round 10's reference; same requests, same order, both fresh):
  ident set 51/51 texts identical; warm 1/1, prefill 4/4 (2.8K-214K), decode-after-context 4/4, needle 1/1 (3/3), short 24/24; score hashes after the prompt
  pass 74/74 and at release 74/74. The same for both R10 reference runs.
- Every other run (GDN_PROD=2 alone, =3, both check runs, C32 off first / band / band at 3,072 / off last): texts 5/5 and score hashes 5/5 + 5/5 against C_y_ref_C20.
- Agents (5 x 56K x 3 turns), R10 against GDN_PROD=2 + CHUNK=6144: texts 15/15, short 24/24, score hashes after the prompt pass 27/27 (release-time 13/27: the
  concurrent requests' schedule noise, as in rounds 8-10).
- Certified select counters: fallback 0 everywhere. Arena: fallback 0. crash lines 0 in every run.

## 4. CAVEAT on CHUNK=6144: it takes memory from the prompt cache (read before making it a default)
- The pass scratch is ~1.1 MB a row. With ARENA=1 the arena sizes itself from what is left: 26.4-26.7 GiB instead of 29.3; without the arena MemAvailable after
  load is 34 GiB instead of 37.
- 5 agents x 56K own system prompt x 3 turns (C31, default profile, fresh engines):
  | | R10 (3,072 rows) | + GDN_PROD=2 + CHUNK=6144 |
  | wall; sum of first-token waits | 119.9 s; 407.5 s | 134.2 s (+12 %); 395.2 s (-3 %) |
  | first turn (cold) first token median / max | 79.9 / 107.5 s | 72.9 / 99.9 s |
  | follow-up first token median / max | 0.82 / 13.10 s | 0.98 / 21.91 s |
  | prompt cache at the end | 15 states, 15.7 of 23.9 GiB, hits 10, misses 5, evicted 15 | 12 states, 12.7 of 20.0 GiB, hits 9, misses 6, evicted 22 |
  The smaller cache lost one hit (a 55K prompt prefilled again) and the wall got worse.
- The same trace with CHUNK=4608 (C32, + GDN_PROD=2 + ROUTE_BAND=2048; R10 re-run on C32 right before it):
  | | R10 (C32) | + GDN_PROD=2 + ROUTE_BAND=2048 + CHUNK=4608 |
  | wall; sum of first-token waits | 119.6 s; 406.6 s | 114.6 s (-4.2 %); 374.1 s (-8.0 %) |
  | first turn first token median / max | (79.9 / 107.5 s on C31) | 76.5 / 100.9 s |
  | follow-up first token median / max | 0.82 / 13.05 s | 0.83 / 14.22 s |
  | prompt cache at the end | 16 states, 16.8 of 24.3 GiB, hits 10, misses 5, evicted 14 | 15 states, 15.7 of 21.4 GiB, hits 10, misses 5, evicted 15 |
  texts 15/15 + 24/24 identical, score hashes after the prompt pass 27/27. No hit lost. => 4,608 rows is the proposed value: most of the gain, 1.4 GiB.
  6,144 stays an option for long single prompts or profiles with spare memory. GDN_PROD=2 and ROUTE_BAND cost no memory.
- NOT TESTED with any larger chunk: 16 agents x 20K on the default profile (heavier cache churn).
- About 0.5 MB of the 1.1 MB a row looks unused by a big prompt chunk (ESTIMATE from cuda_batch_state.zig: head_local 248 KB a row, the generic kernels' K-slice
  partials 82 KB, the unfused DeltaNet rows 82 KB, normed rows 40 KB, PLE partials 41 KB). Capping those at 3,072 rows would give ~1.5 GiB back; NOT DONE (each
  needs a proof that no path uses it beyond that row count).
- 16-user profiles (POOL_CTX / USERS16) NOT TESTED with 6144; leave them at the default.
- The gain depends on how a text's rows spread over experts (section 5): measured on the harness' filler text and the needle text only.

## 5. The floor table: one 3,072-row prompt chunk at 40-110K context (KPROF, serialised launches; 1,230 ms; C20 = round-9 base, cert select on)
Byte floor = the bytes a kernel must move through memory in this dataflow (weights once + activations that do not fit the 24 MiB cache) at 235 GB/s.
FLOP floor = its multiply-adds at the applicable bf16 tensor-core rate: 61.5 TFLOP/s when one operand comes from shared memory per mma pair (NVFP4 tiles: each
k16 step's products are scaled by an fma, so fragments are not reused across the accumulate), 122 when the weight fragment is decoded once and reused
(the MXFP8 prompt tile). The larger is the floor.
| kernel (calls a chunk) | measured ms a chunk | bytes a call -> ms | GFLOP a call -> ms | floor ms a chunk | measured / floor | verdict |
| experts gate/up, NVFP4 tile (48) | 215 | 616 MB (313 experts used) -> 2.62 | 201 -> 3.27 @61.5; the kernel alone, weights cached: 2.83 (69 TFLOP/s) | 136 | 1.58 | streams at the bus AND computes, the two do not overlap fully; kernel config is at its optimum (sec. 7) |
| experts down, NVFP4 tile (48) | 162 | 642 MB (weights 288 + pair rows f32 314 + in 39) -> 2.73 | 101 -> 1.64 @61.5; alone 2.29 | 131 | 1.24 | near the bus |
| MoE write-back: 10 pair rows + shared -> 4 streams (48) | 104 | 472 MB -> 2.01 | - | 96 | 1.08 | AT ITS FLOOR |
| dense MXFP8 prompt tile 2,560 -> 16,512 (48) | 139 | 159 MB -> 0.68 | 260 -> 2.13 @122 | 102 | 1.36 | runs 90 TFLOP/s; MiaAI's tile; nothing found |
| dense MXFP8 prompt tile 6,144 -> 2,560 (48) | 57 | 85 MB -> 0.36 | 97 -> 0.79 @122 | 38 | 1.49 | 82 TFLOP/s |
| DeltaNet fused chain (36) | 114 | 148 MB -> 0.63 | scalar recurrence: chain alone 1.51 ms a call | 54 | 2.1 | the chain waited for its producers: LEVER GDN_PROD=2 (-15 ms); after it one SM's instruction rate carries one head (chain + producers) |
| QSA fused attention, 1,024 rows a call (35.5) | 93 (+23 at 200K) | gather shared by neighbouring rows: not counted | 51.5 -> 0.84 | 30 | 3.1 | FARTHEST FROM ITS FLOOR; per-row chunked online softmax in fixed order; not attempted |
| hyper-connection down, 4 streams -> 320 (92) | 64 | 63 MB -> 0.27 | 20 -> 0.33 | 30 | 2.1 | per-element norm in shared memory; wide form tried: no (sec. 7) |
| hyper-connection up + mix (96) | 62 | 81 MB -> 0.34 | 20 -> 0.33 | 33 | 1.9 | sigmoid + norm + mix on 31.5 M values a call |
| attention-side write-back (52) | 37 | 157 MB -> 0.67 | - | 35 | 1.06 | AT ITS FLOOR |
| shared expert, generic 64 x 64 kernel (144) | 36 | - | 10 -> 0.08 @122 | 12 | 3.0 | its arithmetic (per-block partial, fma scale, K slices) is what is slow; the fast tile gives other bits |
| indexer coarse scores + certified select (150) | 64 (grows) | - | - | - | - | round 9-10 work |
| everything else (route, gate, prep, head, PLE, sums) | ~83 | | | | | |
(The same table on the round-11 set at 3,072 rows, C31: 1,146 ms a chunk; fn_gdn_fused_p2 92.7, listed select 18.8 + coarse by rows 9.5, the rest within 3 %.)
SUM: measured 1,230 ms; if every listed kernel reached its floor: ~845 ms. That is an UPPER BOUND of exact kernel work at 3,072 rows a chunk (+45 %), not an
expectation: by size the gaps are the expert kernels 110 ms (bus and compute not overlapped), QSA attention 63, DeltaNet 60 (15 taken by GDN_PROD=2; the rest is
one SM's instruction rate for a head), hyper-connection 63 (scalar work per element), dense tiles 55 (90 against 122 TFLOP/s), shared expert 24 (other bits).
The two write-backs are at their byte floor.
WHY "30 TFLOP/s": (1) 37 % of the chunk is not tensor-core work at all (write-backs, DeltaNet recurrence, select, glue); (2) the expert tiles stream 1.2 GB
of weights + 0.35 GB of f32 pair rows a layer at the bus rate while they compute; (3) the hyper-connection and the generic kernel carry per-element scalar
work (norms, sigmoid, fma scaling) between mma. Only the two MXFP8 prompt tiles are pure tensor-core kernels, and they run 82-90 TFLOP/s.
THE REAL ROUTING (TF_FLASHNEXT_Q38_XPLAN, 27.5K filler prompt): a 3,072-row chunk uses a median of 313 of 512 experts a layer (148-408); pairs an expert:
median 17, p90 226, max 3,068; 89 % of pairs sit in experts with more than 64 pairs; half the used experts have <= 16 pairs. A 6,144-row chunk uses 342.
So a chunk's expert cost = (weights of ~313 experts once: 865 MB a layer = 3.7 ms at the bus) + (compute of 30,720 pairs: 5.1 ms compute-only), measured
6.9-7.9 ms; with twice the rows the weight part halves per row: that is what CHUNK=6144 buys.

## 6. The draft head's prompt pass and the lane-open zero fill (task items; nothing built)
- Draft head after HEAD_KV (KPROF, C31): generic BF16 projections 7.7 + 5.2 + 4.1 ms, its attention on the last 512 rows 2.0, write-back / norms / mixes ~3, PLE 2.9
  = ~26 ms a chunk = 2.3 %. All of it forms the head layer's input for every prompt row, which the head's keys and values need (later rounds read them): no row
  can be skipped exactly. The kernels are the generic 64 x 64 BF16 tiles (~40 TFLOP/s); a faster tile with the same K slices would be worth < 1 % (ESTIMATE).
- Zero fill at lane open with the arena: 28-50 ms at 109-214K tokens = 0.06 % of those prompts (round 10 numbers). The engine already has ARENA=2 (no zero fill,
  experiment only); a run of it on short prompts was cut for time. Ceiling for a short prompt: under the 8.7 ms lane open of a 96 ms first token.

## 7. Dropped / zero (measured in microbenches that call the engine's own kernels with the engine's launch arguments; `kitC/mb/`)
- Expert tile: units handed out by a ticket counter instead of the fixed stride: gate/up +2..+19 %, down +5..+13 % slower.
- Expert tile: deeper cp.async pipelines (6 / 8 / 10 / 12 stages), fewer or more CTAs (48 / 144; 96 for down), rt4 for down: all slower on real plans; weights-only
  streaming is the same at every stage and CTA count (943 MB in 4.1 ms = 230 GB/s): bus-bound. The engine's instance is the optimum of this design.
- Hyper-connection down as ONE CTA a row tile (448 threads; H read and normed once instead of twice): 64 / 48 rows spill registers (8.6 / 7.3 ms against 0.615),
  32 rows 0.769 ms: slower. The cache already shares most of the double read (forcing it apart costs +18 %).
- DeltaNet: prefetch of the next batches' projection bytes (cp.async): -4 % of the kernel (not kept: the two groups give -15 %); three groups, four groups,
  4 chain threads a state row with 1-4 groups: all slower than 2 threads + 2 groups.
- Shared expert on the 128 x 128 prompt tile: other bits (one accumulate chain against per-block partials + fma scale + K slices). Not built.
- A blocked / prefix form of the DeltaNet recurrence: changes the operation order: not allowed. The chain is inherently sequential over rows.
- Router: the lever of section 1 came from re-profiling per kernel per row after the chunk change; every other kernel costs the same per row at 6,144.

## 8. Full request list on the final binary (C32, all three flags on the round-10 set; fresh engine)
| item | R10 (C31, first / last) | all three (C32) |
| prefill 7,276 / 27,502 / 109,142 / 214,471 tokens, first token | 2.87 / 10.37 / 42.68 / 86.50 s | 2.71 / 9.79 / 39.83 / 80.51 s = 2689 / 2809 / 2740 / 2664 tok/s (-5.6 / -5.6 / -6.7 / -6.9 %) |
| first token of the decode task after 54.6K / 109K / 182K | 21.11 / 42.90 / 72.87 s | 19.87 / 40.11 / 67.84 s |
| decode after 1.9K / 54.6K / 109K / 182K (same texts) | 44.4 / 48.5 / 43.9 / 42.1 tok/s | 44.4 / 48.5 / 44.0 / 42.2 |
| needle 109,126 tokens | 42.7 s, 3/3 | 40.0 s, 3/3 same answer |
| short new prompt (19 tokens) first token, median of 12 | 95.7 ms | 96.6 ms |
| ident set median (no HEAD2 file here) | 65.8 tok/s | 65.5 |
Identity against the base binary's run (C_y_ref_C20) and against the R10 run: ident 51/51; warm 1/1, prefill 4/4, decode 4/4, needle 1/1, short 24/24 texts
identical; score hashes after the prompt pass 74/74, at release 74/74. Arena served 2,762, fallback 0, peak 22.9 GiB; certified select fallback 0; crash lines 0.

## 9. For the integrator
- Take GDN_PROD=2 and ROUTE_BAND=2048 as recipe defaults (no memory cost; +1.5..1.8 % prefill at 3,072 rows).
- CHUNK is a recipe decision. Proposed: 4608 for the one-user / agent profile (+4.8..5.9 % prefill tok/s in all, 5-agent trace wall -4 %, no cache hit lost,
  1.4 GiB less cache room). 6144: +5.6..7.3 % in all for 2.6-2.9 GiB of arena (3.9 GiB of cache budget), and the 5-agent trace lost a hit (wall +12 %).
- Re-run on 923b before quoting: the two check modes once (GDN_PROD=12, ROUTE_BAND_CHECK=1), the fresh pair at 117K / 227K.
- Not in this patch: anything for the QSA fused attention kernel (the group farthest from its floor, 93 ms a chunk + growth with context).
