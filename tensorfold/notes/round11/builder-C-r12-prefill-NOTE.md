# Builder C, round 12: pass scratch trim (6,144-row chunks at less memory than 4,608), certified select for decode rows, a wrong tile choice above 3,072 rows, and three kernels measured at their floor

Patch: `C-r12.patch` = `git diff b37b4e0 HEAD` of Mac `q38/wt/C12` (branch r12-C, HEAD a27a02c), sha256 377dde7f58ea7b35, 381 lines, 7 files;
`q38/mkhandoffC12.sh` printed TREE_EQUAL (base + patch = tree 6ed6888f). Base b37b4e0 = my round-11 handoff tree (793d4ee + C-r11.patch), so it applies after
C-r11.patch. Binary BUILT AND MEASURED from exactly this tree: C42, sha 718a78b857fb1152 (spark-4e5c `~/q38/tfbin/C42`); C41 (sha b323a5a0a1f34b2f) = the same tree without
HC_DN_TALL (commit 21f872a): sections 2-3 were measured on C41 unless they say C42. Build: `./shipC12.sh NAME 16 24g`.
Everything below is MEASURED on spark-4e5c unless it says ESTIMATE; shared windows 2026-11-14 12:23-14:03, 14:11-14:40 and 14:45-15:11 UTC, fresh engines, env flags,
recipe one-user profile (5 x 262,144, fp8 KV, vision on; no HEAD2 file on this box). Full log: `q38/PROGRESS-C.md` (Round 12). Microbenches (the engine's own
kernels, engine launch arguments): `q38/kitC/mb/` (mb_qsa.cu, mb_hcu.cu, mb_hcd.cu). R11 below = round-10 set + GDN_PROD=2 + ROUTE_BAND=2048.

## 1. Flags (all default OFF in the patch)
| flag | what | proposed |
| `TF_FLASHNEXT_Q38_SCRATCH_TRIM=3072` | the pass scratch of a chunk wider than N rows keeps its K-slice partials only for N rows: hc_part, part, ple_part hold every slice for N rows and ONE slice a row past N; native_part holds N rows; head_local (one byte a vocabulary entry a row) holds 256 rows. No kernel, no value changes: these are bytes no launch of a wide pass touches. Users that could exceed a trimmed buffer return error.UnexpectedTensor instead of writing (the generic 32-slice read-out past N rows, the two-rank head past 256 rows, the draft head's lanes past 256 x vocab bytes) | ON |
| `TF_FLASHNEXT_Q38_SCRATCH_CHECK=1` | check mode: 4,096 guard bytes behind each of the 5 trimmed buffers, read back after every pass ("q38 scratch check passes= ... bad=") | off |
| `TF_FLASHNEXT_CHUNK=6144` | NO CODE (existing knob). With the trim it costs LESS memory than the untrimmed 4,608 of round 11 | ON (replaces 4608) |
| `TF_FLASHNEXT_Q38_QSA_CERT_MIN=1` | NO CODE (existing knob, default 64): decode and draft rows take the certified select too (coarse scores + listed cut) instead of the stock scores + select | ON |
| `TF_FLASHNEXT_Q38_HC_DN_TALL=3072` | prompt passes of at least N rows take the 64-row tile of the fused read-out's down projection (fn_hcn_down4). The stock rule (cuda_tiles.zig pickDown) prices a wave of CTAs in proportion to the tile height and takes the 32-row tile for 8,192 rows (and for tails such as 4,694 or 5,000 rows); measured, the 64-row tile is 24 % faster a row there. Both tiles are stock kernels and write the same bytes | ON (matters with CHUNK=8192 and for tails of 3,072+ rows) |
| `TF_FLASHNEXT_Q38_HC_DN_TALL_CHECK=1` | check mode: where the flag changes the tile, the stock tile runs first into scratch and every word of the two outputs is compared on the device ("q38 hcdn tall check ... diff=") | off |
| `TF_FLASHNEXT_Q38_QDUMP=FILE` (+ `_SKIP`, `_STEP`, `_MAX`) | measuring aid: dumps the key lists of fused prompt attention launches (kitC/ql.py reads them; mb_qsa.cu replays them) | off |

## 2. Speed (fresh engines, same prompts, references first and last; first token of a cold prompt)
Reference = the round-11 proposal (R11 + CHUNK=4608).
| prompt tokens | reference first / last | R11 + CHUNK=6144 + SCRATCH_TRIM=3072 | the same + QSA_CERT_MIN=1 (final set, full list) | R11 + CHUNK=8192 + trim |
| 7,276   | 2.71 / 2.71 s = 2685 / 2684 tok/s | 2.71 s 2688 (0.0 %) | 2.70 s 2695 | 2.65 s 2743 |
| 27,502  | 9.91 / 9.98 s = 2774 / 2756 | 9.76 s 2818 (+1.9 %) | 9.76 s 2817 | 9.89 s 2781 |
| 109,142 | 40.41 / 40.73 s = 2701 / 2680 | 39.78 s 2744 (+2.0 %) | 39.88 s 2737 | 40.18 s 2716 |
| 214,471 | 81.95 / 82.36 s = 2617 / 2604 | 80.43 s 2667 (+2.2 %) | 80.68 s 2658 | 81.33 s 2637 |
| arena (room for kept states) | 27,684-27,808 MiB | 28,592 MiB (+0.9 GiB) | 28,270 | 27,388 |
(Round-10 recipe set on the same prompts, round 11: 2538 / 2652 / 2558 / 2480 tok/s; default 3,072-row arena 29,264 MiB.)
Decode (drafts on, the same writing task, same texts):
| after prompt tokens | 1.9K | 54.6K | 109K | 182K |
| 150 tokens, CERT_MIN default, first / last | 44.1 / 44.2 tok/s | | 43.5 / 43.6 | 40.9 / 40.9 |
| 150 tokens, QSA_CERT_MIN = 1 | 44.5 | | 44.3 (+1.7 %) | 41.9 (+2.4 %) |
| 300 tokens, full list without / with CERT_MIN=1 | 44.4 / 44.3 | 48.5 / 48.3 | 43.8 / 44.1 | 42.1 / 42.8 (+1.7 %) |
Short new prompt first token (median of 12): 96.1-97.2 ms in every run (unchanged). Needle 109,126 tokens: 40.2 s, 3/3, same answer.
5 agents x 56K own system prompt x 3 turns (fresh engines):
| | reference (R11 + CHUNK=4608) | CHUNK=6144 + trim | CHUNK=8192 + trim |
| wall; sum of first-token waits | 114.8 s; 372.2 s | 112.8 s (-1.7 %); 358.7 s (-3.6 %) | 113.5 s; 346.4 s (-6.9 %) |
| follow-up first token median / max | 0.81 / 15.97 s | 0.84 / 11.68 s | 0.82 / 2.44 s |
| prompt cache at the end | 13 states, 13.7 of 21.5 GiB, hits 10, misses 5 | 13 states, 13.7 of 22.3 GiB, hits 10, misses 5 | 12 states, 12.7 of 20.9 GiB, hits 10, misses 5 |
No cache hit lost at 6,144 or 8,192 rows with the trim (round 11's untrimmed 6,144 lost one; its wall was +12 %).

CHUNK=8192 with HC_DN_TALL (C42, third window; reference = CHUNK=6144 + trim, first and last):
| prompt tokens | 6,144 + trim, first / last | 8,192 + trim + HC_DN_TALL=3072 | 8,192 + trim without it (C41, 13:34) |
| 7,276   | 2.73 / 2.72 s = 2666 / 2673 tok/s | 2.68 s 2718 (+1.8 %) | 2.65 s 2743 |
| 27,502  | 9.83 / 9.82 s = 2798 / 2801 | 9.80 s 2807 (+0.3 %) | 9.89 s 2781 |
| 109,142 | 40.02 / 40.03 s = 2727 / 2726 | 39.87 s 2737 (+0.4 %) | 40.18 s 2716 |
| 214,471 | 80.80 / 80.77 s = 2654 / 2655 | 80.54 s 2663 (+0.3 %) | 81.33 s 2637 |
| arena | 27,938-28,294 MiB | 26,944 MiB | 27,366-27,388 |
The flag gives 8,192-row chunks +1.0 % on the long prompts (2781 / 2716 / 2637 -> 2807 / 2737 / 2663), which brings them level with 6,144 (+0.3..0.4 %, at the edge of
the ruler) for ~1.2 GiB less arena; only prompts of 6-8K tokens gain (one chunk instead of two: +1.8 %). 6,144 stays the proposed chunk.

## 3. Identity
- HC_DN_TALL check mode (C42, CHUNK=8192 + trim + HC_DN_TALL=3072 + its check; prompts 2.8K-214K): 4,032 read-outs where the tile changed, 10,701,766,656 words of the
  two tiles' outputs compared on the device, 0 differ; texts 5/5, score hashes 5/5 + 5/5 against C_y_ref_C20. Microbench: 3,932,160 bytes, 0 differ. The run without the
  check flag: texts 5/5, hashes 5/5 + 5/5.
- Trim check mode (CHUNK=6144 + SCRATCH_TRIM=3072 + SCRATCH_CHECK=1, the full request list): 4,480 passes (prompt chunks of up to 6,144 rows, decode rounds, head
  passes), guard bytes changed: 0. Against the base binary's run (C_y_ref_C20): ident 51/51; warm 1/1, prefill 4/4, decode 4/4, needle 1/1, short 24/24 texts
  identical; score hashes after the prompt pass 74/74, at release 74/74.
- Final set (CHUNK=6144 + trim + CERT_MIN=1, full list, no check flag): the same counts, all identical; certified select 11,271,871 rows, fallback 0; arena fallback 0; crash 0.
- CERT_MIN=1 device check (QSA_CERT=2): 3,490,048 row selections (prompt rows and the 3,080 decode / draft rows) against the stock scores + select on the device:
  list_diff 0, exact_bit_diff 0, bound_viol 0, fallback 0, largest deviation 0.839 of the bound; texts 3/3 identical to the CERT_MIN=64 run.
- Prefill-only runs (4,608 first / last, 6,144 + trim, 8,192 + trim): texts 5/5, score hashes 5/5 + 5/5 against C_y_ref_C20.
- Agents: texts 15/15 + short 24/24 identical, score hashes after the prompt pass 27/27 for both chunk sizes (release-time 13/27: concurrent schedule noise, as in rounds 8-11).
- NOT TESTED: the trim with two ranks (the two-rank head is guarded at 256 rows: leave the flag off there), training passes, 16-user profiles.

## 4. The scratch account (bytes a row of a chunk; cuda_batch_state.zig Scratch.alloc)
| buffer | bytes a row | a wide prompt chunk uses |
| head_local | 248,320 | nothing (only the two-rank head and the draft head's lanes write it) -> 256 rows |
| y (pair rows fp32) | 112,640 | all |
| native_part (8 K slices) | 81,920 | nothing (the native lane reduces in a cluster, or in the block from 256 rows) -> N rows |
| part (8 output slices) | 81,920 | 1 slice -> 8 for N rows + 1 a row past N |
| hc_part (32 slices x 324) | 41,472 | 1 slice -> the same |
| ple_part (4 slices) | 40,960 | 1 slice -> the same |
| unfused DeltaNet rows (gq gk gv ys ggb) | 65,920 | lanes / late keep: LEFT ALONE |
| the rest (h, norm, proj, hc_up, ple rows, q, ao, moe ...) | ~270,000 | all / left alone |
| each lane's Draft.streams (5 lanes x 20,480) | 102,400 | the prompt chunk's hidden rows for the draft head: left alone |
Saved: 0.65 GiB at any chunk (head_local) and 473 KB a row past 3,072. Arena: 3,072 rows 29,264 MiB; 4,608 untrimmed 27,684-27,868; 6,144 untrimmed 26,406-26,700;
6,144 trimmed 28,270-28,700; 8,192 trimmed 27,366-27,388 (untrimmed 8,192 did not start with 5 x 262,144 lanes).

## 5. Measured at the floor, nothing built (the three kernels the round-11 table called farthest from their floors)
- QSA fused attention (fn_attn_fused_kv8, 90-109 ms a chunk). REAL key lists (QDUMP, 40 launches over a 27.5K and a 109K prompt): neighbouring query rows differ in
  115-303 of their 512 blocks; no two neighbours share a list; a 4-row group selects 777-1,371 distinct blocks; 32-key tiles line up across 4 rows 0.0-0.1 % of the time; a
  launch of 1,024 rows selects EVERY block of the context (97.6 MB of codes at 93K: that is its growth with context). Microbench on those lists (ms a launch, 1,024 rows):
  engine 2.38 at 4K / 3.22 at 93K; without fetch + widen 1.63 / 1.59; only barriers + softmax glue 0.85; the mma are ~0.45. Tried, all byte-identical: touch-prefetch
  1-6 tiles ahead: 30-55 % SLOWER (the cost is the traffic from the cache into the SM, 2.2 GB a launch, not latency); a fourth CTA an SM (128 registers, 24.8 KB): -1 % / +5 %;
  a row-group CTA cannot hold the union window of 4 rows (130-170 KB against the SM's 100 KB of shared memory). The round-11 "30 ms floor" counted the mma only.
- Hyper-connection down (fn_hcn_down4, 64 ms a chunk): already the fused form (norm in shared memory before the mma). Upstream's second form: 0.655 ms against 0.615; its
  benchmark form WITHOUT the norm: 0.468 ms: the norm is 24 % (at most 13.5 ms a chunk) and it is two fp32 multiplies of three varying operands: no exact shortcut.
- Hyper-connection up + mix (fn_hcn_up3_k5, 62 ms a chunk): engine 0.610 ms a call; upstream's debug forms: projection alone 0.431, epilogue alone 0.44-0.45 (they
  already overlap); without the sigmoid 0.54. An exact table for bsig (bf16 in, bf16 out; 3,072 entries in shared memory; proven on all 65,536 inputs, 0 bits differ;
  kernel outputs 0 of 16,711,680 bytes differ): -5 % when the inputs stay in the table's range, +11 % when they do not. DROPPED (best case 3 ms a chunk).
- Context growth of a prompt chunk, 36K -> 202K: +94 ms (8 %): coarse scores +39.9 (0.0745 -> 0.3525 ms a call: 13.2 GFLOP + 25.9 MB of half scores written and read back),
  fused attention +18.8, listed select +17.1, others +16. No lever built (a fused coarse + cut kernel must compute the scores twice to avoid the 26 MB round trip).
- Decode step at 182K against 1.9K (no drafts): +3.1 ms: stock block scores +1.25, stock select +1.00, attention gather +0.44. CERT_MIN=1 moves the first two to the
  certified kernels (section 2); the gather is inherent.

## 6. For the integrator
- Recipe: add `TF_FLASHNEXT_Q38_SCRATCH_TRIM=3072`, set `TF_FLASHNEXT_CHUNK=6144` (instead of 4608), add `TF_FLASHNEXT_Q38_QSA_CERT_MIN=1` and `TF_FLASHNEXT_Q38_HC_DN_TALL=3072`.
- The full request list (ident 51, decode, needle, short) and the 5-agent trace were run on C41 with the first three; HC_DN_TALL was measured on prefill only (C42).
- Re-run on 923b before quoting: the trim check once (SCRATCH_CHECK=1 on the full list), QSA_CERT=2 with CERT_MIN=1 on a decode run, the fresh prefill pair.
- CERT_MIN=1 with many users: shared rounds of 2+ lanes take the lanes kernels (ATTN_LANES), not this path; NOT measured with 16 users here.
- CHUNK=8192 + trim starts and keeps every hit of the 5-agent trace; with HC_DN_TALL it is level with 6,144 on long prompts and costs ~1.2 GiB more: an option, not the default.
- STILL OPEN, by size: the coarse scores' 26 MB a call round trip at 200K (section 5); pickDown's rule itself (a wave of the 64-row tile costs 1.53x, not 2x, a 32-row wave:
  a rule fitted to that would also be right below 3,072 rows; not changed here because only 3,072 rows were measured).

## 7. Operational note for the harness owner (builder A)
- The window S140923 reached its cap at 14:03:41 (start + 280 min, whatever the leases say). A queued chain of mine started an engine on spark-4e5c at 14:04:00, just after
  the window's stop-all; I killed it at 14:05:05. DeepSeek's restore sized its session store in that minute: "session store: 3.13 GiB on every rank (at most 12.0; free device
  memory before it, ranks 0-3: 18.21 / 27.76 / 5.13 / 27.89 GiB)" (head ~/q38/runs/S140923/run.log). Production was healthy (health 200, answer check) from 14:08:00 to the next
  window (14:10:33) with that smaller store; the next restore starts from clean boxes. My chain scripts now refuse to start an engine without a lease stamp (kitC/runQ4.sh).
- After that restore the CLOSING flag stayed (q38-win failed/inactive, q38-guard inactive, health 200): removed at 14:10 as shared/NOTES.md says.
- Suggestion: let the engine starters of every kit check `~/q38/shared/OPEN` / CLOSING through the head before `docker run`, or let qstop.sh run a second time at the end of the restore.
