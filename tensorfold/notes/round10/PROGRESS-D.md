# PROGRESS-D (builder D, head spark-3caa, MoE many-user lever)

2026-11-13T17:00:59Z start. Reading docs. No lease held, no engine started.
17:01:54 head: MemAvail 12.5 GiB with DeepSeek up (total 119.7 GiB); MiaAI ghcr image already on head; no models/tfbin. Started rsync of checkpoint (96 GiB) from 923b at 250 MB/s with evict loop (head ~/q38/kit-D/copy.sh, log copy.log); then kp25 tfbin.
- 17:08 UTC SETUP DONE on head (no window needed): checkpoint 96 GiB at ~/q38/models/nvidia-fp4-local (rsync from 923b over the fabric, 250 MB/s cap, evict loop; rc 0, evicted after),
  ~/q38/tfbin/kp25, build tree ~/q38/build-D/{zig,zcache,tf} copied from 3d24 (tf verified against Mac wt/D base 52b32e1: 0 of 7352 files differ), image MiaAI ghcr + nvcr pytorch were already on head.
  Head with DeepSeek up: MemTotal 119.7 GiB, MemAvailable 10-12 GiB => NO builds on head outside a window (8 GiB build would leave < 8).
- Scripts (Mac q38/kitD -> head ~/q38/kit-D): run.sh BIN VARIANT MIN "ENV" (container q38-run-D, --memory 24g, refuses unless lease D covers MIN and MemAvailable >= 95 GiB;
  starts wd.sh watchdog: kills at MemAvailable < 6 GiB / lease gone / window closing / deadline; logs MemAvailable every 2 s to kit-D/wd.log), stop.sh [OUTDIR], build.sh NAME (3 CPUs, 8g, refuses < 17 GiB avail).
- Mac worktree q38/wt/D branch builder-D (base 52b32e1). First change: log aid TF_FLASHNEXT_Q38_XSTAT=FILE (cuda_batch_moe.zig Xstat: per MoE block picks + weights + lane row counts). Build name D1.
- READ (source, verified): 512 experts/layer, top-10 + shared, 48 layers; decode expert kernel takes items of <= 16 pairs per expert (an expert with 17+ rows = 2 items = weights read again);
  KPROF 16 users (3d24 U2_kp16): expert kernels 53.7 % of serialised GPU time (gate/up 1306 us a call for 186 blocks = 225 GB/s), router logits+topk+plan+shared glue together ~2 %.
  => certified router: bound < 1 % even if the router cost went to zero (ARITHMETIC from measured KPROF): not built.
  => exact skips: experts are SwiGLU (silu, no exact zeros); routing weights are a softmax over the top-10 (exactly 0 only if a logit gap > ~88): will count zeros in the XSTAT dump.
- 17:10:40 acquiring lease D 60 for window W1 (head ~/q38/kit-D/w1.sh, log kit-D/w1.log; releases the lease itself at the end)
- 17:10-17:19 W1 (lease D): MEASURED head memory. With DeepSeek stopped: used 7.6 GiB (gateway 31 MiB, vision encoder 1.26 GiB host + 1.67 GiB GPU, system), MemAvailable 112-114 GiB.
  Engine weights + fixed = 79 GiB (MemAvailable 112 -> 32.9 GiB just before the stream pool). The engine sizes streams from MemAvailable - 9 GiB reserve = 23.4 GiB on the head:
  5 x 262K needs 24.23 GiB -> REFUSED by the engine (and would leave ~7 GiB); 16 x 64K needs 25.08 -> REFUSED (would leave 7.9 GiB; workers have ~4.5 GiB more because no gateway/vision).
  HEAD BUDGET: 119.7 total - 7.6 base - 79 engine - 12 spare = 21 GiB of streams. DECISION: head runs one user as 3 x 262K (14.5 GiB) and many users as 16 x 32K (12.5 GiB; ~20 GiB spare).
  Container --memory 12g (host RSS ~3.4 GiB anon + bounded page cache). Watchdog never fired (min seen 28 GiB). Both refusals were clean exits; nothing hung.
  Build D1 (XSTAT aid) BUILD_OK in ~3.5 min incremental at 3 CPUs / 8g inside the window (sha 5ca89705da5de16b), head ~/q38/tfbin/D1.
- 17:20 W2 starting (head ~/q38/kit-D/w2.sh, log w2.log): one user 3x262K kp25 -> D1 XSTAT 16 users -> D1 XSTAT 8 users -> kp25 ref 16 users GPROF. Releases lease itself.
- 17:24 W2 FIRST RUN on head, MEASURED (kp25, recipe env, vision on, 3 x 262K): READY in 208 s (cold page cache), GPU 78,678 MiB (+1,667 vision encoder), MemAvailable 32 GiB at READY, 30.5 GiB in the bench,
  container 8.9 GiB of 12g. ttft 178 ms; 3-prompt chat 50.6 / code 82.3 (3 runs). Workers (923b, 9 runs): 51.1 / 81.6-82.7, ttft 182 ms => head = workers, no difference.
- 17:33 W2 XSTAT 16 users (D1, 16 x 32K; MemAvailable 32 GiB at READY, 20 GiB in the bench): distinct 16 = 268.8 / 275.2 tok/s (max ttft 1.05 / 1.13 s) WITH the per-layer sync of the log aid; held 16 in flight 179.3 tok/s.
  Head 16 users = workers (266-273).
  EXPERT STATISTICS, MEASURED (735 whole passes, 2 x distinct-16; analysis kit-D/xstat.py, numpy in the pytorch image):
  | passes of 13-16 lanes: 75.9 rows a pass | 222 distinct experts a layer of 512 (43 %) | 3.4 rows an expert | a lane alone 29.6 distinct, lanes pairwise Jaccard 0.084 |
  | round 7's "159 distinct" was inferred from an assumed 225 GB/s; the measured union is 222 => 29.3 GiB of expert bytes a pass |
  | marginal row: distinct = 142.7 + 1.05 x rows => +1.05 experts a layer = 139 MB = 0.62 ms a row | layer 0: 310, layers 36-47: 199 |
  | items of 16 pairs re-read 1.8 % of the expert bytes (3.6 experts a layer have > 16 rows; most rows on one expert 23); 32-pair items 0.2 % |
  | routing weights exactly zero: 0 of 21.0 M (smallest 0.0028) => no exact skip | adjacent-id share of used experts 0.435 = chance (0.434) => storage order by co-activation has nothing to give |
  | SIMULATED sub-passes: one pass 221 distinct/layer; two half-lane sub-passes grouped by expert overlap 289 (x1.31), random halves 298 (x1.35), every lane alone 454 (x2.05) => any split reads MORE; the single pass is already the minimum |
- LEVER BUILT (D2, flag TF_FLASHNEXT_Q38_EXP_PAIRS=N, default 16 = stock): decode expert items of up to N pairs (the kernel's own r0 loop takes rows 17+ on the block the warp has just read).
  ESTIMATE: <= 1.8 % of expert bytes = <= ~1.2 % of a 16-user pass; more with repeated prompts (copies). Building D2 in the window now.
- 17:43 W2 results, MEASURED on head (16 x 32K):
  kp25 reference: distinct 16 = 270.0 / 271.7 / 271.1 (max ttft 1.12-1.14 s), distinct 8 = 210.8  (workers: 266-273, 212) => head = workers.
  D1 with XSTAT: distinct 8 = 211.0 / 214.6, distinct 1 = 79.5.
  EXPERT STATISTICS TABLE (final; raw Mac q38/results_D_xstat_raw.txt; head ~/q38/prof-D/x8.bin, x16.bin):
  +----------------------+-------+----------+--------+----------+-----------+------------+-----------------+----------------+
  | load                 | rows  | distinct | of 512 | rows an  | GiB a pass| extra row  | 16-pair items   | a lane alone / |
  |                      | a pass| a layer  |        | expert   | (experts) | (experts)  | re-read         | pair Jaccard   |
  +----------------------+-------+----------+--------+----------+-----------+------------+-----------------+----------------+
  | 8 users (5-8 lanes)  | 49.0  | 166.4    | 32.5 % | 2.94     | 21.8      | +2.40/layer| 1.1 %           | 36.0 / 0.118   |
  | 16 users (13-16)     | 78.5  | 228.3    | 44.6 % | 3.44     | 30.1      | +1.22/layer| 1.9 %           | 30.7 / 0.085   |
  | prompt passes (16u)  | 25.4  | 61.1     | 11.9 % | 4.16     | 8.5       | -          | 6.9 %           | -              |
  +----------------------+-------+----------+--------+----------+-----------+------------+-----------------+----------------+
  extra row = 319 MB / 1.42 ms at 8 users, 162 MB / 0.72 ms at 16 users (at 225 GB/s). Zero routing weights: 0 of 25.7 M. Adjacent-id share = chance.
  SIMULATED split of a 16-user pass into two sub-passes by expert overlap: x1.31 bytes (random x1.36; each lane alone x2.06). DEAD: one pass is the minimum for its rows.
  D2 BUILD_OK 17:38 (sha 0316f46dd74229f2) = D1 + TF_FLASHNEXT_Q38_EXP_PAIRS. Next: W3 = head ~/q38/kit-D/w3.sh D2 64 (ref / on / ref at 16 users with GPROF + texts, one-user ident off / on).
- 17:45 W2 done, lease released by the script. kp25 held 16 in flight 179.7 tok/s; texts kp25 vs D1+XSTAT 107/107 identical. Min MemAvailable in the 16-user run 18.1 GiB.
  GPROF head 16 users (kp25): 72.6 rows a pass, 180.8 ms a pass: MoE 117.4 (65 %), GDN 32.5, hc 7.6 + 6.8, QSA 8.3, head 7.0, PLE 0.9.
  ARITHMETIC: 72.6 rows -> ~221 distinct x 1.019 = 225 expert reads a layer x 2.7648 MB x 48 = 29.9 GB in <= 117.4 ms => the MoE group moves >= 254 GB/s
  (LPDDR5X-8533 x 16 channels = 273 GB/s theoretical): the MoE group as a whole is at ~93 % of the memory bus. Router + shared expert + glue hide inside that. No work-per-byte lever left.
- 17:46 W3 starting: acquire D 60, head ~/q38/kit-D/w3.sh D2 64 (log w3.log, run dirs ~/q38/runs/D_E_*).
- 17:49 SECOND LEVER written (branch builder-D, build D3 = D2 + it): TF_FLASHNEXT_Q38_ROW_TRIM=1. WHY (arithmetic from measured numbers): at 16 users the dense qmmf lane kernels take ~20 % of GPU time
  (KPROF 3d24: 5.9 of 28.6 ms) = ~37 ms a pass for 3.8 GiB, 2.3x their byte time. A pass of 65-128 rows runs two 64-row blocks and every block multiplies all four 16-row tiles:
  128 rows of mma for ~73-78 live rows = 128 x 4.08e9 x 2 flop / 37 ms = 28 TFLOP/s = the box's measured mma ceiling (prefill: ~30). So the dense side is COMPUTE-bound on padded rows
  (this is also why round 7's 128-row tile gave zero). ROW_TRIM: a block multiplies only its live 16-row tiles (qmmf.cu: stage loop as a generic lambda over a compile-time tile count,
  dispatched 1..4; flag travels in bit 16 of the kernel's `group` argument, so no ABI change; stock path = same code with L = 4). ESTIMATE: 78 rows -> 80 of 128 tiles' work: up to -14 ms of a 180 ms pass (<= +7 %); less where bytes bind.
  Build D3 starts 17:52 inside the window (overlaps W3's first reference's held run and the next engine load: noted).
- 17:57 D3 BUILD_OK (sha 51c11631eda83102, 4.5 min) = D2 + TF_FLASHNEXT_Q38_ROW_TRIM. Head ~/q38/tfbin/D3. Next window W4: head ~/q38/kit-D/w4.sh D3 F "-e TF_FLASHNEXT_Q38_ROW_TRIM=1" "-e TF_FLASHNEXT_Q38_ROW_TRIM=1 -e TF_FLASHNEXT_Q38_EXP_PAIRS=64" with OFF=~/q38/runs/D_E_off1/ident.json
- 17:56 W3 ref1 (D2, flags off), MEASURED: distinct 16 = 271.7 / 272.9 / 271.3; copies 16 = 343.6; held 16 in flight 179.4; waves of 16 = 152.0 (last first token 1.75 s);
  GPROF cumulative (all loads): 67.5 rows, 176.8 ms a pass, MoE 114.1, GDN 32.2, hc 7.4 + 6.7, QSA 8.0, head 7.4. Min MemAvailable 15.9 GiB (waves).
- 18:02 W3 RESULT, MEASURED (D2, head, same window, flag off first): TF_FLASHNEXT_Q38_EXP_PAIRS=64 is NEGATIVE.
  | run | distinct 16 (three) | copies 16 |
  | ref (16-pair items, stock) | 271.7 / 272.9 / 271.3 | 343.6 |
  | EXP_PAIRS=64 | 264.4 / 265.0 / 265.7 (-2.5 %) | 320.7 (-6.7 %) |
  CAUSE (from the kernel source): units are dealt to warps by a fixed stride; an item of 17-64 pairs makes one warp walk the same weight block 2-4 times in a row while its
  neighbours finish, so the launch ends later; stock's second item of 16 pairs runs on another warp at the same time, and re-reads only 1.9 % of the bytes.
  => DEAD, negative. Flag stays in the tree default 16 (= stock) but is NOT handed off. W3 stopped early (kill of w3.sh; its trap removed the engine and released the lease). No second reference run.
- 18:03 W4 starting: acquire D 60; head ~/q38/kit-D/w4.sh D3 F "-e TF_FLASHNEXT_Q38_ROW_TRIM=1" (ref1 / trim / ref2 at 16 + 8 users with GPROF, texts; one user off / on ident). Log kit-D/w4.log, runs ~/q38/runs/D_F_*.
- 18:22 W4 RESULT, MEASURED (D3, head, one window): TF_FLASHNEXT_Q38_ROW_TRIM=1 is ZERO.
  | run | distinct 16 (three) | copies 16 | 8 users | held 16 in flight | GPROF rows / ms a pass: GDN, hc, QSA, head, MoE |
  | ref1 (flag off) | 272.5 / 273.5 / 274.3 | 339.7 | 210.3 | 179.8 | 76.5 / 183.7: 33.27, 7.94 + 7.02, 8.41, 7.09, 118.9 |
  | ROW_TRIM=1 | 272.4 / 271.2 / 265.5 | 339.3 | 211.4 | 178.7 | 76.9 / 184.5: 33.29, 7.96 + 6.99, 8.36, 7.16, 119.6 |
  The flag engaged (engine log line) and every dense group is unchanged to 0.05 ms. So the padded row tiles' mma work is NOT what the dense kernels wait for: my 28 TFLOP/s
  arithmetic was a coincidence (the box's mma capacity is far above it). The lane kernels' time is their staged loads (bytes at ~110-200 GB/s by shape) and small launches. DEAD (zero); not handed off.
  Also shows D3 flag-off = D2 flag-off = kp25 (271-274): the refactor costs nothing.
- 18:23 NEXT (W5, same window): my statistics say an extra row at 16 users costs 0.72 ms of expert bytes (8 users 1.42, one user ~2.4) while the pass yields ~0.27 token a ms,
  so the draft gate tuned for one user (0.4 / 6 drafts) may be too tight at 16 users. Round 4 tested looser gates at 16 users before the stack (zero); round 7 re-fit only for one user.
  Test on the current stack: CONFIDENCE -0.3 / DRAFTS 6, -0.4 / 8, -0.3 / 8, -0.25 / 8 against W4's references (texts with 16 in flight each). head ~/q38/kit-D/w5.sh.
- 18:24 MEASURED from the XSTAT dump (kit-D/lanehist.py, 564 passes of 13-16 lanes): rows a lane 2: 17.1 %, 3: 12.3 %, 4: 14.8 %, 5: 10.5 %, 6: 8.5 %, 7: 34.5 % (= the cap, bonus + 6 drafts), mean 5.10.
  A THIRD OF THE LANES STOP AT THE DEPTH CAP with the running product still >= 0.4, while an extra row costs ~1 ms (0.72 expert bytes + dense) against 3.7 ms a token on average.
  ESTIMATE: DRAFTS 8 at 16 users +2..3 % (one user measured +0.3 % in round 7 because a row costs 2.4 ms there). Gate threshold itself: breakeven p ~0.27 vs 0.4 now: <= 1 %, flat.
  W5 plan (after W4's second reference): D3 variants DRAFTS=8 | DRAFTS=10 + WIDE_ROWS=192 | DRAFTS=8 + CONFIDENCE=-0.3 | ref3; texts with 16 in flight against D_F_ref1.
- 18:32 W4 ref2 (D3 flag off): distinct 16 = 269.3 / 267.2 / 269.0, copies 333.8, 8 users 208.4, held16 176.9; GPROF 75.5 rows, 183.8 ms, MoE 119.0, GDN 33.2. (refs drift ~1.5 % within the window; ROW_TRIM sits between them: ZERO.)
  ROW_TRIM texts vs ref1: held 16 in flight 107/107, waves of 16 128/128 identical; ref2 vs ref1 107/107. W4 killed after ref2 (the one-user ident of a zero lever is not worth window time).
- 18:32 W5 started: acquire D 60; w5.sh D3 G ~/q38/runs/D_F_ref1 d8 | d10w192 | d8c03 | ref3. Log kit-D/w5.log, runs D_G_*.
- 18:47 W5 so far, MEASURED (D3, head; refs in this window: ref1 272.5 / 273.5 / 274.3 held 179.8; ref2 269.3 / 267.2 / 269.0 held 176.9):
  | DRAFTS=8 | distinct 16 = 275.5 / 273.9 / 273.4 | 8 users 210.3 | held16 176.0 | GPROF 75.4 rows, 180.9 ms | texts 107/107 | => inside the reference spread (zero to +1 %)
  | DRAFTS=10 + WIDE_ROWS=192 | 263.3 / 265.5 / 268.2 | 205.5 | 176.6 | 77.0 rows, 182.7 ms | 107/107 | => negative
  Rows a pass did not grow with the deeper cap (75-77), so the pile at 7 rows a lane is not simply TF_FLASHNEXT_DRAFTS: the lane core's allocator (core/lanes/depth.zig, allocateDrafts on learned
  per-depth rates and timed row costs) decides the counts. Follow-up queued: head ~/q38/kit-D/w6.sh (XSTAT with DRAFTS=8 after w5 ends; log kit-D/w6.log) to see the rows-a-lane histogram.
  Handoff dir written: Mac q38/handoff/D/{xstat-log-aid,dead-exp-pairs,dead-row-trim} (patches apply on 52b32e1; NOTE.md to come). Branch builder-D commit 0c0eda1 (local only).
- 19:00 W5 DONE (lease released by the script), MEASURED:
  | DRAFTS=8 + CONFIDENCE=-0.3 | 265.7 / 259.7 / 263.1 | 8 users 209.8 | held16 175.7 | 77.7 rows, 181.4 ms | 107/107 | negative
  | ref3 (0.4 / 6) | 266.3 / 270.7 / 273.0 | 210.4 | 177.6 | 70.8 rows, 179.4 ms | 107/107 |
  Within W5's load mix DRAFTS=8 raises rows a pass 70.8 -> 75.4 for +1.5 ms (0.33 ms a row) but tok/s stays inside the reference spread (274.3 vs 273.4 / 270.0; held 176.0 vs 177.6 / 179.8).
  => draft depth / gate at 16 users: ZERO to negative; MiaAI's 0.4 / 6 stays. DEAD.
- 19:00 w6 (XSTAT with DRAFTS=8) running under its own 12-min lease. Handoff NOTE written: Mac q38/handoff/D/NOTE.md.
- FOUND for builder C (ESTIMATE): 208 of 880 passes in the 16-user runs are prompt passes that read 8.5 GiB of expert bytes alone; riding in the decode pass (KEEP_INPASS, U3) would save ~40 ms per arriving prompt.
- 19:05 w6 DONE (own lease released), MEASURED: XSTAT with DRAFTS=8, 2 x distinct 16 (coding): 272.5 / 264.8 tok/s; rows a lane 9: 39.4 % (new cap), mean 7.05; ~108 rows a pass.
  So deeper chains do fill (coding prompts are confident) and the rows are cheap, but tokens a second do not move: DEAD, confirmed with the histogram.
  CORRECTION of my 19:00 note: passes without a lane table in x16.bin are 144 lone-stream decode passes (1-7 rows, bench tails) + only 64 prompt passes (kit-D/lone.py), not 208 prompt passes.
  PREFILL_MANY already batches 139 prompts into 64 passes; the KEEP_INPASS estimate for builder C shrinks to <= ~2 % on churn loads.

## BUILDER D END STATE (19:07 UTC 2026-11-13)
- Lease D released (w6's trap; lease.sh status shows only A, B, C). No q38-run-D / q38-build-D / analysis container, no kit-D script running on the head.
  Head MemAvailable 111 GiB (the window is still open for A, B, C, so DeepSeek is down by their leases, not mine); GPU apps: vision encoder only (1,667 MiB). Checkpoint, tfbin, build-D, prof-D evicted from page cache.
- On the head (kept for the next round): ~/q38/models/nvidia-fp4-local (96 GiB), ~/q38/tfbin/{kp25,D1,D2,D3}, ~/q38/build-D (3.8 GiB), ~/q38/kit-D, ~/q38/prof-D/{x8,x16,x16d8}.bin (360 MB), ~/q38/runs/D_*. Disk free 895 GiB.
- Mac: q38/PROGRESS-D.md, q38/kitD/ (scripts + tools), q38/wt/D (branch builder-D, commit 0c0eda1, not pushed), q38/handoff/D/ (NOTE.md + 3 patch dirs), q38/results/rD/ (logs + run dirs), skill reference qwen38-fastest-build.md (new section).
- RESULT: no positive lever in the many-user MoE area. EXP_PAIRS negative (-2.5 %), ROW_TRIM zero, draft depth / gate zero to negative; router / exact skips / storage order / sub-passes dead by measurement or arithmetic.
  Handoff = the XSTAT log aid only (measuring; exact). Recipe unchanged.
- NOT DONE: one-user ident 51/51 for the two flags (both dead, so no window time spent); waves-16 for EXP_PAIRS (killed as negative). ROW_TRIM has held 107/107 + waves 128/128.
- RESUME: ./h.sh 'bash ~/q38/shared/lease.sh acquire D 60' ; ./h.sh 'nohup setsid bash ~/q38/kit-D/w5.sh D3 TAG ~/q38/runs/D_F_ref1 "name|-e FLAG=v" > ~/q38/kit-D/w5.log 2>&1 < /dev/null &'

## Round 9 (builder D, head; sampled requests + upstream decode kernels)
- 2026-11-13T23:00:06Z start. Reading docs. No lease, no engine.
- 23:03 BASE = 0b5cf9b (handoff/BASE.txt). Mac worktree q38/wt/D9 branch r9-D. I2 binary (sha 09108beff56b8be2) copied 923b -> head ~/q38/tfbin/I2.
- 23:06 W9a started (lease D 85 min): head ~/q38/kit-D/w9a.sh I2 S1: on | off | log (STEP_LOG=2, gate off, HDUMP) | on2; seeded sampled ruler kit-D/sq.py (40 held chat prompts x 256 tok, seed 9000+id, T0.7 think off/on, greedy ref, sparkDash-method prose seeded). Log kit-D/w9a.log, runs ~/q38/runs/D_S1_*. Script releases the lease itself unless kit-D/KEEP_LEASE exists.
- 23:15 READ (source, base 0b5cf9b; no run needed) HOW SAMPLING WORKS IN THIS ENGINE:
  * The sampler is KEYED Gumbel-max (core/lanes/sampling.zig): noise for token id at a position = splitmix64(seed, position, id). The token at a position is a pure
    function of (that row's top-k logits, seed, position). Request field "seed" is honoured (server.zig resolveSampling); without it the seed is fresh entropy
    (so round 8's 55.1 vs 49.5 compared different texts: NOT a valid pair). With top_k set (server default --top-k 20) the draw reads ONLY the row's 20 best BF16 logits
    (top-p / min-p / temperature are applied inside those 20; no whole-row log-sum-exp on the verify side).
  * Draft acceptance when sampling = EXACT MATCH: the draft head draws its token with the same key (drawPickWith), the verify row draws with the same key, a draft is kept
    when the two tokens are equal. So the output for a seed does not depend on drafts at all (seed-identical AND distribution-exact; it is the Gumbel-coupled form of
    speculative sampling, not the accept/reject-with-resample rule). Nothing to change there.
  * MiaAI's TF_FLASHNEXT_SAMPLE_CAND=1 selects the 20 candidates on the GPU (fn_row_cand), HOST_DRAW_THREADS=7 draws whole rows on host threads when candidates do not apply.
  * What our levers do when sampled: CERT gate (cuda_lanes_verify.zig) => whole head (1.18 GiB) for every sampled pass; HEAD2 off for sampled chains; the draft gate
    (running product >= 0.4) is fed the DRAWN token's temperature-1 probability under the draft head, so a sampled chain stops at the first unlikely draw.
  * sparkDash 1.8.8's own decode bench (server/collectors/DecodeBench.js) is temperature 0, top_p 1, thinking OFF. The "prose T0.7 think-on" row of qk.py sd is MiaAI's
    reporting variant, not sparkDash's protocol.
- 23:15 UPSTREAM DECODE KERNELS, read in research/src/tf-main against base 0b5cf9b:
  | PR | what | in our base? |
  | #1171 half tiles for small decode projections | qmmf half tile, below 96 blocks | YES: ported in round 6 as TF_FLASHNEXT_Q38_HALF_BELOW=96 (recipe default HALF=96; qmmf.cu comment names #1171) |
  | #1925 MTP head experts on the last row only | | YES: ported in round 6 as TF_FLASHNEXT_Q38_MTP_LAST=1 (recipe default) |
  | #1351 NVFP4 decode experts in equal waves (+ two warps a unit when every half unit fits) | src/cuda/experts_waves.zig, nvfp4_expert_waves / nvfp4_expert_half | NO. Exact (same unit program; only which warp takes which unit). Upstream +0.6 % one stream. To port: waves form. |
  | #1702 MTP head norm of 10,240-value rows in tiles | rms_tiles.cuh, tf_rms_rows_sums_tiles | NO. One norm a chain step; size it with a per-kernel profile (KPROF) before porting. |
- 23:32 W9a FIRST RESULTS, MEASURED (head, I2 = base 0b5cf9b, one user 3 x 262K, 40 held chat prompts x 256 tokens, request seed 9000+id, server top_k 20 / top_p 0.95):
  THE SAMPLED RULER IS VALID: the engine honours "seed"; same seed => levers off vs recipe default give IDENTICAL text, 40/40 (thinking off) and 40/40 (thinking on,
  reasoning + answer compared). So sampled output is seed-identical under the levers (lossless holds for sampling).
  | bench (tokens / decode time) | levers off | recipe default | change (per-prompt ratio, 95 % interval) |
  | T 0.7, thinking off | 45.55 | 47.17 | +3.6 % (+3.73 +- 0.29) |
  | T 0.7, thinking on  | 48.55 | 50.25 | +3.5 % (+3.53 +- 0.25) |
  | greedy, thinking off (same 40 prompts) | 46.71 | 50.57 | +8.3 % |
  | sparkDash-method prose T0.7 think-on, 6 seeded rounds | (running) | mean 52.98 (47.1 .. 56.6 by seed) | |
  => The recipe is FASTER than levers-off when sampling too (round 8's 55.1 vs 49.5 was two different random texts; one text swings 47-57 by seed),
  but it keeps only +3.6 % of its +8.3 %: sampled passes take the whole LM head (CERT gate) and sampled is 7 % slower than greedy on the same prompts (47.2 vs 50.6).
- 23:50 W9a log run + offline study, MEASURED (head, I2, TF_STEP_LOG=2, gate off = every chain 6 deep; kit-D/gsim.py; kit-D/hstudyk.py on 6,025 dumped head rows of SAMPLED passes):
  * Draft acceptance when sampling = greedy's: first draft kept 0.676 (T0.7 think off) / 0.671 (think on) / 0.681 (greedy); deeper 0.60-0.70 in all three.
    The keyed (Gumbel-coupled) draw costs no acceptance. The drawn token is not the head's best in 13-16 % of tested drafts (kept 0.26-0.36 of those).
  * Stop rule, SIMULATED on the logged rounds: stock rule (drawn token's probability, line 0.4) 45.7 / 46.0; best alternative from this log (head's top probability, 0.3-0.4)
    45.8 / 46.8 => 0 .. +1.7 %. Small. A keyed-race-margin signal needs the D9a log (QSAMP lines).
  * WHERE THE SAMPLED DEFICIT IS: levers off sampled vs greedy -2.5 %; levers on -6.7 %. The extra 4 % = the certified head not serving sampled passes.
  * CERTIFIED TOP-K HEAD, exact scheme (thought through): the keyed draw reads only the row's top_k (20) BF16 logits and ids. theta = k-th largest LOWER bound of the row;
    at least k entries are >= theta, so every entry whose UPPER bound < theta - 2 bf16 steps is strictly below the k-th largest rounded logit: not in the top k, cannot tie.
    Exact rows for all other entries => fn_row_cand sees the same 20 values and ids => same token for the same seed. No normalisation over the vocabulary is needed
    (top-p / min-p / temperature act inside the 20). NOT covered (stay on the whole head): top_k = 0 or > 64, grammar masks, logprobs.
  * QUANTIFIED (7-row windows; exact tiles of 3,880 a window; share of head bytes incl. the coarse copy):
    | coarse copy | bound | k=1 | k=5 | k=10 | k=20 | k=40 |
    | MXFP8 (the engine's copy today) | +-3.1 logits | 180 tiles, 0.55 | 665, 0.67 | 1098, 0.78 | 1662 tiles, 0.93 (63 % of windows over the fallback line) | 2411, 1.12 |
    | int8, scale per 32              | +-0.74       | 15 tiles, 0.50 | 58, 0.51 | 110, 0.53 | 205 tiles (106 runs), 0.55 | 372, 0.60 |
    => With the MXFP8 copy a certified top-20 head saves nothing (DEAD by measurement). With an int8-per-32 copy it reads 55 % of the head: ~2.4 ms of a ~58 ms sampled round
    = about +4 % for sampled requests (ESTIMATE), and greedy windows would drop 180 -> 15 exact tiles as well. Needs a lane kernel that reads int8 codes.
- 23:51 D9a built on head (sha 032fc1be224d0860; base + EXP_WAVES (#1351 wave form) + RMS_TILES (#1702 port) + SGATE knobs/QSAMP log). W9b started (lease renewed 100 min, KEEP_LEASE set: script does NOT release): D9a S2: log (STEP_LOG=1 gate off) | ref | wr (EXP_WAVES=1 RMS_TILES=10) | ref2. Log kit-D/w9b.log, runs D_S2_*. I must release lease D by hand.
- 00:01 D9b built (sha 771c2b0de0db2f91) = D9a + certified TOP-K head (TF_FLASHNEXT_Q38_CERT_TOPK=K) + SM8 coarse copy (TF_FLASHNEXT_Q38_CERT_CODE=sm8: sign-magnitude 7-bit integer codes in MXFP8's container, new qmmf mode SM8, fn_cert_select_k, fn_cert_bound check). Queued head kit-D/w9c.sh (waits for w9b): D9b S3: sm8chk (CERT=2 check) | sm8 | sm8g | ref3. Log kit-D/w9c.log.
- 00:06 D9a log run (QSAMP), MEASURED + SIMULATED (gsim.py on D_S2_log; same 2,973 / 3,089 rounds as the I2 log = D9a's new code is inert with flags off):
  kept rate against the keyed-race margin: 0-1: 0.25-0.28, 1-2: 0.36, 2-3: 0.46, 3-4: 0.55, 4-6: 0.63, one-token set: 0.856 (n 4,184);
  against the stock signal (drawn token's temperature-1 probability): >= 0.97: 0.933 (n 2,980), 0.9-0.97: 0.74, 0.5-0.6: 0.42, < 0.1: 0.14.
  => the stock signal already separates better than the race margin or the tempered-set probability. Simulated tok/s (think off / on):
  stock 0.4: 45.7 / 46.0 | head's top probability 0.3-0.4: 45.8 / 46.8 | tempered-set p: 45.1 / 45.9 | margin logistic: 45.4 / 46.7 | sqrt(drawn x top) 0.2-0.4: 45.8 / 46.7.
  => sampled stop rule: 0 .. +1.7 % at best (SIMULATED). One real run of SGATE=1 (head's top probability, line 0.35) queued to settle it; otherwise the stock rule stays.
- 00:24 W9b, MEASURED (D9a, head): ref (flags off) = I2: s07off 47.12 / g0off 50.55 / sd 53.00 (I2: 47.17 / 50.57 / 52.98; texts 40/40 + 40/40 identical) => ruler repeats to 0.1 % across engine starts.
  wr (EXP_WAVES=1 + RMS_TILES=10; upstream #1351 wave form + #1702 port): texts identical 40/40 (sampled) + 40/40 (greedy); greedy 50.55 -> 50.62 (+0.14 %; per prompt +0.28 +- 0.20);
  sampled 47.12 -> 46.70 with a wide spread (+-1.10): a disturbance (my own ssh analysis ran on the head during that run; sampled draws use 7 host threads). To be repeated clean.
  LESSON: no analysis commands on the head while a timed sampled run is in flight.
- 00:32 W9b ref2 (D9a flags off, last): s07off 46.73 / g0off 50.71. So references in this window: sampled 47.12 -> 46.73 (-0.8 %), greedy 50.55 -> 50.71 (+0.3 %).
  wr (waves + rms tiles) = 46.70 / 50.62 sits ON ref2 (0.0 % / -0.2 %) and +0.14 % over ref: EXP_WAVES + RMS_TILES = ZERO within +-0.3 % (texts identical 40/40 + 40/40).
  The sampled reference moved between engine starts while greedy did not: runs after the first keep the checkpoint in page cache (container at 10.6 of 12 GiB), sampled draws use host threads.
  => compare sampled runs only with a reference taken in the same state (neighbouring run), which the S3 queue does (ref3 last).
- 00:37 CERTIFIED TOP-K HEAD WITH THE SM8 COPY WORKS (D9b, head, check mode CERT=2, CERT_CODE=sm8, CERT_TOPK=20, T0.7 sampled, 40 prompts; MEASURED, engine's own check lines):
  top-k passes 3,254, fallback to the whole head 0, exact tiles a pass 366 of 3,880 (9.4 %), 57.8 launches a pass;
  11,462 sampled rows: the 20 best (values + columns, as fn_row_cand selects them) of the certified row == the whole head's row in ALL (topk-mismatch 0);
  bound |coarse - exact| <= c h tested on every entry of every checked row: 0 violations of 3.23 billion; best-mismatch 0, value-mismatch 0 of 310 M recomputed entries.
- 00:49 sm8 speed run first numbers (D9b CERT_CODE=sm8 CERT_TOPK=20): s07off 49.00, s07on 51.70 (refs 47.12 / 46.73 and 50.25). Final suite queued behind w9c: head kit-D/w9e.sh D9b F: best (ident held sdreal x2 s07off s07on + SGATE via knobfile) | ref | u16ref | u16best (16 x 32K: distinct 16 x3, held 16 in flight) | off (sdreal x2, s07off). Lease D renewed 150 min; KEEP_LEASE set -> I release by hand. Log kit-D/w9e.log, runs D_F_*.
- 00:54 W9c sm8 run, MEASURED (D9b, head, one user, same 40 seeded prompts; refs: D9a flags off first 47.12 / 50.55 / sd 53.00, last 46.73 / 50.71; I2 s07on 50.25):
  | bench | recipe default (MXFP8 copy, greedy-only cert) | CERT_CODE=sm8 + CERT_TOPK=20 | change |
  | T0.7 think off | 47.12 (46.73 last) | 49.00 | +4.0 % (+4.9 % vs last) |
  | T0.7 think on  | 50.25 | 51.70 | +2.9 % |
  | greedy think off | 50.55 (50.71 last) | 51.43 | +1.7 % (+1.4 %) |
  | sparkDash-method prose T0.7 think-on, 6 seeded rounds | 53.00 | 55.13 | +4.0 % |
  Engine line: "SM8 copy of the head made (625 MiB), mean residual 0.0087 of a row's norm" (MXFP8: 0.0266). Greedy passes need ~10 exact tiles a call instead of 47.
  Texts vs reference: to be compared when no timed run is in flight (sq.py cmp). Branch r9-D commit 0c6dfb9 (local).
- 01:03 IDENTITY + PAIRED SPEED of the certified top-k head on the SM8 copy (D9b, head; sq.py cmp, same seeds; MEASURED):
  | pair (reference D9a flags off -> CERT_CODE=sm8 CERT_TOPK=20) | texts | totals | per-prompt ratio, 95 % |
  | T0.7 think off, 40 x 256 | 40/40 identical | 47.12 -> 49.00 (+4.0 %) | +3.83 +- 0.18 % |
  | T0.7 think on, 40 x 256 (ref I2) | 40/40 identical (reasoning + answer) | 50.25 -> 51.70 (+2.9 %) | +2.51 +- 0.77 % |
  | greedy think off, 40 x 256 | 40/40 identical | 50.55 -> 51.43 (+1.75 %) | +1.57 +- 0.24 % |
  | sparkDash-method prose T0.7 think-on, 6 seeded rounds | 6/6 identical | 52.81 -> 54.94 (+4.0 %) | +4.03 +- 0.25 % |
  check-mode run (CERT=2): texts 40/40 + 40/40 identical too. SM8 copy without TOPK (sm8g): sampled 47.03 (= reference, sampled passes on the whole head), greedy 51.31 (+1.0..1.5 %).
  => SPLIT: SM8 copy alone = greedy +1.0..1.5 %; + TOPK=20 = sampled +3.8..4.0 %. Both exact.
- 01:15 W9c DONE. ref3 (D9b flags off, LAST reference): s07off 46.76 / s07on 49.76 / g0off 50.56. References first -> last: sampled think off 47.12 -> 46.76, think on 50.25 -> 49.76,
  greedy 50.55 -> 50.56. sm8 + topk against the LAST reference: sampled think off +4.8 %, think on +3.9 %, greedy +1.7 %.
  Handoff patch written: Mac q38/handoff/D/r9-cert-topk-sm8/D9b-cumulative.patch (sha256 7071adf8d945b507..., applies on 0b5cf9b; = the tree of binary D9b 771c2b0de0db2f91).
  w9e final suite running since 01:10 (best first). Lease D to 02:49 (renew before 02:00 for the later runs).
- 01:25 F_best so far: ident 51 saved, held ALL 64.18 (n 104). sdreal inside w9e FAILED (head ~/q38/kit/sdw/sd lacks src/shared/llmPrompts.js; not touching the shared kit): queued head kit-D/w9f.sh D9b G off | best = sparkDash 1.8.8 itself from ~/c1fast/sd (read-only mounts), 2 runs an engine, after w9e. Log kit-D/w9f.log, runs D_G_*.
- 01:37 F_best (D9b, CERT_CODE=sm8 CERT_TOPK=20; one engine), MEASURED: ident 51 saved; held 107 x 256 greedy: 64.18 (n 104); s07off 48.92, s07on 52.22 (second engine start: repeats 49.00 / 51.70);
  against ref3: texts 40/40 + 40/40 identical, +4.6 % / +4.9 %. Engine: 10,558 top-k passes, 0 fell back to the whole head, 359 exact tiles a pass (9.3 %), 56.8 launches.
  SAMPLED STOP RULE, MEASURED in the same engine through the knob file (SGATE=1: head's top probability, line 0.35): think off 48.92 -> 48.29 (-1.3 %, +-1.2), think on 52.22 -> 52.24 (+0.05 %);
  texts 40/40 + 40/40 identical (a stop rule never changes the output). => ZERO to negative: DEAD. MiaAI's rule (drawn token's probability, 0.4) stays for sampled streams.
- 01:55 FINAL-SUITE IDENTITY, one user (D9b: flags off "ref" vs CERT_CODE=sm8 CERT_TOPK=20 "best"; MEASURED): ident 51/51 identical, held 107/107 identical (greedy, 256 tok),
  sampled 40/40 (again, third engine pair). Speed: held-104 63.41 -> 64.18 (+1.2 %); s07off 47.19 -> 48.92 (+3.66 %, per prompt +3.61 +- 0.26).
  Lease D renewed to 03:45. Running: u16ref / u16best (16 x 32K) / off; then w9f (sparkDash 1.8.8 itself, off + best).
- 02:07 16 USERS (head, 16 x 32K, D9b; MEASURED, greedy code prompts): flags off: distinct 16 = 271.1 (cold) / 278.0 / 287.6, held 16 in flight 189.2;
  CERT_CODE=sm8 CERT_TOPK=20: 283.8 / 287.0 / 284.2, held 16 in flight 191.9 (+1.4 %). Texts held-107 with 16 in flight: 107/107 identical to flags off AND to the one-user texts.
  OPEN RISK found by arithmetic: 16 SAMPLED users in one wide pass keep ~80 exact tiles a row -> over the 1,500-tile line -> the coarse pass is wasted (est. -1.4 %).
  D9c adds TF_FLASHNEXT_Q38_CERT_TOPK_ROWS (default 16: a top-k pass of more rows goes straight to the whole head); to be built after w9f and measured with 16 sampled users.
- 02:45 w9f done (sparkDash 1.8.8 itself, off + best, 2 runs each). D9c (commit f5209dd: + CERT_TOPK_ROWS, default 16) building on head; w9g queued behind the build: D9c H: s16ref | s16cap0 | s16cap16 (16 SAMPLED users, held-107 x 256 at T0.7 seed 4242, 16 in flight, twice) | one (one-user s07off g0off). Log kit-D/w9g.log. KEEP_LEASE set: I release by hand.
- 02:46 sparkDash 1.8.8 ITSELF (head ~/c1fast/sd + sdrun.mjs decode; its protocol is temperature 0, top_p 1, thinking off, 256 tokens; 2 runs x 5 rounds an engine; D9b; MEASURED):
  | metric (mean of 5 rounds; run 1 / run 2) | levers off (MiaAI configuration) | best = recipe default + CERT_CODE=sm8 CERT_TOPK=20 | change |
  | c1 prose | 56.61 / 56.21 | 61.43 / 61.14 | +8.6 % |
  | c1 code | 91.96 / 91.61 | 99.81 / 98.72 | +8.2 % |
  | c1 structured | 105.46 / 105.04 | 114.19 / 113.08 | +8.0 % |
  | catalog prose (16 prompts) / structural (18) / code (32) | 43.5, 43.6 / 65.9, 66.0 / 80.9, 80.9 | 47.5, 47.6 / 71.7, 71.4 / 87.3, 87.6 | +9.2 / +8.6 / +8.1 % |
  | first token | ~90 ms | ~68 ms | |
  (sparkDash's own test is greedy; the sampled "prose T0.7 think-on" row people quote is MiaAI's variant: on my seeded form of it: off 51.40, default 53.00, best 55.13.)
  Raw: head ~/q38/runs/D_G_off, D_G_best (sdrun_1.log, sdrun_2.log).
- 03:13 16 SAMPLED USERS (D9c, head 16 x 32K, held-107 x 256 at T0.7 seed 4242, 16 in flight, run a = cold prompts, run b = prompts cached; MEASURED):
  | config | run a | run b | texts vs reference |
  | flags off (reference) | 189.7 | 190.2 | a == b 107/107 (16 sampled users in flight are reproducible) |
  | sm8 + topk, no row limit (= D9b) | 181.3 (-4.4 %) | 185.1 (-2.7 %) | 107/107 identical |
  | sm8 + topk, TOPK_ROWS=16 (D9c default) | 182.5 (-3.8 %) | 188.9 (-0.7 %) | 107/107 identical; greedy held-16 107/107 identical |
  => EXACT but NEGATIVE with several sampled users in one pass: rows of different streams share few exact tiles (~80 a row), so the pass runs the coarse copy and then
  most of the head. A lone stream's 4-7 rows share theirs (359 tiles a pass). FIX being measured: TOPK_ROWS=8 (env on D9c, run s16cap8 queued: kit-D/w9h.sh, log w9h.log, runs D_H2_*).
- 03:28 D9c (sha 3afe212327473c17), MEASURED: one user best: s07off 48.73 / g0off 51.24, texts 40/40 + 40/40 identical to D9b's. 16 sampled users with TOPK_ROWS=8 (env):
  run a 188.9 / run b 190.2 (reference 189.7 / 190.2), texts 107/107 identical => the many-sampled-user loss is gone (zero). Recommended: CERT_TOPK_ROWS=8.

## BUILDER D ROUND 9 END STATE (03:30 UTC 2026-11-14)
- Lease D RELEASED 03:27:57 (status: leases A, B only; the window stays open for them, guard 04:06). No q38-run-D / build / study container; no kit-D script running;
  KEEP_LEASE removed. Head MemAvailable 117.5 GiB (DeepSeek is down by A's and B's leases, not mine); GPU: vision encoder only (1,667 MiB). Page cache evicted.
- SAMPLED TABLE (head, one user, 40 held chat prompts x 256 tok, same seeds; tokens / decode time; all pairs text-identical):
  +---------------------------------------+------------+----------------+----------------------------+
  | bench                                 | levers off | recipe default | best: + sm8 copy + top-k 20|
  +---------------------------------------+------------+----------------+----------------------------+
  | T 0.7, thinking off                   | 45.55-45.66| 47.12-47.19    | 48.73-49.00 (+7.1 % on off)|
  | T 0.7, thinking on                    | 48.55      | 49.76-50.25    | 51.70-52.22 (+6.5..7.5 %)  |
  | greedy, thinking off (same prompts)   | 46.71      | 50.55-50.71    | 51.24-51.43 (+10 %)        |
  | sparkDash-method prose T0.7 think-on  | 51.40      | 53.00          | 55.13 (+7.3 %)             |
  | sparkDash 1.8.8 itself prose/code/str | 56.4/91.8/105.3 | not run   | 61.3/99.3/113.6            |
  | 16 sampled users (held, 16 in flight) | not run    | 189.7 / 190.2  | 188.9 / 190.2 (ROWS=8)     |
  +---------------------------------------+------------+----------------+----------------------------+
  PER LEVER when sampling: CERT (MXFP8, greedy-only) = nothing for sampled passes (they took the whole head) -> FIXED by CERT_TOPK on the SM8 copy (+3.7..4.8 %);
  the other levers (fused read-out, half tiles, MTP_LAST, LATE_SKIP, KEEP_INPASS, LANE_REUSE) are kernel / first-token levers and keep their gain: default vs off = +3.5 % sampled
  (not split one by one: no flag HURTS; ATTN_LANES / MANY / GDN_MANY act only with several users; COPY_ROWS only on copy windows: 2 of 2,973 rounds in chat).
  Draft acceptance sampled = greedy (0.676 vs 0.681): nothing to fix. Stop rule for sampled chains: alternative signals measured zero to negative.
- HANDOFF: Mac q38/handoff/D/r9-cert-topk-sm8/ (D9c-cumulative.patch sha256 ac9f022ac811cd86..., NOTE.md, tools); q38/handoff/D/r9-note-for-B.md (HEAD2 on sampled chains).
- DEAD / ZERO this round: EXP_WAVES + RMS_TILES (upstream #1351 wave form, #1702): zero; sampled stop-rule signals (SGATE): zero to negative; certified top-20 on the MXFP8 copy: dead by study.
  Upstream #1171 and #1925 were already in the base (HALF=96, MTP_LAST=1).
- NOT DONE: #1351's half-unit kernel (two warps a unit; applies only to one-row passes); one-launch gather of the exact tiles (57 launches a sampled pass, est. +0.4 %);
  per-lever leave-one-out under sampling; edit / agent / needle / image / qeval on the new flags (integrator); HEAD2 for sampled chains (B).
- On the head: ~/q38/tfbin/{I2,D9a,D9b,D9c}, ~/q38/build-D/tf = Mac wt/D9 f5209dd, ~/q38/kit-D (sq.py, gsim.py, hstudyk.py, study.sh, run9.sh, w9a/c/e/f/g/h.sh + logs),
  ~/q38/prof-D/h9.{x,n} (293 MB head-row dump of sampled passes), runs ~/q38/runs/D_{S1,S2,S3,F,G,H,H2}_*. Mac: q38/results/rD9 (logs + runs), q38/kitD, q38/shipD9.sh.
- RESUME: cd ~/.hermes/cache/scratch/q38; ./shipD9.sh; ./h.sh 'bash ~/q38/shared/lease.sh acquire D 60'; ./h.sh 'cd ~/q38/kit-D && bash build.sh NAME';
  ./h.sh 'cd ~/q38/kit-D && nohup setsid bash w9g.sh D9c TAG "name|on|-e FLAG=v|s07off g0off" > wX.log 2>&1 < /dev/null &'; release: ./h.sh 'bash ~/q38/shared/lease.sh release D'.

## Round 10 (builder D, head; leave-one-out audit + sampled path + #1351 half unit)
- 2026-11-14T04:32:42Z start. Read PROGRESS-D/A/B, notes. Head: MemAvailable 111 GiB, window open (leases A->06:01, B->05:41), no D engine. Base 7061396 (integ9), Mac worktree q38/wt/D10 branch r10-D. No lease held yet.
- 04:39:41 D10a (commit b895a0f on r10-D): knob-file words for the default levers (cert, cert_topk, cert_topk_rows, hcf, hcf_half, half_below, mtp_last, late_skip, keep_inpass, lane_reuse, head2, head2_sampled, deep_off, stop2, gdn_many, best_lanes, attn_lanes, qsa_cert) + parsed knob table (a knob read = a clock read) + CERT_TOPK_ROWS default 8 (FOUND: integ9 has D9b, not D9c: the 16-sampled-user loss -3..4 % is in integ9). Lease D acquired (to 06:36), building D10a on the head (kit-D/build_D10a.log).
- 04:43:15 D10a BUILD_OK sha 528ae962f5987f4d (head ~/q38/tfbin/D10a). head2_r9.bin copied 923b -> head ~/q38/prof-D (sha fce7c913060d45f2 = the measured file). New kit: kitD/run10.sh (round-9 default env on top of run9.sh + knob file), aud.py (per-prompt interleaved leave-one-out through the knob file, ref twice per prompt = ruler noise), aud16.sh + aud16rep.py (16 users), wq.sh (resumable job runner: keeps lease D, restarts the engine before its 112-min deadline, releases the lease at the end unless kit-D/KEEP_LEASE). STARTED 04:46 UTC: head kit-D/wq.sh D10a A jobA1.txt (log kit-D/wA1.log, out ~/q38/runs/D_A_k1): ident, held-107 greedy x 14 variants, chat40 T0.7 x 16 variants, first token, chat40 T0.7 thinking x 9. Report: python3 -I kit-D/aud.py rep FILE.
- 04:52:09 D10b BUILD_OK sha 911f92797210d918 (commit 75cff9e) = D10a + TF_FLASHNEXT_Q38_EXP_HALF (port of upstream #1351 nvfp4_expert_half: two warps a unit; gate-up when 2 x items x 20 <= resident warps; knob exp_half) + TF_FLASHNEXT_Q38_CERT_GATHER (exact tiles of a certified pass in one launch a row tile through a tile list in the kernel's unused part argument, group = -1; only flagged tiles; knob cert_gather). Built while the audit's ident ran (aud.py waits on kit-D/PAUSE). Audit pair phase started 04:50 UTC on D10a. QUEUE: jobB1.txt (D10b: EXP_HALF + GATHER via knobs, incl. no-draft streams), jobA2.txt (16 users), env engines, sparkDash, fresh set.
- 04:56:54 QUEUE on the head (kit-D/chain.sh under ONE lease, KEEP_LEASE while it runs; it releases lease D and removes the engine at its end; stop after the current job: touch kit-D/STOPCHAIN): after wA1: jobB1 (D10b: EXP_HALF, CERT_GATHER; out D_B_b1) -> jobA2 (16 users, knob audit, out D_A_u16 + /s sampled) -> jobA3 (env engines eref1, edeep, ehead2, emx, efast0, ecopy16, eoff, eref2: knob-vs-env check, levers that only an env flag can switch, sparkDash 1.8.8 itself off vs default, edit set) -> jobA4 (16 users env: copy16, many0) -> jobA5 (fresh set, every 4th prompt, out D_A_k1f). Logs kit-D/w<TAG>_<job>.log, chain.log.
- 05:07:28 SAFETY NOTE (measured on the head): the round-9 default CACHE_ELASTIC lets the prompt cache grow to all the memory the reserve leaves (110 MiB a kept state, budget 20,042 MiB at 3 x 262K): GPU use 78.8 -> 89.6 GiB after 160 prompts, MemAvailable 30 -> 21 GiB, heading for ~10 GiB (reserve 9; watchdog line 6). run10.sh now starts engines with TENSORFOLD_MEMORY_RESERVE_GIB=13 on the head (3 x 262K and 16 x 32K still fit); the engine already running keeps 9 until its restart (~06:31).
- 06:14:54 AUDIT RESULT 1, MEASURED (head, D10a sha 528ae962f5987f4d = integ9 + knob words, one engine 3 x 262K, recipe round-9 defaults, 107 held prompts (every 3rd of 320) x 256 tokens greedy thinking off; every prompt sent once per variant back to back, order rotated, reference twice; lever value = time with ONLY that lever off / reference time - 1; raw head ~/q38/runs/D_A_k1/held_g0.jsonl, Mac q38/results_D10_held_g0.txt):
  set held mode g0off: prompts 107; reference (mean of ref, ref2) 67.19 tok/s
  lever off            texts   tok/s lever value %   per-prompt % +- 95%     chat     code     math
  ref               104/104    67.18         +0.01        +0.03 +- 0.05    +0.01    +0.03    +0.00
  cert              104/104    63.88         +5.18        +4.96 +- 0.18    +5.57    +4.39    +4.05
  hcf               104/104    66.07         +1.69        +1.61 +- 0.09    +1.77    +1.57    +1.43
  hcfhalf           104/104    66.47         +1.09        +1.06 +- 0.08    +1.11    +1.12    +0.91
  half              104/104    67.33         -0.21        -0.20 +- 0.07    -0.22    -0.13    -0.28
  mtplast           104/104    66.97         +0.33        +0.40 +- 0.10    +0.26    +0.50    +0.50
  lateskip          104/104    66.97         +0.33        +0.34 +- 0.10    +0.36    +0.39    +0.11
  keepin            104/104    67.20         -0.02        -0.02 +- 0.08    -0.02    +0.01    -0.09
  reuse             104/104    65.61         +2.41        +3.16 +- 0.74    +1.16    +5.74    +4.89
  head2             104/104    65.10         +3.20        +3.52 +- 0.39    +2.92    +3.70    +4.11
  deep              104/104    65.49         +2.59        +3.47 +- 1.08    +0.75    +6.77    +7.22
  stop2             104/104    66.37         +1.24        +1.36 +- 0.37    +1.38    +0.86    +0.98
  multi             104/104    67.16         +0.05        +0.05 +- 0.09    +0.09    +0.01    -0.16
  ref2              104/104    67.20         -0.01        -0.03 +- 0.05    -0.01    -0.03    -0.00
  ruler noise: ref2 against ref per prompt -0.06 % +- 0.11 (95 %), sd of one pair 0.55 %; ref2 texts == ref 107/107
  READ: every text identical in every variant (104 prompts with > 16 tokens; ref2 == ref 107/107). Ruler noise +-0.05..0.11 %. HALF=96 is NEGATIVE at one user (-0.20 +- 0.07). KEEP_INPASS and the many-user group (GDN_MANY, BEST_LANES, ATTN_LANES, QSA_CERT) are zero for decode speed at one user, as designed (first-token / many-user levers). LANE_REUSE is worth +2.4 % of DECODE speed (code +5.7 %), not only first token: fresh lane buffers cost in the first passes. MTP_LAST +0.33, LATE_SKIP +0.33: small but outside noise. DEEP is chat +0.75 / code +6.8 / math +7.2.
- 06:53:41 AUDIT RESULT 2, MEASURED (same engine, 40 held chat prompts x 256, temperature 0.7 thinking off, seed 9000 + id, server top_k 20 / top_p 0.95; raw D_A_k1/chat_s07off.jsonl):
  set chat40 mode s07off: prompts 40; reference (mean of ref, ref2) 48.73 tok/s
  lever off            texts   tok/s lever value %   per-prompt % +- 95%     chat
  ref                39/39     48.75         -0.05        -0.04 +- 0.08    -0.05
  cert               39/39     47.23         +3.16        +3.01 +- 0.21    +3.16
  topk               39/39     47.24         +3.16        +2.99 +- 0.25    +3.16
  hcf                39/39     47.94         +1.64        +1.66 +- 0.14    +1.64
  half               39/39     48.84         -0.23        -0.22 +- 0.12    -0.23
  mtplast            39/39     48.65         +0.15        +0.18 +- 0.13    +0.15
  lateskip           39/39     48.58         +0.31        +0.31 +- 0.12    +0.31
  head2              39/39     48.72         +0.02        -0.01 +- 0.11    +0.02
  deep               39/39     48.59         +0.28        +0.52 +- 0.75    +0.28
  stop2              39/39     48.39         +0.69        +0.56 +- 0.53    +0.69
  h2s_ON             39/39     49.48         -1.51        -1.67 +- 0.56    -1.51
  ref2               39/39     48.71         +0.05        +0.04 +- 0.08    +0.05
  ruler noise: ref2 against ref per prompt +0.08 % +- 0.16 (95 %), sd of one pair 0.51 %; ref2 texts == ref 40/40
  READ: seeded texts identical in every variant (39 prompts > 16 tokens; ref2 == ref 40/40). CERT's whole value when sampling IS the top-k form (cert off = topk off = +3.16 %). HEAD2 is zero for sampled chains (it is switched off there). h2s_ON is an ADD-ONE row: HEAD2_SAMPLED=1 is 49.48 against 48.73 = +1.5 % FASTER (builder B's lever, off in the recipe): recommend default on. HALF negative again (-0.22 +- 0.12). DEEP +0.3 (inside its noise), STOP2 +0.6, FUSED +1.6, LATE_SKIP +0.3, MTP_LAST +0.15 (noise 0.13).
- 06:57:39 AUDIT RESULT 3, MEASURED (same engine; a NEW short prompt each request, 30 rounds, variants interleaved; raw D_A_k1/ttft.jsonl): first token median: reference 118.6 / 118.5 ms; KEEP_INPASS off 161.5 (lever = -43 ms); LANE_REUSE off 139.5 (lever = -21 ms); every other lever 118.7-120.4 (zero for the first token; noise ~1.5 ms).
- 07:27:13 AUDIT RESULT 4, MEASURED (same build, second engine start; 40 held chat prompts x 256, temperature 0.7 THINKING ON, seeded; raw D_A_k1/chat_s07on.jsonl):
  set chat40 mode s07on: prompts 40; reference (mean of ref, ref2) 52.24 tok/s
  lever off            texts   tok/s lever value %   per-prompt % +- 95%     chat
  ref                40/40     52.26         -0.05        -0.07 +- 0.08    -0.05
  cert               40/40     50.51         +3.42        +3.31 +- 0.20    +3.42
  topk               40/40     50.49         +3.47        +3.37 +- 0.19    +3.47
  hcf                40/40     51.36         +1.72        +1.69 +- 0.10    +1.72
  head2              40/40     52.24         -0.01        -0.03 +- 0.12    -0.01
  deep               40/40     52.11         +0.25        +0.42 +- 0.46    +0.25
  stop2              40/40     51.88         +0.68        +0.48 +- 0.61    +0.68
  h2s_ON             40/40     52.73         -0.93        -1.18 +- 0.51    -0.93
  ref2               40/40     52.21         +0.05        +0.07 +- 0.08    +0.05
  ruler noise: ref2 against ref per prompt +0.14 % +- 0.16 (95 %), sd of one pair 0.53 %; ref2 texts == ref 40/40
  READ: thinking text behaves like plain sampled text: CERT top-k +3.4, FUSED +1.7, HEAD2 zero (off for sampled chains), DEEP +0.25 (noise 0.46), STOP2 +0.7 (noise 0.6), HEAD2_SAMPLED on = +0.9 % faster (thinking off: +1.5). Reasoning + answer identical per seed in every variant (40/40). Nothing is mis-tuned for thinking. Engine: 38,826 top-k passes, 0 whole-head fallbacks, 321.5 tiles and 54.2 launches a pass. wA1 ended 07:26 (0 crash lines; min MemAvailable 10.6 GiB with reserve 9, 15.0 GiB with reserve 13). Chain started jobB1 (D10b) 07:26.
- 07:45:56 jobB1 (D10b) STOPPED by me: with TF_FLASHNEXT_Q38_CERT_GATHER=1 the texts are WRONG (ident 2/51 against D10a; ref2 != ref): the one-launch gather has a bug. Triage in the same engine through the knob file (kit-D/tri.py, 8 ident prompts x 64 tokens against D10a's texts): both new flags off 8/8 identical (so D10b flags-off = D10a), EXP_HALF on 8/8, CERT_GATHER on 3/8. No-draft streams: EXP_HALF on 6/6 identical, 29 against 28 tok/s (rough). Gather is NOT handed off until fixed and proven; chain continues with jobA2 (16 users, D10a). The partial D_B_b1 pair files are invalid (gather was on in the reference).
- 07:50:40 jobB1 (D10b) STOPPED by me 07:46: with TF_FLASHNEXT_Q38_CERT_GATHER=1 the texts are WRONG (ident 2/51 against D10a; ref2 != ref): the one-launch gather has a bug. Triage in the same engine through the knob file (kit-D/tri.py, 8 ident prompts x 64 tokens against D10a's texts): both new flags off 8/8 identical (D10b flags-off = D10a), EXP_HALF on 8/8, CERT_GATHER on 3/8. No-draft streams: EXP_HALF on 6/6 identical, ~29 against ~28 tok/s (rough, to be paired). The partial D_B_b1 pair files are INVALID (gather was on in their reference). D10c (commit 7f9792e) adds bisect forms: CERT_GATHER=2 (list = the joined runs' tiles), 3 (old launches, no joining). jobA3 now builds D10c under the idle eref1 engine and then runs engine b2 (D10c, EXP_HALF=1, gather through knobs: tri.py bisect, sampled pair, no-draft pair, held pair; out D_B... = runs/D_A_b2). Chain is on jobA2 (16 users, D10a) since 07:46.
- 08:50:10 AUDIT RESULT 5, 16 USERS, MEASURED (head, D10a, ONE engine 16 x 32K with the pool + 128-row passes, knob file; 2 cycles, each: ref, 9 variants in rotated order, ref2; a variant = 2 x distinct-16 (code prompts, random tags) + the 107 held prompts x 256 with 16 in flight; value = reference / lever-off - 1; raw head ~/q38/runs/D_A_u16/u16.tsv + h16_*.jsonl):
  references: distinct-16 mean 285.6 (sd 7.2, 8 runs: [270, 280, 284, 287, 289, 290, 292, 293]); held 16 in flight mean 199.7 (sd 0.3, n 4: [199.3, 199.7, 199.9, 200.0])
  lever off  distinct-16 runs              value %   held-16 runs        value %  texts == ref  errors
  ref        270/289 292/287                 +0.50   199.3 199.7           +0.11  107/107 107/107  0 0
  gdn        282/279 276/285                 +1.88   193.3 193.6           +3.24  107/107 107/107  0 0
  best       289/284 286/286                 -0.22   198.2 198.7           +0.64  107/107 107/107  0 0
  attn       276/279 288/285                 +1.34   196.2 196.6           +1.69  107/107 107/107  0 0
  lateskip   289/286 286/286                 -0.30   196.6 196.5           +1.62  107/107 107/107  0 0
  ft         279/281 283/285                 +1.31   187.4 186.6           +6.81  107/107 107/107  0 0
  lone       289/293 299/283                 -1.76   198.3 198.9           +0.57  107/107 107/107  0 0
  cert       288/287 284/282                 +0.08   199.8 199.9           -0.06  107/107 107/107  0 0
  hcf        295/290 289/285                 -1.37   199.7 199.1           +0.16  107/107 107/107  0 0
  half       296/285 286/293                 -1.41   199.9 199.3           +0.06  107/107 107/107  0 0
  ref2       293/290 284/280                 -0.43   199.9 200.0           -0.11  107/107 107/107  0 0
  READ: the held-16-in-flight ruler repeats to +-0.15 % (4 references 199.3-200.0); distinct-16 swings +-2.5 % (270-293), so only the held column can judge a lever under ~3 %. Texts with 16 in flight identical in every variant (107/107, both cycles). GDN_MANY +3.2 %, ATTN_LANES +1.7 %, LATE_SKIP +1.6 %, BEST_LANES +0.6 %, first-token pair (KEEP_INPASS + LANE_REUSE) +6.8 % under churn (107 requests arriving through 16 lanes); the lone-chain group (HEAD2, DEEP, STOP2, MTP_LAST) +0.6 % (tails). ZERO at 16 users: CERT (-0.06), FUSED (+0.16), HALF (+0.06).
- 09:00:05 16 SAMPLED USERS, MEASURED (same 16 x 32K engine, D10a, held-107 x 256 at T0.7 seed 4242, 16 in flight; knob file): CERT_TOPK_ROWS=8 (D10a default) 190.7 | CERT_TOPK off 190.6 | no row limit (= integ9 today) 187.7 (-1.6 %). So integ9 as merged LOSES ~1.6 % with 16 sampled users and the row limit removes the loss (it is in the r10 patch, default 8). The second reference of that pass was cut: the shared window's 300-minute guard fired at 08:58 UTC (window opened 04:01), killed every q38-run container and dropped all leases; wq.sh re-acquires by itself (new window) and goes on with jobA3.
- 09:24:21 WINDOW HARNESS PROBLEM (details: Mac q38/handoff/D/r10-note-harness.md): the 300-minute cap at 08:58 did not close cleanly: the guard's pkill killed the window's own restore mid-way, DeepSeek stayed DOWN until the guard's own restore finished at ~09:22 (24 min instead of ~2), and the CLOSING marker was left behind, which blocks every acquire. After the guard ended and prod verified (health 200, session store 12 GiB, image check in ~/q38/guard.log) I removed the stale marker file (~09:24; state only, no script edited). My wq.sh then re-acquired (new window). Real length of the first window: 04:01 -> 09:22 UTC = 5 h 21 min of DeepSeek downtime (A, B, C, D leases back to back).
- 09:54:23 GATHER BUG FOUND BY BISECT (D10c sha e6c3d3ee59730029, engine b2, knob file; tri.py 10 ident prompts x 64 tokens against D10a texts): gather off 10/10; old launches without joining (form 3) 10/10; list of the joined tiles with a SYNCHRONOUS cuMemcpyHtoD (form 2) 7/10; flagged list sync (form 1) 10/10 here but 3/8 on D10b; flagged list sync + wait after the launch (form 5) 7/10; flagged list copied with cuMemcpyHtoDAsync ON THE PASS'S STREAM (form 4) 10/10. So the kernel and the flagged-only choice are right; the synchronous host-to-device copy of the tile list is not reliably seen by the launch that follows it on this engine / GB10 (cause not understood: stated as measured), while a copy queued on the stream is. D10e (commit f0552a0) = gather with the stream copy only (+ a guard: a list last sent on another stream waits for that stream before the host list is rewritten), EXP_HALF, knob words, CERT_TOPK_ROWS; D10d (gather removed) is not needed. jobA4 builds D10e under an idle engine; jobA5 proves it (ident 51 drafted and no-draft, sampled + held pairs with exphalf / gather off as variants, CERT=2 device check with gather on, 16 in flight) and then the fresh set. Trimmed to save window time: markers skip ehead2 / efast0 / ecopy16 held runs and b2's held pair.
- 10:14:54 D10c engine b2 (EXP_HALF=1 in the environment, gather through the knob file), MEASURED:
  set chat40 mode s07off: prompts 40; reference (mean of ref, ref2) 48.78 tok/s
  lever off            texts   tok/s lever value %   per-prompt % +- 95%     chat
  ref                39/39     48.79         -0.01        +0.04 +- 0.10    -0.01
  exphalf            39/39     48.79         -0.01        -0.02 +- 0.16    -0.01
  g3_ON              39/39     47.44         +2.84        +2.82 +- 0.26    +2.84
  g2_ON              27/39     49.40         -1.25        -1.20 +- 0.17    -1.25
  g4_ON              39/39     49.61         -1.67        -1.59 +- 0.18    -1.67
  ref2               39/39     48.78         +0.01        -0.04 +- 0.10    +0.01
  ruler noise: ref2 against ref per prompt -0.09 % +- 0.19 (95 %), sd of one pair 0.61 %; ref2 texts == ref 40/40
  set chat24 mode g0off: prompts 24; reference (mean of ref, ref2) 28.02 tok/s
  lever off            texts   tok/s lever value %   per-prompt % +- 95%     chat
  ref                23/23     27.98         +0.12        +0.29 +- 1.73    +0.12
  exphalf            23/23     27.65         +1.34        +1.51 +- 3.26    +1.34
  ref2               23/23     28.05         -0.12        -0.29 +- 1.73    -0.12
  ruler noise: ref2 against ref per prompt -0.23 % +- 3.46 (95 %), sd of one pair 8.48 %; ref2 texts == ref 24/24
  READ: one-launch gather (stream copy, flagged tiles only; form 4) = +1.6 % for sampled requests (per prompt +1.59 +- 0.18), seeded texts 39/39 identical. Old launches without joining (form 3) are 2.8 % SLOWER (the join exists for a reason). Form 2 (synchronous copy) gave wrong text in 12 of 39 prompts. EXP_HALF (#1351 half-unit kernel): ZERO for sampled drafted streams (-0.01 +- 0.16); no-draft streams 28.02 against 27.65 = +1.3 % but that ruler is +-3.3 % here (24 prompts x 128 tokens; sd of one pair 8.5 %): NOT shown; a longer no-draft pair is queued (jobA5 e1). eref1 (D10a, plain env start): held-104 67.51, T0.7 48.93, edit set 197.26.
- 10:24:46 KNOB-FILE vs ENV-FLAG CHECK, MEASURED (DEEP; held-104 x 256 greedy; texts 107/107 identical): env engine edeep (TF_FLASHNEXT_DRAFTS=6, no DEEP) 64.73 against env engine eref1 67.51 = +4.29 % (chat +2.27, code +8.79, math +9.34); knob deep_off=1 inside one engine: +2.59 % (chat +0.75, code +6.77, math +7.22); builder B's own env pairs in round 9: +2.19 % (chat +0.5, code +5.8, math +7.0). The category DIFFERENCES agree (code minus chat 6.5 env / 6.0 knob / 5.3 B) but the env engine is ~1.5 points slower in EVERY category: that is an engine-start offset, not the lever (the same size builder C reported). CONCLUSION: two engine starts differ by up to ~1.5 % on this box even with identical texts, so env-flag pairs cannot judge a lever under ~2 %; the in-engine knob pairs (ruler +-0.1 %) are the audit's numbers, and env engines are used only for levers a knob cannot switch (their rows carry +-1.5 %).
- 10:40:15 ENV ENGINES so far (D10a; +-1.5 % engine-start noise; reference eref1 held-104 67.51 / T0.7 48.93 / edit set 197.26): emx (CERT_CODE=mxfp8 + CERT_TOPK=0 = the round-8 head) held 65.74 (SM8 + top-k worth +2.7 % greedy), T0.7 46.87 (+4.4 % sampled); ecopy16 (COPY_ROWS=16) edit set 164.25 (COPY_ROWS=32 worth +20 % on file-rewrite prompts).
- 11:26:29 ENV ENGINES COMPLETE (D10a, head, order eref1, edeep, emx, ecopy16, eoff, eref2; MEASURED; texts: held 107/107 identical in every engine against eref1, seeded T0.7 40/40 identical off / mxfp8 / default, edit set 24/24):
  | engine | held-104 x 256 greedy | T0.7 chat40 seeded | edit set (24) | sparkDash 1.8.8 itself (greedy; 2 runs x 5 rounds) |
  | eref1 default (first) | 67.51 | 48.93 | 197.26 | |
  | edeep (DRAFTS=6, no DEEP) | 64.73 | | | |
  | emx (mxfp8 copy, no top-k = round-8 head) | 65.74 | 46.87 | | |
  | ecopy16 (COPY_ROWS=16) | | | 164.25 | |
  | eoff (levers off = MiaAI configuration) | 58.18 | 45.63 | | c1 prose 56.31 / 56.33, code 91.67 / 91.84, structured 105.29 / 105.26; catalog 43.6 / 66.1 / 81.1; first token ~95 ms |
  | eref2 default (last) | 66.27 | 48.30 | 194.63 | c1 prose 61.35 / 61.33, code 119.45 / 122.60, structured 164.91 / 168.81; catalog 47.7-48.2 / 77.0 / 98.0; first token ~70 ms |
  READ: the two references differ by 1.9 % (67.51 first, 66.27 last; T0.7 48.93 / 48.30; GPU 71 C): env-engine rows carry +-1 % on this box. Default vs levers off: held +13.9..16.0 % (against eref2: chat +12.6, code +17.2, math +16.7), T0.7 +5.9..7.2 % (per prompt +5.63 +- 1.55 against eref2). sparkDash 1.8.8 itself, levers off -> default: c1 prose +8.9 %, c1 code +32 % (91.8 -> 121.0), c1 structured +58 % (105.3 -> 166.9), catalog prose +10 %, structural +16.6 %, code +21 %. (Round 9's best was 61.3 / 99.3 / 113.6: DEEP + HEAD2 + STOP2 added the code / structured jump.) SM8 + top-k against the round-8 head: greedy +0.8..2.7 %, T0.7 +2.9 % (+-0.8) to +4.4 %. COPY_ROWS=32: edit set +18.5..20 %.
- 11:44:27 16 USERS, env engines (D10a, 16 x 32K; MEASURED; +-1 % engine-start noise): u16ref1 default: distinct-16 291.4 / 288.3 / 291.3, held 16 in flight 198.5; u16copy16 (COPY_ROWS=16): 288.2 / 283.3 / 284.4, held 193.8 => COPY_ROWS=32 is +2.4 % on held with 16 in flight (distinct inside its +-2.5 % noise): NOT negative at 16 users. Texts 107/107 identical (copy16 vs ref; ref vs the knob engine's reference). D10e BUILD_OK sha a8f8b3f767ca9356 (built under the idle u16ref1 engine). jobA5 started 11:41 (e1 = D10e with EXP_HALF=1 + CERT_GATHER=1).
- 12:19:35 D10e (sha a8f8b3f767ca9356; EXP_HALF=1 + CERT_GATHER=1 in the environment; levers switched OFF through the knob file as variants), engine e1, MEASURED: identity set 51/51 identical to D10a's. T0.7 chat40 seeded: gather +1.61 % (per prompt +1.56 +- 0.21), exphalf -0.12 +- 0.22 (zero), texts 39/39 identical in every variant, ruler +-0.10. NO-DRAFT streams (chat40 x 256, {draft:false}; kit-D/ndpair.py): a prompt's FIRST request is 9.8 % (+-0.5) slower than its repeats whatever the variant (NEW FINDING, cause not looked at; it wrecked the plain pair ruler: sd 8.9 % a pair), so only repeats are compared: EXP_HALF on against off +1.05 % +- 0.09 (26 pairs; reference against reference +0.13 +- 0.12); the D10c run (24 x 128) gives +1.01 +- 0.42. Texts identical. => upstream #1351's half-unit kernel: +1.0 % for one-row (no-draft) streams, zero for drafted streams. Held pair (gather / exphalf off as variants) running; then e2 (CERT=2 device check with gather), e16 (16 in flight), fresh set.
- 12:39:40 D10e engine e1, held-107 x 256 greedy (levers ON in the environment, OFF as variants), MEASURED:
  set held mode g0off: prompts 107; reference (mean of ref, ref2) 67.74 tok/s
  lever off            texts   tok/s lever value %   per-prompt % +- 95%     chat     code     math
  ref               104/104    67.75         -0.02        -0.08 +- 0.16    -0.04    +0.09    -0.06
  exphalf           104/104    67.70         +0.05        +0.03 +- 0.25    +0.03    +0.09    +0.13
  gather            104/104    67.63         +0.16        +0.05 +- 0.29    +0.17    +0.12    +0.14
  ref2              104/104    67.72         +0.02        +0.08 +- 0.16    +0.04    -0.09    +0.06
  D10e's reference texts (EXP_HALF + GATHER on) against D10a's: held 107/107, T0.7 seeded 40/40, identity set 51/51 identical. Greedy: gather +0.16 % and EXP_HALF +0.05 %: both inside the ruler (+-0.16..0.29) = zero for greedy drafted text (a greedy pass has ~10 exact tiles in a few launches already).
- 12:49:23 DEVICE CHECK of the gather (D10e, engine e2: CERT=2 + CERT_GATHER=1 + EXP_HALF=1; 24 seeded T0.7 prompts + 24 greedy; engine's own check lines; MEASURED): 3,000 certified calls, 10,627 rows: best-mismatch 0, value-mismatch 0 of 94.8 M recomputed entries; top-k passes 1,907, 6,585 rows: topk-mismatch 0 (the 20 best values + columns of every sampled row == the whole head's), bound violations 0 of 2.38 billion entries; whole-head fallbacks 0; gather 125 tiles a call in this mix. 0 crash lines. Engine e1 totals: 51,796 gather calls, 0 fallbacks.
- 12:59:43 e2 check-mode texts: T0.7 24/24 identical to D10a's. e16 (D10e, EXP_HALF + GATHER on, 16 x 32K): distinct-16 277.5 (cold) / 293.7, held 16 in flight 200.1, texts 107/107 identical to D10a's 16-user reference. Handoff written: Mac q38/handoff/D/r10-levers/ (D10e-cumulative.patch sha256 43f8c151a4053682..., NOTE.md, SHA256), q38/handoff/D/r10-audit/AUDIT.md (+ tools), q38/handoff/D/r10-note-harness.md. QUEUED behind the fresh set: jobC1 (D10e, env CERT_GATHER=1; variants rec = HALF=0 + HEAD2_SAMPLED=1 added, old = gather off: the RECOMMENDED sampled stack against today's default in one engine; T0.7 thinking off 40 + on 30, greedy 36 held; out runs/D_A_c1; log kit-D/wA_jobC1.txt.log, chain2.log). It releases lease D at its end.
- 13:15:09 FRESH SET (builder A's out-of-fit prompts, every 4th = 31; D10a engine k1f; knob pairs; fitted levers; MEASURED):
  set /home/pico/q38/kit-D/fresh.jsonl:4 mode g0off: prompts 31; reference (mean of ref, ref2) 59.96 tok/s
  lever off            texts   tok/s lever value %   per-prompt % +- 95%    agent     chat     code     lang     math   struct
  ref                31/31     59.90         +0.10        +0.04 +- 0.22    +0.15    +0.09    -0.30    +0.39    -0.24    +0.35
  cert               31/31     56.92         +5.34        +5.05 +- 0.44    +4.08    +5.62    +4.90    +6.15    +3.86    +3.68
  hcf                31/31     58.73         +2.10        +2.06 +- 0.51    +1.65    +2.20    +2.31    +1.97    +1.68    +2.43
  head2              31/31     58.60         +2.32        +2.70 +- 0.75    +5.09    +2.04    +3.09    +0.93    +4.78    +3.19
  deep               31/31     58.69         +2.16        +3.89 +- 2.43    +9.21    +0.20    +4.25    -0.43    +8.90    +7.49
  stop2              31/31     59.15         +1.38        +1.60 +- 0.93    +1.10    +1.22    +1.54    +1.57    +0.26    +3.04
  ref2               31/31     60.02         -0.10        -0.04 +- 0.22    -0.15    -0.09    +0.30    -0.39    +0.24    -0.35
  READ: out of the fit the fitted levers hold: HEAD2 +2.3 % (non-English +0.9, agent +5.1), DEEP +2.2 % (chat +0.2, code +4.3, math +8.9, struct +7.5, agent +9.2; non-English -0.4 inside noise), STOP2 +1.4 %, CERT +5.3 %, FUSED +2.1 %. Texts 31/31 identical in every variant. Chain 1 ended 13:11 (lease D released by its trap, head MemAvailable 111 GiB). Chain 2 (jobC1) was blocked 3 min by a stray stopped shell of mine whose command line matched the wait pattern; killed; it starts now.
- 13:42:07 RECOMMENDED SAMPLED STACK IN ONE ENGINE (D10e, engine c1, env CERT_GATHER=1; variants: rec = + HALF=0 + HEAD2_SAMPLED=1 through knobs, old = gather off = today's integ9 default; seeded; MEASURED):
  | bench | old (integ9 default) | + CERT_GATHER | + HALF=0 + HEAD2_SAMPLED=1 (recommended) | recommended vs old | texts |
  | T0.7 thinking off, 40 x 256 | 49.10 | 49.93 | 50.85 | +3.56 % | 39/39 identical in every variant |
  | T0.7 thinking on, 30 x 256 | 51.97 | 52.80 | 53.44 | +2.83 % | 30/30 (reasoning + answer) |
  ruler +-0.11..0.13 %. Skill reference qwen38-fastest-build.md got a round-10 section.
- 13:48:00 jobC1 complete: greedy 36 held prompts: old 64.50, + gather 64.50, + HALF=0 + HEAD2_SAMPLED=1 64.68 (+0.27 %; texts 34/34 with > 16 tokens identical); c1's reference texts == D10a's (T0.7 40/40, thinking 30/30); 0 crash lines.

## BUILDER D ROUND 10 END STATE (13:48 UTC 2026-11-14)
- Lease D RELEASED 13:47:07 by chain 2's trap (status: leases A, B, C only; the window stays open for them). No q38-run-D / build / sd container, no kit-D script, no KEEP_LEASE / PAUSE; head MemAvailable 117 GiB; GPU: vision encoder only (1,667 MiB). qwen38-gw, dsgx10-vision, sz-memguard untouched (Up 4 days). DeepSeek is down by A / B / C leases, not mine.
- WINDOW TIME I used (DeepSeek down): lease D 04:36 -> 08:58 (window 1, cut by the 300-minute cap; prod was back only 09:22-09:24) and 09:24 -> 13:47 (window 2): ~8 h 45 min of engine time on the head, back to back with A / B / C leases.
- DELIVERED: (1) audit: handoff/D/r10-audit/AUDIT.md (table lever x mode, noise, verdicts, recommended defaults: HALF=0, HEAD2_SAMPLED=1, CERT_GATHER=1, CERT_TOPK_ROWS=8) + tools; (2) sampled path: one-launch gather built and proven (+1.6 % sampled; D10e), HEAD2 on sampled chains measured in the stack (+0.9..1.5 %, builder B's lever), thinking text audited (same picture as plain sampled text), sparkDash 1.8.8 itself off vs default 2 runs a side, recommended sampled stack as one configuration +3.6 % / +2.8 %; (3) #1351 half-unit kernel ported: +1.05 % no-draft streams, zero drafted; handoff/D/r10-levers/ (D10e-cumulative.patch sha256 43f8c151a4053682, binary D10e a8f8b3f767ca9356, NOTE.md); harness note handoff/D/r10-note-harness.md.
- NOT DONE: HEAD2_FAST and MANY not re-measured (no knob; builder B's / round-6 numbers stand); QSA_CERT / ELASTIC not audited for their own purpose (long prompts; builder C's area); 16 users with the three recommended changes together; edit / agent / needle / image / qeval on D10e; why a synchronous list copy fails; why a no-draft stream's first request is 9.8 % slower than repeats (a possible +10 % for --no-drafts); the greedy thinking-on pair (dropped for window time).
- On the head: ~/q38/tfbin/{D10a, D10b (broken gather: do not use), D10c (bisect forms), D10e}, ~/q38/build-D/tf = Mac wt/D10 f0552a0, ~/q38/prof-D/head2_r9.bin, kit-D (run10.sh, wq.sh, chain.sh, aud.py, aud16.sh, tri.py, ndpair.py, cmpcat.py, job*.txt + logs), runs ~/q38/runs/D_A_* and D_B_b1 (invalid: gather bug). Mac: q38/results/rD10 (runs + logs), q38/results_D10_*.txt, q38/kitD, q38/shipD10.sh, q38/pushkitD.sh, branch r10-D (local, not pushed).
- RESUME: cd ~/.hermes/cache/scratch/q38; ./shipD10.sh; ./pushkitD.sh; write kitD/jobX.txt (ENGINE name|on|-e FLAGS|pc_3_262144|D10e, then commands); start head kit-D/chain.sh "D10e A jobX.txt" detached with its output in chainX.log (it acquires and releases lease D by itself); table: python3 -I ~/q38/kit-D/aud.py rep ~/q38/runs/D_A_<name>/<file>.jsonl on the head.

## Round 11 (builder D, head; sampled draft side, chat drafts, no-draft first request)
- 2026-11-14T13:50:01Z start. Read PROGRESS-D r8-10, r10 notes, B's tree design note. Base f0552a0 (r10-D), new worktree wt/D11 branch r11-D. No lease yet.
- 13:56:40 SOURCE READ (no run): item 3 mechanism candidate: the n-gram table (ple) is read by O_DIRECT positional reads into a row cache (cuda_ple_reader.zig); drafted streams read a token's rows AHEAD while the head drafts (DRAFT_AHEAD), a no-draft stream has no read-ahead, so on text the row cache has not seen its one-token gather waits for drive reads (4 rows a thread, 2 reads a row, one after the other); repeats hit the row cache. To be measured (GATHER_LOG=1, kit fr.py groups A-F). D11a (commit da7326b on r11-D, Mac wt/D11): QSTEP carries a time (us) + QCT line (chain end, steps computed); TF_FLASHNEXT_Q38_SCHAIN=STOP2,DEEP_P,DEEP_BASE,MAX (sampled chain's own gate values; knobs s_stop2 s_deep_p s_deep_base s_max; deep_p deep_base chain_max for all chains); TF_FLASHNEXT_Q38_NGRAM_SPLIT=N (a gather of <= N rows: one row a thread; knob ngram_split). Kit: run11.sh (window cap rule), wq11.sh, chain11.sh, lr.py, fr.py, jobR1.txt (engines fr, tm, lg). Tree shipped to head build-D (verified).
- 13:57:24 FIRST REQUEST, DRAFTED STREAMS, MEASURED from round-10 pair files (Mac results/rD10; time of a prompt's first request against the same variant's later requests, per variant): drafted greedy held -0.48 % +- 0.41 (n 101), +0.13 +- 0.45 (n 101); drafted T0.7 -0.24 +- 0.45, -0.56 +- 0.26, -0.59 +- 0.37; thinking -0.89 +- 0.56, -0.64 +- 0.42; fresh set +0.97 +- 1.64 => NO first-request penalty for drafted streams (benchmarks' first prompt and real users' first message are not affected). No-draft streams: +10.9 % time +- 0.6 (two runs) = the finding. Waiting for the window cap (opened 09:23:48, restore 14:03:48) before acquiring: no build / engine start inside the last 10 minutes.
- 14:08:11 window capped 14:03:48 (restore, CLOSING). Started head kit-D/go11.sh D11a 'D11a R jobR1.txt' detached (log kit-D/go11.log; chain log kit-D/wR_jobR1.txt.log): waits out CLOSING, acquires lease D, builds D11a (kit-D/build_D11a.log), runs engines fr, tm, lg (out ~/q38/runs/D_R_{fr,tm,lg}); the chain releases lease D at its end. Stop: touch kit-D/STOPCHAIN.
- 14:11:16 HARNESS: the 280-min restore of window S140923 completed (prod verified health=200 at 14:08:00, DONE written) but shared/CLOSING was left behind again (w_shared.sh's own 'no live lease, closing' line ran after the restore and re-touched it; q38-win ended failed, q38-guard inactive). With q38-win and q38-guard both inactive and DeepSeek health 200 on :18900, I removed the stale marker at 14:10 (the documented rule in shared/NOTES.md). DeepSeek was up 14:08 -> my acquire.
- 14:16:24 D11a BUILD_OK sha d63742a1053c806c (head ~/q38/tfbin/D11a, built 14:11-14:14 with no engine up). Window opened 14:10:4x (leases A, B, C, D); chain R started 14:14:27: engine fr loading.
- 14:20:05 ITEM 3 CAUSE FOUND, MEASURED (engine fr, D11a, GATHER_LOG=1, no-draft stream, one-row passes; engine's own lines per 100 passes): FIRST request of a text: 16 n-gram rows a pass are 'read by the gather' from the drive (O_DIRECT, 2 positional reads a row, 4 rows a thread one after the other): gather 4.0-4.7 ms a pass (slowest part 3.9-4.5 ms = ~0.45 ms a read), the pass waits 4.5-5.5 ms at the n-gram layer (layer 1) with only layer 0 queued on the GPU. REPEAT of the text: 16 row-cache hits, gather 0.3-0.4 ms, wait 0.9 ms (hidden behind layer 0, ~0.75 ms). Difference ~4 ms of a ~36 ms pass = the 10-11 %. A drafted stream reads its tokens' rows ahead while the head drafts (DRAFT_AHEAD; pending token included, cuda_lanes_draft.zig:72), which is why it shows no first-request penalty. It is NOT the prompt cache and not buffers. Fix under test: TF_FLASHNEXT_Q38_NGRAM_SPLIT (one row a thread; fr.py group D).
- 14:33:37 ITEM 3, MEASURED (engine fr, D11a, 8 chat prompts a group x 256 tokens greedy; first request against the 3rd; kit fr.py; raw head ~/q38/runs/D_R_fr/fr.jsonl): A no-draft first -11.46 % +- 1.24 (25.74 vs 29.11 tok/s; uniform over the reply: first quarter -12.1, last -11.2), 2nd = 3rd (+0.01); B drafted first -0.29 +- 0.70 (57.51 vs 57.75): no penalty; C no-draft right after a DRAFTED run of the same prompt: -0.15 +- 0.17: no penalty (the drafted run left the rows in the row cache); F no-draft after a 4-token request of the same prompt (prompt cache hit, rows cold): -10.48 +- 1.01 => the cause is the n-gram row cache, not the prompt cache; D no-draft first with ngram_split=256 (one row a thread): -5.48 +- 0.46 (half of the penalty gone; the rest = two reads a row one after the other, ~0.9 ms + thread wake, against ~0.75 ms hidden behind layer 0); E drafted with split: +0.18 (no change). Texts identical in every group. NEXT: D11b = small gather: cache hits copied inline, each missing row's codes and scales read by two threads at once.
- 14:36:15 D11b (commit 18ebaeb): TF_FLASHNEXT_Q38_NGRAM_SPLIT=N (<= 256) = small gather: rows in the row cache copied on the calling thread, each missing row's codes and scales read by two pool threads woken together (Pool.runWide), rows kept in the cache; falls back to the stock path on any failure. QUEUED on the head: go11.sh D11b 'D11b S jobR2.txt' (waits for chain R; builds D11b with no engine up; engines fr2 (fr.py groups A B D E), id (NGRAM_SPLIT=256 in env: ident 51, held-107 pair with nosplit variant, no-draft pair, sampled pair), i16 (16 x 32K: held 16 in flight texts)); log kit-D/go11b.log, wS_jobR2.txt.log; out runs/D_S_*.
- 14:42:27 engine tm (D11a, rec stack, STEP_LOG=1): identity set 51/51 identical to D10e's (D_A_e1); greedy prompts.jsonl:6 (54 prompts) 67.64 tok/s (the step log costs nothing).
- 14:50 REAL ROUND COSTS, MEASURED (engine tm = D11a, rec stack, real default gate, STEP_LOG=1 with times; one user 3 x 262K; medians over 2.6-2.9k rounds a mode; Mac results/rD11/tm):
  chain (absorb + head steps): greedy 1.32 ms first step + 1.17 a further step (12 steps 14.1 ms); sampled 1.37 + 1.23 (12 steps 14.85).
  verify pass by rows (pending + drafts): greedy 2: 38.2 | 3: 41.0 | 4: 44.2 | 5: 47.1 | 6: 50.6 | 7: 54.6 | 8: 56.4 | 9: 60.2 | 10: 62.4 | 11: 65.4 | 12: 70.1 | 13: 72.0 ms
                                         sampled 2: 38.5 | 3: 41.3 | 4: 44.3 | 5: 47.8 | 6: 50.8 | 7: 54.5 | 8: 57.1 | 9: 62.1 | 10: 64.6 | 11: 66.9 | 12: 69.2 | 13: 72.8 ms
  => a verify row costs ~3.1-3.3 ms at one user in today's stack (rows 2->7: 3.27 ms a row; rows 5->12: 3.28; 7->13: 2.9), NOT 2.4 (the round-8/9 simulators' value) and not cheaper for rows 5-12. (Rows are the policy's own choice, so content-selected; an unbiased depth sweep runs in engine lg.) Sampled passes cost +0.2..0.7 ms (top-k certified head).
  SAME 54 mixed prompts (prompts.jsonl:6), greedy vs T0.7 seeded, real policy (head rounds; tok/s from the round times):
  +--------+------+--------+---------+----------+--------+--------+-------+----------+-----------+-------+
  | mode   | cat  | rounds | steps/r | drafts/r | kept/r | accept | tok/r | chain ms | verify ms | tok/s |
  | greedy | all  | 2661   | 4.29    | 3.91     | 2.58   | 0.661  | 3.58  | 5.37     | 48.24     | 66.8  |
  | greedy | chat | 2062   | 3.58    | 3.13     | 1.91   | 0.611  | 2.91  | 4.43     | 45.29     | 58.6  |
  | greedy | code | 332    | 6.27    | 6.08     | 4.53   | 0.746  | 5.53  | 8.07     | 56.85     | 85.2  |
  | greedy | math | 267    | 7.36    | 7.21     | 5.33   | 0.739  | 6.33  | 9.25     | 60.33     | 90.9  |
  | T0.7   | all  | 2945   | 3.98    | 3.54     | 2.32   | 0.656  | 3.32  | 5.09     | 46.67     | 64.1  |
  | T0.7   | chat | 2253   | 3.33    | 2.84     | 1.70   | 0.599  | 2.70  | 4.27     | 44.30     | 55.6  |
  | T0.7   | code | 397    | 5.71    | 5.39     | 4.07   | 0.755  | 5.07  | 7.27     | 52.53     | 84.7  |
  | T0.7   | math | 295    | 6.63    | 6.37     | 4.70   | 0.738  | 5.70  | 8.44     | 56.87     | 87.3  |
  | T0.7 thinking, chat30 | 2634 | 2.74 | 2.28 | 1.39 | 0.610 | 2.39 | 3.62 | 43.82 | 50.4 |
  +--------+------+--------+---------+----------+--------+--------+-------+----------+-----------+-------+
  whole requests: greedy 67.64 tok/s, T0.7 64.92 (-4.0 %), on the same prompts. READ: (1) on the SAME mixed prompts sampled is 4 % behind greedy, not 'half the gain': the T0.7 ruler of rounds 9-10 is 40 CHAT prompts, and on chat DEEP / STOP2 / HEAD2 are small for greedy too (audit: +0.75 / +1.4 / +2.9); sampled code keeps greedy's speed (84.7 vs 85.2). (2) accept rates are equal by category (0.60/0.61, 0.755/0.746, 0.738/0.739) but sampled chains are ~8 % SHORTER (drafts a round 3.54 vs 3.91) at the same accept rate: the gate sees the DRAWN token's probability, which is lower than the top probability. Whether that stop is too early is the simulation on the always-16 log (engine lg).
- 15:02 SAMPLED LEVERS BY CATEGORY, MEASURED (engine tm, D11a, rec stack on (HEAD2_SAMPLED=1, CERT_GATHER=1, HALF=0), T0.7 seeded, 53 mixed prompts (prompts.jsonl:6) x 256, knob pairs, reference twice; raw head ~/q38/runs/D_R_tm/held6_s07off.jsonl):
  reference 64.63 tok/s; lever value = time with the lever off / reference - 1; texts 53/53 identical in every variant; ruler ref2 vs ref +0.16 +- 0.35
  +----------------+--------+----------------+-------+-------+-------+----------------------------------+
  | lever          | all    | per prompt     | chat  | code  | math  | greedy audit r10 (all/chat/code/math) |
  | DEEP 0.9,6     | +1.87  | +2.22 +- 1.20  | +0.94 | +3.41 | +5.35 | +2.59 / +0.75 / +6.77 / +7.22    |
  | STOP2 0.5      | +1.33  | +1.00 +- 0.60  | +1.72 | +0.25 | +0.50 | +1.24 / +1.38 / +0.86 / +0.98    |
  | HEAD2_SAMPLED  | +2.91  | +2.86 +- 0.47  | +3.05 | +2.08 | +3.30 | HEAD2 +3.20 / +2.92 / +3.70 / +4.11 |
  +----------------+--------+----------------+-------+-------+-------+----------------------------------+
  ANSWER to 'why do DEEP / STOP2 / HEAD2 gain so little when sampling': they do not. On mixed text they are worth +1.9 / +1.3 / +2.9 % sampled against +2.6 / +1.2 / +3.2 % greedy. The round-10 sampled audit showed ~0 because (a) its ruler is 40 CHAT prompts, where DEEP is +0.75 % for greedy too, and (b) HEAD2 was switched off for sampled chains in that reference (HEAD2_SAMPLED is the lever; +2.9 % here).
- 15:18 SAMPLED DRAFT STUDY on the always-16 log (engine lg, D11a, T0.7 seeded, 107 held prompts, 5,324 rounds, HEAD2_SAMPLED on; Mac results/rD11/lg; MEASURED unless marked):
  * HOW a sampled chain drafts (source + log): the draft head draws with the SAME key (seed, position, token id) the main model will use (Gumbel-max over the tempered top-k/top-p set); its draw is the head's top token in 91.3 % of drafts. Kept when it is the top token 0.785, when it is not 0.309.
  * The gate's input (the drawn token's temperature-1 probability) is OVER-confident as a 'will be kept' probability: p 0.5-0.6 -> kept 0.41; 0.7-0.8 -> 0.52; 0.9-0.97 -> 0.71; 0.97-0.995 -> 0.82; >= 0.995 -> 0.969. A drawn token that is NOT the head's top is kept more often than a top token of the same probability (p 0.3-0.5: 0.43-0.53 against 0.28-0.36): the shared key pulls both draws to it.
  * 'The right confidence from the two distributions and the key': with shared Gumbel noise P(match) = sum_i 1 / sum_j max(p_j/p_i, q_j/q_i) (p target, q draft; >= (1-TV)/(1+TV)); the draft side knows q and the noise but not p. Every draft-side signal was tested as a predictor of 'kept' (held half, AUC): drawn p 0.867; top p 0.867; tempered-set p 0.761; keyed-race margin 0.762; logistic on all (drawn, top, tempered, margin, set size, depth, is-top) 0.875. => no signal beyond the drawn probability (+0.008 AUC); what is wrong is the CALIBRATION, not the ranking.
  * SIMULATED on this log with the real costs (verify 32.0 + 3.17 x rows, chain 1.37 + 1.23 a step; the sequence simulator reads 4 % low in absolute terms (61.5 vs 64.1 measured): relative numbers only): re-tuned thresholds on the raw signal (product line 0.2-0.3, base 3-8, depth 16): +0.7..1.2 % on the held half; a CALIBRATED gate (calibration table drawn-p x is-top fitted on the fit half; product of calibrated probabilities, line 0.15, early stop 0.3, deep 0.5 from 6-8, depth 16): +2.1 % sequence sim / +1.5 % per-round sim on the held half (chat +2.3 / +1.4, code +1.6 / +1.0, math +2.4 / +1.4; thresholds picked on the held half: optimistic). To be built as SGATE mode 4 and measured with knob pairs; the same calibration question for GREEDY chains is next (greedy always-16 log running).
- 15:26 GREEDY chains have the same mis-calibration (always-16 greedy log, 107 held prompts, 19,108 drafts; the gate's input = head probability after HEAD2's odds factor): 0.5-0.6 -> kept 0.38; 0.7-0.8 -> 0.50; 0.9-0.97 -> 0.73; 0.97-0.995 -> 0.83; 0.995-0.999 -> 0.926; 0.9995-0.99995 -> 0.969; above -> 0.993; the SAME curve in chat, code and math (e.g. 0.9-0.97: 0.724 / 0.716 / 0.764). AUC of the gate input 0.883.
  CALIBRATED GATE, SIMULATED out of sample (calibration + lines chosen on the even prompts, evaluated on the odd prompts; real costs; sequence sim / per-round sim): one rule for every chain: propose a draft while the product of calibrated probabilities >= 0.15, compute the next head step while it is >= 0.30, depth up to 16:
    greedy  all +1.56 / +1.92 %  (chat +1.20 / +1.06, code +1.77 / +1.41, math +3.36 / +1.83); drafts a round 3.55 -> 3.82
    T0.7    all +2.55 / +2.16 %  (chat +2.56 / +1.67, code +3.19 / +1.94, math +1.44 / -0.41); drafts a round 3.33 -> 3.90
  (re-tuned lines on the RAW signal: sampled +0.7..1.2 %.) BUILT as D11c (commit 99f2bbe): TF_FLASHNEXT_Q38_GCAL=FILE (tables g / st / sn = greedy / sampled-draw-is-top / sampled-draw-is-not-top + 'set' lines; knobs gcal, gcal_c, gcal_stop2, gcal_deep_p, gcal_deep_base). Table: kitD/gcal_r11.txt (fitted on all 107 log prompts, ids 0 mod 3; head ~/q38/prof-D/gcal_r11.txt).
  QUEUE on the head (go11.sh D11c, log kit-D/go11c.log): build D11c -> jobR2 (tag S: n-gram small gather: fr2, id, i16) -> jobR3 (tag T, engine gc: DRAFTS=16 + GCAL; knob pairs on FIT prompts ids 1 mod 9 (36), greedy + T0.7: reference = gcal=0 chain_max=12 (today's default), variants d16 (stock lines at depth 16), cal, c10, c20, s20, s40, cal12). Proof on ids 2 mod 3 + fresh set + identity comes after the lines are chosen.
- 15:32 CHAT BREAKDOWN (item 2), MEASURED on the greedy always-16 log (engine lg, 107 held prompts: chat 3,583 rounds / code 637 / math 437):
  +------+----------------------------------------------------------+----------------------------------------------+---------------+
  | cat  | position of the first wrong draft (share of rounds)      | rank of the true token in the head's list    | head top-1 p  |
  |      | 0     | 1     | 2     | 3-5   | 6-10  | 11-15 | none(16) | 2     | 3     | 4-8   | not in top 8          | at the miss   |
  | chat | 0.282 | 0.239 | 0.164 | 0.215 | 0.085 | 0.013 | 0.003    | 0.322 | 0.128 | 0.206 | 0.344                 | median 0.59   |
  | code | 0.140 | 0.107 | 0.097 | 0.239 | 0.256 | 0.100 | 0.061    | 0.520 | 0.152 | 0.126 | 0.202                 | median 0.75   |
  | math | 0.096 | 0.082 | 0.096 | 0.295 | 0.233 | 0.128 | 0.069    | 0.484 | 0.157 | 0.142 | 0.216                 | median 0.84   |
  +------+----------------------------------------------------------+----------------------------------------------+---------------+
  (entropy of the head's row at the miss, median nats: chat 1.08, code 0.67, math 0.56.) Chat: half of the rounds lose their 1st or 2nd draft, and at the miss the true token is the head's 2nd choice only 32 % of the time (34 % not in its top 8): chat misses are mostly not 'near misses'.
  Sampled (T0.7) misses: in 73-85 % the draft drew the head's top token and the main model drew another; true token = head's rank 2 in 24 % (chat) / 38 % (code) / 42 % (math).
  OPTION (a) SIBLING ROW, SIMULATED on that log with the MEASURED costs (default policy; one rank-2 sibling a round; a verify row 3.11 ms):
    oracle (sibling exactly at the miss): all +3.3 %, chat +3.4 %   | priced by the head, any position (needs the in-kernel fork): all +0.68 %, chat +0.73 %, code +0.97 %
    first position only (the alias-lane shape; +0.4 ms estimate for the repeated pending row): all +0.53 %, chat +0.58 % (lands 26 %, 0.26 rows a round)
    => below the ~1 % bar (round 9's +1.0..1.6 % used 2.4 ms a row; the real row is 3.1 ms): NOT BUILT. (b) chat-specific lines and (c) more candidates at small top-2 gaps: the calibration curve is the SAME in chat, code and math, so one calibrated gate serves all (simulated chat +1.1..1.2 % greedy, +1.7..2.6 % sampled): that is the lever being measured (D11c GCAL).
- 15:35:02 D11c BUILD_OK sha 5bccf01950764374 (head ~/q38/tfbin/D11c; = D11a + small n-gram gather + GCAL). Chain R ended (engines fr, tm, lg done; logs on Mac results/rD11). Chain S (jobR2) started 15:34 on D11c, then chain T (jobR3).
- 15:40 UNBIASED ROW COST, MEASURED (engine lg, same 6 chat prompts x 128 tokens, gate off, chain depth capped by the knob chain_max = 1..16; verify-pass median ms by rows):
  greedy  2: 39.1 | 3: 42.2 | 4: 45.3 | 5: 49.0 | 6: 52.3 | 7: 54.5 | 9: 59.1 | 11: 63.3 | 13: 67.0 | 15: 71.1 | 17: 76.7   => a row: 3.3 ms (rows 2-5), 2.8 (5-7), 2.1 (7-13), 2.4 (13-17)
  T0.7    2: 39.1 | 3: 42.6 | 4: 45.8 | 5: 49.4 | 6: 53.1 | 7: 55.2 | 9: 62.4 | 11: 66.4 | 13: 70.0 | 15: 73.5 | 17: 78.5   => a row: 3.4 (2-5), 2.9 (5-7), 2.5 (7-13), 2.1 (13-17)
  So rows 5-12 cost 2.3-2.6 ms on chat text with the gate off (later rows are mostly wrong drafts there), 3.1-3.3 ms where the real policy goes deep (code / math). The first sibling row would be a 3.3 ms row.
- 15:49 ITEM 3 FIX, MEASURED (engine fr2, D11c sha 5bccf01950764374, same fr.py groups, 8 prompts a group x 256 greedy; knob ngram_split=256 for groups D, E):
  +---------------------------------------------+------------------------+--------------------------+
  | group (first request against the 3rd)       | stock gather           | small gather (SPLIT=256) |
  | no-draft stream, first request              | -11.15 % +- 1.01 (25.73 vs 29.16 tok/s) | -3.92 % +- 0.37 (27.76 vs 28.90) |
  | no-draft stream, repeats (2nd vs 3rd)       | +0.03 +- 0.16          | +0.00 +- 0.11            |
  | drafted stream, first request               | -0.40 +- 0.78 (57.37 vs 57.65) | +0.33 +- 0.47 (57.54 vs 57.38) |
  +---------------------------------------------+------------------------+--------------------------+
  engine's own lines: cold gather 3.9-4.0 ms -> 1.35-1.47 ms a pass (wait at the n-gram layer 4.5 -> 1.9 ms); warm 0.3-0.5 -> 0.16-0.29 ms (wait 0.9 -> 0.67-0.77 ms). Texts identical in every group.
  => a no-draft stream's first pass over a text: 25.73 -> 27.76 tok/s (+7.9 %); the penalty against repeats falls from 11.2 % to 3.9 % (what is left = one round of 32 drive reads at once, ~1.4 ms, against ~0.75 ms hidden behind layer 0). Drafted streams: no penalty before, no change after. Identity + pairs: engine id (running).
- 16:14:44 engine id (D11c, NGRAM_SPLIT=256 in the environment): identity set 51/51 identical to D10e's; held-107 greedy: reference texts 107/107 identical to D10e's.
- 16:28 SMALL GATHER (TF_FLASHNEXT_Q38_NGRAM_SPLIT=256), engine id + i16 (D11c), MEASURED, knob pairs (variant nosplit = knob ngram_split=0):
  drafted greedy held-107: lever -0.02 % (per prompt +0.02 +- 0.68): zero, texts 104/104 in every variant, reference texts == D10e's 107/107, identity set 51/51;
  drafted T0.7 chat24: +0.31 % (+0.25 +- 0.29), texts 23/23; no-draft chat24 x 128 (first requests mixed in): +3.0 % (+-2.8), texts 23/23;
  16 users (16 x 32K, flag on): held 16 in flight texts 107/107 identical to D10a's 16-user reference; 0 crash lines in every engine.
  => exact; +7.9 % for a no-draft stream's first pass over a text (fr2), zero elsewhere. Chain S ended 16:27; chain T (engine gc: GCAL fit pairs) started.
- 16:48 GCAL fit sweep, greedy, first 31 of 36 FIT prompts (ids 1 mod 9: chat-heavy), engine gc (D11c, DRAFTS=16 + GCAL), MEASURED, time against today's default (gcal=0 chain_max=12); negative = faster:
  stock lines at depth 16 -0.38 % | calibrated 0.15/0.30 -0.56 % | line 0.10 +0.55 | line 0.20 -0.91 | stop 0.20 +1.10 | stop 0.40 -1.09 | calibrated at depth 12 -0.87; ruler +-0.26. Texts identical in every variant.
  READ: the calibrated gate is faster by ~0.6-1.1 % on chat-heavy greedy text (simulation said +0.8..1.2 for chat) and the lines want to be HIGHER than the simulated optimum (0.20 / 0.40 better than 0.15 / 0.30; lower is worse). Second sweep queued (jobR3b, tag U, engine gc2): c20s40, c25s50, c20s50, c30s60 on the same fit prompts, greedy + T0.7.
- 17:03 GCAL fit sweep, MEASURED (engine gc, D11c; FIT prompts ids 1 mod 9; time against today's default = knobs gcal=0 chain_max=12; negative = faster; texts identical in every variant):
  greedy, 36 prompts (ref 62.63 tok/s; ruler +-0.26, ref2-ref drift 0.5): d16 -0.33 | cal(0.15/0.30) -0.30 | c10 +0.70 | c20 -0.70 | s20 +1.74 | s40 -0.81 | cal12 -0.71   (chat -0.8..-1.5, math +0.3..+1.4 on ~4 prompts)
  T0.7, first 17 prompts (chat; ref 49.86): d16 -0.26 | cal -1.51 | c10 -0.28 | c20 -2.19 | s20 +0.57 | s40 -1.74 | cal12 -2.12
  READ: sampled chat gains 1.5-2.2 % (simulated +1.4..2.3); greedy gains only 0.3-0.8 % on this set (simulated +1.2..1.9: the simulator over-read greedy, as the ledger warns). Higher lines (0.20, stop 0.40) beat the simulated optimum in both modes; depth 16 adds nothing over 12.
- 17:24 GCAL fit sweep complete, T0.7, 36 FIT prompts (ref 60.28 tok/s; MEASURED; negative = faster): d16 -0.58 | cal(0.15/0.30) -1.46 | c10 -0.46 | c20 -1.96 (chat -1.82, code -2.85, math -1.60) | s20 +0.68 | s40 -1.08 | cal12 -1.52. Texts 36/36 in every variant.
  Per prompt the spread is wide (+-2.7 %): replies of 17-54 tokens swing by +-10..60 % (one or two rounds more or fewer: e.g. id 190, 28 tokens at 115 tok/s under DEEP, loses 2 rounds because the calibrated product of 0.995-0.999 drafts (kept 0.93-0.95) falls below the line where the raw product does not). Replies of >= 200 tokens: T0.7 -1.9 % (cal) / -2.3 % (c20); greedy +0.0 / -0.4 %. => GCAL is a SAMPLED lever (~+2 %); for greedy it is ~+0.5 % at best on this set.
- 17:37 GCAL second sweep, greedy, 36 FIT prompts, engine gc2 (ref 62.40; ref2-ref drift 1.0 % in this run, so +-0.55): c20s40 -1.08 (chat -1.76, code -0.09, math +1.15) | c25s50 -0.64 | c20s50 -0.96 | c30s60 +0.92. With the first sweep: greedy chat is ~1-1.7 % faster under the calibrated gate, math ~0.3-1.2 % slower (few prompts), code ~0.
  An ADAPTIVE line (line = draft cost x the stream's own tokens/ms) was SIMULATED: no better than fixed lines on the held half (greedy +0.7..1.2 against +0.5..1.25; sampled +1.2 against +1.1..2.3): not built.
- 17:45 GCAL second sweep, T0.7, 36 FIT prompts (ref 60.22; drift +-0.46): c20s40 -1.23 | c25s50 -1.22 | c20s50 -1.28 | c30s60 +0.15. With the first sweep (c20 with stop 0.30: -1.96): LINES CHOSEN on the fit prompts: sampled 0.20 / 0.30, greedy 0.20 / 0.40, depth stays 12 (no DRAFTS change). Written into kitD/gcal_r11.txt (head ~/q38/prof-D/gcal_r11.txt).
  PROOF queued (go11.sh D11c 'D11c V jobR4.txt', log kit-D/go11e.log, wV_jobR4.txt.log; out runs/D_V_gp, D_V_g16): engine gp = recipe defaults + rec stack + GCAL + NGRAM_SPLIT=256 (DRAFTS=12): identity set vs D10e; pairs against knob gcal=0 on 107 UNTOUCHED prompts (ids 2 mod 3) greedy + T0.7; held-107 texts vs D10e; fresh set; thinking chat30; seeded chat40 texts vs D10e; engine g16 (16 x 32K) held 16 in flight texts. The window caps at 18:50 (run11 refuses starts after 18:38; wq11 waits and re-acquires).
- 17:44:05 HANDOFF DRAFT: Mac q38/handoff/D/r11-levers/ D11c-cumulative.patch (ONE cumulative patch against 7061396, replaces D10e's; sha256 687a16e566188530; git apply --check OK on wt/I9), gcal_r11.txt (sha256 7d877c8796e4ab02), tools/. NOTE.md to be written with the proof numbers.
- 17:57:37 engine gp (D11c, GCAL + NGRAM_SPLIT=256 + rec stack): identity set 51/51 identical to D10e's (median 68.6 tok/s against 67.7 in engine id without GCAL). Proof pairs running.
- 18:15 GCAL PROOF, GREEDY, MEASURED on 106 UNTOUCHED prompts (ids 2 mod 3; engine gp, D11c; lines 0.20 / 0.40; against knob gcal=0; ref 64.99 tok/s): calibrated gate -0.12 % time (chat -0.19, code -0.34, math +0.69), per prompt -0.10 +- 0.74; ruler +-0.16; texts 105/105 identical. => ZERO for greedy out of the fit (the fit set's -0.7..-1.1 % did not hold; simulation said +1.2..1.9 %: over-read again). The greedy table is DROPPED from the recommended file; sampled proof running.
- 18:34 GCAL PROOF, T0.7, MEASURED on 106 UNTOUCHED prompts (ids 2 mod 3; engine gp; lines 0.20 / 0.30; against knob gcal=0; ref 62.15 tok/s; seeded): calibrated gate -0.40 % time (chat -0.52, code -0.52, math +0.62), per prompt -0.20 +- 0.82; ruler +-0.17; texts 106/106 identical.
  VERDICT GCAL: greedy +0.1 % (zero), sampled +0.4 % (at the edge of the ruler) on untouched prompts. The fit-set pairs (36 prompts: -1.1 / -2.0 %) and the simulation (+1.6 / +2.3 %) both over-read. The lever stays in the patch behind its flag, default OFF, NOT recommended as a default; reason for the miss: the stock lines (0.4 / 0.5 / DEEP 0.9) were already tuned on the raw probabilities, so a monotone re-mapping of the same signal mostly re-labels the same decisions (AUC unchanged), and what it changes (a few more drafts on lukewarm chains, a few fewer on long sure ones) nets out.
  LESSON: 36 prompts x one pass is not enough to pick gate lines (replies under ~60 tokens swing +-10..60 %); pick on >= 100 prompts and prove on another >= 100.
  Skipped to save window time: the fresh-set pair (marker .done_6). Still to run: held-107 texts (running), thinking pair, seeded chat40 texts, 16 in flight.
