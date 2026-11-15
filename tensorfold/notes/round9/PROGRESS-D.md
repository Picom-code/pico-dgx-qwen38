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
