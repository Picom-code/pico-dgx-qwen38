# Builder A (spark-923b picom@spark-923b; integration + per-lane loops at 9-16 users) - progress
Started 2026-11-13 17:01 UTC. Source: Mac q38/wt/A (git worktree of work/tf, branch builder-A, base 52b32e1 = kp25 = patches 0101-0109).
Box dirs: ~/q38/build-A, ~/q38/tfbin/A*, ~/q38/runs/A_*, container q38-run-A, port 18301. Lease name A.
- 17:01 state: no window (lease status unit=inactive), DeepSeek health 200. handoff/{B,C,D} created (empty).
- 17:08 HARNESS FIX: head ~/q38/shared/lease.sh race (acquire while the old unit was still exiting left a stale lease and no window); fixed, noted in Mac q38/shared/NOTES.md.
- 17:10 LEVER A1 written (wt/A, uncommitted yet): TF_FLASHNEXT_Q38_ATTN_LANES=1 = a shared round's attention (prep, pool, scores, select, chunks, merge) in one launch
  a kernel for up to 16 text lanes (by-value lane table, fp8 and bf16 caches; idea = upstream #1739). Files: kernels/cuda/q38_attn_lanes.cuh (new), flashnext_attn.cu,
  flashnext_qsa_tile.cuh, flashnext_batch.cu, cuda_glue.zig, cuda_launch.zig, cuda_batch.zig, cuda_batch_state.zig, cuda_knobs.zig.
  ESTIMATE: 960 -> 72 launches a 16-user verify pass + 480 -> 36 in the draft chain = ~+1.5 % at 16 users.
- 17:22 A1 BUILD_OK on 923b (~/q38/build-A, 3 CPUs 8g while DeepSeek up; sha 7e568f39fb19ff2c; tree verified 0 of 7353 mismatches), ~/q38/tfbin/A1.
  Mac helpers: q38/A/shipA.sh NAME (ship + verify + build), A/w.sh 'cmd' (run on 923b), box ~/q38/kit-A/{eng.sh (container q38-run-A),libA.sh,w1.sh}.
- 17:13 FIRST ACQUIRE: window was already open (B, D leases); lease A registered, DeepSeek health 000 (stopped) = harness open path works. Run script 923b ~/q38/kit-A/w1.sh, log ~/q38/runs/A_w1.log.
- 17:42 A1 MEASURED so far (923b, tfb pc_16_262144, whole stack; one window):
  | run | 16 users distinct x3 | held 107, 16 in flight | waves of 16 |
  | A_ref1 (flag off) | 266.5 271.8 272.3 (270.2) | 178.0 | 149.7 tok/s, last first token 1.64 s |
  | A_on (ATTN_LANES=1) | 275.3 273.9 276.3 (275.2) | 180.7 | 154.1, 1.57 s |
  texts: A1 flag off vs kp25 (U_mu) waves 96/96 identical; on vs off held16 107/107, waves16 96/96 identical. => +1.8 % / +1.5 % / +2.9 % so far (ref2 pending).
- 17:52 A1 PAIRED RESULT (923b, one window, ref first + last). MEASURED:
  | run | 16 users distinct x3 (mean) | held 107 with 16 in flight | waves of 16 (last first token) | 12 users | 9 users |
  | A_ref1 off | 266.5 271.8 272.3 (270.2) | 178.0 | 149.7 (1.64 s) | - | - |
  | A_on ATTN_LANES=1 | 275.3 273.9 276.3 (275.2) | 180.7 | 154.1 (1.57 s) | 252.0 | 218.8 |
  | A_ref2 off | 268.3 268.1 267.6 (268.0) | 177.6 | 150.9 (1.82 s) | 250.4 | 214.1 |
  => 16 users +2.3 % (275.2 vs 269.1), held-16 +1.6 %, waves +2.5 %, 9 users +2.2 %, 12 users +0.6 %.
  IDENTITY: held 107/107 with 16 in flight (on vs ref1, ref2 vs ref1), waves-16 96/96, one-user ident 51/51; flag-off A1 vs kp25 waves 96/96. crash lines 0.
  SHIPS as default (ATTN_LANES=1). Patch 0110 generated (Mac A/mkpatchA.sh; TREE check below).
- 18:05 KPROF 16 users with ATTN_LANES on (923b ~/q38/runs/A_kp_on/kprof.txt): attention small kernels 1.22 -> 0.25 ms of a 28.8 ms profile unit. What is still a launch a lane:
  fn_shift_window 4.1 calls/unit (7.6 us), fn_ple_conv 1.7, fn_gdn_front 1.7, fn_gdn_rows 0.9, fn_gdn_rows_late 0.85 (log flushes): together 0.15 ms of 28.8 (0.5 %, ~9 of ~330 launches a unit).
  => per-lane launch loops are done; PLE conv/shift a lane = 32 launches a pass = ~0.04 % (ESTIMATE): NOT built.
  Biggest non-expert kernel at 16 users: fn_gdn_rows_late_lanes 1.957 ms/unit (6.8 %; 462 us a call for 16 lanes = 28.9 us a lane-layer = the state read 3.1 MB + end-state write 3.1 MB at ~215 GB/s).
- 18:10 LEVER A2 written + building (wt/A commit "A2"): TF_FLASHNEXT_Q38_LATE_SKIP=T. A drafted window whose rows + waiting log rows <= T writes no DeltaNet end states
  (state_out null in fn_gdn_rows_late); kept whole, its rows wait in the log (pend += rows) exactly like a partly kept window; above T the pass writes as before.
  Same chain steps in the same order from the same start state => same bits (to be proven by texts). ESTIMATE: +1.5..2.5 % at 16 users, +0.3..0.7 % one user (if the replayed rows stay hidden behind the bytes).
- 18:27 ATTN_LANES, GPU time by group at 16 users (923b, A1, GPROF; MEASURED):
  | run | pass ms (rows) | qsa-attn | gdn | moe | head | 16 users x2 | held-16 |
  | A_gp_off | 197.53 (79.8) | 8.471 | 34.45 | 130.42 | 7.53 | 266.0 268.5 | 178.1 |
  | A_gp_on  | 193.88 (79.0) | 5.682 | 34.22 | 129.86 | 7.53 | 267.7 270.1 | 180.3 (texts 107/107) |
  => qsa-attn group -2.8 ms a pass (-33 %), pass -1.8 %. LEVER CONFIRMED: +1.5..2.3 % at 16 users. (A_on2, run while the A2 build used 3 CPUs: 264.1 279.6 265.8, held 177.8,
  waves 152.1, texts 107/107: contaminated by the build, not used. LESSON: no build on the box while a paired run is measuring.)
  HARNESS: lease renewed twice (to 18:41, then 19:51): renew works; window shared with B, C, D leases.
- 18:28 w3 running on 923b (A2: LATE_SKIP T=16 / 32 / 8 vs ref at 16 users; log ~/q38/runs/A_w3.log), then w4 (one user) by hand.
- 19:04 LATE_SKIP at 16 users (923b, A2 sha 1f489b0116336b78 = A1 + LATE_SKIP; every run with ATTN_LANES=1; one window, ref first + last). MEASURED:
  | run | 16 users distinct x3 (mean) | held 107, 16 in flight | texts vs A_ref1 |
  | A2_ref1 (skip off) | 268.8 278.9 274.0 (273.9) | 180.9 | 107/107 |
  | T=16 | 277.6 277.1 270.5 (275.1) | 182.8 ; waves of 16 156.4 (1.57 s), 96/96 | 107/107 |
  | T=32 | 270.5 281.0 271.1 (274.2) | 182.6 | 107/107 |
  | T=8  | 271.2 278.1 280.4 (276.6) | 183.0 | 107/107 |
  | A2_ref2 (skip off) | 268.9 271.1 265.8 (268.6) | 179.9 | 107/107 |
  => skip on (mean of three T) 275.3 / 182.8 vs refs 271.3 / 180.4 = +1.5 % distinct, +1.3 % held-16; T does not matter in 8..32. Exact (texts identical with 16 in flight).
  Smaller than the +1.5..2.5 % estimate's top; GPROF pair (w5) and one user (w4, T=16) are running.
- 19:35 LATE_SKIP one user (923b, A2, tfb default 5 x 262K, whole stack; held 107 x 256 one in flight). MEASURED so far:
  | run | wall | ms/round by drafts bucket [2,3) / [3,4) / [4,7) |
  | A2_1ref (skip off) | 346.2 s = 58.3 tok/s | 49.57 / 53.25 / 61.53 |
  | A2_1t (T=16) | 344.8 s = 58.5 tok/s | 49.41 / 53.00 / 61.20 | texts 107/107 identical
  => one user +0.4 % (ref2 pending). Ident (A2_1ref vs A_ref1) 51/51.
- 19:10 INTEGRATION BRANCH Mac q38/wt/I (branch integ): A1 + A2 + C-stack.patch (applies cleanly) + D xstat.patch (aid). Patches 0110-0113 regenerated by Mac A/mkpatches.sh:
  6a8e032 + patches/01* = integ HEAD tree (TREE_EQUAL 6649fb10). D: no positive lever (exp-pairs negative, row-trim zero, draft depth zero) -> only the XSTAT aid is integrated.
  B: nothing in handoff yet. C: NOTE.md present (KEEP_INPASS=4096, LANE_REUSE=N recommended; PREFIX_WAIT verdict in RESULTS.txt, not there yet).
  Final-suite script ready: box ~/q38/kit-A/wF.sh (prep off on cmp mu mia), rsA.sh (documented path, container q38-run-A).
- 19:44 LATE_SKIP one user FINAL of the pair: ref 346.2 s / T=16 344.8 s / ref2 345.4 s (107 held x 256): +0.3 %; ms/round buckets ref 49.57 53.25 61.53, ref2 49.50 53.13 61.34,
  T=16 49.41 53.00 61.20 (-0.2..-0.3 %). Texts held 107/107 (both pairs), ident 51/51, edit set 4 in flight 24/24. SHIPS as default LATE_SKIP=16 (16 users +1.3..1.5 %, one user +0.3 %).
- 19:50 NOTE: handoff/C/C-stack.patch changed at 19:34 UTC (57,546 bytes; LANE_REUSE is now a count). The integ branch must be rebuilt from the newest patch (done at the next step).
  PLAN (window cap: opened ~17:10 by B/D, guard at 300 min = ~22:10): this window: w5 GPROF -> build I1 -> quick identity check -> recipe.sh prepare -> many-user final (mu + mia16).
  Next window: one-user final pair (levers off vs default, full suite).
- 19:57 LATE_SKIP, GPU time by group at 16 users (923b, A2, ATTN_LANES on in both; MEASURED):
  | run | pass ms (rows) | gdn | qsa-attn | moe | 16 users x2 | held-16 |
  | A2_gp_t16 (T=16) | 191.46 (79.6) | 31.21 | 5.65 | 130.31 | 279.8 277.6 | 182.1 (texts 107/107) |
  | A2_gp_ref (off)  | 192.24 (78.2) | 34.08 | 5.64 | 128.65 | 263.9 276.5 | 179.7 |
  => DeltaNet group -2.9 ms a pass (-8.4 %); held-16 +1.3 %. Both levers of mine together: 16-user pass 197.5 -> 191.5 ms (-3.1 %): qsa-attn 8.47 -> 5.65, gdn 34.45 -> 31.21.
- 19:58 integ branch rebuilt with C-stack.patch of 19:34 UTC (sha256 384a3efbcf6fb790, copy Mac handoff/used/): commits A1 ee9a3f4, A2 6a3ce14, C 2cc7445, D 73d83af;
  patches 0110-0113 regenerated, TREE_EQUAL 7dd1f75c. Build I1 started on 923b (6 CPUs, 12g; box idle, no engine).
- 20:04 I1 BUILD_OK (923b, sha 5761e00049a040f7 = integ 73d83af). recipe.sh (Mac repo/tensorfold, pushed to 923b ~/q38/repo_tf with patches 0110-0113): new defaults
  ATTN_LANES=1, LATE_SKIP=16, KEEP_INPASS=4096, LANE_REUSE=number of streams. `recipe.sh prepare` rc 0 with 0101-0113 (image tensorfold-qwen38:v1.1.4-eab8582 rebuilt, prepareA.log).
  Lease A renewed to 21:59 (window guard 22:10:14). Running on 923b: wF.sh "prep mu mia" (log ~/q38/runs/A_wF1.log), then wF2.sh "off onl cmp" (A_wF2.log; one-user pair, lite 'on').
- 20:14 FINAL MANY USERS so far, documented path, USERS16=1, image 0101-0113 (run F_mu; MEASURED): 1 / 4 / 8 / 12 / 16 users = 81.6 / 153.7 / 219.9 / 260.0 /
  276.9 + 283.0 + 278.6 (last first token 0.95 / 0.70 / 0.71 s), copies 16 347.9; waves of 16 160.1 tok/s (last first token 1.12 s), waves of 4 101.3 (0.55 s).
  (round 7 on this path: 81.2 / 151.5 / 212.3 / 247.9 / 269.4 + 266.1 + 272.7 (1.2 s), copies 336.8, waves 151.5 (1.63 s) / 97.2 (0.71 s).)
- B handoff: draft-study logging patch (aid) present; two-stage draft head (+1.0 % one user, B_T1, needs a fitted 30.7 MB file) still being built by B: NOT integrated yet.
- 20:33 FINAL MANY USERS, documented path, ONE window (923b, image 0101-0113; raw box ~/q38/runs/F_mu, F_mia16, log A_wF1.log). MEASURED:
  | users | MiaAI config 16 x 64K (all levers off) | USERS16 recipe default |
  | 1 / 4 / 8 / 12 | 75.4 / 140.1 / 189.9 / 209.1 (0.24 / 1.22 / 2.88 / 4.91 s) | 81.6 / 153.7 / 219.9 / 260.0 (0.19 / 0.53 / 0.70 / 0.99 s) |
  | 16 x3 | 218.4 219.2 217.3 (6.80 / 6.73 / 6.76 s) | 276.9 283.0 278.6 (0.95 / 0.70 / 0.71 s) => +28 % |
  | 16 copies | 245.4 (6.52 s) | 347.9 (0.65 s) |
  USERS16: waves of 16 160.1 tok/s (last first token 1.12 s), waves of 4 101.3 (0.55 s); 136,607-token prompt + 15 short: recall 3/3, shorts' last first token 1.01 s;
  227,621-token prompt 3/3 (2141 tok/s); held 107 with 16 in flight 194.1 tok/s, texts 107/107 identical to A_ref1 (kp25-equivalent texts). crash lines 0.
  (MiaAI waves + text compare: see A_wF1.log tail.)
- 20:35 C FINAL handoff read (RESULTS.txt). C-stack.patch changed again at 20:08 UTC (sha fa62a400): it only drops the measured-zero flag TF_FLASHNEXT_Q38_CACHE_SIBLINGS.
  DECISION: the repo ships the patch that was BUILT AND MEASURED (19:34 version, sha 384a3efb, copy in handoff/used/): the dead flag stays in patch 0112, off.
  PREFIX_WAIT (C table 4: 8 requests on a new 25.5K system prompt, last first token 86.7 -> 12.9 s; texts identical) is NOT a default (my final runs did not have it):
  recipe.sh gets it as an option PREFIX_WAIT=60000; AGENTS profile (pool of 5 x 131K: follow-up first token 85 s -> 1 s for 5 agents with 56K prompts) as option AGENTS=1.
- 20:57 FINAL ONE USER, "levers off" column (923b, documented path, image 0101-0113, every lever of ours off; run F_off; MEASURED): first token 184 ms;
  chat 47.6 / code 77.0 (3-prompt, 9 runs); sparkDash-method prose 56.6, prose T0.7 thinking 55.1, code 91.8, code8 87.4, structured 105.7;
  held same 104: all 58.01 (chat 49.79 / code 78.92 / math 73.98); edit set 137.2 wall (152.4 decode, 12.59 tok/round); agent set 54.2 wall (71.0 decode);
  4 users 146.9 / copies 150.0 (1.25 s); 5 users 162.9 / 167.6 (1.66 s); 16 queued on 5 slots 153.0 (37.7 s); wave 5 prose / code 132.7 / 214.0.
  F_on (recipe default, lite: held stride 3, no sparkDash-real) started 20:57 in the same window; ends ~21:46. Tables: Mac results/final_r8_manyuser.txt, final_r8_levers.txt written.
- 21:27 FINAL ONE USER, documented path, ONE window, same image 0101-0113 (923b; F_off 20:34-20:57, F_on 21:00-21:27; raw box ~/q38/runs/F_off, F_on, log A_wF2.log). MEASURED:
  | bench | levers off | recipe default | change |
  | first token, short prompt | 184 ms | 111 ms | -40 % |
  | 3-prompt chat / code, 9 runs | 47.6 / 77.0 | 50.1 / 84.7 | +5.3 / +10.0 % |
  | sparkDash-method prose / code / code8 / structured | 56.6 / 91.8 / 87.4 / 105.7 | 60.9 / 99.1 / 94.0 / 113.6 | +7.6 / +8.0 / +7.6 / +7.5 % |
  | held, same 104: all / chat / code / math | 58.01 / 49.79 / 78.92 / 73.98 | 62.38 / 53.44 / 85.72 / 79.28 | +7.5 / +7.3 / +8.6 / +7.2 % |
  | edit set wall (decode) | 137.2 (152.4) | 176.1 (198.1) | +28.4 % |
  | agent set wall (decode) | 54.2 (71.0) | 59.1 (76.0) | +9.0 % |
  | 4 users distinct / copies (last first token) | 146.9 / 150.0 (1.25 s) | 160.4 / 158.1 (0.29 s) | +9.2 / +5.4 % |
  | 5 users distinct / copies | 162.9 / 167.6 (1.66 s) | 175.0 / 192.3 (0.43 s) | +7.4 / +14.7 % |
  | 16 queued on 5 slots | 153.0 (37.7 s) | 171.4 (33.6 s) | +12 % |
  | wave 5 prose / code | 132.7 / 214.0 | 135.2 / 225.6 | |
  | prefill 7.4K / 29K / 117K; needle 182K; 227,621 prompt; image | | 2503 / 2519 / 2340 tok/s; 3/3 (2198); 3/3 (2134); PASS 0.42 s | |
  IDENTITY default vs off: ident 51/51, held 107/107, edit 24/24, agent 16/16 texts identical. crash lines 0. qeval ran (score: see below). held-320 and sparkDash 1.8.8 real NOT run this round (time).
- 21:28 B's two-stage draft head arrived in handoff (patches 0110 logging + 0111 head, no NOTE yet; B log: block form +1.9 % one user held, ident 51/51, held 107/107).
  Branch integ2 (Mac q38/wt/I2) = integ + B 0110 + B 0111, both applied cleanly (copies in handoff/used/). I2 building on 923b; file head2_r512.bin (sha c1f89b71, 30.7 MB) +
  head2_rot_r512.rot copied 3d24 -> 923b ~/q38/prof. w6.sh queued: I2 one user HEAD2 off / on with the whole recipe environment (held 107 x 256, texts), ends before 21:57.
- 21:34 COMMITTED + PUSHED private repo fc98d64 on tensorfold-fp4-local (author Pico noreply; gitleaks: no leaks; gateway + engine key values absent; repo private, anonymous 404):
  patches 0110-0113 (TREE_EQUAL 7dd1f75c against integ 73d83af), recipe.sh defaults (ATTN_LANES=1, LATE_SKIP=16, KEEP_INPASS=4096, LANE_REUSE=streams; options AGENTS=1,
  PREFIX_WAIT), README (round-8 tables on top), notes/final_r8_{923b,manyuser,levers}.txt, notes/round8 (this log, builders' notes), harness/round8 (lease harness + my run scripts),
  raw logs. NOTE: the repo recipe.sh has two OPTION blocks (AGENTS, PREFIX_WAIT) that the box copy used for the final runs did not have; defaults are the same.
- 21:35 w6 running (I2 one user: HEAD2 off, then on). qeval F_on: 33/35 (math 12/14, prose 5/5, reason 16/16).
- 21:49 B's TWO-STAGE DRAFT HEAD on the INTEGRATED build (923b, I2 sha 09108beff56b8be2 = integ + B 0110 + B 0111; tfb default one user, whole recipe environment incl.
  ATTN_LANES / LATE_SKIP / KEEP_INPASS / LANE_REUSE; 107 held x 256, one in flight; one pair, no last reference). MEASURED:
  | run | wall | ms/round | tok/round | texts vs A2_1ref |
  | I2_1ref (HEAD2 off) | 336.1 s = 60.1 tok/s | 54.45 | 3.456 | 107/107 (flags-off code is inert) |
  | I2_1h2 (HEAD2=/prof/head2_r512.bin, block form) | 329.7 s = 61.2 tok/s | 53.31 | 3.452 | 107/107 |
  => +1.9 % wall, -2.1 % ms/round; same texts. Agrees with B's paired +1.6 % (3d24). B's NOTE: the rotation was fitted on the same 104 held prompts the bench uses ->
  DECISION: ship as an OPTION (HEAD2=1), not a default, until it is re-fitted on broader text; patches 0114 (logging aid) + 0115 (head), default off.
- 21:52 lease A RELEASED (engine stopped, no q38-run container on 923b). Window still open on other leases / until the guard (22:10).
- 21:57 HARNESS CHECK COMPLETE (end to end): after the last lease (mine) was released at 21:52:01 the shared window closed BY ITSELF (run.log: "no live lease, closing 21:52:09"),
  every q38-run container was stopped, DeepSeek restarted and verified by the harness: health 200, 17*23 = 391, session store 12.00 GiB on every rank, disk tier 28.2 GiB,
  image request answered. Window S131710: 17:10:14 -> 21:56:40 (17,186 s; opened by B/D, used by all four). lease status: unit inactive, no leases.
  One fix made to the harness today (lease.sh race, 17:08). renew worked 5 times; release works.
- 21:58 NEXT (second, short window, mine): documented path with patches 0101-0115 (B's logging + two-stage head, default OFF): recipe.sh head2 (file from the checkpoint +
  shipped rotation; must reproduce sha c1f89b71), recipe.sh prepare, then one user default / HEAD2=1 / default (script box ~/q38/kit-A/wG.sh, log ~/q38/runs/A_wG.log).
  Patches 0114-0115 generated from integ2 (Mac q38/wt/I2, HEAD 0b5cf9b): TREE_EQUAL 414713a2. recipe.sh: new command `head2`, option HEAD2=1. Pushed to 923b ~/q38/repo_tf.
- 22:02 SECOND WINDOW (opened by my acquire in 8 s: "opened a new window"; lease A to 23:32). `recipe.sh head2` on 923b wrote head2_r512.bin with sha256 c1f89b717e2288f1 = the
  file B tested (bit for bit, from the checkpoint + the shipped 5.2 MB rotation). `recipe.sh prepare` rc 0 with patches 0101-0115 (prepareB.log).
- 22:19 G_ref (documented path, image 0101-0115, recipe default = HEAD2 off; MEASURED): first token 118 ms, chat 50.7 / code 85.2 (9 runs), held-104 62.46 (53.52 / 85.72 / 79.36);
  texts vs F_on (image 0101-0113): ident 51/51, held 107/107 identical => patches 0114-0115 are inert when off; the round-8 tables stand for the 0101-0115 image.
  G_h2 (HEAD2=1) running, then G_ref2.
- 22:36 G_h2 (documented path, image 0101-0115, HEAD2=1; MEASURED, same window as G_ref): first token 112 ms (ref 118), chat 52.1 / code 85.6 (ref 50.7 / 85.2),
  held-104 63.81 (chat 54.62 / code 87.82 / math 81.30) against G_ref 62.46 (53.52 / 85.72 / 79.36) = +2.2 % (chat +2.1, code +2.4, math +2.4);
  texts vs G_ref: ident 51/51, held 107/107 identical; image PASS; 4 users 159.5; engine line "q38 head2: on, rank 512, block form, re-score 933 of 79591 rows". G_ref2 running.
- 22:50 TWO-STAGE HEAD, documented path, paired with reference first and last (923b, image 0101-0115, one window; MEASURED):
  | run | first token | 3-prompt chat / code (9 runs) | held-104 all (chat / code / math) | texts |
  | G_ref (default, HEAD2 off) | 118 ms | 50.7 / 85.2 | 62.46 (53.52 / 85.72 / 79.36) | = F_on: ident 51/51, held 107/107 |
  | G_h2 (HEAD2=1) | 112 ms | 52.1 / 85.6 | 63.81 (54.62 / 87.82 / 81.30) | = G_ref: ident 51/51, held 107/107; image PASS |
  | G_ref2 (default) | 120 ms | 49.8 / 80.8 | 62.42 (53.48 / 85.70 / 79.34) | = G_ref: held 107/107 |
  => HEAD2=1: held +2.2 % (63.81 against 62.44), every category +2.1..2.4 %. Ships as an OPTION (B's caveat: rotation fitted on these prompts' head inputs). Pico's call to make it default.
- 22:52 lease A RELEASED; no q38-run container, 0 GPU processes on 923b.
- 22:53 COMMITTED + PUSHED private repo 08fe08b on tensorfold-fp4-local (author Pico noreply; gitleaks no leaks; key values absent; private): patches 0114-0115 (TREE_EQUAL
  414713a2 against integ2 0b5cf9b), recipe.sh `head2` command + HEAD2=1 option, head2/ (rotation 5.2 MB, mkhead2.py, draft vocabulary), README, tables, logs, builders' notes.
  Skill reference qwen38-fastest-build.md updated (round-8 state + method lessons).
- 22:56 SECOND WINDOW CLOSED BY ITSELF after my release (22:51:46 -> closing 22:51:57 -> DeepSeek verified 22:55:52): health 200, 17*23 = 391, session store 12.00 GiB on every
  rank, disk tier 28.2 GiB, image answered. Window S132157: 21:57:28 -> 22:55:52 (3,505 s), opened and used by A only. No q38-run container on any of the four boxes.

## Builder A END STATE (22:57 UTC 2026-11-13)
- No lease, no window (unit inactive), DeepSeek up and verified. 923b: no engine, no build. Binaries ~/q38/tfbin/A1, A2, I1 (= patches 0101-0113), I2 (= 0101-0115).
  Image tensorfold-qwen38:v1.1.4-eab8582 on 923b = 0101-0115 (prepareB.log); ~/q38/hf/q38/head2_r512.bin made by `recipe.sh head2`.
- Source: Mac q38/wt/A (builder-A: A1 ee9a3f4, A2 6a3ce14), wt/I (integ = + C 2cc7445 + D 73d83af), wt/I2 (integ2 = + B 57248c2 + B 0b5cf9b). Repo 08fe08b.
- REMAINING: (1) HEAD2 as a default needs a rotation fitted on broad text + a paired run on prompts outside the fit (+2.2 % one user waiting); (2) held-320 in full and
  sparkDash 1.8.8 itself were not re-run on the final image; (3) PREFIX_WAIT as a default needs the 16-user bench with it on (does the shared chat-template opening make distinct
  prompts wait?); (4) C's indexer design (long-prompt prefill +15 %, estimate); (5) per-lane PLE launches (0.04 %, not worth it); (6) C's final patch dropped the dead
  CACHE_SIBLINGS flag: patch 0112 still carries it (off).
- RESUME: cd ~/.hermes/cache/scratch/q38; tail -40 PROGRESS-A.md; build: A/shipI2.sh NAME (tree wt/I2 -> 923b ~/q38/build-A); patches: TREE=$PWD/wt/I2 A/mkpatches.sh;
  window: ./h.sh 'bash ~/q38/shared/lease.sh acquire A 60'; runs: box ~/q38/kit-A/{libA.sh,wF.sh,wG.sh}; release: ./h.sh 'bash ~/q38/shared/lease.sh release A'.
