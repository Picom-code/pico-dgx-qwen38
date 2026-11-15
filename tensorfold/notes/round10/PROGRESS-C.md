# Builder C (spark-4e5c pico@spark-4e5c): first-token time + prefill. Progress log (resumable)
Started 2026-11-13 17:00 UTC. Base = work/tf 52b32e1 (patch 0109 state). Worktree Mac q38/wt/C branch builder-C.
Box build dir ~/q38/build-C, binaries ~/q38/tfbin/C1.., runs ~/q38/runs/C_*, container q38-run-C, port 18301. Lease name C.
Recipe KV = fp8 (MiaAI default; kept as is).
## Log
- 17:02 workspace created; no lease held. Reading upstream (#1866 KEEP_INPASS, BF16 128x128 tile, PREFIX_WAIT, a third engine prefill).
- 17:16 C1 built (sha 6a1ebfca964ef206 = base + PLOG/GPROF=2 measuring aids + PREFIX_WAIT port behind TENSORFOLD_PREFIX_WAIT_MS, default off). Lease C acquired 45 min; window-1 measure script runW1.sh started on 4e5c (log ~/q38/runs/C_w1.log, runs C_w1_*; deadline 36 min).
- 17:27 WINDOW 1 first results (MEASURED, spark-4e5c, C1 = base + measuring aids, recipe one-user config 5 x 262K, fp8 KV, vision on):
  | item | with PLOG+GPROF | no profilers |
  | new short prompt (19-25 tokens), first token, median of 12 | 166 ms | 162 ms |
  | same prompt again (resumes at the history mark) | 90 ms | (see C_w1.log) |
  | prefill 7,276 / 29,044 / 116,494 tokens | 2236 / 2525 / 2368 tok/s (3.25 / 11.50 / 49.19 s) | |
  | agent, 1 user, 20K system + 1.5K a turn: turn 0 / follow-ups | 8.15 s / 0.84-0.86 s (cache hit every turn) | |
  WHERE A NEW SHORT PROMPT'S 162 ms GOES (PLOG): lane open + restore of the 2-token template state 23 ms; pass over rows [2, plen-7) 66-78 ms; keep of the
  history mark (plen-7 = end of the user text) 5 ms; SECOND pass over the last 7 rows (the generation prompt) 57-60 ms. A pass of 7 rows costs 58 ms and one
  of 16 rows 78 ms (byte floor), so the mark that cuts the prompt in two costs ~55 ms = one third of the first-token time. Long prompts pay the same 69 ms
  7-row pass at their end. => LEVER 1: keep the mark's state INSIDE the pass (upstream #1866's idea). Building as C2: TF_FLASHNEXT_Q38_KEEP_INPASS=N.
  Prompt chunk by group (GPROF=2, a 3072-row chunk at 3K-6K context, 1127 ms = 0.367 ms a row): MoE 423 (37.5 %), GDN 263 (23.4 %), hc-attn-side 164 (14.6 %),
  qsa-attn 162 (14.3 %), hc-mlp-side 101 (9.0 %), ple 11. First chunk of a cold engine: MoE 700 ms (cold expert pages: +277 ms once).
- 17:27 C2 building = C1 + TF_FLASHNEXT_Q38_KEEP_INPASS (our own in-pass keep: DeltaNet chain launched twice around the mark, conv/tail windows shifted copies,
  attention rows copied after the pass; cuda_batch.zig markRow, cuda_snap.zig prepare/finish, cuda_lanes.zig step).
- 17:35 PROMPT-PASS TABLE (MEASURED, C1, GPROF=2 = GPU ms by group for every prompt chunk; 4e5c; raw: box ~/q38/runs/C_w1_one/engine.log, parser kitC/gp.py):
  | chunk (rows @ context) | GPU ms | us/row | MoE | GDN | hc-attn-side | hc-mlp-side | QSA attn | PLE |
  | 3072 @ 3K   | 1127 | 367 | 423 | 263 | 164 | 101 | 162 | 11 |
  | 3072 @ 12K  | 1147 | 373 | 416 | 269 | 168 | 102 | 178 | 11 |
  | 3072 @ 30K  | 1191 | 388 | 428 | 268 | 170 | 102 | 208 | 11 |
  | 3072 @ 49K  | 1229 | 400 | 437 | 266 | 169 | 104 | 236 | 15 |
  | 3072 @ 67K  | 1265 | 412 | 435 | 271 | 168 | 104 | 271 | 11 |
  | 3072 @ 86K  | 1312 | 427 | 438 | 275 | 171 | 106 | 309 | 11 |
  | 3072 @ 104K | 1335 | 435 | 438 | 271 | 169 | 104 | 340 | 11 |
  | 1123-1447 rows (the tail-rule halves) | 497-648 | 442-449 | 227-285 | 100-133 | 64-82 | 39-50 | 63-93 | 4-7 |
  | 3072, FIRST chunk of a fresh engine | 1418 | 462 | 701 (cold expert memory, once) | 288 | 169 | 105 | 121 | 32 |
  Whole prompts: 7,276 tok: GPU 3042 of 3186 ms wall (418 us/row; MoE 44 % because of the cold first chunk); 29,044: 10,993 of 11,387 (378.5 us/row; MoE 37.6, GDN 22.9,
  QSA 15.1, hc-attn 14.4, hc-mlp 8.8 %); 116,494: 47,053 of 48,745 (404 us/row; MoE 34.7, GDN 21.7, QSA 20.4, hc-attn 13.6, hc-mlp 8.3 %).
  FINDINGS: (1) every group is flat with context EXCEPT QSA attention: 121 ms a chunk at 0K -> 340 ms at 104K (+2.1 ms per 1K tokens of context = the indexer's
  block scores fn_qsa_scores (scalar fp32, 512 fma per row x block x layer) and fn_qsa_select (4 radix passes over every block a row)). That growth is ~10 % of a
  117K prefill and ~18 % (estimate) of a 227K one; without it the 117K prompt would run at ~0.363 ms a row = ~2,700 tok/s (a third engine measured 2,848).
  (2) wall - GPU = 3.5 % (host between chunks + the last 7-row pass + keeps). (3) partial chunks cost 0.445 ms a row against 0.367: expert row tiles are less
  full (round 7's reading), so the tail rule's two halves of ~1,400 rows cost ~20 % more a row than a full chunk; chunk 8192 was already measured +0..2 %.
  (4) the first big chunk after a start is +290 ms (cold expert pages): the 8K prefill number is 2,236 with it and ~2,430 without.
  NEXT for prefill: KPROF at 117K to split the QSA growth between scores / select / the fused attention, then an exact faster form of whichever dominates.
- 17:43 LEVER 1 "mark kept inside the pass" (TF_FLASHNEXT_Q38_KEEP_INPASS=64, build C2 sha c1755dc91c9ae22c) FIRST RESULT, MEASURED (4e5c, one user, PLOG+LHASH on both sides):
  new short prompt first token 167.5 -> 124.1 ms median of 12 (-26 %); same prompt again 90 ms (restores the in-pass-kept state); 36 marks kept in pass.
  IDENTITY vs flag off (two fresh engines, same prompts): score hashes after the prompt pass ("lhash first") 59/59 identical, at release 59/59; texts short 24/24,
  prefill 2/2, agent trace 5/5, held 40/40 (96 tokens). Next: =4096 (long prompts' last mark too), paired timing in one engine (knob file, C3), ident 51 set.
  NOTE prompt-cache budget: it is what the streams leave of the memory budget: 4,282 MiB in window-1 runs, 1,199 MiB in the x0 run (a build was running beside it),
  i.e. 39 / 10 short states of 110 MiB. A kept state = 110 MiB (DeltaNet states + conv windows + n-gram tail) + ~17 KiB a token (fp8 KV, 12 layers + head).
  For agents this is the thing to measure next (8 agents x 25K tokens need > 4 GiB of states): window 3 does.
- 17:57 LEVER 1 PAIRED TIMING in ONE engine (C3 sha 5ef94b6d1c4eae37, knob file switches the flag between prompt sets; fresh tagged prompts of 28-31 tokens, 16 each; MEASURED):
  | keep_inpass | new short prompt, first token median ms (reps a / b / c) | prefill 7.3K / 29K s | agent follow-up turn (700 new tokens) |
  | 0 (off)  | 174.5 / 176.8 / 175.7 | 3.05, 3.00, 2.98 / 11.72, 11.63 | 0.57 / 0.58 s |
  | 64       | 134.9 / 136.5         | 3.02, 3.02 / 11.73, 11.60       | 0.57 / 0.58 s |
  | 4096     | 140.0 / 140.7 (its tag is 2 tokens longer) | 2.96, 2.96 / 11.74, 11.64 | 0.53 / 0.53 s |
  => new short prompt -22..-23 % (175.7 -> 135.7 ms); agent follow-up turns -8 % with 4096; long cold prompts -1..-3 % at 7K, 0 at 29K.
  IDENTITY =4096 vs off (fresh engines, C2): lhash first 59/59, end 59/59; texts short 24/24, prefill 2/2, agent 5/5, held 40/40; 60 marks kept in pass.
  C2/C3 run a marked pass's DeltaNet without the fused kernel (a 1,394-row last chunk: 792 ms against 654 + 77 keep + 72 = 803 cut: only -11 ms).
  C4 (sha 33b783b775296b4e, BUILD_OK 17:53) runs each side of the mark as the cut pass's chunk would (fused kernel from 64 rows): window 4 measures it + ident set.
- 18:03 window-3 script: part 1 (paired timing) done; parts 2-3 (KPROF, 16-lane prefix wait) did NOT run: I pushed an engC.sh with a syntax error while the script
  was running (lesson: pushkit.sh now runs bash -n first; do not push kit changes under a running script). C5 built (sha 5812074a53e5de5e) = C4 + LEVER 3
  TF_FLASHNEXT_Q38_LANE_REUSE=1 (a released stream's 73 fixed device buffers, 110 MiB, reused for the next stream instead of freed + allocated; lane open is 22-27 ms
  of a short prompt's first token) + openparts timing in PLOG. Lease C renewed to 19:37. runW5.sh started 18:03 on 4e5c (log ~/q38/runs/C_w5.log, runs C_w5_*):
  1 identity i0 vs i1 (KEEP_INPASS=4096 + LANE_REUSE=1; ident 54 set, needle 120K, short, agent, prefill; lhash) 2 paired timing in one engine (4 flag combos x 3)
  3 16 lanes: bursts of 8 x 28K / 16 x 12K shared NEW system prompt + 8 shared-prefix agents, prefix wait off / on 4 agent cache capacity: 5 agents x 60K, recipe
  default vs pool of 5 x 131K (variant ag131072) 5 KPROF of a 117K prompt.
- 18:22 IDENTITY, C5 (sha 5812074a53e5de5e), flags off (C_w5_i0) vs TF_FLASHNEXT_Q38_KEEP_INPASS=4096 + TF_FLASHNEXT_Q38_LANE_REUSE=1 (C_w5_i1), fresh engines, MEASURED:
  ident set 51/51 texts identical; needle 109,126 tokens 3/3 and the same answer text (46.6 -> 46.0 s); short 24/24; agent trace (20K + 4 turns) 4/4; prefill 2/2;
  score hashes after the prompt pass 71/71 identical, at release 71/71; 72 marks kept in pass.
  Timing in those two runs (PLOG on both): new short prompt 168.1 -> 118.1 ms (-30 %); agent follow-up turn (1.5K new tokens at 20K) 0.88 -> 0.79 s (-10 %);
  prefill 2,766 tok 1.19 -> 1.11 s, 7,276 tok 3.03 -> 2.90 s.
  WHERE A LANE'S OPENING GOES (PLOG openparts): sequence buffers 5.9 ms (0.5 ms with LANE_REUSE), the MTP draft head's buffers 16-17 ms (45 fixed-size device
  allocations + a host array per request), restore of the kept 2-token state ~1 ms.
  => C6 (sha 30d28bb25f0310b0, BUILD_OK 18:14) extends LANE_REUSE to the draft head's fixed buffers (a size-keyed idle pool, <= 1 GiB, zero-filled on reuse).
  runW5 stopped after its identity part; runW6.sh (C6) started 18:22: identity both levers vs C_w5_i0, paired timing 4 flag combos x 3, 16-lane prefix wait,
  agent cache capacity (default vs pool 5 x 131K), KPROF 117K, then flags-off ident on C6 against C5's. Log box ~/q38/runs/C_w6.log.
- 18:32 C6 IDENTITY + first result (MEASURED): flags off (C_w5_i0) vs KEEP_INPASS=4096 + LANE_REUSE=1 on C6 (C_w6_i1): ident 51/51, needle 109K 3/3 same text,
  short 24/24, agent 4/4, prefill 2/2, lhash first 71/71, end 71/71, 72 in-pass keeps. New short prompt first token 168.1 -> 99.7 ms (-41 %); lane opening
  22.5 ms -> 0.7 ms (seq 0.5 + draft 0.1); agent follow-up 0.88 -> 0.78 s; prefill 2.8K 1.19 -> 1.09 s, 7.3K 3.03 -> 2.88 s.
  C7 (building next) makes LANE_REUSE=N the number of idle sets (16 users arriving together open 16 lanes: 16 x 23 ms before their batched pass).
- 18:57 PAIRED TIMING, C6, ONE engine, knob file, 3 reps x 4 combos, fresh tagged prompts of 29 tokens (16 each), MEASURED (medians; reps a / b / c):
  | keep_inpass / lane_reuse | new short prompt first token ms | agent follow-up (700 new tok at 6-9K) | prefill 2.7K / 7.3K / 29K s (rep a) |
  | 0 / 0       | 173.7 / 176.8 / 175.5 | 0.57 s | 1.18 / 3.03 / 11.71 |
  | 4096 / 0    | 133.8 / 133.4 / 132.3 | 0.52-0.53 s | 1.14 / 2.95 / 11.68 |
  | 4096 / 1    | 112.4 / 112.9 / 112.6 | 0.49-0.50 s | 1.09 / 2.94 / 11.65 |
  | 0 / 1       | 155.5 / 156.6 / 153.1 | 0.55 s | 1.14 / 2.97 / 11.59 |
  => both levers: new short prompt 175.3 -> 112.6 ms (-36 %); agent follow-up turn -13 %; cold prompts -8 % at 2.7K, -3 % at 7.3K, -0.5..-1.5 % at 29K.
- 18:57 LEVER 3 PREFIX_WAIT (port of upstream #1321), C6, 16 lanes (USERS16 profile without vision), ONE engine, knob file off / 60000, MEASURED (rep a; rep b off equal):
  | load | wait off | TENSORFOLD_PREFIX_WAIT_MS=60000 |
  | 8 requests at once on a NEW 25.4K-token system prompt: first / median / LAST first token | 26.8 / 81.1 / 85.7 s | 10.6 / 12.2 / 13.4 s |
  | 16 requests at once on a NEW 11K-token system prompt | 4.9 / 74.0 / 75.9 s | 4.8 / 9.2 / 11.0 s |
  | 8 agents, shared 20K system prompt, 3 turns: wall; sum of all first-token waits | 92.2 s; 484 s | 38.7 s; 123 s |
  Identity run (same prompts, two fresh engines, lhash) is in runW7.sh (C_w7_pw0 / pw1).
- 19:15 AGENTS AND THE PROMPT CACHE (item 3), C6 + KEEP_INPASS=4096, 5 agents at once, each its OWN 56K-token system prompt, 3 turns (+1.5K tokens a turn), MEASURED:
  | configuration | prompt cache budget | follow-up turn first token median / max | cache hits of 10 follow-ups | wall |
  | recipe one-user default: 5 lanes x 262,144 windows | 4,081 MiB | 85.1 s / 146.0 s | 0 (every follow-up re-prefilled 57K tokens) | 365.6 s |
  | 5 lanes sharing a POOL of 5 x 131,072 tokens (TF_FLASHNEXT_Q38_POOL_CTX=131072, --parallel 5) | 13,395 MiB | 1.01 s / 13.0 s | 10 | 138.0 s |
  Texts 15/15 identical between the two. CAUSE: a kept state is a device COPY: 110 MiB + ~17 KiB a token = 1.09 GiB at 56K; a request keeps ~3 marks (end of
  system text, history mark, ..), so ONE request fills the 4 GiB budget and pushes every other agent's states out; eviction takes a prompt's own earlier marks
  only after other conversations' states. The budget is what the lanes' full-window reservation leaves (28.4 GiB room - 5 x 4.85 GiB).
  => (a) config: the pool profile gives the cache 13.4 GiB with no code change (one stream can still take 262K; all five together 655K tokens);
     (b) code, C8 (sha 640b02a74dedb731): TF_FLASHNEXT_Q38_CACHE_SIBLINGS=1 = eviction takes a prompt's earlier marks before any conversation's newest state.
  runW7.sh (C8) queued after runW6: 16 lanes lane reuse 0/16, prefix-wait identity (two fresh engines), agents on the default profile with (b).
- 19:39 runW6 finished (19:25): C6 flags-off ident vs C5 flags-off: 51/51, lhash 52/52 (same reference). KPROF run printed no table (N = 200,000 launches is more
  than a 117K prompt makes: ~3K launches a chunk); repeated with N = 20,000 at the end of runW7.
- 19:39 16 LANES (USERS16 profile without vision), C8 (sha 640b02a74dedb731), ONE engine, knob file, alternating, 3 reps, MEASURED (qk.py distinct N: N coding prompts at once, 512 tokens):
  | users | flags off: tok/s total; LAST first token | KEEP_INPASS=4096 + LANE_REUSE=16 |
  | 16 | 263.8, 259.9, 269.4; 1.13, 2.02, 1.32 s | 271.3, 271.3, 273.7; 1.04, 0.70, 0.71 s |
  | 8  | 215.5, 211.7, 206.4; 0.72, 0.69, 0.73 s | 216.0, 216.0, 211.9; 0.49, 0.49, 0.50 s |
  | 4  | 150.1, 151.9, 154.7; 0.51, 0.52, 0.53 s | 158.1, 154.6, 149.8; 0.37, 0.38, 0.38 s |
  | 1 short prompt (qk.py ttft, median of 7) | 184, 185, 185 ms | 118, 119, 118 ms |
  => last first token at 8 users -31 %, at 4 users -27 %, at 16 users 1.49 -> 0.82 s mean (noisy: one 2.0 s outlier off); 16-user throughput +2.9 % (mean of 3),
  8 / 4 users inside noise. One short prompt on the 16-lane profile 185 -> 118 ms (-36 %).
  IDENTITY with 16 in flight: held 107 prompts x 128 tokens, flags off then on in the same engine: texts 107/107 identical (the 160.0 -> 176.1 tok/s of that
  pair is NOT a speed claim: the second run found the first run's kept states).
- 19:53 PREFIX WAIT IDENTITY + second timing (C8, two FRESH 16-lane engines, the SAME prompts; C_w7_pw0 = all flags off, C_w7_pw1 = TENSORFOLD_PREFIX_WAIT_MS=60000 +
  KEEP_INPASS=4096 + LANE_REUSE=16), MEASURED:
  | load | off | on |
  | 8 at once on a new 25.5K system prompt: first / median / last first token | 28.4 / 82.7 / 86.7 s | 10.7 / 11.6 / 12.9 s |
  | 16 at once on a new 11K system prompt | 4.9 / 75.0 / 76.9 s | 4.7 / 6.4 / 9.0 s |
  | 8 agents, shared 20K system prompt, 3 turns: wall; sum of first-token waits | 92.2 s; 503.9 s | 35.9 s; 106.5 s |
  texts: burst-8 8/8, burst-16 16/16, agents 24/24 identical; score hashes after the prompt pass (lhash first) 48/48 identical; 29 requests waited.
  CAVEAT, reported as found: the release-time hash (lhash end) differs in 47 of 48 streams although the tokens fed are equal in all 48 and every text is identical.
  With one user that hash was identical in every lever run (71/71). Hypothesis (NOT proven yet): at release the lane's score buffer holds the last scored row of
  its last shared window, which depends on how the round was grouped, not on the stream. runW8 tests it: the flags-off bursts again on a third fresh engine,
  lhash end against C_w7_pw0 (identical configuration twice). If those differ too, the hash is schedule noise under concurrency; if they are equal, it is a real
  difference of the levers and must be found before shipping PREFIX_WAIT / LANE_REUSE at many users.
  runW8.sh chained after runW7 (lease C renewed to 20:57): that check + the agents profile as it would ship (pool 5 x 131K + every lever).
- 20:08 runW7 finished. MEASURED:
  * TF_FLASHNEXT_Q38_CACHE_SIBLINGS=1 (C8) on the default one-user profile, 5 agents x 56K: follow-up first token median 84.9 s (off: 85.1 s), texts 15/15: ZERO.
    Why: the store makes room for ALL of a new request's marks at its lookup (reserve), which empties a 4 GiB cache of every other agent's state before any
    sibling exists. DROPPED: code removed from the tree (the handoff tree is C7's; upstream's rework of this is #2699, not ported). The config fix stands
    (pool profile: 85 s -> 1.0 s).
  * default one-user profile as it would ship (C8, KEEP_INPASS=4096 + LANE_REUSE=5): new short prompt 108.2 ms (tt.py, 27 tokens), qk.py ttft 122 ms (the recipe's
    ttft figure is 182 ms on 923b; 184-185 ms flags off on this box), prefill 7.3K / 29K / 117K = 2514 / 2540 / 2349 tok/s.
  * KPROF of a 117K prompt (serialised launches, 20,000 a table = ~13 chunks; table 1 = middle of the prompt, table 2 = its end), ms a table:
    | kernel | middle | end | growth |
    | qsa_tile::scores (indexer block scores, scalar fp32) | 807 | 1547 | +740 (74 % of the QSA growth) |
    | fn_qsa_select (top blocks a row) | 323 | 513 | +190 (19 %) |
    | fn_attn_fused_kv8 | 1210 | 1282 | +72 (7 %) |
    | expert tiles gate/up + down | 2812 + 2130 | 2855 + 2143 | flat |
    | MXFP8 prompt tile (bf16 / f32 out) | 1856 + 752 | 1871 + 757 | flat |
    | fn_gdn_fused | 1504 | 1511 | flat |
    | hyper-connection write-back + down/up | 1405 + 494 + 861 + 820 | same | flat |
    => the growth of a long prompt's cost IS the indexer's block scores (design note handoff/C/NEXT-qsa-certified-select.md), not attention itself.
- 20:13 "lhash end" CHECK (runW8, C8): the flags-off bursts a second time on a third fresh 16-lane engine against C_w7_pw0 (IDENTICAL configuration twice):
  texts 24/24 identical, lhash first 24/24 identical, lhash end 3/24 identical. => the release-time hash is not reproducible between two identical concurrent
  runs: it is schedule noise (the lane's score buffer at release depends on how its last shared round was grouped), NOT an effect of the levers. With one user
  it is stable and was identical in every lever run (71/71). The identity criteria for many users stay: lhash first + texts.
- 20:33 runW8 / runW9 finished. MEASURED:
  * agents profile as it would ship (C8, pool 5 x 131K + KEEP_INPASS=4096 + LANE_REUSE=5 + PREFIX_WAIT): 5 agents x 56K: follow-ups median 0.99 s (max 13.8),
    wall 135.4 s, texts 15/15 = the default profile's; cache 15.6 GiB; short prompt 110.5 ms; cold 7.3K / 29K 2445 / 2523 tok/s.
  * HANDOFF BINARY C9 (sha d14ca644ac3405d6, built from the handoff tree after the dropped flag was removed; tree = base + C-stack.patch, TREE_EQUAL) with the three
    flags vs flags-off reference C_w5_i0: ident 51/51, needle 109K 3/3 same text, short 24/24, agent 4/4, prefill 2/2, lhash first 71/71, end 71/71, 72 in-pass keeps;
    short 168.1 -> 100.7 ms, agent follow-up 0.88 -> 0.77 s.
## END STATE (20:35 UTC 2026-11-13)
- Lease C RELEASED 20:32; no q38-run-C container, no run script on 4e5c. Raw results pulled: Mac q38/results/rC (+ rC.tgz; engine logs as engine.q38.log.gz).
- HANDOFF: Mac q38/handoff/C/: C-stack.patch (sha256 fa62a400a3293ca8, wt/C HEAD 4c8eedb), NOTE.md (flags, how to build, what to change in recipe.sh),
  RESULTS.txt (all tables), NEXT-qsa-certified-select.md (the unbuilt prefill lever with its measurements). Builder A integrates; nothing pushed by C.
- SHIPPED-READY (exact, flags default off in the patch): TF_FLASHNEXT_Q38_KEEP_INPASS=4096, TF_FLASHNEXT_Q38_LANE_REUSE=N, TENSORFOLD_PREFIX_WAIT_MS=60000,
  and the agents profile (config only). DROPPED: CACHE_SIBLINGS (zero). NOT BUILT: certified indexer select (prefill at long context), #1846 BF16 tile.
- WHAT REMAINS (in order of measured size):
  1 long-prompt prefill: indexer block scores grow with context (10 % of a 117K prefill, ~18 % est. of 227K): certified select, design + KPROF split in handoff.
  2 the prompt cache for agents on the DEFAULT profile: a state is a 1 GiB device copy at 56K; the store reserves room for all of a request's marks at lookup.
    Real fixes: upstream #2699's policy (newest state per conversation survives) or page-shared KV (upstream TF_FLASHNEXT_KV_PAGES). Until then: the pool profile.
  3 a short prompt's remaining ~100 ms = one byte-bound pass (~78 ms for 17 rows) + keep 5 ms + HTTP/template: no lever seen.
  4 KEEP_INPASS covers one mark a pass, text only, one GPU; the first big chunk after an engine start is +290 ms (cold expert pages).
  5 engine starts on 4e5c take ~200 s each (model load): plan windows around it; flags can be switched in a running engine with TF_FLASHNEXT_Q38_KNOBFILE.
- RESUME: cd ~/.hermes/cache/scratch/q38; tail -60 PROGRESS-C.md; ./shipC.sh NAME (build); ./pushkit.sh (kit -> box, bash -n first); ./hc.sh 'cmd' (run on 4e5c);
  ./h.sh 'bash ~/q38/shared/lease.sh acquire C 60'; box: bash ~/q38/kit-C/runW9.sh $(( $(date +%s) + 1800 )) (identity of a binary named in the script).

# Round 9 (builder C, started 2026-11-13 23:00 UTC)
Box spark-4e5c. Levers: (1) certified indexer select (long-prompt prefill), (2) agent prompt cache on the default profile, (3) partial chunk / cold chunk.
- 23:01 DECISION: handoff/BASE.txt absent at start -> base = round-8 integration commit integ2 0b5cf9b (patches 0101-0115). Worktree Mac q38/wt/C, new branch r9-C off 0b5cf9b.
  If A writes a different BASE later, rebase the handoff patch onto it (the indexer files were not touched by round 8 levers).
- 23:11 CODE READ (corrects round 8's note): the indexer runs per 256-row group (att_rows = 256), not per 16 rows: stock scores = grid (32 x blocks/256) CTAs of
  8 rows, so each layer's pooled keys (6.6 MB at 104K) are read 32 times a group = ~30 GB a 3,072-row chunk at 104K: the score kernel is likely memory-bound as
  much as compute-bound. top = 512 blocks of 4 tokens; select = one 1024-thread CTA a row (4 radix passes + placement).
- 23:11 C10 BUILDING (branch r9-C, base 0b5cf9b). Three flags, all default off:
  * TF_FLASHNEXT_Q38_QSA_CERT=1|2 (+ _CERT_MIN=64 rows, _SLACK=100 millionths, _STATS=1): kernels/cuda/q38_qsa_cert.cuh. fn_q38_qsa_coarse = block scores on
    tensor cores (bf16 mma, 64-block tiles staged once for all 256 rows, + each tile's largest key norm); fn_q38_qsa_select_cert = per row: top-th largest coarse
    score v, eps = slack * sum_h|q_h| * max|k| / sqrt(128); blocks with F > v+2eps certainly listed, F < v-2eps certainly not; the blocks in between (<= 1024)
    get the exact scalar sum (same ops, same order = same bits) and are ranked exactly (ties: lowest block first, as stock); else fallback = exact sums for the whole
    row + stock select inside the kernel. Mode 2 = check: stock scores + stock select also run, device counters: rows whose list differs, exact-bit diffs, bound violations.
  * TF_FLASHNEXT_Q38_QSA_LOOP=L: the stock exact kernel with L 8-row tiles a CTA over the same 256 blocks (keys read once a CTA; identical code per sum).
  Knob-file words: qsa_cert, qsa_cert_min, qsa_slack, qsa_stats, qsa_loop. Decode rows (< CERT_MIN rows) stay on the stock kernels.
- 23:14 C10 BUILD_OK sha 97a016b5b2bbe87c (first compile). 23:15 lease C acquired to 00:05 (window already open by A, D). runX1.sh started on 4e5c (log ~/q38/runs/C_x1.log, dir C_x1): check mode (30K + needle 120K), then paired off/loop8/loop32/cert/off at 29K+117K, off/cert/off at 227K, GPROF=2.
- 23:20 FIRST CHECK (MEASURED, C10, qsa_cert=2, 27,417-token prompt): 329,459 row selections checked on the device: list_diff=0, exact_bit_diff=0, bound_viol=0, fallback=0; blocks needing the exact sum: 7.9 a row on average, 29 at most (of up to 6.8K); largest |coarse-exact|/eps = 0.0019 (slack 100e-6 has ~500x margin). Engine: 3.5 GiB prompt cache, READY 209 s.
- 23:22 CHECK at 109,102 tokens (needle 3/3): cumulative 1,720,550 row selections checked: list_diff=0, exact_bit_diff=0, bound_viol=0, fallback=0, redone=0; in-between blocks 9.8 a row avg, 33 max (of up to 27K blocks); max ratio 0.0020. (>= 1 M selections criterion met at 27K + 109K; 227K still to check.)
- 23:30 WINDOW X1 DONE (lease C released 23:31). C10, ONE engine (recipe one-user profile, 3.5 GiB cache), knob file, GPROF=2 + PLOG on every side, fresh tagged prompts. MEASURED:
  | prompt tokens | off (first / last) | QSA_LOOP=8 | QSA_LOOP=32 | QSA_CERT=1 |
  | 27.3K  | 2476 / 2457 tok/s | 2545 (+3.2 %) | 2472 (0) | 2561 (+3.8 %) |
  | 109K   | 2322 / 2310       | 2368 (+2.2 %) | 2317 (0) | 2454 (+6.0 %) |
  | 214K   | 2142 / 2140 (99.8 s) | -          | -        | 2345 (+9.5 %; 91.3 s) |
  QSA group GPU ms of the whole prompt (GPROF): 109K: 9,820-9,906 off, 8,974 loop8, 9,850 loop32, 7,389 cert; 214K: 26,128-26,166 off -> 18,009 cert.
  A 3,072-row chunk at ~200K context: 1,578 ms off -> 1,350 ms cert; growth with context 1.9 -> 1.07 ms a chunk per 1K tokens.
  CHECK MODE (qsa_cert=2, the stock kernels also run, device compare): 1,720,550 row selections (27K prompt + 109K needle, 3/3): 0 lists differ, 0 exact-sum bit
  differences, 0 bound violations, 0 fallbacks; 9.8 in-between blocks a row (max 33-37); largest |coarse - exact| = 0.2 % of the bound.
  => CERT works and is exact on every row checked. LOOP=8 gives +2-3 % by itself (the exact kernel IS partly memory-bound) but cert replaces that kernel; LOOP=32 = 0
  (too few CTAs). Remaining QSA growth under cert: the select kernel (6 passes over each row's scores + ~100 block scans) is now the largest part (estimate).
- 23:32 NEXT: C11 = select v2 (2 radix passes of 12 bits on the coarse scores instead of 4 of 8: any bracket [v_lo, v_hi] of the top-th value certifies; contiguous
  block ranges a thread with ONE block scan for the candidate list and ONE for the placement instead of two scans per 1,024 blocks) + lever 2 below.
- 23:32 LEVER 2 DESIGN (agent prompt cache on the default profile). Facts from the code: a kept state is one device buffer (110 MiB fixed + ~17 KiB a token); the
  default profile's cache budget is what five FULL 262K windows leave at start (3.5 GiB of 27.7), but a stream only allocates caches for its own prompt + max_tokens.
  So 5 agents x 56K (1.09 GiB a state, ~3 marks a request) cannot fit ANY policy in 3.5 GiB (upstream #2699's order would keep at most 3 of 5 newest states), and
  page-shared KV (TF_FLASHNEXT_KV_PAGES) is not in this tree's Flash Next family at all (only nemotron): a full port, not attempted.
  BUILT INSTEAD: TF_FLASHNEXT_Q38_CACHE_ELASTIC=1: the cache budget = start-up budget + (lanes x full window - live streams' bytes - idle LANE_REUSE sets);
  a stream that opens first takes its room back (Store.q38Resize evicts in the usual order before the stream allocates; if borrowed states block it the request
  re-queues with the existing LaneMemory path). Five full-window streams still fit (then the cache is today's). Replies cannot depend on it (a kept state is
  the same bytes whenever resumed). Files: core/prompt_cache.zig (q38Resize), native/cuda.zig (hook), families/flashnext/cuda_lanes.zig + _load + knobs.
- 23:36 C11 BUILD_OK sha 6a974cff8618d0d7 (select v2 + elastic cache; first compile). Lease C acquired 85 min; runX2.sh started (log ~/q38/runs/C_x2.log; dirs C_x2 = knob-file engine: check mode v2 at 30K/120K/235K, paired timing, decode at 120K; C_x2_ref = fresh engine new flags off; C_x2_on = fresh engine QSA_CERT=1 + CACHE_ELASTIC=1: ident set, needle, short, agent1, prefill, 5 agents x 56K, 16 agents x 20K; texts + lhash compared).
- 23:44 SELECT v2 CHECK MODE (C11, qsa_cert=2; MEASURED): 4,478,408 row selections (27K + 109K needle 3/3 + 214K prompt): list_diff=0, exact_bit_diff=0, bound_viol=0, fallback=0, redone=0; in-between 11.1 a row avg, 39 max; max ratio 0.0021.
- 23:52 SELECT v2 TIMING (C11, one engine, knob file, GPROF=2; MEASURED): 27K 2445 / 2452 off -> 2567 cert (+4.8 %); 109K 2310 / 2302 -> 2445 (+6.0 %);
  214K 2130 -> 2267 (+6.4 %). v1 (C10) gave +3.8 / +6.0 / +9.5 %. => v2 is WORSE at long context: a thread walking its own contiguous block range makes every warp
  load touch 32 separate cache lines (v1's strided loop is coalesced). v2 DROPPED (kept for the record: kitC/q38_qsa_cert.v2.cuh.txt); tree back to the v1 select.
  X2 parts 2-3 still run C11: valid for the reference engine and for the elastic cache; the certified select's identity run is repeated on the final binary.
  runX3.sh queued behind X2: KPROF of a 109K prompt with cert on (which of coarse / select / fused attention holds the remaining growth).
- 00:13 DECODE at long context (C11, needle answers, 34 tokens each after a 109K prompt; MEASURED, small sample): cert off 75.7 / 76.7 tok/s, cert for prompt rows 75.6,
  cert for every row (qsa_cert_min=1) 76.0: no difference; a longer decode test (tt.py decode: the same 200-token task after 0 / 30K / 120K tokens) runs in window X4.
- 00:13 REFERENCE ENGINE (C11, every new flag off, fresh; C_x2_ref; MEASURED): ident 51 saved; needle 109K 3/3 2356 tok/s; short first token 101.0 ms median of 12;
  1 agent 20K x 4 turns: follow-ups 0.78 s; prefill 2.8K / 7.3K / 27.5K = 2498 / 2527 / 2536 tok/s.
  REGRESSION REPRODUCED on the default profile: 5 agents x 56K own system prompts x 3 turns: follow-up first token median 64.8 s (max 102.8 s), wall 323.8 s,
  cache 3.4 GiB: "3 states, 3182 MiB ... evicted 116". (Round 8 measured 85.1 s / 365.6 s on the pre-integration base.)
- 00:13 DECISION: runX3 (KPROF on C11) cancelled: C11's select v2 is dropped, so its table would not describe the shipping kernel; KPROF goes on C12 if time allows.
  C12 (tree now) = select v1 (C10's, restored) + TF_FLASHNEXT_Q38_QSA_HALF=1 (coarse scores stored as IEEE halves: half the bytes, 2 radix passes instead of 4,
  one barrier a block scan; the half rounding |f| 2^-11 is added to the bound per block) + elastic cache. Build waits for X2 to end (no build beside a measured engine).
- 00:18 REFERENCE (default profile, flags off, C_x2_ref; MEASURED): 16 agents x 20K own system prompts x 3 turns on 5 lanes: follow-up first token median 123.5 s (max 185.9 s), wall 445.5 s, sum of first-token waits 5,273 s; every one of the 48 requests missed the cache (7 states, 3.3 GiB, evicted 336). Short prompt after it 110.2 ms.
- 00:30 LEVER 2 RESULT (C11, default one-user profile, fresh engines, same requests in the same order; MEASURED):
  | load | flags off (C_x2_ref) | TF_FLASHNEXT_Q38_CACHE_ELASTIC=1 (+ QSA_CERT=1) (C_x2_on) |
  | 5 agents x 56K own system prompt x 3 turns: follow-up first token median / max; wall | 64.8 s / 102.8 s; 323.8 s | 0.98 s / 12.0 s; 120.0 s |
  | 1 agent 20K x 4 turns: follow-ups | 0.78 s | 0.79 s |
  | short prompt first token (median of 12) | 101.0 ms | 100.0 ms |
  | prefill 2.8K / 7.3K / 27.5K tok/s | 2498 / 2527 / 2536 | 2523 / 2536 / 2540 |
  | needle 109K | 3/3, 2356 tok/s | 3/3, 2434 tok/s (same answer) |
  Engine line: "q38 elastic: the prompt cache takes 3659 MiB at start-up and up to 21607 MiB more of the 5 lanes' room while streams leave it"; after the 5-agent
  run: 22 states, 23.0 GiB of 23.9 GiB (hits 93, misses 6). Cost found: a NEW kept state's device buffer is a fresh allocation (~0.15 ms a MiB): the last chunk of a
  109K prompt takes +271 ms (995 -> 1266 ms), of a 27.5K prompt +90 ms; the static cache pays it only until its 3.4 GiB is cycling.
- 00:32 METHOD FINDING: TF_FLASHNEXT_Q38_GPROF=2 (a sync after every kernel group) inflates the STOCK indexer's time by ~2 % of a 109K prompt but not the certified
  path's: full 3,072-row chunks of the 109K needle: stock 46,075 ms with GPROF vs 45,090 ms without; cert (v2) 43,378-43,447 vs 43,277. So window X1's gains
  (+6.0 % at 109K, +9.5 % at 214K, measured with GPROF on both sides) are OVERSTATED by ~2 points. Honest (PLOG only, fresh engines): cert v2 at 109K +4.2 % on the
  full chunks (45,090 -> 43,277 ms), +0.9 % at 27.5K. Window X4 measures v1 and the 16-bit form WITHOUT GPROF (kitC/cw.py: chunk walls from PLOG).
- 00:35 X2 DONE. IDENTITY of C11 with TF_FLASHNEXT_Q38_QSA_CERT=1 (select v2) + TF_FLASHNEXT_Q38_CACHE_ELASTIC=1 against the flags-off engine (both fresh, same requests,
  same order; MEASURED): ident set 51/51 texts identical; needle 109K same answer (3/3); short 24/24; 1 agent 4/4; prefill 3/3; 5 agents x 56K 15/15; 16 agents x 20K
  48/48; short after 24/24; score hashes after the prompt pass (lhash first) 144/144 identical; lhash end 83 identical / 61 differ = the 63 requests that ran
  concurrently (round 8 showed the release-time hash is not reproducible between two IDENTICAL concurrent runs; with one user it is equal). 18,992,917 certified
  row selections in that engine: fallback 0, redone 0; in-between 1.2 a row (max 7) at these shorter contexts. crash lines 0, elastic waits 0.
- 00:35 16 AGENTS x 20K on the default profile (5 lanes): ELASTIC ALONE DOES NOT FIX IT (MEASURED): follow-up median 118.7 s (off: 123.5 s), wall 443 s (445 s).
  Cause from the engine line: "kept 15; 50 states, 22.8 GiB of 22.8 GiB, evicted 280": under 5 concurrent prefills a request keeps a state at EVERY chunk boundary
  (15 x ~470 MiB = 7 GiB a request), 16 requests = 112 GiB of mostly redundant states, so LRU throws out the waiting conversations' newest states.
  => next run adds TF_FLASHNEXT_Q38_CACHE_SIBLINGS=1 (round 8's eviction tier: a request's earlier marks go before any conversation's newest state; it was zero
  on a 4 GiB cache, is still in the base tree, off). This is upstream #2699's first rule ("redundant states are evicted before each conversation's newest one").
- 00:36 C12 building (v1 select + QSA_HALF 16-bit form + elastic). Then runX4 (paired timing without GPROF, check mode of the 16-bit form, decode test).
- 00:39 C12 BUILD_OK sha d728a479da80b7df. runX4.sh started 00:39 (log ~/q38/runs/C_x4.log, dir C_x4). Lease C to 01:50.
- 00:45 16-BIT FORM CHECK (C12, qsa_cert=2 qsa_half=1; MEASURED): 1,720,550 row selections (27K + 109K needle 3/3): list_diff=0, exact_bit_diff=0, bound_viol=0, fallback=0; in-between 21 a row avg, 69 max; largest |stored - exact| / bound 0.73 (the half rounding is a hard bound of |f| 2^-11; the bound is |f| 2^-11 x 1.01 + eps).
- 00:56 CORRECTION of the 00:32 finding: the skew is NOT GPROF: window X4 ran without GPROF and its knob-file engine shows the same slow stock path (27K 2462 / 2450,
  109K 2309 / 2304 tok/s against 2536 / 2356 in the fresh flags-off engine). Cause: TF_FLASHNEXT_Q38_KNOBFILE re-read the file at EVERY knob use, and the stock path
  reads the QSA_LOOP knob between its two launches of every 256-row group (144 file reads a chunk); the certified path reads its knobs once per call. So every
  knob-file pair of this round overstates cert vs off by ~2 points; cert-vs-cert pairs are fair. Fixed in C13: the knob file's text is cached for 200 ms.
- 00:56 X4 (C12, knob file, no GPROF; MEASURED; the "off" rows carry the knob-file penalty, see above):
  | prompt | off (first / last) | QSA_CERT=1 (fp32 scores, select v1) | QSA_CERT=1 + QSA_HALF=1 |
  | 27.3K | 2462 / 2450 | 2546 | 2546 |
  | 109K  | 2309 / 2304 | 2440 (44.79 s) | 2446 (44.62 s) |
  | 214K  | 2131 (100.5 s) / (running) | 2331 (91.73 s) | 2345 (91.24 s) |
  16-bit form check at 214K: cumulative 4,478,408 selections, list_diff=0, exact_bit_diff=0, bound_viol=0, fallback=0, max in-between 76; v1 again at 27K: 0 diffs.
  => DECISION: recommend QSA_CERT=1 + QSA_HALF=1 (0.4-0.5 % better than fp32 scores at 109K / 214K, same lists); QSA_HALF=0 stays as the plain form.
- 00:57 C13 committed (7a5f12e + knob cache + ELASTIC implies the sibling eviction order unless TF_FLASHNEXT_Q38_CACHE_SIBLINGS=0). Build after X4 ends; then runX5:
  A knob engine (honest now) off / cert / cert+half / off at 27K + 109K + 214K; B fresh engine QSA_CERT=1 QSA_HALF=1 CACHE_ELASTIC=1 on the reference's request list.
- 01:01 X4 DONE. DECODE vs CONTEXT (C12, the same 200-token writing task after N prompt tokens; MEASURED): flags off 43.9 / 46.3 / 44.7 tok/s after 45 / 27K / 110K tokens (second pass 46.6 / 46.4 / ..): decode does NOT slow down with context (the indexer scores one row a step: negligible). Cert for every row (qsa_cert_min=1): 43.5 / 41.8 / 43.6: no gain, so decode rows stay on the stock kernels (CERT_MIN=64 default). C13 shipped 01:01, build + runX5 chained on the box (log ~/q38/runs/C_x5.log).
- 01:05 C13 BUILD_OK sha c11dfbcea69a2849 = tree e7a68cf = handoff patch (mkhandoffC9.sh: TREE_EQUAL; handoff/C/r9-qsa-cert-elastic/C-r9.patch). runX5 running; lease C to 02:20.
- 01:15 SECOND CORRECTION (the 00:56 explanation is NOT confirmed): C13 caches the knob file, yet its knob engine's stock path is still slower than the fresh
  flags-off engine: full 3,072-row chunks of a 109K prompt (PLOG walls, kitC/cw.py): 45,593 / 45,972 ms (C13 knob engine, off first / last) against 45,090 ms
  (C_x2_ref); X4 46,131; X2 part 1 46,074. The certified path is the same in every engine (43,277-43,447 ms). So the STOCK indexer's time differs by up to 2 %
  between engine instances and 0.8 % inside one (cause not found: neither GPROF nor the knob file; suspected: where an instance's score / key buffers land in
  memory, the stock score kernel being memory-bound). Consequence: I report BOTH an in-engine pair (reference first and last in the same engine) and the
  fresh-engine pair, and call the gain a range.
- 01:15 X5 part A so far (C13, ONE engine, knob file cached, PLOG only; MEASURED; first-token time and full-chunk walls):
  | prompt | off first | QSA_CERT=1 | QSA_CERT=1 + QSA_HALF=1 | off last |
  | 27.3K first token | 11.28 s (cold start chunk +240 ms) | 10.67 s | 10.63 s | 11.07 s |
  | 27.3K full chunks (8 x 3072) | 10,016 ms | 9,440 | 9,433 | 9,847 ms |
  | 109K first token | 47.12 s = 2322 tok/s | 44.56 s = 2452 | 44.36 s = 2460 | 47.04 s = 2316 |
  | 109K full chunks (35 x 3072) | 45,593 ms | 43,429 | 43,295 | 45,972 ms |
  => in-engine: 109K +5.7 % (v1) / +6.1 % (half) on first-token time; 27K +3.7 % / +4.1 % against the last reference.
- 01:16 NOTE: one shell call (a long heredoc for this log + a status check) was refused by the tool's command-size limit; nothing in it ran. Log entries now go
  through the file tool.
- 01:19 X5 PART A DONE (C13 sha c11dfbcea69a2849 = the handoff tree; ONE engine, knob file cached, PLOG only, reference first and last; MEASURED):
  | prompt | off first | QSA_CERT=1 | QSA_CERT=1 + QSA_HALF=1 | off last | gain of CERT+HALF vs mean off |
  | 27.3K first token | 11.28 s | 10.67 s (2565 tok/s) | 10.63 s (2566) | 11.07 s (2460) | +4.1 % vs the last off (the first off holds the cold-start chunk) |
  | 109K first token  | 47.12 s (2322) | 44.56 s (2452) | 44.36 s (2460) | 47.04 s (2316) | +6.1 % |
  | 214K first token  | 100.36 s (2134) | 91.51 s (2336) | 91.08 s (2349) | 100.17 s (2132) | +10.1 % |
  | 214K full chunks (69 x 3072) | 98,399 ms | 89,998 | 89,560 | 98,667 ms | +10.0 % |
  | a 3,072-row chunk at ~200K context | 1,661 ms | 1,405 | 1,405 | 1,649 ms | -15 % |
  8,949,742 certified row selections in that engine: fallback 0, redone 0, in-between max 76. crash lines 0.
  Part B running: fresh engine C13 with QSA_CERT=1 QSA_HALF=1 CACHE_ELASTIC=1 (sibling order implied), the reference's request list (dir C_x5_on).
- 01:35 X5 PART B DONE = FINAL IDENTITY + LEVER 2 on the HANDOFF BINARY (C13 sha c11dfbcea69a2849, fresh engine, env flags QSA_CERT=1 QSA_HALF=1 CACHE_ELASTIC=1
  [sibling order implied], against the flags-off engine C_x2_ref; same requests, same order; MEASURED):
  IDENTITY: ident set 51/51 texts identical; needle 109K same answer 3/3; short 24/24; 1 agent 4/4; prefill 3/3; 5 agents x 56K 15/15; 16 agents x 20K 48/48;
  short after 24/24; score hashes after the prompt pass 144/144 identical; lhash end 82 identical / 62 differ (the 63 concurrent requests: schedule noise, see
  round 8). 11,183,608 certified selections: fallback 0, redone 0, in-between max 54. crash lines 0, elastic waits 0.
  | load (default one-user profile, 5 lanes) | flags off | CERT + HALF + ELASTIC |
  | 5 agents x 56K x 3 turns: follow-up first token median / max; wall | 64.8 s / 102.8 s; 323.8 s | 0.99 s / 12.2 s; 120.7 s |
  | 16 agents x 20K x 3 turns: follow-up median / max; wall; sum of first-token waits | 123.5 s / 185.9 s; 445.5 s; 5,273 s | 15.3 s / 95.7 s; 170.3 s; 2,083 s |
  |   turn 1 / turn 2 median (16 agents queue on 5 lanes) | 119.8 / 124.6 s | 48.1 / 13.4 s; 33 of 48 requests resumed (0 before) |
  | 1 agent 20K x 4 turns: follow-up | 0.78 s | 0.84 s (C11 elastic run: 0.79 s) |
  | short prompt first token, median of 12 (before / after the agent runs) | 101.0 / 110.2 ms | 100.2 / 106.6 ms |
  | prefill 2.8K / 7.3K / 27.5K | 2498 / 2527 / 2536 tok/s | 2549 / 2519 / 2527 |
  | needle 109K | 46.3 s = 2356 tok/s | 44.9 s = 2431 (+3.1 %) |
  NOTES: (1) with ELASTIC alone (C11) 16 agents were NOT fixed (118.7 s); the sibling eviction order is what makes it work, and it only works with the larger budget.
  (2) every newly kept state is a fresh device allocation (~0.15 ms a MiB): +0.09 s on a 27.5K prompt, +0.27 s on 109K, ~+0.05 s on a 1-agent follow-up turn.
  (3) fresh-engine pair shows NO cert gain at 27.5K (10.84 -> 10.88 s) and +3.1 % at 109K, against +4.1 % / +6.1 % inside one engine: see 01:15.
  runX6 (fresh-engine pair at 109K + 214K, cert + half only, no elastic) queued 01:30.
- 01:46 X6 DONE = FRESH-ENGINE PAIR (C13, env flags, PLOG + LHASH only, the SAME prompts; MEASURED). This is the number to quote:
  | prompt | flags off | QSA_CERT=1 + QSA_HALF=1 | gain |
  | 109,617 tokens: first token; full 3,072-row chunks | 46.55 s = 2355 tok/s; 45,026 ms | 44.75 s = 2450 tok/s; 43,196 ms | +4.0 %; +4.2 % |
  | 213,604 tokens | 98.26 s = 2174 tok/s; 96,590 ms | 90.40 s = 2363 tok/s; 88,886 ms | +8.7 %; +8.7 % |
  | chunk at the start / middle / end of the 214K prompt | 1,140 / 1,403 / 1,625 ms | 1,144 / 1,290 / 1,399 ms | growth 2.3 -> 1.2 ms a chunk per 1K tokens |
  texts 2/2 identical, score hashes first 3/3 and end 3/3 identical. The fresh flags-off engine again has the FAST stock path (45,026 ms; knob engines 45.6-46.1 s).
  Each fresh engine's first long chunk is +230 ms (1,383 / 1,372 ms against ~1,150): the cold-start chunk, both sides.
## ROUND 9 END STATE (01:50 UTC 2026-11-14)
- Lease C RELEASED 01:45:46. spark-4e5c: no q38-run-C container, no run script, no build, 0 GPU processes. Raw results: Mac q38/results/rC9.tgz (361 KB).
- HANDOFF: Mac q38/handoff/C/r9-qsa-cert-elastic/: C-r9.patch (git diff 0b5cf9b..e7a68cf, sha256 2a2a3370e1cc0580, 1,286 lines, 14 files, TREE_EQUAL) + NOTE.md.
  Built and measured binary of exactly that tree: C13 sha c11dfbcea69a2849 (box ~/q38/tfbin/C13). Builder A integrates; nothing pushed by C.
- SHIPPED-READY (exact; flags default off in the patch; proposed recipe defaults):
  1 TF_FLASHNEXT_Q38_QSA_CERT=1 + TF_FLASHNEXT_Q38_QSA_HALF=1: prefill 109.6K 2355 -> 2450 tok/s (+4.0 %), 213.6K 2174 -> 2363 (+8.7 %), 27.5K 0 (fresh engines).
    Check mode: 4,478,408 (16-bit) + 2,048,591 (fp32) row selections against the stock kernels on the device: 0 lists differ, 0 exact-sum bit differences,
    0 bound violations, 0 fallbacks. ident 51/51, needle same answer, lhash first 144/144.
  2 TF_FLASHNEXT_Q38_CACHE_ELASTIC=1 (turns the sibling eviction order on): default profile, 5 agents x 56K follow-up 64.8 -> 0.99 s (wall 324 -> 121 s);
    16 agents x 20K 123.5 -> 15.3 s (wall 446 -> 170 s); 1 agent 0.78 -> 0.84 s; short prompt and cold prefill unchanged; texts identical everywhere.
- DROPPED: select v2 (per-thread block ranges: slower at 214K); QSA_LOOP as a lever of its own (+2-3 % alone, superseded; the flag stays in the patch, off);
  porting #2699 as a whole / page-shared KV (cannot fit a 3.5 GiB cache; KV pages are not in this tree's Flash Next family).
- NOT DONE: lever 3 (partial chunks, cold first chunk): no build. Reading of the data: a partial chunk's higher per-row cost is a fixed cost per chunk
  (ESTIMATE ~130 ms from two points) and round 7 measured bigger chunks at 0..2 %; the cold first chunk is +230-240 ms once per engine start (a warm pass at
  start-up would remove it). KPROF split of what is left of the indexer growth under cert: not run.
- WHAT REMAINS, by size:
  1 the indexer still grows 1.2 ms a chunk per 1K tokens under cert (coarse GEMM vs select vs fused attention unknown: KPROF on C13 first).
  2 elastic cache: a newly kept state is a fresh device allocation (0.15 ms a MiB; +0.27 s at 109K, ~+0.05 s on a one-agent follow-up): keep freed state
    buffers as spare storage instead of freeing them.
  3 why a knob-file engine runs the stock indexer 1.5-2 % slower than a fresh env-flag engine (not GPROF, not the file reads).
  4 shared rounds (ATTN_LANES) and decode rows still use the stock score + select kernels (decode does not grow with context, so nothing to gain there).
  5 A re-runs on 923b (check mode once, fresh pair at 117K / 227K, 5-agent trace) before any number is quoted publicly.
- RESUME: cd ~/.hermes/cache/scratch/q38; tail -60 PROGRESS-C.md; ./shipC.sh NAME; ./pushkit.sh; ./mkhandoffC9.sh; box: bash ~/q38/kit-C/runX5.sh DL BIN "FLAGS"
  (knob pair + identity against C_x2_ref), runX6.sh DL BIN (fresh pair at 109K / 214K), python3 -I ~/q38/kit-C/cw.py ENGINE.log (chunk walls).

# Round 10 (builder C, started 2026-11-14 04:32 UTC)
Box spark-4e5c. Base = integ9 7061396 (patches 0101-0118). Worktree Mac q38/wt/C10 branch r10-C. Ship script q38/shipC10.sh. Binaries C20.. (C20 = base, unchanged).
Levers planned: (1) KPROF at 117K/227K under cert -> fix the growing part; (2) spare buffers for kept states; (3) partial chunks + cold first chunk; (4) big groups vs byte / tensor-core rate, a third engine read; (5) decode vs context to 200K.
- 04:33 window already open (leases A, B). 4e5c idle (116 GiB free, 0 GPU processes). Building C20 = base 7061396 unchanged for the first measuring window.
- 04:37 C20 BUILD_OK sha 367fc37e8b93522d (= base 7061396 unchanged). kitC/engC10.sh = engC.sh + recipe R9=1 defaults (QSA_CERT+HALF, ELASTIC, CERT sm8/TOPK 20, DEEP 0.9,6 / DRAFTS 12, STOP2 0.5; no HEAD2 file on 4e5c: draft side differs from 923b, prefill does not). Lease C acquired 55 min. runY1.sh started (log box ~/q38/runs/C_y1.log): A KPROF 120K+235K (dir C_y1_kp_C20), B reference fresh engine (C_y_ref_C20: prefill 8K/30K/120K/235K, decode after 2K/60K/120K/200K, ident, needle, short), C GPROF 30K+235K.
- 04:45 WHILE Y1 RUNS (code written on the Mac, not built yet; tree wt/C10):
  * LEVER 2 generalised = device arena, TF_FLASHNEXT_Q38_ARENA=1 (new file cuda_q38_arena.zig; hook in cuda_lanes.zig q38Elastic). Old logs (rC9 C_x6_on PLOG) show the
    allocation cost is not only the kept state: lane open of a 109,617-token prompt = 202 ms (sequence caches 174 ms + draft caches 24 ms), 214K 146 ms, 2.8K 32 ms
    (seq 12.7 + draft 18.6), and the last chunk with its kept state +120..290 ms over its rows' cost. So one start-up allocation of the lanes' + cache's planned room
    (base + 5 x 4.85 GiB), served through memory.zig's existing sub-allocator hook (best fit, merge on free, zero fill on hand-out), instead of only spare state
    buffers. ESTIMATE before measuring: 2.8K -3 %, 7.3K -2 %, 27K -1.3 %, 109K -1 %, agent follow-up turn -10 %. Idle engine then holds the planned room (MemAvailable ~8 GiB = the reserve).
  * LEVER 3b = TF_FLASHNEXT_Q38_WARM=N: one N-row made-up prompt pass on a throw-away sequence at start-up (cold first chunk +230..290 ms once per engine start).
  * sz-memguard on 4e5c only kills containers labelled sizing=1 (not ours); read-only check.
- 04:43 Y1 PART A = KPROF under the certified select (C20 = base, recipe defaults, 109K prompt then 214K prompt; 8 tables of 20,000 launches = ~13 chunks each; MEASURED, ms a table;
  tables 0-2 = the 109K prompt, 3-7 = the 214K prompt from its start to its end). Prefill with KPROF on: 109,364 tok 44.97 s = 2432 tok/s; 213,886 tok 90.95 s = 2352 tok/s.
  | kernel | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 (175-214K) | grows? |
  | expert tiles gate/up (tf_nvfp4_tile_gu_rt4_s4) | 2903 | 2852 | 2628 | 2755 | 2861 | 2873 | 2873 | 2855 | flat |
  | expert tiles down (tf_nvfp4_tile_f32_rt2_s3) | 2336 | 2143 | 1971 | 2090 | 2141 | 2152 | 2155 | 2148 | flat |
  | MXFP8 prompt tile bf16 + f32 | 1832+739 | 1839+747 | .. | .. | .. | .. | .. | 1891+764 | flat |
  | fn_gdn_fused | 1490 | 1504 | 1385 | 1505 | 1514 | 1516 | 1513 | 1518 | flat |
  | hyper-connection: writeback_moe_sh + down4 + up3_k5 + slices | 1379+856+814+488 | .. | .. | .. | .. | .. | .. | 1387+886+829+493 | flat |
  | fn_attn_fused_kv8 (+ fn_attn_fused) | 1125 (+151) | 1228 | 1136 | 1198 | 1269 | 1337 | 1390 | 1430 (+214) | +0.13 ms a chunk per 1K |
  | fn_q38_qsa_select_cert16 | 281 | 497 | 534 | 324 | 542 | 775 | 1048 | 1389 | +0.48 ms a chunk per 1K (largest) |
  | fn_q38_qsa_coarse16 | 122 | 285 | 317 | 154 | 312 | 467 | 624 | 777 | +0.29 ms a chunk per 1K |
  | table total | 15889 | 16179 | 15097 | 15739 | 16431 | 16949 | 17446 | 17953 | |
  READING: what still grows under cert = the select (107 ms a chunk at ~195K, 8 %), the coarse scores (60 ms, 4.6 %), fused attention's gather (+23 ms).
  Select at 195K: 2040 launches x 256 rows x 4 passes over a 100 KB row of halves = 209 GB in 1.39 s = 150 GB/s: it is at the memory rate already, so the fix is
  FEWER PASSES, not a faster pass. Coarse: 26.7 TFLOP in 0.777 s = 34 TFLOP/s (bf16 mma), 65 GB moved = 84 GB/s: compute-bound.
  DESIGN (next build): sampled cut for the 16-bit select, TF_FLASHNEXT_Q38_QSA_SAMPLE=1: a 1/8 sample of the row (whole cache lines) gives a threshold tlo at sample rank
  k + 6 sqrt(k) (k = 512 x sample share); ONE pass over the row compacts every block with key >= tlo in block order (~800-1,000 of 50K) into shared memory; if that list
  holds >= 512 blocks, the 512th largest stored half is found inside it (exactly the old v), and classification / exact in-between sums / ranks / placement run on the
  list only. Excluded blocks are certainly not listed when tlo's interval lies below v's (checked); any miss (list too short / too long / tlo too close / non-finite)
  falls back to the old four-pass body in the same kernel. Passes over the row: 4 -> 1.125. ESTIMATE: select 107 -> ~40 ms a chunk at 195K = -2.7 % of a 214K prompt, -1.4 % at 109K.
- 04:57 Y1 PART B = REFERENCE on the base (C20 sha 367fc37e8b93522d, fresh engine, recipe defaults incl. QSA_CERT+HALF+ELASTIC, PLOG+LHASH only; box dir ~/q38/runs/C_y_ref_C20; MEASURED):
  | item | value |
  | prefill 2,766 / 7,276 / 27,502 / 109,142 / 214,471 tokens (first token) | 1.27 s / 3.14 s = 2318 tok/s / 10.67 s = 2578 / 44.28 s = 2465 / 91.03 s = 2356 |
  | full 3,072-row chunks: 27.5K / 109K / 214K | 382.4 / 400.7 / 420.5 us a row (2615 / 2496 / 2378 tok/s); chunk at start / middle / end of 214K: 1165 / 1295 / 1396 ms |
  | the first 3,072-row chunk of the engine (in the 7.3K prompt, AFTER a 2.8K warm prompt) | 1386 ms against 1160: +226 ms: the cold cost belongs to the first chunk of that SIZE |
  | last (partial) chunk with its kept state | 1130 rows 555 ms; 2924 rows 1222; 1620 rows @109K 986; 2501 rows @214K 1542; 640 rows @182K 1006 ms |
  | decode, the same 300-token task after N prompt tokens (drafts on; texts differ) | 1.9K: 44.4 tok/s; 54.6K: 48.7; 109K: 44.1; 182K: 41.9 |
  | needle 109,126 | 3/3, 44.7 s = 2440 tok/s |
  | short new prompt first token, median of 12 | 100.4 ms |
  | ident set | 51 saved (median 65.3 tok/s; no HEAD2 file on this box) |
  certified select counters: 12,105,751 rows, fallback 0, redone 0, max in-between 58. crash lines 0.
  Part C (GPROF) skipped (KPROF table has the groups); its engine stopped.
- 04:58 TENSOR-CORE AND BUS RATES OF THIS BOX (kitC/mb/mb_mma.cu, standalone, GPU idle otherwise; MEASURED): GB10 = 48 SMs, L2 24 MiB.
  | bf16 m16n8k16 mma, operands from | TFLOP/s |
  | registers only (1-8 accumulators) | 121-123 (the ceiling) |
  | B fragment from shared memory at every pair of mma (the coarse indexer kernel's inner loop, nothing else) | 61.5 |
  | B from global memory, cache-resident (32 KiB a block) | 83.6 |
  | B streamed from global memory (0.5 / 8 GiB in all) | 3.8 / 3.7 = 240 / 233 GB/s of operands (the bus) |
  => "30 TFLOP/s" is NOT the tensor-core ceiling (122); it is a quarter of it. A kernel that needs a fresh operand from shared memory per mma pair tops out near 60;
  the engine's coarse kernel does 34, the prompt as a whole ~30. One 3,072-row chunk reads every expert of every layer once (~53.6 GiB = ~240 ms at the bus rate):
  the expert tiles (218 + 164 = 382 ms a chunk) are at ~63 % of their byte floor.
- 04:59 C21 building (= 41f2913: arena + sampled cut + warm pass; all three flags default off).
- 05:03 C21 BUILD_OK sha adc288c0ebb1d29a (first compile). Lease C renewed 60 min. runY2.sh C21 "chk on off" started (log box ~/q38/runs/C_y2.log): chk = ARENA=1 QSA_SAMPLE=1 WARM=3072 + QSA_CERT=2 device check (30K, needle 120K, 235K); on = the reference list with the three flags (dir C_y_on_C21); off = the same list, new flags off (C_y_off_C21); texts + lhash against C_y_ref_C20.
- 05:06 HOW THIRD ENGINE PREFILLS FASTER (source read, research/src/a third engine at 8a88720; NOT measured by me): docs/model-preparation.md: "Checkpoint NVFP4 routed experts
  use W4A4"; ignition/recipes/qwen38/tuning.py numerical_policy "qsa16-gdn-sequential-k1-dense-mxfp8-auto-vision-nvfp4-a16-moe-a4-route-fp32"; moe.py experts_a4_prefill
  ("A16 experts prepared with A4 capability run large prefill calls with quantized activations"); cli --native-dense-activations ("A8 for FP8 weights and A4 for FP4
  weights"), dense MXFP8 projections pick "BF16 or quantized activations" by autotuning; prefill chunk 4096; one scratch arena for all operators; per-GPU autotuned tiles.
  => its prompt pass quantises the ACTIVATIONS (4-bit into the routed experts, optionally 8-bit into dense projections) and runs native FP4 / FP8 tensor-core products:
  fewer operand bytes and a higher mma rate than bf16 activations. That is a different arithmetic from this engine (W4A16: bf16 activations into 4-bit weights), so every
  hidden state differs: NOT portable under the rule "same output as this engine with the trick off". What IS portable and exact: one scratch arena (our ARENA lever),
  chunk geometry (measured here in round 6: 4096 / 8192 = -3..+2 %, zero). DECISION FOR PICO (not mine): the official checkpoint's own metadata carries the experts'
  activation scales (W4A4 is how NVIDIA's engine runs it); whether "the official arithmetic" is W4A4 or this engine's W4A16 decides if an A4 prompt pass could ever count.
- 05:11 Y2 chk (C21, fresh engine, ARENA=1 WARM=3072 QSA_SAMPLE=1 + QSA_CERT=2 device check; MEASURED): engine READY in 180 s; "q38 arena: 29152 MiB of device memory
  taken once", start-up cost 4,040 ms; "q38 warm: a 3072-row prompt pass at start-up took 1324 ms"; MemAvailable 7 GiB with the arena held (36-37 GiB without).
  4,492,081 row selections (27K + 109K needle 3/3 + 214K) checked against the stock kernels on the device: list_diff 0, exact_bit_diff 0, bound_viol 0, fallback 0.
  SAMPLED CUT: exact on the 1,715,030 rows it served, but it GAVE UP on 1,636,388 (49 %): on this text a 1/8 sample in 32-block runs does not predict how many blocks
  lie above its threshold (block scores are strongly clustered by position), so half the rows pay the sample pass AND the four-pass body. => the sampled form is DROPPED;
  replaced (next build) by a deterministic form: pass 1 = an 11-bit histogram of the row's keys (2,048 bins, 1.6 % wide), which gives the exact count above any bin edge
  before the second pass; pass 2 lists every block from one bin below the cut's bin (count known, ~600-900); cut, classification, exact sums, placement on the list.
  2 passes over the row instead of 4, no dependence on the text. ESTIMATE: select 107 -> ~58 ms a chunk at 195K = -1.8 % of a 214K prompt, -0.9 % at 109K.
  Y2 restarted as "on off" with ARENA=1 WARM=3072 only (the sampled cut would confound the pair).
- 05:24 Y2 "on" (C21 sha adc288c0ebb1d29a, fresh engine, TF_FLASHNEXT_Q38_ARENA=1 + TF_FLASHNEXT_Q38_WARM=3072; reference C_y_ref_C20, same request list; MEASURED):
  | item | reference (C20) | ARENA + WARM (C21) |
  | prefill first token 2,766 / 7,276 / 27,502 / 109,142 / 214,471 | 1.27 / 3.14 / 10.67 / 44.28 / 91.03 s | 1.24 / 2.91 / 10.59 / 44.05 / 89.97 s |
  | = tok/s at 7.3K / 27.5K / 109K / 214K | 2318 / 2578 / 2465 / 2356 | 2501 / 2597 / 2478 / 2384 |
  | lane open (PLOG) at 2.8K / 27.5K / 109K / 214K | 29.5 / 25.7 / 114.3 / 208.8 ms | 8.7 / 10.0 / 28.3 / 50.3 ms |
  | last chunk with its kept state: 1130 rows @7K / 2924 @27K / 1620 @109K / 640 @182K / 2501 @214K | 555 / 1222 / 986 / 1006 / 1542 ms | 524 / 1164 / 814 / 515 / 1232 ms |
  | first 3,072-row chunk of the engine | 1386 ms | 1198 ms (later ones 1156) |
  | short new prompt first token (median of 12) | 100.4 ms | 96.2 ms |
  | needle 109K | 44.7 s | 44.0 s, same answer 3/3 |
  IDENTITY: ident 51/51 identical; warm 1/1, prefill 4/4, decode 4/4, needle 1/1, short 24/24 texts identical; lhash first 74/74, end 74/74. crash lines 0.
  Arena: 29,310 MiB taken once (4.6 s at start), served 2,762 allocations, fallback 0, peak 26.6 GiB in use; MemAvailable 7-8 GiB while it is held.
  NOTE: this pair is confounded by the knob bug below (both sides ran the sampled cut); C22 repeats it clean.
- 05:29 BUG FOUND in round 9's patch 0116 (mine to fix, the integrator's to merge): cuda_knobs.zig q38EnvInt's cache struct is ONE type for all its knobs, so
  QSA_CERT, QSA_CERT_MIN, QSA_SLACK, QSA_HALF, QSA_STATS, QSA_LOOP (and my new knob) all return the FIRST one's value. MEASURED proof: the Y2 "off" engine had no
  QSA_SAMPLE in its environment and ran the sampled path on 986,938 rows (miss 0.6 %: there the multiplier read 1 -> default 6; in the chk engine it read QSA_CERT=2
  -> multiplier 2 -> the 49 % misses). So the shipped recipe (QSA_CERT=1) runs slack 1 millionth instead of 100, CERT_MIN 1 instead of 64 (decode rows take the
  certified path), QSA_STATS on; and round 9's "knob-file engines run the stock indexer 1.5-2 % slower" = they ran QSA_LOOP=32 (inference). No wrong output seen
  (check modes: 0 differences on > 12 M rows at slack 1-2), but the designed 100x margin was not there. Written up for A: handoff/C/URGENT-r9-knob-cache.md.
  FIX in tree (7f8a2fb+): the struct names `env`; a "q38 qsa knobs:" line prints what the engine reads.
- 05:30 The sampled cut with its intended margin misses only 0.6 % of rows, yet it bought little: 109K 44.28 -> 44.17 s, full chunks unchanged, last chunk at 214K
  1395 -> 1369 ms. Reading: the select's cost is its two passes with a block-wide barrier every 1,024 blocks (classification, placement), not the radix passes.
  => C22's listed cut has NO barrier pass: histogram pass (atomics), list pass (a warp reserves list slots with one atomic add), then cut / classification /
  exact sums on the ~900 listed blocks and placement through a bitmap prefix count. Committed (wt/C10), builds after the Y2 "off" run ends.
- 05:40 Y2 off (C21, new flags off but the sampled cut ran through the knob bug; MEASURED): 7.3K 3.09 s, 27.5K 10.68, 109K 44.17, 214K 90.05 s; full chunks 214K 88,167 ms (base 89,130): the sampled cut = -1.1 % at 214K, -0.3 % at 109K. ident 51/51, texts identical, lhash 74/74 + 74/74 vs C_y_ref_C20. C22 (42bd434: knob fix + listed cut without barriers + arena + warm + tile knob words) building; lease C renewed 75 min.
- 05:43 C22 BUILD_OK sha b9fe3dda8f321e1b (first compile). runY2.sh C22 "chk tile off on nd" started (log box ~/q38/runs/C_y4.log): chk = ARENA+LIST+WARM with QSA_CERT=2; tile = expert tile sweep (13 configs, GPROF=2, knob words); off / on = the reference list (dirs C_y_off_C22, C_y_on_C22) vs C_y_ref_C20; nd = no-drafts decode after 2K/60K/120K/200K.
- 05:49 C22 chk (fresh engine, ARENA=1 QSA_LIST=1 WARM=3072 + QSA_CERT=2, knob fix in: slack 100, CERT_MIN 64; MEASURED): 4,490,634 row selections checked on the device against the stock kernels (27K + 109K needle 3/3 + 214K): list_diff 0, exact_bit_diff 0, bound_viol 0, fallback 0; in-between 29.5 a row avg, 89 max (slack 100 now); largest deviation 0.64 of the bound. LISTED CUT: served 3,919,434 rows, gave up on 0. crash lines 0.
- 05:54 EXPERT TILE SWEEP (C22, ONE engine, GPROF=2, knob words tile_pairs / tile_gu / tile_down = TF_FLASHNEXT_PROMPT_EXPERT_TILE_CFG, never swept before; 12.8K-token
  prompts; MoE group ms of a full 3,072-row chunk; MEASURED):
  | pairs,gu,down | MoE ms | whole chunk GPU ms |
  | 64,1,2 (default; first / last) | 405.5 / 412.5 | 1120 / 1138 |
  | down instance 0 / 1 (rt4_s3 / rt4_s4) | 422.6 / 416.5 | 1140 / 1137 |
  | gate-up instance 0 / 2 / 3 / 4 | 416.9 / 489.2 / 451.8 / 459.0 | 1141 / 1208 / 1180 / 1182 |
  | pairs 32 / 48 / 96 / 128 / 256 | 428.7 / 431.6 / 472.8 / 484.8 / 551.2 | 1153 / 1157 / 1197 / 1213 / 1279 |
  => the default is the best of every instance and tile size: ZERO, nothing to change (the knob words stay as a measuring aid).
  PARTIAL CHUNKS, MoE (same run): 3,072 rows 409 ms; 1,760-1,792 rows 289-295; 516-544 rows 180-208 => MoE = ~127 ms a chunk + 0.092 ms a row: the fixed part is
  reading every expert once whatever the row count; that is the partial chunk's higher cost per row (0.445 against 0.367), not a tile fault. Bigger chunks would
  spread it, and round 6 measured 4,096 / 8,192 at -3..+2 %: no exact lever here.
  Other groups of a full chunk at 3-12K context (GPROF): GDN 265, hc-attn-side 168, hc-mlp-side 102, QSA 163 ms.
- 06:08 C22 "off" = base + the knob fix only (fresh engine, no new flag; slack 100, CERT_MIN 64 as designed; dir C_y_off_C22; MEASURED) against C20 (base, slack 1 through the bug):
  | item | C20 base (bug: slack 1) | C22 off (fix: slack 100) |
  | prefill 7,276 / 27,502 / 109,142 / 214,471 first token | 3.14 / 10.67 / 44.28 / 91.03 s | 3.10 / 10.67 / 44.62 / 91.78 s |
  | full chunks 109K / 214K | 43,085 / 89,130 ms | 43,289 / 89,678 ms (+0.5 / +0.6 %) |
  | in-between blocks a row (exact sums) | 12.6 avg, 58 max | 27.1 avg, 84 max |
  | decode after 1.9K / 54.6K / 109K / 182K | 44.4 / 48.7 / 44.1 / 41.9 tok/s | 44.4 / 48.5 / 44.0 / 42.0 |
  | short first token | 100.4 ms | 99.7 ms |
  ident 51/51, warm 1/1, prefill 4/4, decode 4/4, needle 1/1 (3/3), short 24/24 identical; lhash first 74/74, end 74/74.
  => the fix changes no output; the designed margin costs ~0.5 % of a long prompt (more blocks take the exact sum). QSA_SLACK is tunable: worst-case fp32 summation
  error of the two sums is ~1.6e-5 x sum|q||k| (128 products, pessimistic tensor-core rounding), so slack 32 = 2x over that; 100 = 6x. C23 measures 32.
- 06:22 C22 PAIR, ONE BINARY (sha b9fe3dda8f321e1b), fresh engines, env flags, PLOG + LHASH only, the same request list; off = knob fix only, on = TF_FLASHNEXT_Q38_ARENA=1
  + TF_FLASHNEXT_Q38_QSA_LIST=1 + TF_FLASHNEXT_Q38_WARM=3072 (dirs C_y_off_C22 / C_y_on_C22; MEASURED):
  | item | off | on | change |
  | prefill 7,276 tokens, first token | 3.10 s = 2347 tok/s | 2.90 s = 2505 | +6.7 % (warm pass + arena) |
  | 27,502 | 10.67 s = 2576 | 10.58 s = 2600 | +0.9 % |
  | 109,142 | 44.62 s = 2446 | 43.80 s = 2492 | +1.9 % |
  | 214,471 | 91.78 s = 2337 | 88.92 s = 2412 | +3.2 % |
  | full 3,072-row chunks 109K / 214K | 43,289 / 89,678 ms | 42,896 / 87,544 ms | -0.9 / -2.4 % (listed cut) |
  | a chunk at the end of the 214K prompt | 1,406 ms | 1,329 ms | -77 ms |
  | last chunk + kept state: 1620 rows @109K / 640 @182K / 2501 @214K | 1,059 / 914 / 1,722 ms | 808 / 498 / 1,197 ms | arena |
  | first 3,072-row chunk of the engine | 1,360 ms | 1,197 ms | warm pass |
  | short new prompt, first token (median of 12) | 99.7 ms | 97.0 ms | -2.7 % |
  | needle 109K | 44.8 s | 43.8 s, 3/3 same answer | |
  | decode after 1.9K / 54.6K / 109K / 182K (drafts on) | 44.4 / 48.5 / 44.0 / 42.0 tok/s | 44.4 / 48.4 / 43.9 / 42.1 | same |
  IDENTITY of "on" against the BASE binary's run (C_y_ref_C20): ident 51/51; warm 1/1, prefill 4/4, decode 4/4, needle 1/1, short 24/24 texts identical;
  lhash first 74/74, end 74/74. Listed cut served 9,960,461 rows, gave up on 0; arena served 2,762 allocations, fallback 0, peak 26.6 GiB. crash lines 0.
  Against the base as shipped (C20, slack 1 through the bug): 7.3K 3.14 -> 2.90 s, 27.5K 10.67 -> 10.58, 109K 44.28 -> 43.80 (+1.1 %), 214K 91.03 -> 88.92 (+2.4 %).
- 06:29 DECODE AGAINST CONTEXT, one token a step (C22, --no-drafts engine, recipe defaults, the same 150-token task after N prompt tokens; MEASURED):
  | context tokens | 1,908 | 54,617 | 109,041 | 181,890 |
  | tok/s | 24.9 | 24.5 | 24.3 | 24.0 |
  | ms a step | 40.2 | 40.8 | 41.2 | 41.7 |
  => a decode step grows 1.5 ms (+3.7 %) from 2K to 182K context (the indexer's one row over all blocks + attention's gather). With drafts on: 44.4 / 48.5 / 44.0 / 42.0
  tok/s (texts differ per context, so those are not a pair). Round 9 measured the certified path on decode rows as no faster, and with the knob bug it WAS on for decode
  rows in the base: C22 off (decode rows back on the stock kernels) reads the same. No lever built: the growth is small and its fix is not known.
  Side reading: the no-drafts engine prefills 181,890 tokens in 73.64 s against 77.02 s with drafts: the MTP draft head's own prompt pass is 4.4 % of a long prompt's
  first-token time (2.1 % at 55K). Not touched this round (TF_FLASHNEXT_KEEP_LATE / Q38_LATE_SKIP already defer part of it).
- 06:30 C23 building = C22 + TF_FLASHNEXT_Q38_QSA_ROWS (coarse scores cut by rows: a block takes 64 rows x 256 key blocks) + arena hands a kept state's range out
  without the zero fill. Lease C renewed 75 min. Then runY2 C23 "chk on pf32 off" (pf32 = on + QSA_SLACK=32, prefill only).
- 06:33 C23 BUILD_OK sha 8720d0f32d99a9a0 (first compile; tree d19 commits to 2d0e53c). runY2 C23 "chk on pf32 off agon agoff" started (log box ~/q38/runs/C_y5.log; ON = ARENA + QSA_LIST + QSA_ROWS + WARM).
- 06:39 C23 chk (ARENA + QSA_LIST + QSA_ROWS + WARM, QSA_CERT=2; MEASURED): 4,490,634 rows checked: list_diff 0, exact_bit_diff 0, bound_viol 0, fallback 0; listed 3,919,434,
  list_miss 0; the in-between counts equal C22's to the block (132,584,847): the coarse scores by row group are the same values.
- 06:55 C23 "on" (sha 8720d0f32d99a9a0; fresh engine; TF_FLASHNEXT_Q38_ARENA=1 QSA_LIST=1 QSA_ROWS=1 WARM=3072; engine line "q38 qsa knobs: cert=1 cert_min=64
  slack=100.0e-6 half=1 list=4096 rows=1 loop=0 stats=1" = the knob fix read back; dir C_y_on_C23; MEASURED):
  prefill 7,276 / 27,502 / 109,142 / 214,471: 2.92 / 10.56 / 43.64 / 88.23 s = 2494 / 2604 / 2501 / 2431 tok/s; full chunks 109K 42,745 ms, 214K 86,856 ms
  (C22 on: 42,896 / 87,544: QSA_ROWS = -0.4 / -0.8 %); chunk at the end of 214K 1,318 ms; last chunks 798 (1620 rows @109K) / 485 (640 @182K) / 1,167 (2501 @214K);
  short 96.7 ms; needle 43.5 s 3/3; decode 44.2 / 48.2 / 43.8 / 42.0 tok/s.
  IDENTITY vs the base binary's run (C_y_ref_C20): ident 51/51, warm 1/1, prefill 4/4, decode 4/4, needle 1/1, short 24/24, lhash first 74/74, end 74/74. crash 0.
- 06:59 C23 pf32 (the "on" flags + TF_FLASHNEXT_Q38_QSA_SLACK=32; fresh engine; knobs line "slack=32.0e-6"; MEASURED): prefill 7,276 / 27,502 / 109,142 / 214,471:
  2.89 / 10.50 / 43.39 / 87.82 s = 2515 / 2620 / 2516 / 2442 tok/s (slack 100: 2.92 / 10.56 / 43.64 / 88.23); full chunks 109K 42,487, 214K 86,481 ms;
  in-between 18.4 a row (27 at slack 100; 12.6 at slack 1); texts 5/5, lhash 5/5 + 5/5 identical. => slack 32 = -0.5 % of a long prompt against slack 100.
  RECOMMENDATION for the recipe: QSA_SLACK=32 (2x over the pessimistic worst case of the fp32 sums, 160x over the largest deviation measured); 100 stays the code default.
- 07:15 C23 "off" = REFERENCE LAST (fresh engine, knob fix only; knobs line "list=0 rows=0"; dir C_y_off_C23; MEASURED): prefill 7,276 / 27,502 / 109,142 / 214,471:
  3.10 / 10.64 / 44.31 / 91.15 s; full chunks 109K 43,103, 214K 89,164 ms; short 99.9 ms; decode 44.4 / 48.6 / 44.0 / 42.2; ident 51/51, texts and lhash identical to C_y_ref_C20.
  PAIRED TABLE, ONE WINDOW, fresh engines, reference first (C22 off 05:53) and last (C23 off 06:58); "on" = C23 with ARENA=1 QSA_LIST=1 QSA_ROWS=1 WARM=3072 (MEASURED):
  | prompt tokens | off first | on (slack 100) | on + QSA_SLACK=32 | off last | gain of "on" vs mean off | with slack 32 |
  | 7,276   | 3.10 s 2347 tok/s | 2.92 s 2494 | 2.89 s 2515 | 3.10 s 2345 | +6.3 % | +7.2 % |
  | 27,502  | 10.67 s 2576 | 10.56 s 2604 | 10.50 s 2620 | 10.64 s 2584 | +0.9 % | +1.5 % |
  | 109,142 | 44.62 s 2446 | 43.64 s 2501 | 43.39 s 2516 | 44.31 s 2463 | +1.9 % | +2.5 % |
  | 214,471 | 91.78 s 2337 | 88.23 s 2431 | 87.82 s 2442 | 91.15 s 2353 | +3.7 % | +4.1 % |
  | short new prompt first token | 99.7 ms | 96.7 ms | - | 99.9 ms | -3 % | |
  Against the base as shipped today (C20: slack 1 through the knob bug): 3.14 / 10.67 / 44.28 / 91.03 s -> on + slack 32: 2.89 / 10.50 / 43.39 / 87.82 (+8.7 / +1.6 / +2.1 / +3.7 %).
- 07:24 AGENTS WITH THE ARENA (C23, default one-user profile, 5 agents x 56K own system prompt x 3 turns, fresh engines; MEASURED):
  | | flags off | ARENA + LIST + ROWS + WARM |
  | wall; sum of first-token waits | 125.9 s; 426.5 s | 122.1 s; 412.5 s (-3.3 %) |
  | follow-up first token median / max | 0.96 / 11.49 s | 0.95 / 11.99 s |
  | short prompt after | 100.1 ms | 96.3 ms |
  | MemAvailable at the end | 10 GiB | 2 GiB |
  texts 15/15 + 24/24 identical, lhash first 27/27 (end 20/27: the concurrent requests' schedule noise, round 8). cache: 22 states, 23.0 GiB, hits 10, evicted 8 in both.
  PROBLEM FOUND: arena "fallback=5, largest_miss 1106 MiB": five kept states found no gap (fragmentation), went to cuMemAlloc ON TOP of the arena's fixed 29.3 GiB, and
  the box was left with 2 GiB. Not acceptable as it stands. FIX (C24, tree 49cf032): a kept state is never placed outside the arena: cuda_snap refuses
  (error.Q38ArenaFull), the prompt cache evicts its least valuable states until the state has a gap (new optional vtable hook `placeable`), and an in-pass keep with no
  gap falls back to the cut pass. Lane buffers may still fall back (short-lived, at most one stream's caches).
- 07:25 C24 building (C23 + arena placement rule + TF_FLASHNEXT_Q38_HEAD_KV + head hash aid TF_FLASHNEXT_Q38_HHASH). Lease C renewed 75 min.
  HEAD_KV: the no-drafts engine showed the MTP head's own prompt pass is 2-4.4 % of a long prompt's first-token time; with MTP_LAST only the call's last row's
  output is read, yet every prompt row ran the head's block scores, selection, attention, gate, out projection and write-back. With HEAD_KV=512 all rows but the call's
  last 512 stop after the cache writes. ESTIMATE: -1..-2 % first-token time at 55K-214K.
- 07:31 C24 BUILD_OK sha 6d4d3460d89d453e (second compile: two syntax errors in the first). runY2 C24 "on pfk off agon" started (log box ~/q38/runs/C_y6.log): ON = ARENA + QSA_LIST + QSA_ROWS + WARM + HEAD_KV=512; on / off with TF_FLASHNEXT_Q38_HHASH=1 (head state hashes; they inflate first-token times of those two runs); pfk = prefill only with ON, no HHASH (the timing).
- 07:45 C24 "on" (sha 6d4d3460d89d453e; fresh engine; ARENA=1 QSA_LIST=1 QSA_ROWS=1 WARM=3072 HEAD_KV=512, + HHASH aid; dir C_y_on_C24; MEASURED): prefill 7,276 / 27,502 /
  109,142 / 214,471: 2.86 / 10.34 / 42.67 / 86.60 s (with the hash download inside; clean timing = pfk below); full chunks 27.5K 9,163 / 109K 41,737 / 214K 85,153 ms
  (C23 on, no HEAD_KV: 9,368 / 42,745 / 86,856 = -2.2 / -2.4 / -2.0 %); decode 44.4 / 48.5 / 43.8 / 42.2 tok/s; short 96.9 ms; needle 42.8 s 3/3.
  IDENTITY vs the base binary's run (C_y_ref_C20): ident 51/51, warm 1/1, prefill 4/4, decode 4/4, needle 1/1, short 24/24 texts identical; lhash first 74/74, end 74/74.
  arena served 2,762, refused 0, fallback 0. crash lines 0. Head-state hashes are compared when the off run is in.
- 07:55 C24 pfk = CLEAN TIMING OF THE HANDOFF BINARY (sha 6d4d3460d89d453e, tree 793d4ee, fresh engine, ARENA=1 QSA_LIST=1 QSA_ROWS=1 WARM=3072 HEAD_KV=512, no hash aid;
  MEASURED): prefill 7,276 / 27,502 / 109,142 / 214,471 = 2.87 / 10.39 / 42.90 / 86.79 s = 2538 / 2647 / 2544 / 2471 tok/s; full chunks 27.5K 9,218, 109K 41,996, 214K 85,448 ms;
  texts 5/5, lhash 5/5 + 5/5 vs the base run. Against the mean of the two flags-off references (3.10 / 10.655 / 44.465 / 91.465 s): +8.0 / +2.6 / +3.6 / +5.4 %.
- 08:05 C24 "off" (with the hash aid; MEASURED): 3.10 / 10.67 / 44.4x / 91.41 s; ident 51/51, texts, lhash 74/74 + 74/74 vs C_y_ref_C20.
  HEAD_KV IDENTITY (TF_FLASHNEXT_Q38_HHASH=1 on both; kitC/hh.py): the MTP head's state after every prompt pass (last row's output, ALL keys, ALL values; prompts from 19
  to 214,471 tokens): 86 head states in both runs, 86 identical, 0 differ.
  HANDOFF written: handoff/C/r10-prefill/C-r10.patch (sha256 bbc97f5555304f31, TREE_EQUAL to 793d4ee) + NOTE.md; handoff/C/URGENT-r9-knob-cache.md.
- 08:11 C24 AGENTS WITH THE ARENA'S PLACEMENT RULE (5 agents x 56K x 3 turns, default profile, fresh engine, full flag set; reference = C23 flags off; MEASURED):
  | | flags off | full set (C24) |
  | wall; sum of first-token waits | 125.9 s; 426.5 s | 120.2 s; 407.5 s (-4.5 %) |
  | follow-up first token median / max | 0.96 / 11.49 s | 0.82 / 13.13 s |
  | MemAvailable at the end | 10 GiB | 7 GiB (C23's first form: 2 GiB) |
  | cache at the end | 22 states, 23.0 GiB, evicted 8, hits 10 | 15 states, 15.7 GiB, evicted 15, hits 10 |
  arena: served 1,199, fallback 0, "no gap" answers 11 (each made the cache evict instead of leaving the arena). texts 15/15 + 24/24 identical, lhash first 27/27
  (end 13/27: concurrent schedule noise). crash lines 0. => memory is bounded now; the price is more evictions when states of different sizes churn (7 more here, no
  lost hit). NOT TESTED: 16 agents x 20K on the default profile with the arena (the heavy-eviction case), 16-user profiles (no arena there: POOL_CTX).
## ROUND 10 END STATE (08:12 UTC 2026-11-14)
- Lease C RELEASED 08:11:23. spark-4e5c: no q38-run-C container, no run script, no build, 0 GPU processes. Raw results: Mac q38/results/rC10.tgz.
- HANDOFF (Mac q38/handoff/C/): r10-prefill/C-r10.patch (git diff 7061396..793d4ee, sha256 bbc97f5555304f31, 1,174 lines, 12 files, TREE_EQUAL) + NOTE.md;
  URGENT-r9-knob-cache.md (the shared knob cache of patch 0116: fix is in the patch). Built and measured binary of exactly that tree: C24 sha 6d4d3460d89d453e.
- SHIPPED-READY (exact; flags default off in the patch; proposed recipe defaults): QSA_LIST=1, QSA_ROWS=1, HEAD_KV=512, WARM=3072, ARENA=1 (one-user profile; caveat
  below), QSA_SLACK=32, and the knob fix. Cold prompt first token, full set vs flags off: 7.3K +8.0 %, 27.5K +2.6 %, 109K +3.6 %, 214K +5.4 % (2538 / 2647 / 2544 / 2471 tok/s).
- DROPPED: sampled cut (-0.3 %, text dependent); expert tile config (default is best); partial chunk (the fixed 127 ms is every expert read once); decode growth (+3.7 % at 182K, no fix known).
- WHAT REMAINS, by size:
  1 the prompt pass runs ~30 TFLOP/s of a measured 122 TFLOP/s ceiling (61.5 with one operand from shared memory): the dense prompt tiles / hyper-connection / DeltaNet
    chain are the place; a third engine's lead comes from quantised activations (not portable under the lossless rule).
  2 the draft head's prompt pass after HEAD_KV: projections + read-out still run for every row (~1-2 % of a long prompt).
  3 ARENA under heavy state churn (16 agents x 20K) untested; lane buffers can still fall back; a zero fill remains at lane open (28-50 ms at 109-214K).
  4 select and coarse scores still grow with context (a chunk at the end of 214K: 1,287 ms against 1,140 at its start); fused attention's gather +23 ms.
  5 A re-runs on 923b before any number is quoted publicly (check mode once, the fresh pair, the agent trace).
- RESUME: cd ~/.hermes/cache/scratch/q38; tail -80 PROGRESS-C.md; ./shipC10.sh NAME 16 24g; ./pushkit.sh; ./mkhandoffC10.sh; box: bash ~/q38/kit-C/runY2.sh DL BIN "chk on pfk off agon agoff nd tile" "FLAGS"
  (env LX = extra flags for on/off, PF = flags of pf*, TILES = tile configs); python3 -I ~/q38/kit-C/hh.py OFF/engine.log ON/engine.log; bash ~/q38/kit-C/mb/run_mb.sh.

# Round 11 (builder C, started 2026-11-14 08:14 UTC)
Box spark-4e5c. Base = r10-C 793d4ee. Worktree Mac q38/wt/C11 branch r11-C. Binaries C30.. Levers: floor table by group (bytes at 235 GB/s, FLOPs at 61.5 TFLOP/s, measured), then exact kernels for the largest gap; MoE prompt tiles; draft head prompt pass; lane-open zero fill.
- 08:14 window open (leases A, B, D). 4e5c idle (110 GiB free, no container but sz-memguard). Worktree created. Reading the round-10 KPROF log for the per-kernel table.
- 08:21 lease C acquired 60 min. MICROBENCH 1 = the engine's own prompt expert tile kernels, unchanged source, engine launch arguments (kitC/mb/mb_exp.cu + run_mbx.sh;
  one layer: 512 experts, 2560 x 640; gate/up tf_nvfp4_tile_gu_rt4_s4 grid 96 smem 34,816; down tf_nvfp4_tile_f32_rt2_s3 grid 192 smem 19,200; random weights; MEASURED, ms a launch, median of 7):
  | case | gate/up ms (ns a pair, TFLOP/s) | down ms (ns a pair, TFLOP/s) |
  | A 512 experts x 60 pairs, items of 64 (= a 3,072-row chunk, even routing) | 5.05 (164, 40) | 4.06 (132, 25) |
  | A2 same pairs, skewed counts (exponential, mean 60), items of 64 (768 items) | 6.24 (208, 32) | 3.95 (132, 25) |
  | A3 skewed, ONE item an expert | 8.53 (285) | 4.32 (144) |
  | B 512 experts x 240 pairs, items of 64 (= a 12,288-row chunk) | 12.55 (102, 64) | 9.98 (81, 40) |
  | C 512 x 240, one item an expert | 22.2 (181) | 10.7 (87) |
  | D ONE expert, 480 items of 64 (weights cache-resident = compute only) | 2.91 (95, 69) | 2.32 (75, 43.5) |
  | E 512 experts x 16 pairs (= reading every expert once, almost no rows) | 4.42 = 213 GB/s | 2.59 = 182 GB/s |
  READING: (1) engine KPROF per layer-chunk: gate/up 4.5-4.6 ms, down 3.4 ms: the microbench reproduces the engine (A). (2) Reading every expert of a layer once costs
  4.42 + 2.59 = 7.0 ms (E) = 336 ms a chunk for 48 layers whatever the rows; the rows' own compute is 2.91 + 2.32 = 5.2 ms a layer per 30,720 pairs (D) = 251 ms.
  A 3,072-row chunk pays max-ish of both (9.1 ms a layer in A = 437 ms). (3) Items of the SAME expert that follow each other share the weight read (B: 102 ns a pair
  against 95 compute-only): the cache dedups it, no kernel change needed. So with 4x the rows a chunk the expert kernels cost 5.6 ms per 30,720 pairs instead of 9.1
  (-38 % of the expert kernels = ESTIMATE -13 % of a chunk) IF the engine's bigger chunks really behave like B. Round 6 measured TF_FLASHNEXT_CHUNK=8192 at only
  +0..2 %: something else must get worse with the chunk size. (4) One long item an expert is slower (C, A3), as the round-10 sweep said.
  => runZ1.sh started: GPROF=2 by group with TF_FLASHNEXT_CHUNK = default 3072 / 6144 / 12288 (C24, QSA_LIST + QSA_ROWS + HEAD_KV, no arena), log box ~/q38/runs/C_z1a.log.
- 08:33 Z1 GPROF by group against TF_FLASHNEXT_CHUNK (C24, QSA_LIST + QSA_ROWS + HEAD_KV, no arena, fresh engines, same prompts; us a row of a 54,679-token prompt; MEASURED):
  | chunk rows | total gpu us/row | moe | gdn | hc-attn-side | hc-mlp-side | qsa-attn | first token 7.3K / 27.5K / 54.7K |
  | 3072 (default) | 374.0 | 138.0 | 86.6 | 54.8 | 33.3 | 56.1 | 2.83 / 10.42 / 20.99 s |
  | 6144 | 362.2 | 127.2 | 86.0 | 54.5 | 33.5 | 56.3 | 2.77 / 10.14 / 20.51 s (-2.1 / -2.7 / -2.3 %) |
  texts 3/3 identical, score hashes 4/4 + 4/4 identical between the two chunk sizes. 12288 did not start: its scratch does not fit beside 5 x 262,144 lanes
  ("needs 25.11 GiB, budget leaves 20.25"). MemAvailable after load 34 GiB at 6144 against 37.
  READING: only MoE moves with the chunk size (-7.8 %), far less than microbench case B said (-38 %): the engine's real routing is not even. Fit MoE = 66 ms a chunk +
  116 us a row; compute-only in the microbench is 82 us a row. So a real chunk still re-reads weights per row tile somewhere, or the real spread of pairs over experts
  makes the tile kernel slow (cases A2/A3 were slower than A). Need the real plan: C30 = C24 + TF_FLASHNEXT_Q38_XPLAN (dumps each prompt MoE block's items) building;
  kitC/xp.py reads it and writes the counts for a replay in the microbench.
- 08:44 REAL EXPERT PLANS (C30 = C24 + TF_FLASHNEXT_Q38_XPLAN, sha bad9db64e6f60448; 27.5K filler prompt; kitC/xp.py; MEASURED): a 3,072-row chunk uses a median of 313 of
  512 experts a layer (148-408); pairs an expert: median 17, p90 226, max 3,068 (one expert takes nearly every row); 89 % of pairs sit in experts with more than 64 pairs,
  half the used experts have <= 16 pairs; ~700 items of 64 a layer. A 6,144-row chunk: 342 experts, 1,190 items. (Filler text; real documents may spread more.)
  REPLAY of those plans in the microbench (engine kernels, 4 layers; MEASURED ms a layer): 3,072 rows: gate/up 4.0-4.6 + down 3.0-3.3 (the engine's KPROF: 4.5 + 3.4);
  6,144 rows: 7.0-7.9 + 5.2-5.8 = -13 % a pair (engine GPROF: -7.8 % of the MoE group). Compute-only (one expert): 2.92 + 2.33.
  LEVER TRIED IN THE MICROBENCH, DROPPED: handing units to CTAs by a ticket counter instead of the fixed stride (same tiles): gate/up +2..+19 %, down +5..+13 % SLOWER.
  (The microbench's "bits differ" flag there is its own artefact: random member lists repeat pair ids, so two items write the same output row.)
- 08:47 MICROBENCH 2 (kitC/mb/mb_gdn.cu): the DeltaNet chain's own arithmetic (state in registers, k and q from shared memory, a barrier every 16 rows), one CTA a head on
  48 SMs: 0.43 us a row = 1.3 ms per 3,072 rows = 48 ms a chunk for 36 layers. The engine's fn_gdn_fused takes 3.2 ms (KPROF) = 1.04 us a row = 115 ms a chunk.
  HYPOTHESIS: the chain waits for its 128 producer threads (conv + SiLU + norms for 16 rows a batch, with un-prefetched global loads), not for its own arithmetic.
  The recurrence itself must stay sequential over rows (a blocked / prefix form changes the operation order: not allowed), but feeding it faster is exact.
  Next: microbench of the REAL fn_gdn_fused with engine arguments, then producer variants.
- 08:56 MICROBENCH 3 = the REAL fn_gdn_fused (flashnext_batch.cu included unchanged, engine launch arguments: grid 48, 384 threads, 82,496 shared bytes; 3,072 rows,
  random inputs; kitC/mb/mb_gdnf.cu; MEASURED, ms per 3,072 rows; outputs compared byte for byte with the engine kernel: out + xs + end state = 43,253,760 bytes):
  | kernel | ms | registers | bytes differing |
  | engine fn_gdn_fused | 2.72 (engine KPROF with real data: 3.2) | 166 | - |
  | chain only (stage filled once) | 1.51 | | (diagnostic) |
  | producers only (conv + SiLU + norms, no chain) | 2.66 | 80 | (diagnostic) |
  | prefetch of the next batches' projection bytes (cp.async into a dummy word), 384 threads | 2.61-2.65 | 165 | 0 |
  | two producer groups, 512 threads, first form (chain's k held in registers) | 4.1-6.0 (spills: 128 registers + stack) | 128 | 0 |
  | two producer groups, 512 threads, chain re-reads k from shared memory for its second loop | 2.31 (-15 %) | 106 | 0 |
  | three producer groups, 640 threads | 2.38 | 96 + 32 B stack | 0 |
  READING: the hypothesis holds: the chain (1.5 ms) waits for 4 producer warps (2.66 ms alone: one warp's serial instruction stream, not the SM's issue rate and
  not load latency: prefetch gave -4 %). fn_gdn_fused needs 166 registers a thread because the chain keeps k (64 values) live across the tree, so a CTA cannot
  hold more than 384 threads; reading k again from shared memory (same values, same arithmetic) brings it to 106 and lets a second producer group in.
  Why not more: q and k of a key head are formed by each of its three value-head CTAs (inherent to one CTA a head without a memory round trip), and with
  20 warps the SM's issue rate is the limit (ESTIMATE ~2.0 ms if nothing waited). GDN chain floor = its sequential arithmetic, 1.5 ms per 3,072 rows a layer.
  => LEVER 1 (to build): TF_FLASHNEXT_Q38_GDN_PROD=2 -> fn_gdn_fused_p2; ESTIMATE -0.4..-0.6 ms a layer-chunk x 36 = -15..-22 ms a chunk = -1.3..-1.9 % of prefill.
- 08:58 C31 BUILD_OK sha ec07be368c7be43f (first compile; tree r11-C: XPLAN aid + kernels/cuda/q38_gdn_prod.cuh + cuda_gdn_fused.zig TF_FLASHNEXT_Q38_GDN_PROD=2|3,
  12|13 = device check against fn_gdn_fused on every launch). Lease C renewed 90 min. runZ3.sh C31 started (log box ~/q38/runs/C_z3.log): chk (GDN_PROD=12) -> off -> p2 -> p3 -> off,
  each a fresh engine on the round-10 set, warm 3K + prefill 8K / 30K / 120K / 235K, texts + score hashes against C_y_ref_C20.
- 09:00 WINDOW CUT: the shared window S140401 hit its 300-minute cap at 08:58:10 (my chk engine was killed 20 s after it started; nothing lost). Its restore was itself
  killed by the guard unit at 09:01:30 (pkill of qws.sh in the middle of its stop-all), so ~/q38/shared/CLOSING stayed behind and `lease.sh acquire` waits for ever.
  The guard is restoring DeepSeek (health 000 at 09:11). PLAN: wait until q38-guard.service is done and health is 200, then remove the stale CLOSING flag (a flag file,
  not a harness script; noted in shared/NOTES.md) and acquire. No GPU job meanwhile.
- 09:05 PREPARED while waiting: kitC/mb/mb_hcd.cu = the engine's fn_hcn_down4 with engine arguments + variants. Source reading: the hyper-connection down projection
  runs TWO CTAs a row tile (column tiles 0-159 / 160-335), and each loads and norms the same 64 rows x 10,240 values of H: H (63 MB a call) is read and normed twice.
  KPROF: 0.683 ms a call = 29.5 TFLOP/s, 184 GB/s if H is really read twice. Variant "wide": one CTA a row tile of 448 threads (14 warps), H read and normed once.
- 09:23 ARITHMETIC ON THE REPLAY (ESTIMATE, to be tested when the window is back): gate/up of a real 3,072-row layer = 4.56 ms x 96 CTAs = 438 CTA-ms for ~3,500 units
  (item x column group). ~427 full 64-row items of the big experts cost 117 us a unit (compute-only rate) = 250 CTA-ms; the other ~270 short items (half the used
  experts have <= 16 pairs) take the remaining ~188 CTA-ms = ~139 us a unit = 2.6 GB/s a CTA while they stream 368 KB of weights for one row tile. Case E (every unit
  short) reads 213 GB/s over 96 CTAs = 2.2 GB/s a CTA: the same per-CTA rate. HYPOTHESIS: a short unit is bound by its own cp.async pipeline (3 K32 groups in flight a
  CTA), not by the bus; in the real mix the bus averages only ~135 GB/s. If so, deeper pipelines (more stages) for short items would cut the expert kernels by up to ~25 %
  with the same arithmetic. mb_exp.cu now instantiates the engine's own tile template with 6 / 8 / 12 stages and varies the grid (48 / 96 / 144) to test it.
  DeepSeek health 503 at 09:22 (starting under the guard); builder A fixed qws.sh (NOTES 09:20).
- 09:26 window open again (A fixed the harness; leases A, B, C, D). Lease C acquired 120 min (until 11:26).
- 09:28 MICROBENCH 4 (kitC/mb/mb_hcd.cu; the engine's fn_hcn_down4, engine arguments: 3,072 rows, N 320, K 10,240, grid 96 x 224 threads; outputs compared; MEASURED):
  | kernel | ms a call | bytes differing |
  | engine fn_hcn_down4 (two CTAs a row tile) | 0.615 (KPROF in the engine: 0.683) = 32.7 TFLOP/s | - |
  | the two column tiles far apart in time (H certainly read twice) | 0.726 | 0 |
  | ONE CTA a row tile, 448 threads, 64 / 48 rows | 8.6 / 7.3 (register spills: 448 threads leave 146 registers, the kernel needs ~200) | 0 |
  | ONE CTA a row tile, 448 threads, 32 rows | 0.769 | 0 |
  | engine's other row tiles: fn_hcn_down2 / down8 | 0.804 / 2.22 | 0 |
  => DROPPED: the cache already shares most of the double H read (far apart costs +18 %), and the one-CTA form either spills or loses the weight reuse of 64-row tiles.
  The down projection is neither at its byte floor (63 MB = 0.27 ms) nor at the tensor-core rate (0.33 ms at 61.5 TFLOP/s): its per-element norm (2 multiplies + a bf16
  rounding on 31.5 M values a call, done in shared memory between loads and mma) is what is left; MiaAI's second form (fn_hcn_dv2) already tried interleaving it.
- 09:33 MICROBENCH 5 (mb_exp.cu, the engine's tile template instantiated with other stage counts and grids; real plans; MEASURED, ms a layer):
  | case | gate/up engine (4 stages, 96 CTAs) | 48 CTAs | 144 CTAs | 3 stages | 6 / 8 / 10 stages | down engine (3 stages, 192 CTAs) | 96 CTAs | 6 / 8 / 12 stages | rt4 |
  | E weights only (512 x 16 pairs) | 4.11 | 4.13 | 4.11 | 4.10 | 4.09 / 4.22 / 4.11 | 2.52 | 2.47 | 2.54 / 2.62 / 2.52 | 2.50 |
  | D compute only (one expert) | 2.83 | 4.19 | 3.73 | 3.08 | 4.26 / 4.45 / 4.33 | 2.29 | 2.96 | 2.60 / 4.28 / 4.18 | 2.79 |
  | real 3,072-row layers (4) | 3.58-4.19 | 4.9-5.4 | 4.5-5.3 | 3.8-4.4 | 5.0-5.6 | 2.96-3.25 | 3.4-3.8 | 3.3-4.6 | 3.3-3.7 |
  | real 6,144-row layers (4), per 61,440 pairs | 6.4-7.0 | 9.0-9.7 | 8.1-8.9 | 6.9-7.5 | 9.0-10.3 | 5.1-5.6 | 6.1-6.7 | 5.8-8.3 | 5.9-6.4 |
  => the 09:23 hypothesis is FALSE: weights-only streaming is the same at any CTA count and stage count = bus-bound (943 MB in 4.1 ms = 230 GB/s). The engine's
  instance (MiaAI's choice, confirmed by my round-10 sweep) is the best in every row. EXPERT KERNELS: NOTHING LEFT in this design: a real 3,072-row layer costs
  3.8 + 3.1 = 6.9 ms against 5.1 compute-only and 3.7 ms for its weight bytes at the bus rate (sum 8.8, max 5.1); at 6,144 rows it is 3.3 + 2.6 = 5.9 per 30,720
  pairs, 15 % over compute-only. The lever is rows a chunk (fewer weight bytes a row), not the kernel.
- 09:35 runZ3.sh C31 started (log box ~/q38/runs/C_z3.log): chk (GDN_PROD=12) / off1 / p2 / p3 / c6 = p2 + TF_FLASHNEXT_CHUNK=6144 / c8 = p2 + CHUNK=8192 / off2.
- 09:34 C31 CHECK MODE (TF_FLASHNEXT_Q38_GDN_PROD=12 on the round-10 set; fresh engine; prompts 2.8K / 7.3K / 27.5K / 109K / 214K; MEASURED): every fn_gdn_fused_p2 launch
  against fn_gdn_fused on the same inputs, on the device: 4,320 launches, 46,190,462,976 words (out + xs + end state) compared, 0 differ. Texts 5/5 and score hashes
  5/5 + 5/5 identical to the base binary's run (C_y_ref_C20). crash lines 0.
- 09:40 C31 off1 (round-10 set, no new flag; fresh; MEASURED): 7,276 / 27,502 / 109,142 / 214,471 tokens first token 2.86 / 10.34 / 42.68 / 86.52 s; full chunks
  2,306 / 9,172 / 41,796 / 85,174 ms; chunk mid 1,131 / 1,152 / 1,198 / 1,238 ms.
- 09:46 C31 p2 = TF_FLASHNEXT_Q38_GDN_PROD=2 (fresh, round-10 set + the flag; MEASURED): first token 2.83 / 10.21 / 42.20 / 85.50 s (off1: 2.86 / 10.34 / 42.68 / 86.52 =
  -1.0 / -1.3 / -1.1 / -1.2 %); full chunks 2,278 / 9,053 / 41,311 / 84,186 ms (off1 2,306 / 9,172 / 41,796 / 85,174: -1.2..-1.3 %, ~15 ms a chunk as the microbench said).
  Texts 5/5, score hashes 5/5 + 5/5 identical to C_y_ref_C20. crash 0.
- 09:52 C31 p3 = GDN_PROD=3 (three groups, 640 threads): 2.84 / 10.26 / 42.46 / 85.98 s (-0.6 %): worse than two groups (as in the microbench). Identity the same. => p2 is the form.
- 09:57 C31 c6 = GDN_PROD=2 + TF_FLASHNEXT_CHUNK=6144 (fresh, round-10 set; MEASURED): first token 2.75 / 9.97 / 40.69 / 82.20 s = 2643 / 2759 / 2682 / 2609 tok/s
  (p2 alone: 2.83 / 10.21 / 42.20 / 85.50 => the chunk size gives -2.8 / -2.4 / -3.6 / -3.9 %; against off1: -3.8 / -3.6 / -4.7 / -5.0 %). Texts 5/5, score hashes
  5/5 + 5/5 identical to C_y_ref_C20 (the base binary at 3,072 rows): the chunk size changes no output. COST: the pass scratch grows ~3.4 GiB and the arena sizes itself
  smaller: 25,886 MiB instead of 29,264 (that much less room for kept prompt states).
- 10:01 c8 = CHUNK=8192: the engine does not start with 5 x 262,144 lanes (memory budget, as 12,288 at 08:33). 6,144 is the largest step tried that fits this profile.
- 10:07 C31 off2 = REFERENCE LAST (fresh, round-10 set): 2.87 / 10.41 / 42.82 / 86.77 s; full chunks 2,318 / 9,223 / 41,940 / 85,442 ms. Texts + hashes identical.
  PAIRED (one window, fresh engines, same prompts, reference first and last; MEASURED), first token s:
  | prompt tokens | off first | GDN_PROD=2 | GDN_PROD=3 | GDN_PROD=2 + CHUNK=6144 | off last | PROD=2 vs mean off | PROD=2 + CHUNK=6144 vs mean off |
  | 7,276   | 2.86 | 2.83 | 2.84 | 2.75 | 2.87 | -1.2 % | -4.0 % |
  | 27,502  | 10.34 | 10.21 | 10.26 | 9.97 | 10.41 | -1.6 % | -3.9 % |
  | 109,142 | 42.68 | 42.20 | 42.46 | 40.69 | 42.82 | -1.3 % | -4.8 % |
  | 214,471 | 86.52 | 85.50 | 85.98 | 82.20 | 86.77 | -1.3 % | -5.1 % |
  = 2643 / 2759 / 2682 / 2609 tok/s with both (round-10 set alone: 2540 / 2651 / 2553 / 2475).
- 10:09 MICROBENCH 3b (mb_gdnf.cu, every chain / producer split; all outputs byte-identical to fn_gdn_fused; MEASURED ms per 3,072 rows, registers):
  | chain threads a state row (TPR) | producer groups | threads | ms | registers |
  | 2 (engine) | 1 (engine) | 384 | 2.74 | 166 |
  | 2 | 1, separate scopes + k re-read | 384 | 2.67 | 110 |
  | 2 | 2 | 512 | 2.34 | 106 |
  | 2 | 3 | 640 | 2.40 | 96 |
  | 2 | 4 | 768 | 7.5 (spills) | 80 + stack |
  | 4 | 1 / 2 / 3 / 4 | 640 / 768 / 896 / 1024 | 2.84 / 2.49 / 2.63 / 2.57 | 64-77 |
  | 1 | 2 | 384 | 2.62 | 168 |
  => TPR 2 with two producer groups is the best form; more chain threads do not help (one SM's issue rate carries one head: chain alone 1.51 ms, producers alone
  2.69 ms on 4 warps, together 2.34). GDN CHAIN: AT ITS FLOOR for one CTA a head after this lever; what is left (q and k of a key head formed by each of its three
  value-head CTAs: ESTIMATE 0.6 % of a chunk) needs a memory round trip that costs more.
- 10:12 SHARED EXPERT (36 ms a chunk on the generic 64 x 64 MXFP8 kernel at ~40 TFLOP/s; the 128 x 128 prompt tile does 82-89): NOT PORTABLE EXACTLY. Source reading
  (qmmf.cu): the generic kernel forms each 32-K block's products from a zero accumulator (mma0 + mma), then acc = fma(d, 2^e, acc), and folds K slices in order
  (gate/up: 4 slices); the prompt tile runs ONE accumulate chain over K with the block scale folded into the weight operand. Different rounding points: other bits.
  A tile that copies the generic arithmetic would lose most of the tile's speed. DROPPED before building (ceiling was 1.5 % of a chunk).
- 10:13 runZ4.sh C31 started (final pair, log box ~/q38/runs/C_z4.log): reference = round-10 recipe set incl. QSA_SLACK=32, first and last; on = + GDN_PROD=2 + CHUNK=6144;
  full list (warm, prefill x4, decode after 2K / 60K / 120K / 200K, ident 51, needle 120K, short x12). Handoff patch already cut from the C31 tree:
  handoff/C/r11-prefill/C-r11.patch sha256 2c4f370f664c279c, TREE_EQUAL (HEAD ce1f180), binary C31 sha ec07be368c7be43f.
- 10:44 Z4 FINAL PAIR (C31 sha ec07be368c7be43f = the handoff tree ce1f180; fresh engines; reference = round-10 recipe set ARENA + QSA_LIST + QSA_ROWS + WARM=3072 +
  HEAD_KV=512 + QSA_SLACK=32; on = + TF_FLASHNEXT_Q38_GDN_PROD=2 + TF_FLASHNEXT_CHUNK=6144; same request list; MEASURED):
  | item | off first | on | off last | change vs mean off |
  | prefill 7,276 tokens, first token | 2.87 s 2538 tok/s | 2.76 s 2640 | 2.87 s 2538 | -3.8 % time (+4.0 % tok/s) |
  | 27,502 | 10.37 s 2652 | 9.97 s 2758 | 10.37 s 2653 | -3.9 % (+4.0 %) |
  | 109,142 | 42.64 s 2560 | 40.67 s 2684 | 42.72 s 2555 | -4.7 % (+4.9 %) |
  | 214,471 | 86.46 s 2481 | 82.12 s 2612 | 86.53 s 2479 | -5.1 % (+5.3 %) |
  | needle 109,126 | 42.7 s 3/3 | 40.8 s 3/3 same answer | | -4.4 % |
  | decode after 1.9K / 54.6K / 109K / 182K (drafts on, same texts) | 44.4 / 48.5 / 43.9 / 42.1 tok/s | 44.3 / 48.3 / 43.9 / 42.1 | 44.3 / 48.4 / 43.8 / .. | same |
  | first token of that task after 54.6K / 109K / 182K | 21.11 / 42.90 / 72.87 s | 20.28 / 40.84 / 69.19 s | | -3.9 / -4.8 / -5.1 % |
  | short new prompt (19 tokens) first token, median of 12 | 95.7 ms | 95.9 ms | | same |
  | ident 51 median | 65.8 tok/s | 65.6 | | same (no HEAD2 file on this box) |
  IDENTITY of "on" against the BASE binary's run (C_y_ref_C20, round 10's reference at 3,072 rows): ident 51/51; warm 1/1, prefill 4/4, decode 4/4, needle 1/1,
  short 24/24 texts identical; score hashes after the prompt pass 74/74, at release 74/74. Arena: 26,406 MiB (29,264 at 3,072 rows), served 2,762, fallback 0,
  peak 23,050 MiB; MemAvailable 8 GiB. certified select: 11,236,552 rows, fallback 0. crash lines 0. The same for off first.
- 10:50 Z4 done: "on" against off first: decode 4/4, ident 51/51, score hashes 74/74 + 74/74. runZ5.sh queued (kp, ag, z0), log box ~/q38/runs/C_z5.log. Lease C renewed to 12:27.
- 10:55 KPROF OF THE ROUND-11 SET AT 3,072 ROWS (C31, round-10 set + SLACK=32 + GDN_PROD=2, serialised launches, 12-55K context, 16 chunks; kitC/kp3.py; MEASURED ms a chunk):
  all kernels 1,146 ms a chunk (round-9 base at 40-110K: 1,230). gate/up 209.5, down 158.4, prompt tile bf16 139.0, MoE write-back 104.9, fn_gdn_fused_p2 92.7
  (2.574 ms a call; fn_gdn_fused was 3.157 = -18.5 % = -21 ms a chunk), fused attention 89.2, hc down 65.1, hc up 61.8, prompt tile f32 58.1, attention write-back 36.6,
  shared expert 23.9 + 12.5, listed select 18.8 + coarse by rows 9.5, route logits 13.6.
  DRAFT HEAD'S PROMPT PASS AFTER HEAD_KV: generic BF16 projections 7.7 + 5.2 + 4.1, its attention on the last 512 rows 2.0, write-back / norms / mixes ~3, PLE 2.9
  = ~26 ms a chunk = 2.3 %. All of it forms the head layer's input for every prompt row, which its keys and values need (they are read by later rounds), so no
  row can be skipped exactly; the kernels are the generic 64 x 64 BF16 tiles (~40 TFLOP/s); a faster tile with the same K slices would be worth < 1 % (ESTIMATE). NOT BUILT.
- 11:00 KPROF AT 6,144 ROWS A CHUNK (same engine flags; per 3,072 rows = half a chunk; MEASURED): all kernels 1,108.5 (3,072-row chunks: 1,146). gate/up 170.2 (209.5),
  down 137.0 (158.4), every other kernel the same per row EXCEPT tf_route_logits_wide: 40.8 against 13.6 (1.700 ms a call for 6,144 rows against 0.283 for 3,072).
  CAUSE (source): the router launch is a grid of row tiles x 9 expert tiles of 64, and runs all row tiles for one expert tile after the other: the input rows are
  read nine times. 3,072 rows (15.7 MB) stay in the 24 MiB cache; 6,144 rows (31 MB) do not: +1.13 ms a layer of memory traffic.
  => LEVER 3 (code written, commit after ce1f180): TF_FLASHNEXT_Q38_ROUTE_BAND=N: the router launched in bands of N rows (rows are independent in the kernel);
  _ROUTE_BAND_CHECK=1 compares against the single launch on the device. ESTIMATE: -27 ms per 3,072 rows at CHUNK=6144 = -2.4 %; ~0 at 3,072.
- 11:10 AGENTS (C31, default one-user profile, 5 agents x 56K own system prompt x 3 turns, fresh engines; MEASURED):
  | | round-10 set (3,072 rows) | round-11 set (GDN_PROD=2 + CHUNK=6144) |
  | wall; sum of first-token waits | 119.9 s; 407.5 s | 134.2 s (+12 %); 395.2 s (-3 %) |
  | first turn (cold 52K prompts) first token median / max | 79.9 / 107.5 s | 72.9 / 99.9 s |
  | follow-up first token median / max | 0.82 / 13.10 s | 0.98 / 21.91 s |
  | prompt cache at the end | 15 states, 15.7 of 23.9 GiB, hits 10, misses 5, evicted 15 | 12 states, 12.7 of 20.0 GiB, hits 9, misses 6, evicted 22 |
  | arena: served / "no gap" answers / fallback / peak | 1,199 / 10 / 0 / 26.5 GiB | 1,203 / 23 / 0 / 23.4 GiB |
  texts 15/15 + short 24/24 identical, score hashes after the prompt pass 27/27 (release-time 13/27: concurrent schedule noise, as in rounds 8-10). crash 0.
  => CHUNK=6144 takes ~3.9 GiB from the prompt cache's room; with five 56K agents that cost one cache hit (a 55K prefill again: 21.9 s) and the wall got WORSE.
  So CHUNK=6144 is an OPTION for long single prompts (or profiles with spare memory), NOT a default for the agent profile. GDN_PROD=2 has no such cost.
- 11:11 Z5's last part (short prompts with the arena's zero fill off) stopped before it measured; C32 building (= C31 + ROUTE_BAND).
- 11:14 C32 BUILD_OK sha 69638727a9799a29 (first compile). runZ6.sh C32 started (log box ~/q38/runs/C_z6.log): chk (ROUTE_BAND=2048 + its check + GDN_PROD=12, CHUNK=6144) / off1 (round-11 set) / band / band3 (band at 3,072 rows) / off2.
- 11:20 C32 CHECK MODE (round-11 set at CHUNK=6144 + ROUTE_BAND=2048 + ROUTE_BAND_CHECK=1 + GDN_PROD=12; fresh; prompts 2.8K-214K; MEASURED): router bands against the
  single launch on the device: 2,928 passes, 8,940,235,680 logit words, 0 differ; fn_gdn_fused_p2 against fn_gdn_fused at 6,144-row chunks: 2,232 launches,
  44,548,392,960 words, 0 differ. Texts 5/5, score hashes 5/5 + 5/5 identical to C_y_ref_C20. crash 0.
- 11:25 C32 off1 (round-11 set: round-10 set + SLACK=32 + GDN_PROD=2 + CHUNK=6144): 2.76 / 9.97 / 40.68 / 82.11 s (C31 "on": 2.76 / 9.97 / 40.67 / 82.12: the same).
- 11:31 C32 band (+ TF_FLASHNEXT_Q38_ROUTE_BAND=2048): 2.72 / 9.80 / 39.92 / 80.59 s = 2679 / 2805 / 2734 / 2661 tok/s (-1.4 / -1.7 / -1.9 / -1.9 % against off1).
  Texts 5/5, score hashes 5/5 + 5/5 identical to C_y_ref_C20. Then band3 (band at 3,072 rows), off2, and a full list (ident 51, decode, needle, short) on C32 queued.
- 11:37 C32 band3 = GDN_PROD=2 + ROUTE_BAND=2048 at the DEFAULT 3,072 rows: 2.82 / 10.19 / 42.02 / 85.23 s = 2577 / 2699 / 2597 / 2516 tok/s (round-10 set + SLACK=32:
  2.87 / 10.37 / 42.68 / 86.50 => -1.7 / -1.7 / -1.5 / -1.5 %; of that the band is ~0.2 %: at 3,072 rows the router's input rows stay cached). Identity the same.
- 11:43 C32 off2 (round-11 set, no band) = REFERENCE LAST: 2.75 / 9.99 / 40.64 / 82.09 s.
  PAIRED, C32 (sha 69638727a9799a29), fresh engines, reference first and last (MEASURED), first token s:
  | prompt tokens | off first (round-11 set, CHUNK=6144) | + ROUTE_BAND=2048 | off last | change |
  | 7,276 | 2.76 | 2.72 | 2.75 | -1.3 % |
  | 27,502 | 9.97 | 9.80 | 9.99 | -1.8 % |
  | 109,142 | 40.68 | 39.92 | 40.64 | -1.8 % |
  | 214,471 | 82.11 | 80.59 | 82.09 | -1.8 % |
  CUMULATIVE against the round-10 recipe set (2.87 / 10.37 / 42.68 / 86.50 s = 2538 / 2652 / 2558 / 2480 tok/s; C31 pair at 10:44):
  | set | 7.3K | 27.5K | 109K | 214K |
  | GDN_PROD=2 + ROUTE_BAND=2048, 3,072 rows (no memory cost) | 2.82 s 2577 tok/s (+1.5 %) | 10.19 s 2699 (+1.8 %) | 42.02 s 2597 (+1.5 %) | 85.23 s 2516 (+1.5 %) |
  | + CHUNK=6144 (3.9 GiB less cache room) | 2.72 s 2679 (+5.6 %) | 9.80 s 2805 (+5.8 %) | 39.92 s 2734 (+6.9 %) | 80.59 s 2661 (+7.3 %) |
- 11:57 Z7 = FULL REQUEST LIST ON THE FINAL BINARY (C32 sha 69638727a9799a29 = handoff tree b37b4e0; fresh engine; round-10 set + SLACK=32 + GDN_PROD=2 + CHUNK=6144 +
  ROUTE_BAND=2048; MEASURED): prefill 7,276 / 27,502 / 109,142 / 214,471 tokens first token 2.71 / 9.79 / 39.83 / 80.51 s = 2689 / 2809 / 2740 / 2664 tok/s
  (R10: 2.87 / 10.37 / 42.68 / 86.50 => -5.6 / -5.6 / -6.7 / -6.9 % time); first token of the decode task after 54.6K / 109K / 182K 19.87 / 40.11 / 67.84 s
  (R10 21.11 / 42.90 / 72.87); decode 44.4 / 48.5 / 44.0 / 42.2 tok/s (same); needle 109,126 tokens 40.0 s, 3/3; short new prompt 96.6 ms (R10 95.7); ident median 65.5.
  IDENTITY against the base binary's run (C_y_ref_C20) AND against the R10 run (C_y_off1_C31): ident 51/51; warm 1/1, prefill 4/4, decode 4/4, needle 1/1,
  short 24/24 texts identical; score hashes after the prompt pass 74/74, at release 74/74. Arena served 2,762, fallback 0, peak 22.9 GiB; select fallback 0; crash 0.
  Handoff patch re-cut from this tree: C-r11.patch sha256 ff7a98e55d633133, 522 lines, 6 files, TREE_EQUAL (HEAD b37b4e0).
- 11:58 Z8 started (log box ~/q38/runs/C_z8.log): the middle point CHUNK=4608 (+ GDN_PROD=2 + ROUTE_BAND=2048): prefill x4, then the 5-agent trace against R10 on C32.
- 12:04 Z8 CHUNK=4608 (C32, round-10 set + SLACK=32 + GDN_PROD=2 + ROUTE_BAND=2048 + CHUNK=4608; fresh; MEASURED): first token 2.71 / 9.89 / 40.36 / 81.96 s =
  2690 / 2780 / 2704 / 2617 tok/s (R10: 2.87 / 10.37 / 42.68 / 86.50 => -5.6 / -4.6 / -5.4 / -5.2 %); arena 27,868 MiB. Texts 5/5, hashes 5/5 + 5/5 vs C_y_ref_C20.
- 12:16 AGENTS AT 4,608 ROWS (C32, 5 agents x 56K x 3 turns, fresh engines; R10 first, then + GDN_PROD=2 + ROUTE_BAND=2048 + CHUNK=4608; MEASURED):
  | | R10 | + the three at 4,608 rows |
  | wall; sum of first-token waits | 119.6 s; 406.6 s | 114.6 s (-4.2 %); 374.1 s (-8.0 %) |
  | follow-up first token median / max | 0.82 / 13.05 s | 0.83 / 14.22 s |
  | prompt cache at the end | 16 states, 16.8 of 24.3 GiB, hits 10, evicted 14 | 15 states, 15.7 of 21.4 GiB, hits 10, evicted 15 |
  texts 15/15 + 24/24, score hashes after the prompt pass 27/27. crash 0. => 4,608 keeps every hit: THE PROPOSED CHUNK. (R10 wall repeats: 119.9 on C31, 119.6 on C32.)
## ROUND 11 END STATE (12:20 UTC 2026-11-14)
- Lease C RELEASED 12:16:30. spark-4e5c: no q38-run-C container, no run script, no build, 0 GPU processes. Raw results: Mac q38/results/rC11.tgz (381 KB).
- HANDOFF (Mac q38/handoff/C/r11-prefill/): C-r11.patch (git diff 793d4ee..b37b4e0, sha256 ff7a98e55d633133, 522 lines, 6 files, TREE_EQUAL) + NOTE.md.
  Built and measured binary of exactly that tree: C32 sha 69638727a9799a29 (box ~/q38/tfbin/C32). Builder A integrates; nothing pushed by C.
- SHIPPED-READY (exact; flags default off in the patch): TF_FLASHNEXT_Q38_GDN_PROD=2 (-1.2..-1.6 % first-token time), TF_FLASHNEXT_Q38_ROUTE_BAND=2048 (-1.3..-1.8 % with
  bigger chunks, ~0 at 3,072), and the existing knob TF_FLASHNEXT_CHUNK=4608 (proposed) / 6144 (option). Cold prompt first token, 7.3K / 27.5K / 109K / 214K:
  R10 2538 / 2652 / 2558 / 2480 tok/s -> at 4,608 rows 2690 / 2780 / 2704 / 2617 (+6.0 / +4.8 / +5.7 / +5.5 %) -> at 6,144 rows 2689 / 2809 / 2740 / 2664 (+5.9 / +5.9 / +7.1 / +7.4 %).
  Short new prompt first token unchanged (95.7 -> 96.6 ms, noise). Decode unchanged.
- FLOOR TABLE: handoff NOTE section 5. Groups at their floor: both write-backs (93-94 % of the byte floor); expert kernels at the optimum of their design (bus-bound
  streaming + compute); DeltaNet chain at one SM's instruction rate after GDN_PROD=2. Farthest from its floor and untouched: QSA fused attention (3x its FLOP floor).
- DROPPED (measured): expert tile ticket dealing / deeper pipelines / other grids; hyper-connection down as one CTA a row tile; DeltaNet prefetch, 3-4 producer
  groups, 4 chain threads a row; GDN_PROD=3; shared expert on the prompt tile (other bits); CHUNK 8192 / 12288 (do not fit).
- NOT DONE: lane-open zero fill off for short prompts (ARENA=2 run cut; ceiling under 8.7 ms of 96); the draft head's remaining prompt pass (2.3 % of a chunk, needed
  inputs; < 1 % available); trimming the pass scratch (~0.5 of 1.1 MB a row looks unused by a big chunk: would let 6,144-8,192 rows in without cache cost).
- WHAT REMAINS, by size: 1 QSA fused attention kernel (93 ms a chunk + growth; floor ~30); 2 pass scratch trim -> CHUNK 6144 / 8192 for the agent profile (ESTIMATE
  +1.5..2.5 % more); 3 expert kernels' bus/compute overlap (110 ms gap, no idea that survived the replay); 4 hyper-connection scalar work (63 ms gap); 5 16 agents x 20K
  with CHUNK=4608 and the arena untested; 6 A re-runs on 923b.
- RESUME: cd ~/.hermes/cache/scratch/q38; tail -80 PROGRESS-C.md; ./shipC11.sh NAME 16 24g; ./pushkit.sh; ./mkhandoffC11.sh; box: bash ~/q38/kit-C/runZ6.sh DL BIN "chk off1 band c46 band3 off2",
  runZ4.sh DL BIN (full pair), runZ5.sh DL BIN "kp ag" (KPROF tables via kp3.py; agents), runZ2.sh BIN "def 6144" (expert plans via xp.py);
  microbenches: bash ~/q38/kit-C/mb/run_mbx.sh mb_exp.cu | mb_gdnf.cu | mb_hcd.cu (need ~/q38/build-C/tf present; mb_exp reads kit-C/mb/plan_*.txt).

# Round 12 (builder C, started 2026-11-14 12:20 UTC)
Box spark-4e5c. Base = r11-C b37b4e0 (793d4ee + C-r11.patch). Worktree Mac q38/wt/C12 branch r12-C. Binaries C40.. Levers in order: 1 QSA fused attention in prompt geometry,
2 pass scratch trim (CHUNK 6144 / 8192 without cache cost), 3 hyper-connection scalar work, 4 decode at long context, 5 paired prefill + decode table.
- 12:20 window open (leases A, B, D). 4e5c idle (110 GiB free). Worktree created. Reading the fused attention kernel source.
- 12:22 C40 = r11 tree + TF_FLASHNEXT_Q38_QDUMP (dump aid of the fused attention's key lists; commit on r12-C), BUILD_OK sha 876272c8928b886b. Lease C acquired 12:23 (90 min).
  Scripts: shipC12.sh, kitC/runQ1.sh (dump + kitC/ql.py list statistics), kitC/mb/mb_qsa.cu + mb_qsa_main.h + run_mbq.sh (microbench of fn_attn_fused_kv8, generated copy with parts removed).
- 12:25 SOURCE READING, fn_attn_fused_kv8 (flashnext_attn.cu attn_chunks<1, ONE, FUSE, KV8>): one CTA of 128 threads per (query row, KV head); 24 query heads / 2 KV heads = 12 query
  heads a CTA in a 16-row mma tile (4 of 16 rows are padding); a sparse row reads its own list of 512 blocks x 4 keys (+ <= 3 tail keys), ascending, in tiles of 32 keys:
  fetch 32 x 256 fp8 key codes + scales into registers, widen to bf16 in shared memory, 16 mma a warp for scores, max / exp / sum per query head in a fixed tree,
  then the values take the keys' place (fetch + widen again), 16 mma a warp; 5 barriers a tile; 65 tiles a (row, head); chunks of 512 keys merged in the program.
  Every (row, head) CTA fetches and widens its 2,048 keys + values itself: 1.08 MB of codes a CTA = 2.2 GB a launch, served by the cache (neighbour rows share blocks).
- 12:27 MICROBENCH 6 (kitC/mb/mb_qsa.cu; the engine kernel with engine arguments, 1,024 rows, synthetic lists at 100K context, 10 of 512 blocks change a row; MEASURED ms a launch):
  | kernel | ms |
  | engine fn_attn_fused_kv8 (KPROF in the engine: 2.5-2.6) | 2.404 = 28.6 TFLOP/s |
  | copy of its body (bytes identical) | 2.423 |
  | widen without the fp8 -> bf16 conversion (plain store) | 2.331 |
  | no fetch, no widen | 1.620 (-0.78 = 32 %) |
  | scores mma 1/8 | 2.164 |
  | no value mma | 2.313 |
  | no exp | 2.273 |
  | no fetch / widen, scores 1/8 | 1.280 |
  | + no value mma | 1.167 |
  | + no exp (only barriers + softmax glue) | 0.893 (37 %) |
  READING: the mma are ~0.45 ms of 2.4 (the FLOP floor said 0.84 at 61.5 TFLOP/s: they run faster than that); the kernel is a third fetch + widen of codes every row
  repeats, and more than a third per-tile glue (5 barriers, per-element softmax, shared stores). A CTA that takes several query rows over the same staged keys would
  divide the fetch + widen and the barriers by the rows and fill the 4 padding rows of the mma tile (4 rows x 12 heads = 48 = 3 tiles). It is exact only if each row
  keeps its own 32-key tiles in its own list order: needs the real lists' alignment (QDUMP running).
- 12:30 REAL KEY LISTS (C40, QDUMP of 40 fused launches over a 27.5K and a 109K filler prompt; kitC/ql.py; MEASURED): neighbouring query rows differ in 115 (4K context) to
  303 (93K) of their 512 blocks (mean; p90 171-391); no two neighbouring rows have the same list (0.0 %); a group of 4 rows selects 777-1,371 distinct blocks (1.5-2.7x one row),
  a group of 8: 978-2,398; 32-key tiles that line up across 4 rows: 0.0-0.1 %. A launch of 1,024 rows selects EVERY block of the context (23,096 of 23,296 at 93K = 97.6 MB of
  codes for both heads), so the launch's keys do not stay in the 24 MiB cache at long context: that is the growth of this kernel with context.
- 12:33 MICROBENCH 6b (same kernel on the REAL lists; MEASURED ms a launch, 1,024 rows; every non-diagnostic variant byte-identical to the engine kernel, 12,582,912 bytes):
  | variant | 4K context | 93K context |
  | engine fn_attn_fused_kv8 | 2.38 | 3.20-3.23 |
  | no fetch + no widen (diagnostic) | 1.63 | 1.59 |
  | only barriers + softmax glue (diagnostic) | 0.85 | 0.83 |
  | constant scales (no scale loads; diagnostic) | 2.39 | 3.17 |
  | touch-prefetch of the codes 1 / 2 / 3 / 4 / 6 tiles ahead (cp.async into a dummy word, 32 B apart) | 3.71 / 3.62 / 3.61 / 3.61 / 3.59 | 4.41 / 4.25 / 4.26 / 4.33 / 4.29 |
  | the same 64 B apart, 2 / 3 / 4 ahead; keys only 3 ahead | 3.38 / 3.40 / 3.40; 3.41 | 4.17 / 4.14 / 4.13; 4.14 |
  | shared memory a CTA raised: <= 30,000 B (3 CTAs an SM) / 33,000-49,000 (2) / >= 51,300 (1) | 2.40 / 2.97 / 5.0 | 3.22 / 3.83 / 5.34 |
  | one register set for the staged codes + 13-row q / p layout + launch_bounds(128, 4): 128 registers, no spill, 24,800 B (4 CTAs fit) | 2.51 | 3.20 |
  | launch_bounds(128, 4) alone (compiler spills) / (128, 5) | 2.83 / 2.83 | 3.56 / 3.60 |
  READING: (1) the kernel uses 165 registers and 26.6 KB of the SM's ~100 KB shared memory: 3 CTAs an SM; a fourth CTA gives nothing (-1 % / +5 %).
  (2) touching the codes ahead of time makes it 30-55 % SLOWER: the cost of the fetch is the traffic from the cache into the SM (2.2 GB a launch), not load latency.
  (3) sharing staged keys between rows of one CTA cannot work with real lists: tiles never line up, a 4-row union is 1.5-2.7x one row, and a pool that holds the
  union window of 4 rows (keys + values, ~130-170 KB) does not fit an SM's 100 KB of shared memory; 2 rows a CTA would halve the rows an SM runs for a 27-40 % smaller fetch.
  VERDICT, LEVER 1 KERNEL: fn_attn_fused_kv8 is AT THE FLOOR OF ITS DATAFLOW. The round-11 "30 ms floor" counted only its mma (0.45 ms of a 2.4 ms launch); the real bound is
  each (row, KV head) gathering its own 1.08 MB of codes, because real neighbouring rows select different blocks. Nothing to integrate. DROPPED: row-group CTA, prefetch, 4 CTAs an SM.
- 12:44 KPROF AGAINST CONTEXT (C40, round-11 set, 3,072-row chunks, serialised launches, one 214K prompt; kitC/runQ2.sh kl + kp4.py; MEASURED ms a chunk; table 1 = 24-48K, table 8 = 190-214K):
  | kernel | at ~36K | at ~202K | growth | ms a call |
  | all kernels | 1,154.8 | 1,248.9 | +94.1 (8 %) | |
  | fn_q38_qsa_coarse16r (coarse scores, 144 calls a chunk) | 10.9 | 50.8 | +39.9 | 0.0745 -> 0.3525 |
  | fn_attn_fused_kv8 (36) | 90.4 | 109.2 | +18.8 | 2.49 -> 3.04 |
  | fn_q38_qsa_select_cert16l (144) | 19.8 | 36.9 | +17.1 | 0.135 -> 0.256 |
  | expert gate/up, prompt tiles, hc down, expert down, prompt tile f32 | | | +6.0 +4.1 +2.2 +2.0 +1.6 | |
  READING: the coarse scores are the largest growth. A call = 256 rows x 4 heads x 128 x ~50.5K blocks = 13.2 GFLOP (0.11 ms at 120 TFLOP/s), reads 12.9 MB of pooled keys and
  WRITES 25.9 MB of half scores that the select reads back (2 x 0.11 ms at the bus rate): 0.35 ms measured against ~0.28 for bytes + compute in sequence: near its floor in this
  dataflow. A fused coarse + cut kernel would have to compute the scores twice (histogram pass, list pass) to avoid the 26 MB round trip: no clear win; 1-byte coarse codes would
  need a new bound. NOT BUILT (ESTIMATE of the ceiling: -20..-30 ms a chunk at 200K = 1.6-2.4 % there, under 1 % averaged over a 214K prompt).
- 12:46 PASS SCRATCH ACCOUNT (cuda_batch_state.zig Scratch.alloc; bytes a row of a chunk; who uses it past 3,072 rows; source reading):
  | buffer | bytes a row | used by a wide prompt chunk? |
  | head_local | 248,320 | NO: written only by the two-rank head (collective) and the draft head's lanes (lanes x draft ids x 2 bytes) |
  | y (pair rows fp32) | 112,640 | yes (write-back reads them) |
  | native_part (8 K slices x dims fp32) | 81,920 | NO: the native lane reduces in a cluster (sm >= 9.0) or in the block from 256 rows; only < 256-row calls with > 8 slices use it |
  | part (8 output slices) | 81,920 | 1 of 8 slices (from 64 rows the slices fold in the block; the native lane writes one) |
  | hc_part (32 slices x 324 fp32) | 41,472 | 1 of 32 (fused read-out tiles); the generic 32-slice path is the fallback (now guarded) |
  | ple_part (4 slices) | 40,960 | 1 of 4 |
  | proj | 32,960 | yes |
  | gq gk gv ys ggb (unfused DeltaNet rows) | 65,920 | not by fn_gdn_fused; lanes / late keep use them: LEFT ALONE |
  | h, norm, hc_up, ple_keys, ple_gated, ple_nrow (6 x 20,480) | 122,880 | yes / left alone |
  | everything else | ~115,000 | yes |
  | Scratch total | ~944,000 | |
  | + each lane's Draft.streams (chunk x 20,480 B x 5 lanes) | 102,400 | (kept hidden rows; left alone) |
  => LEVER 2 (code written, commit on r12-C): TF_FLASHNEXT_Q38_SCRATCH_TRIM=N (recipe value 3072): slices kept for N rows, one slice a row past N (hc_part, part, ple_part),
  native_part N rows, head_local 256 rows; TF_FLASHNEXT_Q38_SCRATCH_CHECK=1 puts 4,096 guard bytes behind each trimmed buffer and reads them back every pass ("q38 scratch
  check ... bad="). Users that could exceed a trimmed buffer return error.UnexpectedTensor (generic 32-slice read-out past N rows, two-rank head past 256 rows, draft lanes).
  ESTIMATE: -0.65 GiB at any chunk (head_local) and -473 KB a row past 3,072: CHUNK=6144 costs ~1.0 GiB over today's default instead of 3.1; 8192 ~2.2 GiB. No arithmetic touched.
- 12:49 KPROF OF A DECODE STEP AGAINST CONTEXT (C40, no-drafts engine, one row a step, serialised launches; kp4.py; MEASURED ms a step after 1.9K / after 182K tokens; 21.8 / 20.4 tok/s):
  all kernels 40.59 -> 43.72 (+3.13): qsa_tile::scores (the stock exact block scores, 12 calls) 0.006 -> 1.257; fn_qsa_select 0.074 -> 1.073; fn_attn_chunks_kv8 0.443 -> 0.886
  (2,051 selected keys gathered from a 100 MB cache instead of 1,908 resident ones: inherent); everything else +0.4. So 2.25 of the 3.1 ms are the indexer's scores + select of
  ONE row over 45K blocks, 12 layers: 0.105 + 0.089 ms a layer. Round 9 measured the certified path for decode rows (CERT_MIN=1) as no gain, but that was before the listed
  select and the by-rows coarse kernel: re-test queued (flag only: TF_FLASHNEXT_Q38_QSA_CERT_MIN=1).
- 12:52 C41 = r12-C HEAD (QDUMP aid + scratch trim) BUILD_OK sha b323a5a0a1f34b2f. Lease C renewed to 14:51. chainQ3.sh C41 queued (log box ~/q38/runs/C_q3.log; kitC/runQ3.sh):
  tchk (full list, CHUNK=6144 + TRIM=3072 + SCRATCH_CHECK) / dec64 / dec1 (CERT_MIN=1) / p46a / p61t / p81t / p46b (prefill pairs, R11 at 4,608 first and last) / ag46 / ag61t / ag81t.
  First line: with TRIM=3072 the arena at CHUNK=6144 is 28,462 MiB (untrimmed 6,144: 26,406-26,700; 4,608: 27,868; default 3,072: 29,264).
- 12:55 LEVER 3 SOURCE READING: the prompt hyper-connection kernels (flashnext_hcf.cu hcn::down_body / upmix_body = fn_hcn_down4, fn_hcn_up3_k5) ARE already the fused form
  (fn_hc_normalize2 formed in shared memory before the mma; sigmoid gate, norm and mix in the epilogue); upstream also ships its second down form (norm between the k16
  steps, fn_hcn_dv2_4) and debug forms (projection alone, epilogue alone). What is left is the per-element scalar work itself. One exact idea: bsig(x) = bf16(sigmoid(x)) of a
  bf16 x has only 65,536 possible inputs: a table of the device's own results (3,072 entries for 2^-9 <= |x| < 8, computed outside it) gives the same bits and can be proven
  on all 65,536 inputs. Microbench prepared: kitC/mb/mb_hcu.cu (up + mix: engine, debug forms, table in shared / global memory), mb_hcd.cu extended with dv2_4 and dv2_4_nx.
- 13:05 SCRATCH TRIM CHECK RUN (C41 sha b323a5a0a1f34b2f, fresh engine, round-11 set + CHUNK=6144 + SCRATCH_TRIM=3072 + SCRATCH_CHECK=1, the FULL request list; MEASURED):
  guard bytes behind the 5 trimmed buffers read back after every pass: 4,480 passes (prompt chunks of up to 6,144 rows, decode rounds, head passes), bad=0.
  prefill 7,276 / 27,502 / 109,142 / 214,471 first token 2.70 / 9.71 / 39.68 / 80.23 s = 2697 / 2831 / 2751 / 2673 tok/s; decode after 1.9K / 54.6K / 109K / 182K
  44.4 / 48.5 / 43.8 / 42.1 tok/s; needle 3/3 same answer 39.9 s; short 96.1 ms; ident median 65.5. Arena 28,462 MiB (peak 24,553), served 2,762, fallback 0.
  IDENTITY against the base binary's run (C_y_ref_C20): ident 51/51; warm 1/1, prefill 4/4, decode 4/4, needle 1/1, short 24/24 texts identical; score hashes after the
  prompt pass 74/74, at release 74/74; certified select fallback 0; crash lines 0.
- 13:17 DECODE WITH THE CERTIFIED SELECT FOR DECODE ROWS (C41, drafts on, fresh engines, R11 at 4,608 rows; 150 tokens of the same task after 1.9K / 109K / 182K; MEASURED):
  | | 1.9K | 109K | 182K |
  | CERT_MIN default 64 (decode rows on the stock scores + select) | 44.1 tok/s | 43.5 | 40.9 |
  | TF_FLASHNEXT_Q38_QSA_CERT_MIN=1 | 44.5 | 44.3 (+1.8 %) | 41.9 (+2.4 %) |
  texts 3/3 identical; 3,080 more rows took the certified path, fallback 0. One pair only: reference-last + device check (QSA_CERT=2 on decode rows) queued (C_q3b.log).
- 13:34 PREFILL AGAINST THE CHUNK WITH THE TRIM (C41, fresh engines, R11 set = round-10 set + GDN_PROD=2 + ROUTE_BAND=2048; first token s / tok/s; MEASURED; arena = room for kept states):
  | set | 7,276 | 27,502 | 109,142 | 214,471 | arena MiB |
  | R11 + CHUNK=4608 (round-11 proposal), FIRST | 2.71 s 2685 | (9.89 s 2780, round 11) | 40.41 s 2701 | 81.95 s 2617 | 27,684 |
  | R11 + CHUNK=6144 + SCRATCH_TRIM=3072 | 2.71 s 2688 | 9.76 s 2818 | 39.78 s 2744 | 80.43 s 2667 | 28,592 |
  | R11 + CHUNK=8192 + SCRATCH_TRIM=3072 (starts now) | 2.65 s 2743 | 9.89 s 2781 | 40.18 s 2716 | 81.33 s 2637 | 27,388 |
  texts 5/5, score hashes 5/5 + 5/5 against C_y_ref_C20 in every run; crash 0. READING: with the trim, 6,144 rows cost LESS memory than the untrimmed 4,608 (+0.9 GiB of arena)
  and are +0.1 / +1.4 / +1.6 / +1.9 % faster; 8,192 starts but is slower than 6,144 on long prompts (some kernel leaves the cache again, as the router did at 6,144):
  KPROF at 6,144 against 8,192 queued (C_q4.log) to find it.
- 13:40 p46b = REFERENCE LAST (R11 + CHUNK=4608): 2.71 / 9.98 / 40.73 / 82.36 s = 2684 / 2756 / 2680 / 2604 tok/s (first: 2.71 / 9.91 / 40.41 / 81.95 = 2685 / 2774 / 2701 / 2617).
  PAIRED (C41, fresh engines, reference first and last; MEASURED): CHUNK=6144 + SCRATCH_TRIM=3072 against the mean reference: 7.3K 0.0 %, 27.5K -1.9 % time, 109K -1.9 %, 214K -2.1 %.
- 13:51 AGENTS (C41, 5 agents x 56K own system prompt x 3 turns, fresh engines; MEASURED):
  | | R11 + CHUNK=4608 (round-11 proposal) | R11 + CHUNK=6144 + SCRATCH_TRIM=3072 |
  | wall; sum of first-token waits | 114.8 s; 372.2 s | 112.8 s (-1.7 %); 358.7 s (-3.6 %) |
  | follow-up first token median / max | 0.81 / 15.97 s | 0.84 / 11.68 s |
  | prompt cache at the end | 13 states, 13.7 of 21.5 GiB, hits 10, misses 5, evicted 17 | 13 states, 13.7 of 22.3 GiB, hits 10, misses 5, evicted 17 |
  | arena | 27,784 MiB | 28,700 MiB |
  texts 15/15 + short 24/24 identical; score hashes after the prompt pass 27/27 (release-time 13/27: concurrent schedule noise as in rounds 8-11); crash 0.
  => NO HIT LOST at 6,144 rows with the trim (round 11's untrimmed 6,144 lost one and its wall was +12 %). 6,144 + trim is the new proposed chunk.
- 13:57 AGENTS AT 8,192 ROWS + TRIM (same trace): wall 113.5 s; sum of first-token waits 346.4 s (-6.9 %); follow-up max 2.44 s; cache 12 states, 12.7 of 20.9 GiB, hits 10, misses 5,
  evicted 18; arena 27,366 MiB; texts 15/15 + 24/24, score hashes after the prompt pass 27/27. No hit lost either, but single long prompts are slower than at 6,144 (13:34).
- 14:03 DEVICE CHECK OF THE CERTIFIED SELECT ON DECODE ROWS (C41, R11 at 4,608 + QSA_CERT_MIN=1 + QSA_CERT=2; a 109K and a 182K prompt + 3 x 150 decoded tokens; MEASURED):
  3,490,048 row selections (the prompt rows AND the 3,080 decode / draft rows) compared on the device against the stock scores + select: list_diff 0, exact_bit_diff 0,
  bound_viol 0, fallback 0, largest |F - exact| / bound 0.839; texts 3/3 identical to the CERT_MIN=64 run. crash 0.
  Only lease C is live since 14:00 (A, B, D released): finishing with the minimum: KPROF 6,144 / 8,192 (running), two microbenches, one full list on the final set, one reference decode.
- 14:04 WINDOW CUT: the shared window reached its cap (restore started 14:03; lease C dropped by the harness). k61's engine was killed at its start and k81's engine had
  started at 14:04:00 INTO the closing window: I stopped it and every queued script at 14:05 (55 s of an engine start beside the restore; nothing measured is affected;
  no q38-run-C, no script, no microbench left). NOT YET RUN: KPROF 6,144 / 8,192, the two hyper-connection microbenches, the full list on the final set, the reference decode.
  Plan: when DeepSeek is back (health 200) acquire once more for ~30 min: kitC/runQ6.sh "mb fin dec kp".
- 14:06 HANDOFF CUT (Mac q38/handoff/C/r12-prefill/C-r12.patch = git diff b37b4e0..HEAD of wt/C12; mkhandoffC12.sh). Binary of exactly this tree: C41 sha b323a5a0a1f34b2f.
- 14:10 SIDE EFFECT OF THE CUT, MY FAULT (to tell the owner): the restore's log (head ~/q38/runs/S140923/run.log) says DeepSeek came back at 14:08:00 healthy (health 200,
  answer check passed) but with "session store: 3.13 GiB on every rank (at most 12.0; free device memory before it, ranks 0-3: 18.21 / 27.76 / 5.13 / 27.89 GiB)". The rank
  with 5.13 GiB free is very probably spark-4e5c: my queued chain started the k81 engine at 14:04:00, right after the window's stop-all, and it held ~22 GiB of GPU memory
  until I killed it at 14:05:05, while DeepSeek's rank was sizing its session store. So production is UP but with a smaller session store than usual until its next restart.
  I do not restart DeepSeek myself. CAUSE: my chain scripts checked only "GPU holds < 20 GB" before an engine start, not the lease. FIX (kitC/runQ4.sh, runQ6.sh): every
  engine start now needs the stamp ~/q38/runs/C_LEASE_UNTIL (epoch of my lease's or the window cap's end, minus 6 min). LESSON for every builder: a window's cap = its start
  + 280 min whatever the leases say (S140923 started 09:23:41 -> cut 14:03:41); read the unit's start before queueing chains near it.
- 14:11 q38-win failed/inactive, q38-guard inactive, DeepSeek health 200, CLOSING still there = the stale flag of shared/NOTES.md: removing it (the documented fix), then one more
  window of ~45 min (task: "acquire again if cut"). Its release restarts DeepSeek on clean boxes; I will read the restore log's session-store line afterwards.
- 14:11 new window (the stale CLOSING removed per shared/NOTES.md; leases A, C, D). Lease C until 14:55:39; stamp written on the box; runQ6.sh "mb fin dec kp" started (log box ~/q38/runs/C_q6.log).
- 14:12 MICROBENCH 7 (kitC/mb/mb_hcd.cu, the engine's hyper-connection down kernels, engine arguments, 3,072 rows; MEASURED ms a call; bytes against fn_hcn_down4):
  | kernel | ms | bytes differing |
  | engine fn_hcn_down4 | 0.615 | - |
  | engine fn_hcn_dv2_4 (upstream's second form: the norm between the k16 steps) | 0.655 | 0 |
  | engine fn_hcn_dv2_4_nx (upstream's benchmark form WITHOUT the norm: loads + mma only) | 0.468 | (other values) |
  READING: the per-element norm (2 multiplies + a bf16 rounding on 31.5 M values a call) costs 0.15 ms of 0.615 (24 %); without it the kernel still takes 0.468 ms =
  43 TFLOP/s (two CTAs a row tile, each staging the same H rows). The round-11 floor (0.31 ms a call, "30 ms a chunk") is not reachable by removing scalar work: the most an
  exact norm trick could give is 0.15 ms x 92 calls = 13.5 ms a chunk (1.2 %), and the norm is two fp32 multiplies of three varying operands: no table, no shortcut.
  VERDICT: hyper-connection DOWN: AT THE FLOOR OF ITS DESIGN (the fused prologue form already exists upstream; its second form is slower). Nothing built.
- 14:25 FULL REQUEST LIST ON THE FINAL SET (C41 sha b323a5a0a1f34b2f = handoff tree 21f872a; fresh engine; R11 set + CHUNK=6144 + SCRATCH_TRIM=3072 + QSA_CERT_MIN=1; MEASURED):
  prefill 7,276 / 27,502 / 109,142 / 214,471 first token 2.70 / 9.76 / 39.88 / 80.68 s = 2695 / 2817 / 2737 / 2658 tok/s; decode after 1.9K / 54.6K / 109K / 182K (300 tokens,
  drafts on) 44.3 / 48.3 / 44.1 / 42.8 tok/s (the same list without CERT_MIN=1 at 13:05: 44.4 / 48.5 / 43.8 / 42.1); first token of the decode task 19.93 / 40.25 / 68.06 s;
  needle 109,126 tokens 40.2 s 3/3 same answer; short new prompt 97.2 ms (median of 12); ident median 65.0 tok/s. Arena 28,270 MiB, served 2,762, fallback 0, peak 24,553.
  IDENTITY against the base binary's run (C_y_ref_C20): ident 51/51; warm 1/1, prefill 4/4, decode 4/4, needle 1/1, short 24/24 texts identical; score hashes after the prompt
  pass 74/74, at release 74/74; certified select 11,271,871 rows (decode rows included), fallback 0; crash lines 0.
- 14:40 LEASE C RELEASED (14:40:01). spark-4e5c: no q38-run-C, no script, no microbench, 0 GPU processes. Leases A, B, D are still live (the window stays open for them).
- 14:31 dec64b = REFERENCE LAST for the decode pair (CERT_MIN default, 150 tokens): 44.2 / 43.6 / 40.9 tok/s after 1.9K / 109K / 182K (first: 44.1 / 43.5 / 40.9): the ruler repeats
  to 0.1 tok/s, so CERT_MIN=1 (44.5 / 44.3 / 41.9) = +0.8 % (no scores at 1.9K: noise) / +1.7 % / +2.4 %.
- 14:39 MICROBENCH 8 (kitC/mb/mb_hcu.cu; the engine's hyper-connection up + mix kernel fn_hcn_up3_k5, engine arguments, 3,072 rows; MEASURED ms a call; two input scales):
  | kernel | run 1 | run 2 (larger weights) | bytes differing |
  | engine fn_hcn_up3_k5 | 0.610 | 0.605 | - |
  | engine debug form: projection alone | 0.431 | 0.430 | (timing) |
  | engine debug form: epilogue alone | 0.453 | 0.437 | (timing) |
  | no sigmoid (diagnostic) | 0.543 | 0.534 | (timing) |
  | sigmoid from a 3,072-entry table in shared memory (computed outside 2^-9 <= abs(x) < 8) | 0.578 (-5 %) | 0.671 (+11 %) | 0 of 16,711,680 |
  | the table in global memory | 0.682 | 0.749 | 0 |
  The table was proven on all 65,536 bf16 inputs (bits differing 0). READING: the sigmoid is 11 % of the kernel; a table saves 5 % when the inputs stay in its range and
  costs 11 % when they do not (run 2: the computed branch runs beside the table in most warps). Best case 3 ms a chunk (0.3 %). DROPPED.
  VERDICT: hyper-connection UP + MIX: at the floor of its design (projection alone 0.43, epilogue alone 0.44-0.45, together 0.61: they already overlap).
- 14:40 KPROF AT 6,144 AGAINST 8,192 ROWS (C41, both with the trim, 109K prompt, serialised launches; ms per 3,072 rows; MEASURED): all kernels 1,122.0 / 1,120.6;
  expert gate/up 178.3 -> 168.9, expert down 138.9 -> 135.0, BUT the hyper-connection down kernel changes: at 6,144 rows fn_hcn_down4 63.0 ms; at 8,192 rows the engine
  picks fn_hcn_down2 (the 32-row tile): 87.4 ms (+24). CAUSE (cuda_tiles.zig pickDown): the tile is chosen by "waves x rows a tile" with a wave's cost taken as
  proportional to the tile height; the microbench says a 64-row wave costs 1.53x a 32-row wave, not 2x (0.615 against 0.805 ms per 3,072 rows). For 8,192 rows the rule
  gives 22 (32-row) against 24 (64-row) and takes the slower tile; the same happens for other row counts above 3,072 (e.g. a 5,000-row tail).
  => LEVER 4 (to build): TF_FLASHNEXT_Q38_HC_DN_TALL=N: passes of at least N rows take the 64-row down tile (both tiles are stock kernels with the same bytes: microbench
  0 of 3,932,160 bytes differ). ESTIMATE: -20 ms per 3,072 rows at CHUNK=8192 (-1.8 %), which would put 8,192 ahead of 6,144.
- 14:45 C42 = r12-C HEAD 3rd commit (+ TF_FLASHNEXT_Q38_HC_DN_TALL, _CHECK) BUILD_OK sha 718a78b857fb1152. Lease C acquired again (until 15:45:12; the window is open for A, B, D
  anyway; its unit started 14:10:33, cap 18:50). runQ4.sh C42 started (log box ~/q38/runs/C_q7.log): chk8 (CHUNK=8192 + trim + HC_DN_TALL=3072 + its check) / p61r1 (reference:
  CHUNK=6144 + trim) / p81f (8192 + trim + HC_DN_TALL=3072) / p61r2 (reference last).
- 14:51 C42 CHECK MODE (CHUNK=8192 + SCRATCH_TRIM=3072 + HC_DN_TALL=3072 + HC_DN_TALL_CHECK=1; fresh engine; prompts 2.8K / 7.3K / 27.5K / 109K / 214K; MEASURED): in the 4,032
  read-outs where the flag changes the tile, the stock tile ran first into scratch and every word of the two outputs was compared on the device: 10,701,766,656 words, 0 differ.
  Texts 5/5, score hashes 5/5 + 5/5 identical to C_y_ref_C20. crash 0.
- 15:09 PREFILL, THIRD WINDOW (C42, fresh engines, reference = R11 + CHUNK=6144 + SCRATCH_TRIM=3072 first and last; MEASURED):
  | prompt tokens | reference first | 8,192 + trim + HC_DN_TALL=3072 | reference last |
  | 7,276 | 2.73 s 2666 tok/s | 2.68 s 2718 (+1.8 %) | 2.72 s 2673 |
  | 27,502 | 9.83 s 2798 | 9.80 s 2807 (+0.3 %) | 9.82 s 2801 |
  | 109,142 | 40.02 s 2727 | 39.87 s 2737 (+0.4 %) | 40.03 s 2726 |
  | 214,471 | 80.80 s 2654 | 80.54 s 2663 (+0.3 %) | 80.77 s 2655 |
  arena 27,938-28,294 MiB (6,144) against 26,944 (8,192). Texts 5/5, score hashes 5/5 + 5/5 in every run. READING: HC_DN_TALL gives 8,192-row chunks +1.0 % on long prompts
  (2781 / 2716 / 2637 at 13:34 -> 2807 / 2737 / 2663; ESTIMATE was -1.8 % of a chunk: half arrived), which only brings them level with 6,144 (+0.3..0.4 %, the ruler's edge)
  at ~1.2 GiB more memory. 6,144 stays the proposed chunk; HC_DN_TALL is proposed ON (exact, no cost; it also fixes tails of 3,072+ rows).
## ROUND 12 END STATE (15:15 UTC 2026-11-14)
- Lease C RELEASED 15:11:18 (third and last time). spark-4e5c: no q38-run-C container, no run script, no build, no microbench, 0 GPU processes. Raw results: Mac q38/results/rC12.tgz
  (jsons + run logs; engine logs stay on the box under ~/q38/runs/C_q3_* / C_q4_*). Leases A, B, D live at that time.
- HANDOFF (Mac q38/handoff/C/r12-prefill/): C-r12.patch (git diff b37b4e0..a27a02c, sha256 377dde7f58ea7b35, 381 lines, 7 files, TREE_EQUAL) + NOTE.md.
  Binary of exactly that tree: C42 sha 718a78b857fb1152; C41 sha b323a5a0a1f34b2f = the tree without the last commit (most runs). Nothing pushed by C.
- SHIPPED-READY (exact; flags default off): TF_FLASHNEXT_Q38_SCRATCH_TRIM=3072 + TF_FLASHNEXT_CHUNK=6144 (prefill +0.0 / +1.9 / +2.0 / +2.2 % over round 11's 4,608 at
  7.3K / 27.5K / 109K / 214K, 0.9 GiB MORE arena than 4,608, 5-agent trace wall -1.7 % with every cache hit kept); TF_FLASHNEXT_Q38_QSA_CERT_MIN=1 (existing knob: decode +1.7 % at
  109K, +2.4 % at 182K, 0 at short context); TF_FLASHNEXT_Q38_HC_DN_TALL=3072 (+1.0 % at CHUNK=8192; ~0 at 6,144).
  Final set on the full list (C41): 2695 / 2817 / 2737 / 2658 tok/s prefill; decode 44.3 / 48.3 / 44.1 / 42.8 after 1.9K / 54.6K / 109K / 182K; short prompt 97.2 ms; ident 51/51.
- AT THE FLOOR, measured (nothing to build): QSA fused attention (real lists share too little: every row gathers its own 1 MB of codes); hyper-connection down (the norm is
  24 %, no exact shortcut) and up + mix (sigmoid table: -5 % / +11 %).
- DROPPED (measured): attention prefetch, 4 CTAs an SM, row-group CTA; sigmoid table; CHUNK=8192 as default (level with 6,144, more memory).
- WHAT REMAINS, by size: 1 coarse scores at long context (50 ms a chunk at 200K: the 26 MB a call half-score round trip; a fused coarse + cut, or 1-byte codes with a new
  bound: ESTIMATE up to -20..-30 ms a chunk at 200K); 2 pickDown's wave rule below 3,072 rows (a 64-row wave costs 1.53x a 32-row one: re-fit and measure 768-3,072 rows);
  3 the lanes' Draft.streams (5 x chunk x 20 KB) and the unfused DeltaNet rows (66 KB a row) in the scratch: ~0.5 GiB more at 6,144; 4 CERT_MIN=1 with 16 users (not
  measured); 5 the integrator's re-runs on 923b (trim check, CERT=2 with CERT_MIN=1, prefill pair); 6 expert kernels' bus / compute overlap (110 ms gap, no idea).
- LESSONS: (1) a FLOP floor is not a floor: replay REAL data (key lists) in the microbench before designing a kernel: the attention "30 ms floor" assumed shared gathers
  that real rows do not have. (2) Re-profile per kernel after every geometry change: 8,192 rows flipped a tile choice (+24 ms) exactly as 6,144 rows hurt the router in
  round 11. (3) Before trimming memory, read who writes each buffer and add guard bytes + error returns: 45 % of the pass scratch was slices nobody writes at prompt
  geometry. (4) A lever that measured zero two rounds ago (CERT_MIN=1) can pay after its kernels changed: re-test old zeros when their inputs change. (5) The window has a
  hard cap (start + 280 min): chains must check the lease before every engine start, not only the GPU.
- RESUME: cd ~/.hermes/cache/scratch/q38; tail -90 PROGRESS-C.md; ./shipC12.sh NAME 16 24g; ./pushkit.sh; ./mkhandoffC12.sh; box (after `lease.sh acquire C N` on the head AND
  `date -u -d "HH:MM:SS" +%s > ~/q38/runs/C_LEASE_UNTIL` on the box): bash ~/q38/kit-C/runQ4.sh DEADLINE_EPOCH BIN "NAME:one:PARTS:FLAGSET ..." (see its header),
  runQ1.sh BIN (key list dump + ql.py), runQ2.sh BIN "kl kd" (KPROF against context; kp4.py), microbenches: bash ~/q38/kit-C/mb/run_mbq.sh mb_qsa.cu /prof/qdumpC_C40.bin 39 |
  mb_hcu.cu 1 ; run_mbx.sh mb_hcd.cu.

# Round 13 (builder C; agent serving path: shared prefix state, eviction/pin, short-prompt tiles, coarse scores)
- 15:15 UTC start. Base a27a02c, worktree q38/wt/C13 (branch r13-C). No lease held. Reading code (prompt cache / snapshot) before measuring.
- 15:19 BASELINE ARITHMETIC from the round-12 5-agent log (box ~/q38/runs/C_q3_ag61t_C41/engine.log; MEASURED log, computed): a cold 52K request keeps 4 marks
  (48685 / 50221 / 50732 / 52144 = 4 whole copies, 4,074 MiB) whose attention rows are prefixes of each other: distinct bytes ~1.0 GiB + 4 x 110 MiB fixed, so ~64 % of the kept
  bytes are duplicate prefix. After turn 1 of 5 agents: 19 states 19.4 of 19.6 GiB (full, evictions start). keep_ms ~10 ms a 1 GiB state (arena), restore+open 25.8 ms.
  Follow-up at 52K: open 25.8 ms + ONE pass of 1,479 rows 706 ms + first 0.3 ms: the follow-up first token IS the partial pass (fit: ~220 ms + 0.318 ms a row; a full 6,144-row
  chunk is 0.354 ms a row). So: memory lever = delta states (share attention rows between marks); time lever = the partial pass itself (lever 4).
- DESIGN (delta states, TF_FLASHNEXT_Q38_SNAP_DELTA=1; =2 check): Snap = fixed buffer (conv + DeltaNet state + n-gram tail, 110 MiB, a copy) + ref-counted segment chain of
  attention rows [from, at) (kc, vc, ikc, pooled, head rows); base = the lane base (state restored from, or last kept by this lane). Evicted states whose segment a child
  still needs: bytes reported as family spare (orphans) so the budget stays honest. Check mode: at each save the lane rows [0, from) are compared with the chain (host memcmp).
- 15:24 C50 building (delta states: cuda_snap.zig Seg chain, cuda_lanes.zig Lane.base + snap_need/charged/spare, core prompt_cache.zig need hook + reserve estimate,
  prompt_cache_policy.zig q38Delta, cuda_snaps.zig, cuda_rank_engine.zig signature). New kit: kitC/at.py (agent traces: per-turn prompt / resumed / first token; seeded tool
  results 300-2,000 tokens, 0.3-2 s tool pause), kitC/atlog.py (cache side from the engine log + duplicate-prefix arithmetic), kitC/runR1.sh (configs, lease stamp check
  before every engine start, MemAvailable low-water sampler). Mac scripts shipC13.sh / mkhandoffC13.sh. Window now open since 14:10:33 UTC (cap use until 18:40).
- 15:36 C50 BUILD_OK sha 10c6ab5af1c3efe6 (first compile). Lease C acquired 15:27 (until 16:57; window cap 18:40). runR1.sh C50 started on the box (log ~/q38/runs/C_r1.log):
  ref (r12 set) / chk (SNAP_DELTA=2) / dl (SNAP_DELTA=1, + trace D) / ref2; traces A = 1 agent 20K x 6 turns, B = 5 agents x 56K x 6, C = 16 agents x 20K x 4, D = 16 x 20K with 8K shared.
- 15:36 REFERENCE so far (C50 flags off = r12 set; MEASURED): A: follow-ups 0.50-0.79 s, 9 states 4.3 GiB for ONE 24.5K conversation (67 % duplicate prefix);
  B: 5 x 56K: 25/25 follow-ups resumed, median 0.99 s max 2.83 s, wall 118.3 s, held max 16.5 GiB, 43 evictions, 80 % of planned keep bytes are duplicate prefix.
  PARTIAL PASS COST (follow-ups; MEASURED): 460 rows 376 ms, 732 rows 487, 1273 rows 628, 1988 rows 857 => ~260 ms + 0.30 ms a row. ESTIMATE of the cause: a pass of 500+ rows
  touches every expert of every layer, i.e. reads the whole 77 GiB model once (~260 ms at the bus rate); KPROF by pass size will check it (runR2.sh, parts c / k).
  IDEA (not built, sized later): follow-up passes of several agents that wait together merged into ONE pass would save ~260 ms each (PREFILL_MANY takes only fresh prompts <= 256 rows).
- 15:46 REFERENCE C (16 agents x 20K x 4 turns, default profile 5 lanes; MEASURED): 48/48 follow-ups resumed, follow-up median 10.94 s (queueing on 5 lanes: turn 1 36 s, turn 3 8.8 s),
  wall 172.8 s, held max 21.4 GiB (budget 20.5-23.3), 128 evictions, 63 % duplicate prefix. ident 51 saved, short 96.7 ms. MemAvailable low-water 7.7 GiB.
- 15:46 FIRST DELTA RESULT (chk = SNAP_DELTA=2, trace A; MEASURED): 9 keeps, 8 extend a kept state, kept 1,433 MiB where whole copies take 4,300 MiB (33.3 %); check mode compared
  3,469,058,688 bytes of the streams rows with the shared chains: 0 differ. (Check mode costs ~130 ms a pass: timing comes from the dl run.)
- THINKING, written down so it is not lost: (1) under memory pressure the stock order already evicts interior marks first, so the capacity LIMIT in conversations is the same with
  delta states (each conversation needs its rows once + one fixed part either way); delta buys 2-3x fewer held bytes off-pressure, fewer evictions, cheap keeps, shared system-prompt
  rows across agents, not more hits at the limit. (2) The duplicate that remains is kept state vs LIVE lane (needs paged KV in the kernels: not attempted). (3) upstream has a disk tier
  (--learn-conversations, core/rows_store.zig, block files shared between states) wired for Nemotron only on CUDA; a Flash Next CUDA image (parts / gather / scatter / blank, ~170 lines
  as native/cuda_disk.zig) would let evicted conversations come back at disk speed (~0.5 s a GiB vs 21 s re-prefill at 56K): NOT built this round (page cache on a unified-memory box
  next to DeepSeek restarts is the risk to design for: O_DIRECT). (4) merging waiting follow-up passes (each pays ~260 ms fixed): Engine.prompts takes lanes of <= 256 rows without marks;
  agents need multi-block lanes + a mark per lane: NOT built. (5) the pool profile has no elastic cache: added TF_FLASHNEXT_Q38_CACHE_ELASTIC=2 (in tree, builds as C51).
- 15:57 CHECK-MODE RUN DONE (C50, r12 set + SNAP_DELTA=2, fresh engine, same requests as ref; MEASURED): every keep compared the stream rows [0, at) with the shared chain on the host:
  91,217,281,536 bytes, 0 differ (227 keeps, 206 extend a kept state; longest chain 11). Texts vs ref: trace A 6/6, B 30/30, C 64/64, ident 51/51, short 24/24; score hashes after the
  prompt pass 160/160 (release-time 66/160 = concurrent schedule noise as in every round). crash 0, arena fallback 0.
  Memory: A (1 agent, 24.5K): 9 kept states 1,433 MiB vs 4,300 MiB whole (33.3 %); B (5 x 56K x 6 turns): 54 states ALL kept in 11,231 MiB, 0 evictions (ref: 43 evictions, max 16.5 GiB
  held, 21 states); as whole copies those 54 would be 52,375 MiB (21.4 %). C after A+B in the same engine: cache full either way (fixed parts: 100 states x 110 MiB), 82 evictions vs 128.
  Per kept turn: 110 MiB fixed + 18.9 KB a NEW token (a 1,200-token turn: ~132 MiB) instead of 110 MiB + 18.9 KB x the whole history (1.12 GiB at 56K).
- 16:11 DELTA TIMING RUN DONE (dl = C50, r12 set + SNAP_DELTA=1, fresh engine; MEASURED; ref = same binary flags off, first):
  | trace | ref | SNAP_DELTA=1 |
  | A 1 agent 20K x 6: follow-up first token (5 turns) | 0.66 0.79 0.63 0.54 0.50 s | 0.66 0.78 0.63 0.55 0.49 s |
  | A: kept states, MiB held | 9, 4,300 | 9, 1,433 |
  | B 5 x 56K x 6: follow-up median / max; wall; sum of first-token waits | 0.99 / 2.83 s; 118.3 s; 355.5 s | 1.04 / 3.37 s; 119.9 s; 363.2 s |
  | B: states at end, MiB held max, evictions | 11 (21 max), 16,522, 43 | 54, 11,231, 0 |
  | C 16 x 20K x 4 (5 lanes): follow-up median / max; wall; sum of waits | 10.94 / 84.2 s; 172.8 s; 2,044 s | 10.62 / 86.5 s; 165.8 s; 1,938 s |
  | C: resumed follow-ups; evictions | 48/48; 128 | 48/48; 82 |
  | partial pass 460 / 732 / 1273 / 1990 rows (B) | 376 / 487 / 628 / 857 ms | 367 / 477 / 619 / 855 ms |
  | open + restore of a resumed state: 20K / 56K | 13.4 / 26.1 ms | 14.3 / 27.0 ms |
  | short new prompt | 96.7 ms | (next line) |
  D (16 x 20K, 8K shared system block, delta only): 48/48 resumed, median 9.80 s, wall 169.7 s; shared rows whose own state was evicted (orphans) up to 7.3 GiB, counted in the budget.
  Identity vs ref: A 6/6, B 30/30, C 64/64, ident 51/51, short 24/24, score hashes after the prompt pass 160/160. crash 0. MemAvailable low-water 8.2 GiB (ref 7.7: the arena is taken
  once either way, so the box does not see the saving; the cache does).
  READING: time-neutral at one user (keeps were already ~10 ms), -1..2 % on partial passes with keeps, memory of kept states 21-33 % of whole copies, evictions 43 -> 0 (5 agents).
- 16:20 ref2 (flags off, last; MEASURED): A follow-ups 0.67 0.78 0.63 0.55 0.49 s; B median 1.00 s max 11.47 s, wall 121.3 s, 40 evictions, held max 18.8 GiB. So B wall: ref 118.3 / delta 119.9 / ref 121.3: no time difference.
- 16:20 C51 BUILD_OK sha fd78b9de483847ba = HEAD 8e2df5c (delta states + TF_FLASHNEXT_Q38_CACHE_ELASTIC=2: elastic cache with the pool). Lease C renewed to 17:57:10.
  runR1.sh C51 chain started (log ~/q38/runs/C_r1b.log): uref (u16 profile, r12 set) / udl (u16 + ELASTIC=2 + DELTA=1) / eref (one: B, E = 10 agents x 56K x 3, ident, short) / edl (same + DELTA=1) /
  cold (fresh engine: each agent s LAST request of B again, cold: text must equal the resumed one) / uel (u16 + ELASTIC=2 only).
- 16:21 POLICY SIMULATION (kitC/csim.py; SIMULATED on the logged ref traces B and C: arrival order, the marks each request planned, fixed budget, stock eviction order):
  | trace, budget | whole copies | delta | delta, last mark only | delta + pin 5 s (keep fails when all pinned) |
  | B (5 x 56K) 2 GiB: follow-ups resumed | 0/25 | 0/25 | 0/25 | 3/25 |
  | B 4 GiB | 3/25 (39.9K hit tokens a GiB) | 3/25 | 3/25 | 12/25 (161K a GiB) |
  | B 8 GiB | 25/25, 38 evictions | 25/25, 18 | 25/25, 3 | 25/25 |
  | B 16 GiB | 25/25, 31 evictions, 15.4 GiB held | 25/25, 0, 9.9 GiB | 25/25, 0, 8.3 GiB | = delta |
  | C (16 x 20K) 4 GiB | 0/48 | 0/48 | 0/48 | 4/48 |
  | C 8 GiB | 46/48 | 46/48 | 46/48 | 46/48 |
  | C 12 GiB | 48/48, 89 evictions | 48/48, 60 | 48/48, 12 | = delta |
  The simulator reproduces the measured default (all follow-ups resumed at the real 17-24 GiB budget). READING: (1) hits are IDENTICAL for whole copies and delta at every budget:
  the stock order drops superseded marks first, so the limit is one newest state a conversation either way; delta removes the churn and the held bytes above that limit.
  (2) keeping only a request s last mark saves the fixed parts (110 MiB each) and evictions, never a hit on these traces (no rewinds / edits in them: an edit would need the earlier mark).
  (3) a pin across the tool call only matters in OVERLOAD (budget below the conversations newest states: under 8 GiB here), where LRU thrashes; the elastic budget keeps the real
  engine out of that regime on these traces. NOT BUILT (would be: victimBut skips a conversation s newest state used within N ms for keep-driven evictions only, never for a stream s room).
- 16:33 USERS16 POOL PROFILE REFERENCE (C51 flags off, u16 = 16 lanes, pool of 64K windows, r12 set; MEASURED): trace C (16 agents x 20K x 4 turns): prompt cache STATIC 4,626 MiB,
  7/48 follow-ups resumed, follow-up first token median 107.2 s (max 128.0 s), wall 488.0 s, 267 evictions. (Default profile, 5 lanes: 48/48 resumed, 10.9 s median, 172.8 s.)
  => the 16-lane profile loses the cache: this is what TF_FLASHNEXT_Q38_CACHE_ELASTIC=2 (cache borrows the unused part of the POOL) is for; udl run next.
- 16:51 USERS16 + ELASTIC=2 + DELTA=1 (C51 udl, trace C; MEASURED): cache 4,628 MiB + up to 15,105 MiB of the pool; 37/48 follow-ups resumed, follow-up median 11.6 s, wall 248.2 s
  (ref: 7/48, 107.2 s, 488.0 s). PROBLEM FOUND in delta states under pressure: a live stream held a reference on the rows it stood on, so evicting that conversation s states freed
  only the fixed part and the rows stayed as orphans (up to 5.4 GiB; in trace D 4.8 of 6.5 GiB budget were orphans, 215 evictions): evictions cascaded. FIX (commit daa19a6, C52):
  a stream standing on a range holds the struct, not the bytes (Seg.strong / dead): when the last kept state needing rows is evicted they are freed at once and the stream s next keep is
  a whole copy. Chain on C51 stopped (my pkill pattern also ended the ssh shell; engine removed by hand 16:50), C52 building.
- 17:03 C52 BUILD_OK sha ab0bb462a9f63f06 = HEAD daa19a6. Chain on C52 (log ~/q38/runs/C_r1c.log): udl (u16 + ELASTIC=2 + DELTA=1: C, D) / uel (u16 + ELASTIC=2: C) / eref (one: A B E i s) /
  edl (one + DELTA=2 check: A B E i s) / cold (fresh engine: last request of A and B cold). Queued behind it (~/q38/runs/C_queue2.sh, log C_q2r13.log): u5 (u16 + ELASTIC=2 + DELTA=1 +
  LANE_REUSE=5), runR2.sh lad (cold ladder 200-6000 rows) and kp (KPROF by pass size). Lease C until 18:21, stamp 18:19, window cap 18:40.
- 17:03 C52 udl trace C (USERS16 + ELASTIC=2 + DELTA=1 with the fix; MEASURED): 42/48 follow-ups resumed, follow-up median 9.41 s (max 37.3), wall 210.6 s (ref 7/48, 107.2 s, 488.0 s;
  C51 before the fix 37/48, 248.2 s). Budget min 6,508 MiB when all 16 lanes are live: the elastic room subtracts 16 idle lane sets (LANE_REUSE=16 x 640 MiB = 10.2 GiB) AND the
  live streams bytes: a double count when every lane is live. No-code test queued: LANE_REUSE=5 on this profile (u5).
- 17:13 USERS16 POOL PROFILE, 16 agents x 20K x 4 turns (C52 = handoff tree, fresh engines, same requests; MEASURED):
  | | ref (static 4.6 GiB cache) | ELASTIC=2 (whole copies) | ELASTIC=2 + SNAP_DELTA=1 |
  | follow-ups resumed (of 48); cached share of follow-up tokens | 7; 13.9 % | 37; 73.7 % | 42; 83.3 % |
  | follow-up first token median / max | 107.2 / 128.0 s | 12.22 / 71.2 s | 9.41 / 37.3 s |
  | wall; sum of first-token waits | 488.0 s; 5,982 s | 249.5 s; 2,645 s | 210.6 s; 2,238 s |
  | cache: budget min-max, held max, evictions | 4,626 MiB, 4,606, 267 | 5,951-18,162, 14,023, 127 | 6,508-18,719, 7,495 (+ up to 5.0 GiB shared rows), 85 |
  | MemAvailable low-water | 19.0 GiB | 8.4 GiB (arena 29.5 GiB taken once) | 7.8 GiB (arena 30.1) |
  Trace D (8K of the system prompt shared by all 16): ref 9/48, 87.9 s, wall 468.5 s; ELASTIC=2 + DELTA: 43/48, 7.48 s, wall 199.9 s.
  Texts udl vs uel 64/64 identical, score hashes after the prompt pass 64/64. crash 0, arena fallback 0.
  READING: here delta states DO add hits (37 -> 42): with every lane live the budget is tight (6 GiB) and a cold request s four whole copies push out other conversations newest
  states before its own superseded marks exist; the simulator (atomic keeps, fixed budget) does not show this. The remaining 6 misses: the budget floor (idle lane sets counted twice).
- 17:14 HANDOFF PATCH written: handoff/C/r13-agent-cache/C-r13.patch = git diff a27a02c..daa19a6 (branch r13-C, Mac q38/wt/C13), sha256 4763cdf0c53405df, 610 lines, 7 files, TREE_EQUAL
  (tree 1021f294); binary of exactly this tree: C52 sha ab0bb462a9f63f06. NOTE.md follows when the C52 runs are in.
- 17:29 DEFAULT PROFILE REFERENCE on C52 (flags off; MEASURED): A as before (0.67 0.79 0.63 0.55 0.50 s). B: THIS run lost one hit: 24/25 resumed, turn 1 median 18.8 s, wall 139.8 s, 44 evictions
  (earlier refs 25/25, 118.3 and 121.3 s): whole copies are at the edge at 5 x 56K (a cold request holds four 1.05 GiB copies while others keep theirs).
  E (10 agents x 56K x 3 turns): 0/20 follow-ups resumed, follow-up first token median 147.7 s (max 299 s), wall 561.8 s, 176 evictions, held max 16.9 of 17.2-23.1 GiB: the cache thrashes.
  (The simulator said 10 newest states fit: it keeps a request s marks at once and has no concurrent cold passes; the real engine keeps four whole copies a cold request, ten at a time.)
- 17:41 C52 CHECK-MODE RUN (edl = r12 set + SNAP_DELTA=2, default profile, the handoff binary; MEASURED): A: 1,433 MiB (33.3 %), check 3,469,058,688 bytes 0 differ; B: 25/25 resumed, 54 states
  11,231 MiB, 0 evictions, check 48,675,392,384 bytes 0 differ; E (10 agents x 56K x 3 turns): 20/20 follow-ups resumed, wall 130.2 s, sum of first-token waits 918 s, 13 evictions, held max
  18.2 GiB, check 77,983,081,856 bytes 0 differ (ref: 0/20, 561.8 s, 3,977 s, 176 evictions). CAVEAT: E s agents 0-4 use the same system prompts as B s; the delta engine still held B s
  states when E began (turn 0 median resumed 25.7K), the reference engine had evicted most of them (3 resumed). Same requests in the same order on both, so the comparison is fair as a
  sequence, but ~96 s of the wall difference is those five cold prompts; the follow-up result (0/20 vs 20/20) is the cache holding ten conversations or not.
- 17:51 C52 IDENTITY (edl check mode vs eref flags off; MEASURED): A 6/6, B 30/30, E 30/30 (ref cold, delta resumed), ident 51/51, short 24/24, score hashes after the prompt pass 115/115;
  check 80,636,454,656 bytes 0 differ. COLD REPLAY (C52 SNAP_DELTA=1, fresh engine): last request of A (24,497 tok) 1/1 and of B s five agents (55.6-58.9K) 5/5 texts identical to the
  resumed ones. handoff/C/r13-agent-cache/NOTE.md written (placeholders left for u5 and the ladder). Queue2 (u5, ladder, KPROF) running.
- 17:56 u5 (C52, USERS16 + ELASTIC=2 + SNAP_DELTA=1 + LANE_REUSE=5; trace C; MEASURED): 48/48 follow-ups resumed, follow-up first token median 1.47 s (max 18.95 s; turns 1.02 / 1.26 / 2.05 s),
  wall 165.4 s, sum of first-token waits 1,882 s, 20 evictions, budget 13.1-25.3 GiB, MemAvailable low-water 8.1 GiB. (USERS16 ref: 7/48, 107.2 s, 488.0 s; with LANE_REUSE=16: 42/48, 9.41 s, 210.6 s;
  default profile 5 lanes: 48/48, 10.9 s, 172.8 s.) So for 16 agents: follow-up 107 s -> 1.5 s, wall 488 -> 165 s on the 16-lane profile.
- 18:02 u5 short prompt 96.7 ms (USERS16 ref 102.7), texts 64/64, hashes 64/64 vs udl. COLD LADDER (C52, r12 set, default profile, fresh engine; 6 distinct cold single-pass prompts a size, 1 token
  out; first token ms median; MEASURED): 214 tok 236 | 397: 291 | 764: 415 | 1,141: 504 | 1,439: 608 | 1,471: 611 | 1,836: 741 | 2,197: 861 | 2,771: 1,040 | 3,662: 1,355 | 5,466: 2,012.
  = ~170 ms + 0.33 ms a row, marginal cost between sizes 0.24-0.36 ms a row with no step at any tile boundary (768 / 1,536 / 2,304 / 3,072 rows): no tile-choice cliff below 3,072 rows.
  (19 tokens: 97 ms.) KPROF by pass size running (last engine of the window for me).
- 18:07 KPROF BY PASS SIZE (C52, TF_KPROF=1000, cold prompts of 1,384 / 2,752 / 5,462 rows; ms per 1,000 rows; MEASURED): expert gate/up 98.8 / 65.3 / 53.7; expert down 74.6 / 49.8 / 42.3;
  shared projections 69.7 / 63.7 / 65.4; write-back 45.6 / 43.7 / 44.1; DeltaNet 30.7 / 29.0 / 28.9; hc down 24.4 / 23.5 / 22.8; hc up 22.0 / 19.6 / 19.0; fused attention 10.2 / 17.0 / 22.3;
  all 423 / 349 / 341. The fixed part of a pass is in the two expert kernels only (240 / 317 / 525 ms a pass = ~143 ms + 0.07 ms a row; one call a layer): a pass of a few hundred rows reads
  every expert once = the bandwidth floor. Every other kernel is flat per row. LEVER 4 (tile choice below 3,072 rows): NOTHING TO BUILD (no step in the ladder, kernels flat per row;
  pickDown s choices agree with the measured 1.53x wave cost).
- 18:07 Queue2 done 18:05:21; lease C RELEASED 18:06:25, box clean. RE-ACQUIRED 18:06:43 (until 18:39:43; stamp 18:37:44; window cap 18:40:41) for one last pair: trace E alone on fresh engines
  (e2d = SNAP_DELTA=1 first, e2r = flags off last; log ~/q38/runs/C_r1d.log) to remove the caveat on the 10-agent number. Release as soon as e2r ends (~18:29).
- 18:15 CLEAN TRACE E (fresh engine, E only; C52 r12 set + SNAP_DELTA=1; MEASURED): 10 agents x 56K x 3 turns: 20/20 follow-ups resumed, turns 0 / 1 / 2 first token median 134.1 / 21.45 / 4.36 s
  (turn 1 queues behind the other agents cold prompts: 10 agents on 5 lanes), follow-up median 5.48 s, wall 223.0 s, sum of first-token waits 1,646 s, 0 evictions, 60 states in 16.4 GiB
  (whole copies: 62.5 GiB), MemAvailable low-water 7.6 GiB. Reference (flags off) running last.
- 18:28 CLEAN TRACE E REFERENCE (fresh engine, E only, flags off, LAST; MEASURED): 1/20 follow-ups resumed, first token turns 0 / 1 / 2 median 117.2 / 144.3 / 139.6 s, follow-up median 142.6 s
  (max 284 s), wall 595.9 s, sum of first-token waits 4,176 s, 121 evictions, held max 19.4 GiB. Delta (first): 20/20, 5.48 s, 223.0 s, 1,646 s, 0 evictions. Texts 30/30 identical (ref cold,
  delta resumed), score hashes after the prompt pass 30/30. => 10 agents x 56K on the default profile: wall 595.9 -> 223.0 s (2.67x), follow-up first token 142.6 -> 5.5 s.
- 18:28 LEASE C RELEASED (second and last time, 18:28). spark-4e5c: no q38-run-C container, 0 GPU processes, no script. Raw results: Mac q38/results/rC13.tgz.
## ROUND 13 END STATE (18:30 UTC 2026-11-14)
- Lease C RELEASED 18:28:02 (acquired 15:27, released 18:06:25, re-acquired 18:06:43 for the clean trace-E pair). spark-4e5c: no q38-run-C container, 0 GPU processes, no script, no build.
  Raw results: Mac q38/results/rC13.tgz. Engine logs on the box: ~/q38/runs/C_r1_{ref,chk,dl,ref2}_C50, C_r1_{uref,udl}_C51, C_r1_{udl,uel,eref,edl,cold,u5,e2d,e2r}_C52, C_r2_{lad,kp}_C52.
- HANDOFF (Mac q38/handoff/C/r13-agent-cache/): C-r13.patch (git diff a27a02c..daa19a6, sha256 4763cdf0c53405df, 610 lines, 7 files, TREE_EQUAL tree 1021f294) + NOTE.md.
  Binary of exactly that tree: C52 sha ab0bb462a9f63f06. Nothing pushed by C.
- SHIPPED-READY (exact; flags default off):
  1 TF_FLASHNEXT_Q38_SNAP_DELTA=1 (delta prompt states; =2 check mode). Kept bytes 21-33 % of whole copies; per kept turn 110 MiB + 18.9 KB a NEW token. Default profile: 1 and 5
    agents time-neutral (5 x 56K: evictions 43 -> 0); 10 agents x 56K: follow-ups resumed 1/20 -> 20/20, follow-up first token 142.6 -> 5.5 s, wall 595.9 -> 223.0 s.
  2 TF_FLASHNEXT_Q38_CACHE_ELASTIC=2 (elastic cache with the pool) for USERS16: 16 agents x 20K: resumed 7/48 -> 37/48 (whole copies) -> 42/48 (+ delta) -> 48/48 with
    TF_FLASHNEXT_Q38_LANE_REUSE=5; follow-up first token median 107.2 -> 1.47 s, wall 488.0 -> 165.4 s.
  Identity: check mode 91.2 GB (C50) + 80.6 GB (C52) of stream rows against the shared chains, 0 bytes differ; texts A 6/6, B 30/30, C 64/64, E 30/30 (ref cold vs delta resumed),
  ident 51/51, short 24/24, cold replay of last requests 6/6; score hashes after the prompt pass 160/160 (C50), 115/115 + 30/30 + 64/64 (C52).
- MEASURED AT THE FLOOR, nothing built: tile choice below 3,072 rows (cold ladder 214-5,466 tokens: ~170 ms + 0.33 ms a row, no step; KPROF: the fixed part of a pass is the two expert
  kernels, ~143 ms = every expert read once; every other kernel flat per row). A follow-up's first token = open + restore 14-27 ms + ONE partial pass (~260 ms + 0.30 ms a new row).
- SIMULATED, not built: reuse-aware eviction (the stock order already is), last-mark-only admission (no hit gained), pin across tool calls (helps only below 8 GiB budgets).
- NOT DONE: coarse scores at long context (lever 5); merged follow-up passes; disk tier; paged KV for the live lane; exact idle-lane-set accounting (LANE_REUSE=5 is the config answer).
- INCIDENTS / LESSONS: (1) first delta build let live streams pin shared rows: found only on the tight USERS16 profile (orphans 4.8 of 6.5 GiB), fixed in C52 (Seg.strong / dead).
  (2) The USERS16 profile's baseline (static 4.6 GiB cache, 7/48 hits) was the largest finding and needed one run: baseline every profile first. (3) My cache simulator (atomic keeps,
  fixed budget) said whole copies and delta hit equally at every budget; the engine with ten concurrent cold requests says 1/20 vs 20/20: simulate ORDER, measure capacity.
  (4) A pkill pattern "[a]t.py" over ssh also matched my own remote shell (the regex dot): the engine stayed up idle for 20 s beside a build until removed by hand; use "kit-C/[a]t\.py".
  (5) Never overwrite a running chain script: single files were copied through the head instead of pushkit.sh while a chain ran.
- RESUME: cd ~/.hermes/cache/scratch/q38; tail -60 PROGRESS-C.md; ./shipC13.sh NAME 16 24g; ./pushkit.sh (only when no chain runs); ./mkhandoffC13.sh; box (after `lease.sh acquire C N`
  on the head AND an epoch in ~/q38/runs/C_LEASE_UNTIL = min(lease end, window start + 270 min)): bash ~/q38/kit-C/runR1.sh BIN "NAME:one|u16:PARTS:r12+X=Y ..." (parts A B C D E Z i s p n,
  see its header), runR2.sh BIN "NAME:c|k|s|A:r12+KPROF=1000"; analysis: at.py tab / cmp, atlog.py ENGINE.log, csim.py TRACE.json ENGINE.log BUDGETS, kpp.py ENGINE.log LABEL:FIRST:LAST:ROWS.
- 18:29 Final check: lease list has no C, box has no engine. Skills updated (llm-inference-systems-playbook: SKILL.md rules + papers-serving-scheduling-prefill.md round-13 table;
  dsgx10-spark-deepseek-engine: qwen38-fastest-build.md round-13 section). Kit synced to the box.

# Round 14 (builder C; bf16 KV profiles for many users and agents: fit, speed, agents, identity, check modes, PROFILES.md)
- 18:31 UTC start. BASE TREE for every number of this round unless said: Mac q38/wt/C14 (branch r14-C) = integ10 512e3cf (patches 0101-0123, tree 7628b4db) + C-r11 + C-r12 + C-r13
  handoff patches applied with git apply --3way (all clean), commits 2ee5185 / 87ad2f7 / 85eeab7, tree afb40abf. Build C60 started on spark-4e5c 18:31:44 (16 CPUs, box idle,
  window S141041 still open until 18:50:41; no engine of mine). KV dtype is the serve flag --kv-cache bf16|fp8 (MiaAI start.sh KV_DTYPE); MiaAI config.sh: bf16 30,784 bytes a token,
  fp8 17,888. Ship script Mac q38/shipC14.sh. Next: engine script with KV / lanes / pool arguments, run chain with OPEN + lease checks, wait for a NEW window (this one ends 18:50).
- 18:38 BUILDS (spark-4e5c, both BUILD_OK, tree hash check 7358 files each): C60 sha a4ce7004763f9efb = tree afb40abf (integ10 512e3cf + C-r11 + C-r12 + C-r13);
  C61 sha ba9dd31f6211d9e0 = tree f2d4f730 = C60 + builder B's r11 commit b9932b6 ("window buffers sized at run time (mtp.bufRows)" + prediction / recall fields; applies clean;
  Mac q38/wt/C14b, branch r14-C-b). WHY C61: patch 0122 sizes every lane's window buffers for 128 rows at compile time (+0.45 GiB a lane); under bf16 that is what decides whether 16 lanes
  fit, so the many-user bf16 profile is measured on C61 and the fit table shows C60 beside it. head2_r9.bin copied from 923b to 4e5c ~/q38/hf/q38 (sha fce7c913060d45f2 = the measured file).
  KIT (Mac q38/kitC, box ~/q38/kit-C): engC14.sh BIN KV PAR CTX POOL LEV [EXTRA] (lever env = recipe.sh R10 defaults incl. HEAD2 r9; 'off' = MiaAI configuration; refuses a start unless
  box stamps C_LEASE_UNTIL and C_WIN_END are > 4 min ahead), run14.sh "NAME|BIN|KV|PAR|CTX|POOL|LEV|WORDS|PARTS ..." (see header), head ~/q38/keepC.sh (renews lease C, writes both
  stamps every 60 s, releases when the box has ~/q38/runs/C_chain_done or the window end is near). No engine started yet: window S141041 ends 18:50:41 (start + 270 min = 18:40:41).
  ARITHMETIC (ESTIMATE until the start lines are read): fp8 stream = 0.84 GiB fixed (C60) + 17,888 B a token; bf16 = 0.84 + 30,784 B a token => 262K 8.36, 128K 4.60, 64K 2.72,
  48K 2.25, 32K 1.78 GiB; with B's run-time buffers at 32-row windows the fixed part should be ~0.39 GiB.
  PLAN for the new window (order): start-line probes (bf16 5 x 262K, C60 16-lane pool, C61 16-lane pools) -> speed block on C61: fp8 USERS16 ref / bf16 16-lane pool / bf16 12 x 64K /
  bf16 8 x 128K / fp8 ref -> bf16 levers-off engine (identity) -> bf16 default profile 3 x 262K (ident, held, agents 1 / 5 / 10) -> bf16 pool agents (16) -> check modes x2 -> 5-lane pool default candidate.
