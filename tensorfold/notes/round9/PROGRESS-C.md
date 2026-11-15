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
