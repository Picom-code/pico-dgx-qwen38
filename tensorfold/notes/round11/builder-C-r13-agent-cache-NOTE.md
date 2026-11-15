# Builder C, round 13: the agent serving path. Delta prompt states (kept states share their attention rows), the elastic cache for the 16-lane pool profile, and where a follow-up's first token goes

Patch: `C-r13.patch` = `git diff a27a02c HEAD` of Mac `q38/wt/C13` (branch r13-C, HEAD daa19a6), sha256 4763cdf0c53405df, 610 lines, 7 files;
`q38/mkhandoffC13.sh` printed TREE_EQUAL (base + patch = tree 1021f294). Base a27a02c = my round-12 handoff tree, so it applies after C-r12.patch.
Binary BUILT AND MEASURED from exactly this tree: C52, sha ab0bb462a9f63f06 (spark-4e5c `~/q38/tfbin/C52`). C50 (sha 10c6ab5af1c3efe6) = the first commit's delta
states without ELASTIC=2 and without the "a stream does not hold bytes" fix: sections say which binary. Build: `./shipC13.sh NAME 16 24g`.
Everything below is MEASURED on spark-4e5c unless it says SIMULATED or ESTIMATE; shared window 2026-11-14 14:10-18:40 UTC (my lease 15:27-18:06 and 18:06-18:28), fresh engines, env flags.
Raw results: Mac `q38/results/rC13.tgz` (trace jsons + run logs; engine logs stay on the box under `~/q38/runs/C_r1_*` / `C_r2_*`).
"r12 set" = the round-12 final set (QSA_LIST, QSA_ROWS, HEAD_KV, ARENA, WARM, GDN_PROD=2, ROUTE_BAND, SCRATCH_TRIM=3072, CHUNK=6144, QSA_CERT_MIN=1, HC_DN_TALL). Profiles: "default" = the
recipe's one-user profile (5 lanes x 262,144, elastic cache, vision on); "USERS16" = 16 lanes, pool of 64K windows (kit-C/engC10.sh u16). Full log: `q38/PROGRESS-C.md` (Round 13).
Kit (Mac `q38/kitC`, box `~/q38/kit-C`): `at.py` (agent traces), `atlog.py` (the cache's side from the engine log), `csim.py` (cache policy simulator on a logged trace),
`runR1.sh` / `runR2.sh` (the run chains; they refuse to start an engine without a lease stamp), `pp.py`, `kpp.py`.

## 1. Flags (all default OFF in the patch)
| flag | what | proposed |
| `TF_FLASHNEXT_Q38_SNAP_DELTA=1` | DELTA STATES. A kept prompt state is no longer one whole device copy. Its fixed pieces (36 DeltaNet states, conv windows, n-gram tail: 110 MiB) are copied as before; its attention rows (keys, values, indexer keys, pooled blocks of the 12 layers and of the draft head) are a reference-counted chain of row ranges: its own rows [from, at) plus, through `parent`, the ranges of the state its stream resumed or kept last. A restore copies the same bytes to the same places. Why exact: a stream's attention caches are append-only (row r is written once), so two states of one conversation hold the same bytes for the rows both cover. The cache's accounts: a state is charged its own buffers (`snap_charged`), a keep asks the family what it will newly take (`snap_need`), shared rows whose own state was evicted while a later kept state still needs them count as the family's spare bytes (`snap_spare`) so the budget stays honest. A live stream standing on kept rows holds the struct, not the bytes: when the cache has evicted every state that needs them they are freed at once | ON, both profiles |
| `TF_FLASHNEXT_Q38_SNAP_DELTA=2` | check mode: at every keep the stream's own rows [0, at) are downloaded and compared byte by byte with the chain (line "q38 snapdelta keep ... check_bytes= check_diff=", PLOG on) | off |
| `TF_FLASHNEXT_Q38_CACHE_ELASTIC=2` | the round-9 elastic prompt cache ALSO with the pool (`TF_FLASHNEXT_Q38_POOL_CTX`): the cache borrows the part of the pool that live streams are not using and hands it back before a stream opens (and the arena, when on, is sized base + pool). `=1` behaves as before (off with the pool) | ON for USERS16 / pool profiles (instead of no elastic cache there) |

Files: `families/flashnext/cuda_snap.zig` (Seg, chain save / restore / check), `cuda_lanes.zig` (Lane.base, snap_need / snap_charged / snap_spare, elastic room with the pool),
`cuda_rank_engine.zig` (signature), `cuda_knobs.zig`, `core/prompt_cache.zig` (need hook, reserve estimate), `core/prompt_cache_policy.zig`, `native/cuda_snaps.zig`.
Credit: the idea of sharing prefix blocks between cached sequences is vLLM PagedAttention (arXiv 2309.06180), SGLang RadixAttention (2312.07104) and upstream TensorFold's
`TF_FLASHNEXT_KV_PAGES`; "one recurrent checkpoint + shared KV" is how Marconi (2411.19379) and HiState (2610.41508) describe hybrid models. This is a smaller form: only kept
states share, the live lane still gets a copy.

## 2. The per-turn agent table (traces: `at.py run`; each turn = the previous prompt + the reply + a seeded tool result of 300-2,000 tokens, a 0.3-2 s tool pause; agents start together)
"first token" = median / max over the agents of that turn, seconds. "resumed" = follow-up requests that took more than half their prompt from the cache.

1 AGENT, 20K system prompt, 6 turns, default profile (C50: ref = flags off first, delta = SNAP_DELTA=1; C52 ref repeats the ref column to 0.01 s):
| turn | prompt tok | resumed tok | new tok | first token, ref | first token, delta |
| 0 | 18,765 | 0 | 18,765 | 7.20 | 7.21 |
| 1 | 20,070 | 18,758 | 1,312 | 0.66 | 0.66 |
| 2 | 21,720 | 20,063 | 1,657 | 0.79 | 0.78 |
| 3 | 22,879 | 21,713 | 1,166 | 0.63 | 0.63 |
| 4 | 23,788 | 22,872 | 916 | 0.54 | 0.55 |
| 5 | 24,497 | 23,781 | 716 | 0.50 | 0.49 |
| kept states; MiB held; evictions | | | | 9; 4,300; 0 | 9; 1,433; 0 |

5 AGENTS x 56K own system prompt x 6 turns, default profile (C50; ref first / delta / ref last = 118.3 / 119.9 / 121.3 s wall):
| turn | prompt tok | resumed tok | new tok | first token, ref | first token, delta |
| 0 | 52,048 | 0 | 52,048 | 64.13 / 90.89 | 66.40 / 90.68 |
| 1 | 52,811 | 52,041 | 833 | 0.67 / 1.36 | 0.70 / 1.15 |
| 2 | 54,466 | 52,804 | 1,139 | 0.96 / 1.52 | 0.98 / 1.51 |
| 3 | 55,630 | 54,459 | 1,171 | 0.83 / 1.52 | 1.89 / 3.37 |
| 4 | 56,544 | 55,623 | 921 | 1.08 / 2.83 | 1.16 / 2.77 |
| 5 | 57,258 | 56,537 | 940 | 1.35 / 2.31 | 1.04 / 2.52 |
| follow-ups resumed; follow-up median / max | | | | 25/25; 0.99 / 2.83 s | 25/25; 1.04 / 3.37 s |
| states at the end (max); MiB held max; evictions | | | | 11 (21); 16,522; 43 | 54; 11,231; 0 |
The third reference run (C52, flags off) LOST one hit on this trace: 24/25 resumed, turn 1 median 18.8 s, wall 139.8 s, 44 evictions: whole copies are at the edge here.
Delta kept every state in all three delta runs (C50 timing, C50 check, C52 check: 0 evictions, 25/25).

10 AGENTS x 56K x 3 turns, default profile, FRESH engines with this trace only (C52; delta = SNAP_DELTA=1 first, ref = flags off last):
| turn | prompt tok | resumed tok ref / delta | first token, ref | first token, delta |
| 0 | 52,019 | 0 / 0 | 117.23 / 220.78 | 134.09 / 196.74 |
| 1 | 53,273 | 0 / 52,012 | 144.26 / 284.23 | 21.45 / 99.39 |
| 2 | 54,887 | 0 / 53,266 | 139.59 / 209.27 | 4.36 / 6.13 |
| follow-ups resumed; cached share of follow-up tokens | | | 1/20; 5.0 % | 20/20; 97.6 % |
| follow-up median / max | | | 142.6 / 284.2 s | 5.48 / 99.4 s |
| wall; sum of first-token waits | | | 595.9 s; 4,176 s | 223.0 s; 1,646 s |
| states at the end; MiB held max; evictions | | | 15; 19,374; 121 | 60; 16,442; 0 |
Whole copies thrash: a cold request keeps four 1.05 GiB copies and ten requests do so at a time, so each conversation's newest state is pushed out before its next turn.
Delta holds all ten conversations with every mark (60 states that would be 62.5 GiB as whole copies). Turn 1 still waits for the other agents' cold prompts (10 agents, 5 lanes).
The same trace after traces A and B in one engine (delta in check mode): ref 0/20, wall 561.8 s; delta 20/20, wall 130.2 s (there the delta cache also still held five of the
ten system prompts from trace B, which the reference had evicted).

16 AGENTS x 20K x 4 turns:
| | default profile (5 lanes), ref (C50) | default, delta (C50) | USERS16 ref (C52) | USERS16 + ELASTIC=2 (C52) | USERS16 + ELASTIC=2 + delta (C52) |
| turn 0 first token | 65.67 / 125.53 | 60.29 / 117.73 | 116.97 / 123.06 | 115.62 / 121.58 | 115.31 / 121.25 |
| turn 1 | 36.12 / 84.20 | 36.22 / 86.53 | 120.97 / 127.84 | 14.48 / 71.16 | 9.78 / 25.59 |
| turn 2 | 11.09 / 12.76 | 10.35 / 11.76 | 109.71 / 128.02 | 11.68 / 63.20 | 7.52 / 37.29 |
| turn 3 | 8.82 / 10.61 | 9.01 / 13.20 | 70.94 / 111.94 | 8.31 / 42.91 | 10.97 / 13.85 |
| follow-ups resumed (of 48); cached share | 48; 94.7 % | 48; 94.7 % | 7; 13.9 % | 37; 73.7 % | 42; 83.3 % |
| follow-up median / max | 10.94 / 84.2 s | 10.62 / 86.5 s | 107.2 / 128.0 s | 12.22 / 71.2 s | 9.41 / 37.3 s |
| wall; sum of first-token waits | 172.8 s; 2,044 s | 165.8 s; 1,938 s | 488.0 s; 5,982 s | 249.5 s; 2,645 s | 210.6 s; 2,238 s |
| cache budget; evictions | 20.5-23.3 GiB; 128 | same; 82 | 4.6 GiB static; 267 | 6.0-18.2 GiB; 127 | 6.5-18.7 GiB; 85 |
With 8K of the 20K system prompt shared by all 16 agents (USERS16): ref 9/48 resumed, 87.9 s median, wall 468.5 s; ELASTIC=2 + delta 43/48, 7.48 s, wall 199.9 s.
USERS16 + ELASTIC=2 + delta + `TF_FLASHNEXT_Q38_LANE_REUSE=5` (instead of 16; C52): 48/48 resumed, turns 1-3 first token 1.02 / 1.26 / 2.05 s (max 18.95 / 4.74 / 7.33), follow-up
median 1.47 s, wall 165.4 s, sum of first-token waits 1,882 s, 20 evictions, cache budget 13.1-25.3 GiB; short new prompt 96.7 ms (USERS16 ref 102.7 ms); texts 64/64 and score
hashes after the prompt pass 64/64 identical to the other USERS16 runs. Why: the elastic room subtracts LANE_REUSE x 640 MiB of idle lane sets; at 16 that is 10.2 GiB the cache
never gets. => 16 agents on the 16-lane profile: follow-up first token 107 s -> 1.5 s, wall 488 -> 165 s.
Reading: (1) the 16-lane profile had no usable prompt cache (4.6 GiB static): ELASTIC=2 halves the wall and delta adds another 16 %. (2) With hits, 16 agents that start together
still wait 7-11 s: their follow-up passes run one after another (16 x ~0.6 s) and on 5 lanes they queue for a lane. (3) On the default profile delta changes memory, not time,
until the cache is short (10 agents x 56K), where it is the difference between thrashing and holding every conversation.

## 3. Memory per kept turn
- Before: every mark is 110 MiB + 18.9 KB x the WHOLE history (0.47 GiB at 20K, 1.12 GiB at 56K); a cold request keeps 4 marks, a follow-up 1-2.
- After: 110 MiB + 18.9 KB x the NEW tokens since the state the stream stands on (a 1,200-token turn: ~132 MiB).
- Measured: 1 agent, 24.5K after 6 turns: 9 states in 1,433 MiB (whole copies 4,300 MiB: 33.3 %). 5 agents x 56K x 6 turns: all 54 states in 11,231 MiB, 0 evictions (whole copies would
  be 52,375 MiB: 21.4 %; the reference keeps 11-21 states in up to 16.5-18.8 GiB with 40-44 evictions). 64-80 % of the bytes whole copies keep are duplicate prefix (atlog.py, from the plan).
- What is NOT shared: the fixed 110 MiB a state (a recurrent summary, must be copied; it is now the cost of a mark), and the live lane's own copy of the rows it resumed (needs paged
  KV inside the kernels: not attempted).
- The box does not see the saving while `TF_FLASHNEXT_Q38_ARENA=1` is on (the arena takes the lanes' and the cache's planned room once): MemAvailable low-water 7.7-8.4 GiB in every
  default-profile run, off or on. USERS16: 19.0 GiB with the static cache, 7.8-8.4 GiB with ELASTIC=2 (the arena is then 29.5-30.2 GiB). That is the plan's 9 GiB reserve less ~1 GiB, as in round 10-12.

## 4. Identity
- Check mode (`SNAP_DELTA=2`), the handoff binary C52, default profile, traces A + B + E + ident 51 + short: every keep compared the stream's rows [0, at) with the chain it
  shares: 80,636,454,656 bytes, 0 differ (149 keeps, 138 extend a kept state, longest chain 11). C50 (first commit), traces A + B + C + ident + short: 91,217,281,536 bytes, 0 differ.
- C52 delta (check mode) against C52 flags off, both fresh, same requests in the same order: trace A 6/6, B 30/30, E 30/30 texts identical (in E the reference prefilled every
  follow-up cold, the delta engine resumed all 20: cold text = resumed text), ident 51/51, short 24/24; the fresh E pair (SNAP_DELTA=1 against flags off): 30/30 texts, 30/30 hashes; score hashes after the prompt pass 115/115 (at release 70/115: the requests
  that ran concurrently, as in every round).
- C50 `SNAP_DELTA=1` against C50 flags off: A 6/6, B 30/30, C 64/64, ident 51/51, short 24/24; score hashes after the prompt pass 160/160. Reference last (flags off): A 6/6, B 30/30.
- Cold replay (C52, `SNAP_DELTA=1`, fresh engine): the LAST request of trace A (24,497 tokens) and of each of trace B's five agents (55.6-58.9K tokens) sent cold: 1/1 and 5/5 texts
  identical to the resumed ones of the reference run.
- USERS16 (C52): ELASTIC=2 + delta against ELASTIC=2: 64/64 texts identical, score hashes after the prompt pass 64/64.
- crash lines 0 and arena fallback 0 in every run. The unit test added in cuda_snap.zig (a range's bytes add up to the whole state's wherever the chain is cut) was NOT run (no test
  runner on the box in this round).

## 5. Where a follow-up's first token goes (default profile, one agent, 20K-57K history)
| part | measured |
| open + restore of the kept state (a device copy of the rows) | 13.4 ms at 20K, 26.1 ms at 56K (delta: 14.3 / 27.0 ms) |
| ONE partial prompt pass over the new rows | 460 rows 376 ms; 732 rows 487 ms; 1,273 rows 628 ms; 1,990 rows 857 ms = about 260 ms + 0.30 ms a row |
| keeping the new mark (inside the pass) | ~10 ms as a whole copy at 56K, ~1 ms as a delta |
| first decode step | 0.3 ms after the pass (the pass leaves the scores) |
So a follow-up's first token is the size of the tool result, not of the history (0.50 s for 716 new tokens at 24K, 0.67-1.35 s at 52-57K with five agents interleaving).
Short and mid prompts, the same pass seen cold (C52, r12 set, default profile, fresh engine; 6 distinct single-pass prompts a size, 1 token out; first token, median):
| prompt tokens | 214 | 397 | 764 | 1,141 | 1,439 | 1,836 | 2,197 | 2,771 | 3,662 | 5,466 | (19) |
| first token ms | 236 | 291 | 415 | 504 | 608 | 741 | 861 | 1,040 | 1,355 | 2,012 | 97 |
= about 170 ms + 0.33 ms a row; the marginal cost between neighbouring sizes is 0.24-0.36 ms a row with NO step at a tile boundary (768 / 1,536 / 2,304 / 3,072 rows).
KPROF by pass size (C52, TF_KPROF=1000; cold prompts of 1,384 / 2,752 / 5,462 rows; ms per 1,000 prompt rows; the 5,462 column is scaled to its 32,772 rows):
| kernel | 1,384-row passes | 2,752 | 5,462 |
| expert gate/up (tf_nvfp4_tile_gu) | 98.8 | 65.3 | 53.7 |
| expert down (tf_nvfp4_tile_f32) | 74.6 | 49.8 | 42.3 |
| shared projections (tf_qmmf_prompt_mx2 bf16 + f32) | 69.7 | 63.7 | 65.4 |
| read-out write-back (fn_hc_writeback_moe_sh + slices) | 45.6 | 43.7 | 44.1 |
| DeltaNet (fn_gdn_fused_p2) | 30.7 | 29.0 | 28.9 |
| hyper-connection down / up (fn_hcn_down4 / up3_k5) | 24.4 / 22.0 | 23.5 / 19.6 | 22.8 / 19.0 |
| fused attention (fn_attn_fused_kv8) | 10.2 | 17.0 | 22.3 |
| all kernels | 423 | 349 | 341 |
The whole fixed part is in the two EXPERT kernels: 240 ms a 1,384-row pass, 317 ms at 2,752, 525 ms at 5,462 = ~143 ms + 0.07 ms a row; one call a layer in every pass (48). A pass of
a few hundred rows already routes rows to every expert of every layer, so it reads every expert's weights once: that is bytes at the bus rate, the bandwidth floor of an exact pass.
Every other kernel costs the same per row at 1.4K as at 5.5K rows (within 10 %): no tile-choice lever below 3,072 rows. The down-tile rule (round 12's open point) also checks
out on paper with the measured 1.53x wave cost: it picks the 64-row tile for 769-1,536 rows and above 2,304, the 32-row tile up to 768, and the two are level (3.06 against 3
wave units) at 1,537-2,304. NOTHING BUILT for lever "tile choice below 3,072 rows": measured at its floor.

## 6. Policy (levers 3): simulated, not built
`kitC/csim.py` replays a logged trace (arrival order, the marks each request planned) through the cache at a fixed budget with the stock eviction order.
| trace, budget | whole copies | delta | delta, last mark only | delta + 5 s pin on a conversation's newest state |
| 5 x 56K, 4 GiB: follow-ups resumed (hit tokens a GiB) | 3/25 (39.9K) | 3/25 | 3/25 | 12/25 (161K) |
| 5 x 56K, 8 GiB | 25/25, 38 evictions | 25/25, 18 | 25/25, 3 | 25/25 |
| 5 x 56K, 16 GiB | 25/25, 31 evictions, 15.4 GiB held | 25/25, 0, 9.9 GiB | 25/25, 0, 8.3 GiB | = delta |
| 16 x 20K, 4 GiB | 0/48 | 0/48 | 0/48 | 4/48 |
| 16 x 20K, 8 GiB | 46/48 | 46/48 | 46/48 | 46/48 |
- Reuse-aware eviction: the stock order (superseded marks, then a request's earlier marks, then least recently used) is already that for conversations that only grow. Nothing to build.
- Keeping only a request's last mark: saves the fixed parts (110 MiB each) and evictions, no hit on these traces; an edit or a branch would miss the earlier mark. Not built.
- A pin with a time-to-live across the tool call (Continuum 2511.02230, InferCept 2402.01869): only pays in overload (budget below the conversations' newest states), where least
  recently used thrashes. Not built: with the elastic budget and delta states the measured traces are out of that regime; if built it must only refuse KEEPS, never a stream's room.
- The simulator's limit, found by measuring: it keeps a request's marks at once on a fixed budget, so it shows the same hits for whole copies and delta at every budget. The real
  engine with ten cold 56K requests at a time (four whole copies each) thrashes at a budget the simulator calls enough. Trust it for ORDER questions, not for capacity.

## 7. For the integrator
- Recipe: add `TF_FLASHNEXT_Q38_SNAP_DELTA=1` (both profiles). USERS16 / pool profiles: add `TF_FLASHNEXT_Q38_CACHE_ELASTIC=2`, keep `TF_FLASHNEXT_Q38_ARENA=1`, and set
  `TF_FLASHNEXT_Q38_LANE_REUSE=5` there (not 16). NOT measured with LANE_REUSE=5: the 16-user throughput benches (qk.py distinct 16); round 8 measured LANE_REUSE on first token only.
- Re-run on 923b before quoting: check mode once (`SNAP_DELTA=2` + `PLOG=1`, any agent trace: the last "q38 snapdelta keep" line must show check_diff=0), then trace B and E
  (`python3 -I kit-C/at.py run OUT.json 5 6 56000 0 300 2000` and `... 10 3 56000 0 300 2000`) off and on.
- NOT TESTED: two ranks (delta is off there by construction: the rank engine keeps whole copies), vision prompts in a delta chain (the rows are the same buffers; no image trace was
  run), `--learn` paths, very long chains (one range a keep: 100 turns = 100 small copies a restore; no compaction is built; longest chain seen 11).
- Known cost: none measured at one user (open + restore of a resumed state 13.4 -> 14.3 ms at 20K, 26.1 -> 27.0 ms at 56K; a keep copies the new rows only).

## 8. What remains, by size
1. USERS16 budget floor: the elastic room subtracts every idle lane set (LANE_REUSE x 640 MiB = 10.2 GiB at 16) AND the live streams' bytes; with all 16 lanes live the cache has
   6 GiB and 6 of 48 follow-ups still miss. Setting `TF_FLASHNEXT_Q38_LANE_REUSE=5` on that profile removes it (48/48, section 2); the
   exact fix is to subtract only the lane sets that are idle NOW. Not built.
2. A disk tier for evicted conversations: upstream has one (`--learn-conversations`, core/rows_store.zig: block files shared between states) wired for Nemotron only on CUDA. A Flash
   Next CUDA image (parts / gather / scatter / blank, like native/cuda_disk.zig) would bring an evicted 56K conversation back at disk speed instead of 21 s. Design risk: page cache on
   a unified-memory box next to DeepSeek's restart (use O_DIRECT). Not started.
3. Sharing between the kept state and the LIVE lane (paged KV in the kernels, upstream's KV_PAGES for other families): the duplicate that is left. A full port.
4. Follow-up passes of agents that wait together: each pays ~260 ms fixed. `Engine.prompts` (PREFILL_MANY) takes lanes of at most 256 rows without marks; agents need several
   attention blocks a lane and a mark a lane. ESTIMATE: at most 260 of ~620 ms a follow-up when they co-arrive; not built.
5. Chain compaction for very long sessions (one range a keep).
6. Coarse scores at long context (round 12's 26 MB a call round trip): not touched this round.
