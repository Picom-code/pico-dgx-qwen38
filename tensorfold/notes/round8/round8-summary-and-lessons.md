## CURRENT STATE after round 8 (2026-11-13 evening; patches 0101-0115, repo branch tensorfold-fp4-local; four builders in parallel)
Full log: Mac `q38/PROGRESS-A.md` (integrator) + `PROGRESS-B/C/D.md`; tables `q38/results/final_r8_{923b,manyuser,levers}.txt`; handoffs `q38/handoff/{B,C,D}`.
Quote these (spark-923b, `recipe.sh prepare` + `recipe.sh start`, vision on, levers off vs recipe default, ONE window, same image 0101-0113):
| bench | levers off (MiaAI configuration) | recipe default | change |
|---|---|---|---|
| first token, short prompt | 184 ms | 111 ms | -40 % |
| 3-prompt chat / code, 9 runs | 47.6 / 77.0 | 50.1 / 84.7 | +5.3 / +10.0 % |
| sparkDash-method prose / code / structured | 56.6 / 91.8 / 105.7 | 60.9 / 99.1 / 113.6 | +7.6 / +8.0 / +7.5 % |
| held, same 104 | 58.01 | 62.38 | +7.5 % |
| edit set / agent set wall | 137.2 / 54.2 | 176.1 / 59.1 | +28.4 / +9.0 % |
| 16 users distinct (last first token) | MiaAI 16 x 64K: 218.4 / 219.2 / 217.3 (6.8 s) | USERS16: 276.9 / 283.0 / 278.6 (0.95 / 0.70 / 0.71 s) | +28 % |
| 12 / 8 / 4 users | 209.1 / 189.9 / 140.1 | 260.0 / 219.9 / 153.7 | +24 / +16 / +10 % |
| waves of 16 held (last first token) | 127.7 (7.20 s) | 160.1 (1.12 s) | |
| prefill, needle 182K, 227K prompt, image, qeval | | 2503 / 2519 / 2340 tok/s, 3/3, 3/3, PASS, 33/35 | |
Identity: ident 51/51, held 107/107, edit 24/24, agent 16/16; waves-16 128/128 and waves-4 64/64 vs MiaAI's config; held 107 with 16 in flight 107/107.
Not run in round 8: held-320 in full, sparkDash 1.8.8 itself (round 6: 61.33; 59.8 / 97.2 / 110.4). Still trails custom engine R's re-quantised 54.6 / 88.6.

Levers shipped in round 8 (recipe defaults unless said):
- `ATTN_LANES=1` (`TF_FLASHNEXT_Q38_ATTN_LANES`, patch 0110, idea upstream #1739): a shared round's attention (prep, pool, scores, select, chunks, merge) in one
  launch a kernel for up to 16 text lanes; by-value lane table (`__grid_constant__` struct, no upload), fp8 and bf16 caches as separate groups, media lanes and lone
  lanes stay per lane. 16 users +1.6..2.3 %; GPROF QSA group 8.47 -> 5.68 ms of a 197.5 ms pass.
- `LATE_SKIP=16` (`TF_FLASHNEXT_Q38_LATE_SKIP`, patch 0111): with `TF_FLASHNEXT_KEEP_LATE` a drafted window whose rows + waiting log rows <= T passes a null
  state_out to fn_gdn_rows_late (no 3.1 MB end-state write a stream-layer); kept whole, its rows wait in the log (pend += rows) like a partly kept window's.
  16 users +1.3..1.5 % (GPROF DeltaNet 34.1 -> 31.2 ms), one user +0.3 %. T = 8 / 16 / 32 the same.
- `KEEP_INPASS=4096` + `LANE_REUSE=streams` (patch 0112, builder C; ideas upstream #1866 / #1763): the chat template's prompt-cache mark 7 tokens before the end no longer
  cuts every prompt pass in two (a 7-row pass costs 58-72 ms), and a finished stream's 73 + 45 fixed device buffers are reused (22 ms a request).
- Options, not defaults: `PREFIX_WAIT=60000` (port of upstream #1321: 8 requests on a new 25.5K system prompt, last first token 86.7 -> 12.9 s), `AGENTS=1` (pool of
  5 x 131K windows: prompt cache 4 -> 13 GiB, 5 agents' follow-up first token 85 -> 1 s), `HEAD2=1` after `recipe.sh head2` (patch 0115, builder B: two-stage draft head,
  rank-512 4-bit preview in a PCA-of-head-inputs basis + exact re-score of each 256-row block's 3 best rows; one user +1.6..1.9 %, texts identical; not a default because
  the rotation was fitted on the 104 held prompts the bench uses).
Measured dead in round 8 (do not retry): expert items of 64 pairs (-2.5 %), row-trimmed lane tiles (zero), draft depth 8 / 10 or gate 0.3 at 16 users, expert storage
order / exact expert skips / a certified coarse router (statistics say nothing to gain: 228 distinct experts a layer at 16 users, read at >= 254 GB/s = the memory bus),
per-lane PLE conv + shift (0.04 %, not built), prompt-cache sibling-first eviction (zero), rank-256 or SVD-of-head previews for the draft head.
Anatomy facts to reuse: the DeltaNet chain kernel is the state read + write (2 x 3.1 MB a stream-layer at ~215 GB/s; rows are in registers and free); after ATTN_LANES
only ~9 of ~330 launches a profile unit are still per lane; a 16-user pass is 191.5 ms = MoE 130 (bus-bound), DeltaNet 31, hyper-connection 15, head 7.5, attention 5.7.
What is next (estimates): re-fit the two-stage head's rotation on broad text and make it a default (+1.6 %); a tuned preview kernel (+0.7 %); the indexer's block scores
on long prompts (certified coarse score, design note `handoff/C/NEXT-qsa-certified-select.md`: 117K prefill 2,340 -> ~2,700 tok/s).

### Round-8 method lessons (parallel builders, shared window)
- Shared window harness: head `~/q38/shared/lease.sh acquire|renew|release NAME MIN` (unit q38-win runs qws.sh + w_shared.sh). The window opens on the first acquire
  (8 s), stays open while any lease is live, and closes by itself: DeepSeek back and verified 4.5 min after the last release. One window served four builders for 4.8 h.
  A window's guard is 300 min from its opening, whoever opened it: check `systemctl --user list-timers | grep q38-guard` before planning a long suite.
- Never build on a box while its engine is being measured: a 3-CPU zig/nvcc build cost a 16-user run ~1.5 % (the GPU shares the memory bus with the CPUs).
- One builder a box, each with its own worktree (`git worktree add -b builder-X`), build dir, kit copy and container name; hand off ONE cumulative patch against the
  base commit + NOTE.md with flags, numbers and identity proof. The integrator applies with `git apply --3way`, one commit a handoff, then regenerates one patch a
  commit and checks `git write-tree` of base + patches against the branch (Mac `q38/A/mkpatches.sh`). All four handoffs applied without a conflict.
- A handoff patch that keeps changing: record the sha256 of the version that was BUILT AND MEASURED and ship that one (copy in `handoff/used/`), even if a later
  version only drops a dead flag.
- Queue run scripts on the box (`nohup`, each waits for the previous by `pgrep`), with a deadline file the Mac rewrites on every lease renew; never edit a script that
  is running (write wF2.sh instead). GPROF prints every 400 passes: a profile run needs the held-16-in-flight load too, two 16-user benches are too short.
- A lever proven only on the bench's own prompts (a fitted rotation, a tuned list) ships as an option until it is re-fitted on other text, however clean its identity.
- Polling a 5-minute sleep costs one iteration; do not chain two sleeps in one call (the tool times out at ~7 min). Budget ~12 iterations an hour of window.
