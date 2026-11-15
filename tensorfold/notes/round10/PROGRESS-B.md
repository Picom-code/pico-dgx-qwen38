# Builder B progress (spark-3d24 pico@spark-3d24; draft side, one user). Started 2026-11-13.
Workspace: Mac q38/wt/B (source), box ~/q38/build-B, ~/q38/tfbin/B*, ~/q38/runs/B_*, container q38-run-B, port 18301.
## Log
- start: reading state. No lease held.
- 17:11:21 lease B acquired (75 min, until 18:25 UTC); B1 = base + TF_STEP_LOG=2 logging (sha 26c170ae89f5c8cb) built; study runs B_S1 (always-8, top-8, head-state dump) and B_S2 (default gate) launched on 3d24 (driver ~/q38/kit-B/studyB.sh, log ~/q38/runs/B_study.log)
- (c) public drafters CHECKED (HF API, both exist, weights public): incoai/Qwen3.8-Flash-Next-DFlash2 (Apache-2.0, block 8, 5 layers, taps 5,15,25,35,45, selector top-16; card numbers are T=1.0 + thinking at 3 drafts: MT-Bench 2.50 -> 2.83) and shadowpa0327/Qwen3.8-Flash-Next-DSpark (license "other", 664.6M params, taps 1,12,23,34,45, confidence + Markov head; GREEDY block 7: chat mix 4.75 (rev 5bc7f2b) vs MTP-7 4.67, agent 3.70 vs 4.24). Serving needs SGLang (+FP8/NVFP4 checkpoint download, QSA build patched for width 8 that the DSpark card says is NOT public). So acceptance at T=0 is ~equal to MTP per the DSpark card; only draft COST could win (est. 8-bit body ~3 ms + head vs chain 4.8 steps x 1.70 = 8.2 ms => ~+7 % if acceptance and gating hold). Port = L (new runner + target feature taps + drafter KV). NOT measured here (no SGLang on box); decision: do the cheap levers first.
- FOUND: the core engine already supports tree windows (Plan.parents, accept.acceptPath, shape.Shape ranks, plan_lanes tree pricing) but the flashnext CUDA backend refuses non-chain parents (cuda_lanes.zig:564 HeldDraftsNotBuilt). A sibling row needs tree verify in flashnext kernels (DeltaNet state fork + attention mask): L effort; simulate first.
- 17:28 UTC S1 done (B1, always-8, 160 held prompts, 10,996 rounds, 85,174 chain steps with top-8; head-state dump 1.7 GB at 3d24 ~/q38/prof/B_S1.mtp; raw: Mac results/rB/B_S1; table results/rB/B_S1/dsim.txt). Tools: wt/B/kit/dparse.py, dsim.py, gatetrain.py.
  SIMULATED on always-8 rounds (cost 38.6 + 1.70 x chain steps + 2.40 x rows; iid rounds; KNOWN BIAS: always-8 rounds all start right after a miss, depth-0 accept 0.756 vs 0.77-0.80 deeper, so gated policies are under-rated: sim default 3.32 tok/round vs measured 3.46; sim said 0.4/8 = +2.7 % where round 7 measured +0.3 %):
  | policy | sim tok/s | vs default |
  | default gate 0.4 / max 6 | 60.78 | |
  | oracle: verify exactly the kept drafts | 74.68 | +22.9 % |
  | ORACLE sibling: rank-2 row at the first miss | 63.98 | +5.3 % |
  | priced siblings (reach x p2 >= 0.15, max 2 rows) | 61.75 | +1.6 % |
  | rank-2 row at the lowest-p verified draft | 61.19 | +0.7 % |
  | two-threshold stop (compute while product >= 0.5, propose >= 0.4, max 6) | 61.16 | +0.6 % |
  | verify the gate-crossing draft too | 60.3-60.7 | -0.2..-0.8 % |
  At the first miss the target's token is the head's rank 2 in 38.0 %, rank 3 13.9 %, rank 4 7.5 %, outside top-8 28 %. P(top-2 right | top-1 wrong) = 0.34-0.42 at every depth.
  Two-tier head coverage (chain top-1 inside tier 1; tier = tokens seen in any top-8 of the other half of the prompts + lowest ids): 8,192 rows 84.9 %, 16,384 rows 91.8 %, 24,576 rows 95.3 %, 32,768 rows 97.2 %.
  DECISION: a draft tree needs tree verify in the flashnext kernels (L) for ~+1.5 % simulated: NOT built unless S2 (real default rounds) says more.
- 17:39 S2 done (B1, default gate, logging on, 104 held prompts: 61.22 tok/s; 7,511 rounds; raw Mac results/rB/B_S2, table dsim2.txt). MEASURED head rounds: 3.63 drafts verified, 4.20 chain steps computed (0.57 thrown away at the gate), 2.35 kept, 3.347 tok/round, accept 0.647 (chat 0.576 at 3.08 drafts; code 0.766 at 5.10; math 0.774 at 5.47).
  Sibling rows added to these REAL rounds (SIMULATED add-on, a sibling row = 2.40 ms, lands = +1 token): oracle rank-2 at the first miss +4.7 %; priced by the head's own probabilities (reach x p2 >= 0.15, at most 2 rows, 0.56 rows a round) +1.7 %; with rows at the gate-crossing position +1.0 %.
  => (a) DRAFT TREE / SECOND CANDIDATES: +1.5..1.7 % simulated at best, needs tree verify in the flashnext kernels (DeltaNet state fork, attention mask, n-gram keys): NOT BUILT (effort L, gain below the model's own error: the iid sim over-rated gate 0.4/8 by 2.4 points).
- 17:40 (b) LEARNED GATE study done on S1 (38,179 labelled chain steps, 30 % of requests held out; 3d24 ~/q38/runs/B_train.log): test AUC of "draft kept": head probability 0.886; logistic on top-8 features (p1, p2, margin, entropy, depth, product, tail mass) 0.889; MLP on the head layer's 10,240-value output state 0.763 (over-fits 26k samples), state + scalars 0.820. No signal beyond the head's own probability => DEAD (as the brief said for chain probabilities; now also for top-k shape and the hidden state at this data size). A PCA-reduced variant is queued as a last check.
- NEXT: lever 2 as a TWO-STAGE draft head (ideas item 6): rotated rank-r preview over all 79,591 rows, exact 4-bit re-score of the best K. B2 (sha 75083f7649aa3bf7) dumps the draft head's input rows; S3 run + offline study (kit/headstudy.py) in flight.
- 17:53 S3 done (B2, default gate, 104 held prompts 60.98 tok/s, 30,870 draft-head input rows dumped: 3d24 ~/q38/prof/B_S3.mix). Offline replay with the checkpoint's lm_head rows (kit/headstudy.py, Mac results/rB/B_S3/headstudy.txt), MEASURED on real head inputs:
  replay of the 4-bit head == logged top-1 in 99.28 % of steps (tool validated); the shipped 4-bit head == exact BF16 rows top-1 in 96.96 %.
  | draft head variant (bytes a chain step) | pick == full 4-bit head |
  | full 4-bit head, 79,591 rows (102 MB) | 100 % |
  | first-tier list 16,384 rows, fall back when tier prob < 0.5 (21 MB + 12.7 % fall-backs; list fitted on the same prompts: optimistic) | 98.4 % |
  | rotated preview, SVD of the head rows, rank 512 4-bit (20 MB), re-score best 64 | 76.4 % (DEAD: the head's own singular directions are not where the inputs live) |
  | rotated preview, PCA of the head INPUTS, rank 256 4-bit (10 MB), best 64 | 97.6 % |
  | rotated preview, PCA of the head inputs, rank 512 4-bit (20 MB + 5 MB scales), best 32 / 64 | 98.9 % / 99.45 % |
  | same, rotation fitted on 70 % of the requests, tested on the other 30 % (kit/mkhead2.py) | K=16 97.8 %, K=32 98.5 %, K=64 99.0 %; gate (p >= 0.4) flips 0.46 % |
  DECISION: build lever 2 as the two-stage head, rank 512, K = 64 (expected: head 0.55 ms -> ~0.2 ms a chain step, 4.2 steps a round => ~+2 % one user, minus ~1 % of drafts changed).
- 18:00 code written (wt/B/tf branch q38-B): kernels fn_q38_head2_rot / _prev / _exact in zig/kernels/cuda/flashnext_mtp.cu; Head2 in cuda_mtp.zig; plain 4-bit rows in cuda_mtp_load.zig; flag TF_FLASHNEXT_Q38_HEAD2=FILE (+ _K, default 32); file made by kit/mkhead2.py (3d24 ~/q38/head2/head2_r512.bin, 30.7 MB). B3 = first compile (8-bit preview, superseded); B4 = 4-bit preview, building. Lease renewed to 19:12 UTC.
- 18:32 B_T1 (3d24, B4, one window, COPY_LOG on in all runs): ref1 held-104 61.75 tok/s (chat 52.93 / code 84.63 / math 78.44), ident median 63.6;
  HEAD2 rank 512 K=64: held 62.38 (chat 53.34 / code 86.04 / math 79.65), ident median 64.4 => +1.0 % held (MEASURED, ref2 pending). IDENTITY: ident 51/51, held 107/107 texts identical to ref1. No panic / error lines.
  Smaller than the +2 % estimate; T2 queued (t2B.sh: TF_HOSTPROF mtpstat chain-step ms + drafts/kept per round for off, r512 K64, r512 K32, r256 K64 on the identity set). Lease renewed (80 min from 18:35).
- 18:47 B_T1 COMPLETE (MEASURED, 3d24, B4 sha 4c17094ef1252a33, one window, same 104 held prompts, 384 tokens, temp 0):
  | run | held ALL | chat | code | math | ident median |
  | ref1 (HEAD2 off) | 61.75 | 52.93 | 84.63 | 78.44 | 63.6 |
  | HEAD2 rank 512, K 64 | 62.38 | 53.34 | 86.04 | 79.65 | 64.4 |
  | ref2 (HEAD2 off) | 61.78 | 52.95 | 84.73 | 78.53 | 63.5 |
  => two-stage draft head +1.0 % (chat +0.8, code +1.6, math +1.5). Identity on vs ref1: ident 51/51, held 107/107 texts identical; ref2 vs ref1 also 51/51, 107/107. Raw: Mac results/rB/B_T1_*.
  B5 building: preview kernel with 8 rows a warp (fn_q38_head2_prev8, TF_FLASHNEXT_Q38_HEAD2_ROWS=8 default). T2 (mtpstat + round stats for variants) and T3 (KPROF per kernel) are queued on 3d24 and end by themselves.
- 19:12 T2 so far (identity set, TF_HOSTPROF mtpstat, B4): chain step off 1.67-1.71 ms; HEAD2 r512 K64 1.43-1.46 ms (-0.25 ms a step); K32 the same time. Round stats: off 3.803 drafts / 2.369 kept (accept 0.623); K64 3.896 / 2.368 (0.608); K32 3.933 / 2.358 (0.599): the two-stage head's probability reads high, so the gate verifies ~0.1 more drafts a round for the same kept tokens.
  kit/calB.py (offline, 7,718 head inputs): mean(p2 - p_full) = +0.0042 at K64 (+0.0067 at K32); (p >= 0.4) up-flips 0.89 % vs down-flips 0.10 %; odds factor c = 1.06 (K64) / 1.08 (K32) centres it.
  B6 (sha a8673e1cca549f00) = B5 (preview kernel 8 rows a warp) + TF_FLASHNEXT_Q38_HEAD2_C (default 1.06) applied to the gate's probability. T5 queued: ref1, c 1.06, c 1.0, ref2 (ident + held 107 each), ends ~20:45; lease renewed to ~20:53.
- 19:35 KPROF (serialised launches, 12 held prompts, per call): stock 4-bit draft head qmm_kernel<32,...> 558 us (102 MB); two-stage B4: preview 260 + rotation 36 + fn_row_cand 50 + re-score 13 = 359 us (+ fn_row_best 48 in both); B5 preview with 8 rows a warp 214 us. The single-block passes over the score row (fn_row_cand, fn_row_best) are 97 us of the new head.
  T2 complete: rank 256 K64 chain step 1.28-1.31 ms but kept/round 2.254 against 2.369 (ident median 62.2 < 63.7 off): rank 256 is DEAD; K32 = K64 in time, K64 keeps more.
- 19:50 kit/calB2.py: block candidates (best M preview rows of each 256-row block re-scored): M=2 98.9 % agreement (c 1.06), M=3 99.46 % (c 1.02); global top-64 98.9 %.
  B7 written: block form (TF_FLASHNEXT_Q38_HEAD2_MODE=block, default): fn_q38_head2_scan (preview + per-block top 3 + tail log-sum-exp), fn_q38_head2_exact2 (933 rows), fn_q38_head2_fin (best + log-sum-exp into Draft.best; fn_row_cand and fn_row_best are not launched). Build queued behind T5's c106 run; T6 queued behind T5 (ref1, block, ref2, then mtpstat). Lease renewed to ~21:43.
  NOTE ruler: the same flags on different binaries differ (B4 refs 61.75 / 61.78, B6 ref1 62.20): compare on/off within ONE binary only.
- 20:30 T5 COMPLETE (MEASURED, 3d24, B6 sha a8673e1cca549f00 = top-k form, preview 8 rows a warp, gate probability calibrated; one window; 104 held prompts):
  | run (in order) | held ALL | chat | code | math | ident median | drafts / kept a round |
  | ref1 (HEAD2 off) | 62.20 | 53.32 | 85.22 | 79.08 | 64.1 | 3.625 / 2.293 |
  | HEAD2 r512 K64, c 1.06 | 63.05 | 53.93 | 86.85 | 80.53 | 64.7 | 3.619 / 2.271 |
  | HEAD2 r512 K64, c 1.00 | 62.82 | 53.72 | 86.58 | 80.27 | - | 3.684 / 2.289 |
  | ref2 (HEAD2 off) | 61.90 | 53.04 | 84.88 | 78.75 | 63.6 | 3.625 / 2.293 |
  => c 1.06: +1.6 % against the mean of the two references (+1.4 % vs ref1, +1.9 % vs ref2); c 1.00: +1.2 %. The references drifted 0.5 % across the hour.
  Identity: both HEAD2 runs and ref2 against ref1: ident 51/51, held 107/107 texts identical; 0 panic / error lines. Raw: Mac results/rB/B_T5_*.
  B7 (sha 25683938f4de8020, block form) built; T6 running since 20:30 (ref1, block, ref2, mtpstat), ends ~21:25.
- 21:00 T6 so far (B7 sha 25683938f4de8020, block form default c 1.02): ref1 held 61.85 (chat 53.01 / code 84.83 / math 78.68; 3.625 drafts / 2.293 kept); block form 63.02 (53.93 / 86.65 / 80.44; 3.630 / 2.279) = +1.9 % vs ref1; ident 51/51 + held 107/107 identical, 0 error lines. ref2 + mtpstat + file-reproduction check still running (end ~21:27).
  Handoff written: Mac q38/handoff/B/draft-study-logging (patch 0110 + NOTE), handoff/B/two-stage-draft-head (patch 0111 = diff 5436541..305e4f2; base 52b32e1 + 0110 + 0111 reproduces the B7 tree 7cd70d95, checked with git write-tree), handoff/B/study (tools + tables). Source: Mac q38/wt/B/tf branch q38-B (commits 5436541, c1ed232, 305e4f2).
- 21:25:06 all runs done, no q38 container or kit-B script on 3d24, page cache evicted, LEASE B RELEASED

## BUILDER B SUMMARY (2026-11-13 17:01 -> 21:27 UTC, one shared window, lease B 17:10 -> 21:25, released)
| item | result | status |
|---|---|---|
| (a) draft tree / second-candidate rows | SIMULATED on real default rounds: oracle +4.7 %, priced by the head's own probabilities +1.7 % (0.56 extra rows a round); needs tree verify in the flashnext kernels (the core engine has Plan.parents, the CUDA backend refuses non-chains) | NOT BUILT (effort L, gain inside the sim's error) |
| (b) better gate signal | test AUC of "draft kept": head probability 0.886, + top-8 shape features 0.889, MLP on the head's 10,240-value state 0.763 / 0.820 | DEAD (no signal beyond the head's probability at 38k labelled steps) |
| (c) public parallel drafter | both exist with weights (incoai DFlash2, Apache-2.0; shadowpa0327 DSpark, license "other"); DSpark card, greedy block 7: 4.75 vs MTP 4.67 on chat mix, 3.70 vs 4.24 on agent work; needs SGLang + a QSA build for width 8 that is not public | NOT MEASURED here; only draft COST could win (est. +5..7 % one user chat if an 8-bit body holds acceptance); port = L |
| lever 2: two-stage draft head (patch 0111, TF_FLASHNEXT_Q38_HEAD2) | MEASURED +1.6 % held-104 paired (61.85 / 62.15 refs -> 63.02; chat +1.5, code +1.9, math +2.0); chain step 1.69 -> 1.37 ms; ident 51/51, held 107/107 identical | BUILT, in handoff |
| lever 2 variants | rank 256: kept drafts -4.9 %, slower than off (dead); SVD-of-head rotation 76 % agreement (dead); K32 = K64 in time; uncalibrated gate +1.2 % vs calibrated +1.6 % | measured |
| tier-by-list head (frequent rows first) | offline 98.4 % agreement at 16,384 rows + 12.7 % fall-backs, but the list is fitted to the prompts (domain bound) and needs a host round trip a fall-back | superseded by the two-stage head |
| lever 3: 6-bit coarse copy for the certified head | NOT ATTEMPTED: at ~5 verify rows the coarse pass is 3.2 G MACs, so it must stay on the MMA tile kernel (qmmf); a 6-bit decode inside that kernel is a kernel project for <= 0.15 GiB = ~0.5 ms = ~+0.9 % (estimate). A rotated low-rank preview cannot certify (Cauchy-Schwarz on the remainder is ~10x too loose). | open, lowest priority |
| logging aids (patch 0110) | TF_STEP_LOG=2, TF_FLASHNEXT_Q38_MTPDUMP / MTPDUMP2 | in handoff |
Measured facts worth keeping: default gate computes 4.20 chain steps for 3.63 verified drafts (0.57 thrown away at the gate); accept by category chat 0.576 / code 0.766 / math 0.774; at the first miss the target's token is the head's rank 2 in 38 %; depth-0 accept (right after a miss) 0.756 vs 0.77-0.80 deeper, which is why iid-round sims over-rate deeper gates; the shipped 4-bit draft head already differs from exact BF16 rows in 3 % of picks.
Lessons: (1) a background waiter with a watchdog subshell keeps the script's name alive in pgrep: wait on a marker line, not on pgrep. (2) `bash -c "cmd && setsid nohup ... &"` over ssh hangs the ssh: wrap the detached part in ( ... &). (3) A helper container named q38-run* blocks runB.sh's start check; CPU-only analysis containers get another name and run during an engine load. (4) the paired ruler here is +-0.05 % within minutes but drifts 0.5 % in an hour: keep references first and last. (5) a preview's log-sum-exp under-counts tail mass: calibrate the gate's probability (+0.4 % here).
What remains (in order of value): integrate 0111 + file step into the recipe (builder A; run the full one-user suite and 16 users with it; re-fit the rotation on broader prompts); a cp.async tile kernel for the preview (~+0.7 % est.); bf16 rotation; the two-stage head for shared rounds and sampled requests; tree verify (+1.7 % sim); DSpark/DFlash2 acceptance at temperature 0 on our prompts through SGLang before any port; 6-bit coarse copy.
State left on 3d24: no container, no script; builds ~/q38/tfbin/B1..B7 (B7 = handoff); ~/q38/head2/head2_r512.bin (tested file) + head2_rot_r512.rot; dumps ~/q38/prof/B_S1.mtp (1.7 GB), B_S3.mix (158 MB); runs ~/q38/runs/B_*; build tree ~/q38/build-B; kit ~/q38/kit-B. Page cache evicted (117 GiB free).

# Round 9 (builder B, started 2026-11-13 22:59 UTC)
- start: reading state; BASE.txt not present yet; fallback base = integ2 0b5cf9b (wt/I2, patches 0101-0115). No lease held.
- 23:25 UTC base = integ2 0b5cf9b (patches 0101-0115); new branch q38-B9 in Mac wt/B/tf. Tools for this round: Mac wt/B/kit9/.
- 23:40 SIMULATED on the real default rounds of round 8 (B_S2, 7,274 head rounds; tool kit9/dsim3.py, table kit9/dsim3_S2.txt; cost 38.6 + 1.70 x steps + 2.40 x rows; second column = with HEAD2, step 1.37):
  | add-on | HEAD2 off | HEAD2 on |
  | sibling (rank-2) row, priced reach x p2 >= 0.15, any position, 1 row (0.49 rows a round, 25.8 % land) | +1.59 % | +1.57 % |
  | same, only at the LAST verified draft (0.24 rows a round) | +0.76 % | +0.75 % |
  | same, only at the FIRST draft (0.23 rows a round) | +1.04 % | +1.03 % |
  | oracle (row exactly at the first miss) | +4.66 % | +4.64 % |
  | early stop: do not compute chain step j+1 when the product after draft j < T2 (0.50 / 0.55) | +0.55 / +0.61 % | +0.34 / +0.30 % |
  How chains end (measured): 56.9 % at the gate (one step computed and thrown away), 34.1 % at depth 6, 9.0 % weak first draft.
  After a product in [0.40,0.45) the next draft passes the gate in 41.7 % and is kept in 10.1 %; [0.45,0.50) 54.4 / 14.8 %; [0.50,0.55) 60.1 / 18.4 %; [0.55,0.60) 67.6 / 23.9 %.
  "Re-use the thrown-away step next round" is not possible: its token sits at the position the next round's first row resolves anyway, and the step that would continue from it was never computed.
- 23:45 DECISION tree verify: read the backend (cuda_lanes.zig verifyFn, cuda_mtp_ops.zig verify / rollback, cuda_batch.zig attnRows / gdnBlock / pleBlock). A window is the prompt-pass
  forward over consecutive rows: every sequence mechanism is prefix-based (attention writes K/V + indexer keys into the cache by position and pools blocks; DeltaNet chain + late log;
  conv taps; n-gram tail; rollback = "keep the first n rows"). A sibling needs a second cache slot, a visibility mask in 6 attention kernels (x fp8 / bf16 / lane-batched forms), a state
  fork in fn_gdn_rows_late / front / tail, and a path-based rollback. Two designs noted for a follow-up (handoff note): (a) in-kernel fork; (b) "alias lane": the sibling path
  [pending, d1..d(j-1), s] as a second lane of the same pass on the same sequence (duplicate prefix rows are bit-equal and share experts, so cheap), rollback on that lane's own keep data.
  Both are several days of engine work with exactness risk in the caches for +1.0..1.6 % simulated. NOT BUILT this round; the cheap exact levers first (early stop, preview kernel).
- 23:17 UTC (earlier stamps today are ~25 min fast) B8 built (sha 8d74625319d7e60e, commit 6c46c35 on q38-B9): TF_FLASHNEXT_Q38_STOP2=x (early chain stop) and TF_FLASHNEXT_Q38_HEAD2_FAST bits (1 = integer preview fn_q38_head2_scan2 with dp4a, 2 = bf16 rotation fn_q38_head2_rot2). Microbench kit9/mb/mbh2.cu (real head2 file + real head inputs).
  Lease B acquired 23:17 (window was already open since 23:06, guard 04:05). w9a.sh running on 3d24 (log ~/q38/runs/B_w9a.log): microbench; HEAD2 off: ref, STOP2 0.50, 0.55, ref; HEAD2 on: ref, FAST=3, FAST=3+STOP2 0.5, ref.
- 23:20 MICROBENCH (MEASURED, 3d24 GPU, real head2 file + 1,500 real head inputs, engine launch shapes, a 192 MB flush read before every launch; kit9/mb/mbh2.cu):
  | kernel | us a launch |
  | fn_q38_head2_scan (stock preview, 25.5 MB) | 236 |
  | fn_q38_head2_scan2 (input as int8, dp4a) | 116 (= 220 GB/s: at the bus; 65 us when the bytes are cached) |
  | fn_q38_head2_rot fp32 / rot2 bf16 | 26 / 15 |
  Candidates: 97.2 % of blocks keep the same 3 rows, the stock preview best row is among the candidates in 100 % of 1,500 inputs. => -131 us a chain step expected (~+1.0 % one user with HEAD2).
- 23:32 B9_ref1 (B8, HEAD2 off, flags off; MEASURED): held-104 62.49 (chat 53.59 / code 85.59 / math 79.30), ident median 64.4, 3.625 drafts / 2.293 kept a round. Lease renewed to ~01:25.
- 23:30 SIMULATED on the always-8 rounds (S1; kit9/gsim.py + inline): going past depth 6 pays only when the product after 6 drafts is >= 0.8 (11.8 % of rounds have >= 0.9: +1.27 tokens for 7.9 ms; [0.4,0.6): negative), and such rounds are 43 % of code, 55 % of math, 9 % of chat rounds. Round 7 measured 0.4/8 at +0.3 % (tokens +4.0 %, round time +3.7 %). => new flag TF_FLASHNEXT_Q38_DEEP=P[,BASE] (B9, commit on q38-B9): with TF_FLASHNEXT_DRAFTS=12 a lone chain passes BASE=6 drafts only while its product >= P. Build B9 queued behind the next engine load. Weak first drafts (p < 0.2) are still kept in 15.5-16.5 % (break-even 14.7 %): nothing to cut there.
- 23:48 B9_s50 (B8, STOP2=0.5, HEAD2 off; MEASURED): held-104 62.72 (chat 53.88 / code 85.64 / math 79.26) vs ref1 62.49 = +0.37 % (ref2 pending); 3.483 drafts / 2.293 kept a round (ref 3.625 / 2.293); identity vs ref1: ident 51/51, held 107/107; 0 error lines. B9 built (sha 49164345a523c1ad). w9b.sh queued behind w9a (DEEP tests).
- 00:04 B9_s55 (STOP2=0.55; MEASURED): held 62.57 (53.72 / 85.39 / 79.29), 3.371 drafts / 2.249 kept; ident 51/51, held 107/107 vs ref1. Less than 0.50 (62.72): 0.5 is the setting. Per-depth thresholds simulate no better than a flat one (+0.65 vs +0.61 %).
- 00:19 EARLY STOP, part 1 COMPLETE (MEASURED, 3d24, B8 sha 8d74625319d7e60e, one window, HEAD2 off, 104 held prompts x 384, temp 0, COPY_LOG on in all; runs in this order):
  | run | held ALL | chat | code | math | ident median | drafts / kept a round | texts vs ref1 |
  | ref1 | 62.49 | 53.59 | 85.59 | 79.30 | 64.4 | 3.625 / 2.293 | |
  | STOP2=0.50 | 62.72 | 53.88 | 85.64 | 79.26 | 63.8 | 3.483 / 2.293 | ident 51/51, held 107/107 |
  | STOP2=0.55 | 62.57 | 53.72 | 85.39 | 79.29 | 63.5 | 3.371 / 2.249 | ident 51/51, held 107/107 |
  | ref2 | 62.12 | 53.28 | 85.09 | 78.78 | 63.7 | 3.625 / 2.293 | ident 51/51, held 107/107 |
  The references drifted -0.6 % in the hour. Against the reference interpolated to each run's time: STOP2=0.50 +0.56 % (chat +0.7, code +0.3, math +0.2); 0.55 +0.53 %.
  Against the mean of the two references: +0.67 % / +0.43 %. Simulated was +0.55 / +0.61 %. 0 error lines. => STOP2=0.5 is a small exact positive.
- 00:33 B9_h_ref1 (B8, HEAD2 on, FAST off; MEASURED): held 63.27 (54.20 / 86.89 / 80.49), 3.630 / 2.279; texts = ref1 (51/51, 107/107). Lease renewed to ~03:23. B10 (commit: TF_FLASHNEXT_Q38_HEAD2_SAMPLED=1, two-stage head for a lone sampled chain) queued to build at the next engine load.
  (correction: a lease call is capped at 120 min: lease B runs to 02:32; box deadline file set to 02:31:30.) B10 built: sha da979a99fddd03c8.
- 00:48 B9_h_fast (B8, HEAD2 on + HEAD2_FAST=3; MEASURED): held 63.87 (54.69 / 87.90 / 81.21) vs h_ref1 63.27 = +0.95 % (h_ref2 pending); 3.630 / 2.279 a round (same as HEAD2 without FAST); texts = ref1: ident 51/51, held 107/107; 0 error lines.
- 01:03 B9_h_fast_s50 (HEAD2 + FAST=3 + STOP2=0.5; MEASURED): held 64.42 (55.23 / 88.23 / 81.91), 3.484 / 2.281; ident 51/51, held 107/107 vs ref1; 0 error lines. (h_ref1 63.27, h_fast 63.87; h_ref2 running.)
- 01:17 PREVIEW IN INTEGERS + EARLY STOP WITH HEAD2, part 2 COMPLETE (MEASURED, 3d24, B8 sha 8d74625319d7e60e, same window, HEAD2 on in all four, runs in this order):
  | run | held ALL | chat | code | math | ident median | drafts / kept a round | texts vs ref1 (HEAD2 off) |
  | HEAD2 ref1 | 63.27 | 54.20 | 86.89 | 80.49 | 64.0 | 3.630 / 2.279 | ident 51/51, held 107/107 |
  | + HEAD2_FAST=3 | 63.87 | 54.69 | 87.90 | 81.21 | 64.4 | 3.630 / 2.279 | ident 51/51, held 107/107 |
  | + HEAD2_FAST=3 + STOP2=0.5 | 64.42 | 55.23 | 88.23 | 81.91 | 64.7 | 3.484 / 2.281 | ident 51/51, held 107/107 |
  | HEAD2 ref2 | 63.28 | 54.21 | 86.87 | 80.54 | 64.0 | 3.630 / 2.279 | ident 51/51, held 107/107 |
  => HEAD2_FAST=3: +0.94 % over HEAD2 (chat +0.9, code +1.2, math +0.9), drafts and kept unchanged; with STOP2=0.5 on top: +1.81 % over HEAD2 (+1.9 / +1.5 / +1.8).
  No drift in this hour (63.27 / 63.28). 0 error lines in every run. Against the HEAD2-off references of part 1 (62.49 / 62.12) the stack reads 64.42 = +3.4 %.
  w9b.sh (B9, DEEP tests: ref, 0.85 at 12 drafts, 0.9 at 12, 0.8 at 8, ref) started 01:16, ends ~02:31.
- 01:47 DEEP FIRST RESULT (MEASURED, 3d24, B9 sha 49164345a523c1ad, HEAD2 off): d_ref1 held 62.19 (chat 53.33 / code 85.24 / math 78.89; 3.625 drafts / 2.293 kept; ident median 64.0);
  TF_FLASHNEXT_DRAFTS=12 + TF_FLASHNEXT_Q38_DEEP=0.85,6: held 63.28 (chat 53.41 / code 89.48 / math 84.05; 3.984 / 2.469; ident median 65.3) = +1.75 % (chat +0.2, code +5.0, math +6.5; ref2 pending);
  texts vs d_ref1: ident 51/51, held 107/107; 0 error lines. Next in w9b: 0.9 at 12, 0.8 at 8, ref2.
- 02:02 B9_d90 (DRAFTS=12 + DEEP=0.9,6; MEASURED): held 63.46 (chat 53.51 / code 90.05 / math 84.37; 3.919 / 2.444; ident median 65.7) = +2.04 % vs d_ref1 62.19 (0.85 gave 63.28); ident 51/51, held 107/107; 0 error lines.
- 02:18 B9_d80_8 (DRAFTS=8 + DEEP=0.8,6; MEASURED): held 62.52 (53.25 / 86.90 / 80.76; 3.883 / 2.449); ident 51/51, held 107/107. Depth 8 gives a quarter of what depth 12 gives. w9c.sh queued on B10 (= handoff tree): ref, DEEP 0.9 at 15, at 24, STOP2 0.5 + DEEP 0.9 at 12, ref.
  Handoff written (interim): Mac q38/handoff/B/r9-draft-side/ (patch = diff 0b5cf9b..2708a5b, TREE_EQUAL 2168edda, NOTE.md), q38/handoff/B/r9-tree-verify-design/NOTE.md.
- 02:32 DEEP CHAINS, w9b COMPLETE (MEASURED, 3d24, B9 sha 49164345a523c1ad, one window, HEAD2 off, runs in this order):
  | run | held ALL | chat | code | math | ident median | drafts / kept a round | texts vs d_ref1 |
  | d_ref1 | 62.19 | 53.33 | 85.24 | 78.89 | 64.0 | 3.625 / 2.293 | |
  | DRAFTS=12, DEEP=0.85,6 | 63.28 | 53.41 | 89.48 | 84.05 | 65.3 | 3.984 / 2.469 | ident 51/51, held 107/107 |
  | DRAFTS=12, DEEP=0.9,6 | 63.46 | 53.51 | 90.05 | 84.37 | 65.7 | 3.919 / 2.444 | ident 51/51, held 107/107 |
  | DRAFTS=8, DEEP=0.8,6 | 62.52 | 53.25 | 86.90 | 80.76 | 65.0 | 3.883 / 2.449 | ident 51/51, held 107/107 |
  | d_ref2 | 62.01 | 53.16 | 84.99 | 78.74 | 63.8 | 3.625 / 2.293 | ident 51/51, held 107/107 |
  Against the reference interpolated to each run's time: 0.85 at 12 +1.83 %; 0.9 at 12 +2.19 % (chat +0.5, code +5.8, math +7.0); 0.8 at 8 +0.75 %. 0 error lines.
  => DEEP is the round's largest lever: +2.2 % one user on the held mix, almost all of it in code and math. w9c (B10) running: deeper maxima + the stack.
- 03:01 w9c (B10 sha da979a99fddd03c8 = handoff tree): e_ref1 held 62.06 (53.21 / 85.08 / 78.71; 3.625 / 2.293; ident median 63.9); DRAFTS=15 + DEEP=0.9,6: 63.66 (53.52 / 90.51 / 85.94; 3.977 / 2.470; ident median 65.4) = +2.58 % vs e_ref1 (chat +0.6, code +6.4, math +9.2); ident 51/51, held 107/107; 5 streams at 262K still fit. Running: 0.9 at 24, STOP2 + DEEP at 12, ref2. Lease B to 04:10 (guard 04:05).
- 03:17 DRAFTS=24 + DEEP=0.9,6 (B10; MEASURED): held 63.67 (53.57 / 90.16 / 85.98; 4.032 / 2.494); ident 51/51, held 107/107; = the 15-draft result (63.66): the depth saturates at ~15 on this mix. Setting: DRAFTS=15 (a 16-row window), DEEP=0.9,6.
- 03:32 B9_e_all (B10, HEAD2 off: STOP2=0.5 + DRAFTS=12 + DEEP=0.9,6; MEASURED): held 63.99 (chat 54.05 / code 90.54 / math 84.62; 3.688 / 2.357) vs e_ref1 62.06 = +3.1 % (ref2 running); ident 51/51, held 107/107; 0 error lines. w9d.sh queued (sampled chain with the two-stage head, D ruler sq.py, two 7-minute runs before the 04:01 deadline).
- 03:46 w9c COMPLETE (MEASURED, 3d24, B10 sha da979a99fddd03c8 = the handoff tree, one window, HEAD2 off, runs in this order):
  | run | held ALL | chat | code | math | ident median | drafts / kept a round | texts vs e_ref1 |
  | e_ref1 | 62.06 | 53.21 | 85.08 | 78.71 | 63.9 | 3.625 / 2.293 | |
  | DRAFTS=15, DEEP=0.9,6 | 63.66 | 53.52 | 90.51 | 85.94 | 65.4 | 3.977 / 2.470 | ident 51/51, held 107/107 |
  | DRAFTS=24, DEEP=0.9,6 | 63.67 | 53.57 | 90.16 | 85.98 | 65.4 | 4.032 / 2.494 | ident 51/51, held 107/107 |
  | STOP2=0.5 + DRAFTS=12, DEEP=0.9,6 | 63.99 | 54.05 | 90.54 | 84.62 | 64.5 | 3.688 / 2.357 | ident 51/51, held 107/107 |
  | e_ref2 | 62.32 | 53.45 | 85.37 | 79.02 | 64.2 | 3.625 / 2.293 | ident 51/51, held 107/107 |
  Against the reference interpolated to each run's time (the ruler rose 0.4 % this hour): DEEP 0.9 at 15 drafts +2.47 % (chat +0.5, code +6.3, math +9.1); at 24 +2.38 %;
  STOP2 + DEEP at 12 +2.79 % (chat +1.2, code +6.1, math +7.2). 0 error lines. 5 streams at 262K still fit with DRAFTS=24 (start line unchanged).
- 03:56 SAMPLED CHAIN WITH THE TWO-STAGE HEAD (MEASURED, 3d24, B10, builder D ruler sq.py: temperature 0.7, thinking off, seed 9000 + id, 30 held chat prompts x 256, HEAD2 + FAST=3 in both):
  HEAD2_SAMPLED off 47.07 tok/s -> on 47.93 = +1.83 % (per-prompt ratio +1.94 % +- 0.37); texts identical 30/30 (same seeds); 0 error lines.
- 03:57 all runs in; no q38 container or script of mine on 3d24; page cache evicted; LEASE B RELEASED (window closes by the guard 04:05 or when the last lease goes).
- 03:59 handoff NOTE finalized (Mac q38/handoff/B/r9-draft-side/NOTE.md; patch unchanged c850019b). Builder A had already integrated the interim version (R9=1: DEEP 0.9 at 12, STOP2 0.5, HEAD2_FAST 3; A measured the whole stack 63.75 -> 66.56 on held-104); the NOTE now tells A that 15 drafts is better (+2.47 vs +2.19 %) and that HEAD2_SAMPLED is measured (+1.83 %, 30/30).
  Calibration (S1, drafts 6-8, SIMULATED/measured offline): product 0.9-0.95 -> 59 % whole chains, 0.95-0.98 -> 71 %, >= 0.99 -> 92 %; a step with own p >= 0.98 is right 93-97 % given the chain so far. Next candidates: P 0.95 / 0.98 at 15, HEAD2 full stack at 15, sampled with DEEP + STOP2, 16 users with DRAFTS=15.
- 04:03 second window (opened by A at ~04:00 for the final suite): lease B acquired to 05:42; w9e.sh on 3d24 (B10): S4 = always-16 chains log (for a sequence-aware gate simulator), DEEP 0.95 at 15 between references, sampled requests default vs DEEP + STOP2 (D ruler). Log ~/q38/runs/B_w9e.log.
- 04:24 S4 done (B10, always-16 chains, 107 held prompts, 6,358 head rounds, 95,687 chain steps with top-8; Mac results/rB/B9_S4). Sequence-aware simulator kit9/seqsim.py (replays a policy along the real token sequence; a round that stops inside a logged chain continues from the rest of that chain).
  On S4 it gives STOP2 0.5 +0.38 % (measured +0.56), DEEP 0.9 at 12 +2.83 (measured +2.19), at 15 +3.35 (2.47), 0.85 at 12 +3.02 (1.83), 0.8 at 8 +2.09 (0.75), 0.4/8 +1.77 (0.3): right sign and size class, still ~1 point generous to deep policies (a 16-draft log truncates the shallow policy inside long easy chains). S5 (always-31) queued to remove that.
- 04:47 w9e (B10): g_ref1 held 62.22 (53.37 / 85.21 / 78.89); DRAFTS=15 + DEEP=0.95,6: 63.35 (53.34 / 89.75 / 85.09; 3.862 / 2.410) = +1.8 % (ref2 pending): below 0.9 at 15 (+2.47 %). P = 0.9 is the peak of 0.85 / 0.9 / 0.95. ident 51/51, held 107/107.
- 05:02 g_ref2 62.11 => DEEP 0.95 at 15: +1.9 % against the mean reference (62.17); texts identical. Lease B renewed to 05:52.
- 05:16 SAMPLED REQUESTS with DEEP + STOP2 (MEASURED, B10, HEAD2 off, D ruler: temperature 0.7, thinking off, seed 9000 + id, 256 tokens): default -> DRAFTS=15 + DEEP=0.9,6 + STOP2=0.5:
  30 chat prompts 46.73 -> 46.97 (+0.5 %), texts identical 30/30; 20 code prompts 65.99 -> 68.53 (+3.85 %; per-prompt +4.2 % +- 2.9), identical 20/20; 0 error lines. S5 (always-31 log) running, ends ~05:33.
- 05:36 S5 done (always-31 chains); no q38 container or script of mine on 3d24, page cache evicted, LEASE B RELEASED (window stays open on A and D leases).
- 05:45 SEQUENCE-AWARE SIMULATOR on S5 (always-31 chains, 6,329 head rounds, 138,953 chain steps; kit9/seqsim.py, table kit9/seqsim_S5.txt; SIMULATED):
  | policy | simulated | measured |
  | STOP2 0.5 | +0.41 % | +0.56 % |
  | DEEP 0.9 at 12 / at 15 | +2.68 / +3.00 % | +2.19 / +2.47 % |
  | DEEP 0.85 at 12 | +2.83 % | +1.83 % |
  | DEEP 0.8 at 8 | +1.84 % | +0.75 % |
  | gate 0.4, depth 8 for every chain | +1.50 % | +0.3 % (round 7) |
  | STOP2 0.5 + DEEP 0.9 at 12 | +3.15 % | +2.79 % |
  Right sign and order of size, 0.4-1 point generous to deeper / looser policies. Search over base 3-8, P 0.8-0.98, per-depth schedules, a recalibrated product, gate 0.3-0.5:
  nothing more than +0.5 point above the measured setting (best: base 5, P 0.8: +3.97 % against +3.49 % for the measured setting), which is inside the simulator's error and
  in the direction (looser) where it is known to be generous; measured P 0.85 < 0.9 > 0.95. => no further gate build; STOP2 0.5 + DEEP 0.9,6 at 15 drafts stands.
- 05:43 (box clock; a few stamps above run ~8 min fast) B11 built (sha 7c0754a0b17925ee, commit on q38-B9): TF_FLASHNEXT_Q38_DEEP_MANY=1 = shared rounds under the product rule follow DEEP and STOP2 too. Lease B acquired (80 min) in the open window; w9g.sh running: 4 users (5 x 262K): ref, on, ref; 16 users (16 slots): ref, on; reference = DEEP + STOP2 for lone chains only. Log ~/q38/runs/B_w9g.log.
- 05:55 DEEP_MANY at 4 users (MEASURED, 3d24, B11, 5 x 262K, reference = DEEP + STOP2 for lone chains only): 4 distinct code prompts 154.3 / 159.4 / 156.0 -> 154.1 / 153.9 / 155.6; held 107 with 4 in flight 113.0 -> 113.4 tok/s (+0.35 %); texts identical 107/107; 0 error lines. ZERO at 4 users (ref2 and 16 users still running).
- 06:15 DEEP_MANY COMPLETE (MEASURED, 3d24, B11 sha 7c0754a0b17925ee; reference = DRAFTS=15 + DEEP=0.9,6 + STOP2=0.5 acting on lone chains only):
  | load | reference | DEEP_MANY=1 | texts |
  | 4 distinct code prompts (5 x 262K), three runs | 154.3 / 159.4 / 156.0 and 158.9 / 156.8 / 158.2 | 154.1 / 153.9 / 155.6 | |
  | held 107 with 4 in flight | 113.0, 112.8 | 113.4 | 107/107 identical |
  | 16 distinct code prompts (16 slots, 128-row passes) | 274.1 / 279.4 / 274.5 | 273.6 / 272.0 / 274.3 | |
  | held 107 with 16 in flight | 186.7 | 186.6 | 107/107 identical |
  => ZERO (4 users: -1.8 % on code, +0.4 % on held; 16 users: 0). DEAD: the flag stays off and is NOT in the handoff patch (commit on q38-B9 only). Side result: the lone-chain
  settings (DRAFTS=15, DEEP, STOP2) leave 16 users where builder A measured them (272-282).
- 06:16 no q38 container or script of mine on 3d24, page cache evicted, LEASE B RELEASED.

## BUILDER B ROUND 9 SUMMARY (2026-11-13 23:00 -> 2026-11-14 06:16 UTC; leases B 23:17-03:56, 04:02-05:36, 05:40-06:16, all released)
| lever | simulated first | measured (3d24, held-104 x 384, temp 0, paired, references first and last) | identity | status |
|---|---|---|---|---|
| DEEP: lone chain past 6 drafts only while product >= 0.9 (`TF_FLASHNEXT_DRAFTS=15 TF_FLASHNEXT_Q38_DEEP=0.9,6`) | always-8 log: only chains with product >= 0.8 after 6 drafts gain | +2.47 % (chat +0.5, code +6.3, math +9.1); 12 drafts +2.19; 24 = 15; P 0.85 +1.83, 0.95 +1.9, 0.8 at 8 +0.75 | ident 51/51, held 107/107 each | in handoff; A integrated (at 12) |
| early stop (`TF_FLASHNEXT_Q38_STOP2=0.5`) | +0.55 % | +0.56 % (0.55: +0.53); +0.86 % on top of HEAD2 + FAST | 51/51, 107/107 | in handoff; A integrated |
| preview in integers + bf16 rotation (`TF_FLASHNEXT_Q38_HEAD2_FAST=3`) | microbench 236 -> 116 us, 26 -> 15 us | +0.94 % over HEAD2 | 51/51, 107/107 | in handoff; A integrated |
| two-stage head for sampled chains (`TF_FLASHNEXT_Q38_HEAD2_SAMPLED=1`) | D's estimate +1.6..2 % | +1.83 % (seeded, temperature 0.7) | seeded texts 30/30 | in handoff; measured after A's integration |
| DEEP + STOP2 on sampled requests | | chat +0.5 %, code +3.85 % | seeded texts 30/30, 20/20 | same flags |
| stack, HEAD2 off: STOP2 + DEEP at 12 | | +2.79 % (chat +1.2, code +6.1, math +7.2) | 51/51, 107/107 | |
| DEEP_MANY (shared rounds follow the same rules) | not simulated | 4 users -1.8 % / +0.4 %, 16 users 0 | 107/107 in flight | DEAD, not in handoff |
| tree verify (rank-2 sibling row) | any position +1.6 %, first +1.0 %, last +0.75 %, oracle +4.7 % | not built | | design note in handoff |
| re-use of thrown-away chain steps | not possible (the step that would continue was never computed) | | | dropped |
| better gate schedules (base, P, per-depth, recalibrated product) | sequence-aware sim: nothing > +0.5 point over the measured setting | P 0.85 / 0.95 measured below 0.9 | | no build |
Handoff: Mac q38/handoff/B/r9-draft-side/ (patch c850019b = diff 0b5cf9b..2708a5b, TREE_EQUAL 2168edda; NOTE.md; microbench; gsim.py, seqsim.py + table, dparse.py),
q38/handoff/B/r9-tree-verify-design/ (NOTE.md, dsim3.py + table). Source: Mac q38/wt/B/tf branch q38-B9 (6c46c35 B8, b8956b7 B9, 2708a5b B10 = handoff, + B11 DEEP_MANY dead).
For the integrator: use DRAFTS=15 (A's recipe has DEEP_MAX=12: +0.3 point left) and consider HEAD2_SAMPLED=1 (+1.8 % sampled).
What remains (in order): tree verify by the alias-lane route (+0.8..1.6 % simulated, days of work); DEEP on text outside the held mix (the 0.9 line was set on these 104 prompts:
check the fresh set); HEAD2 for shared rounds needs an MMA preview (scalar/dp4a preview is per-row compute); 6-bit coarse copy for the certified head (unchanged from round 8).
State on 3d24: no container, no script; binaries ~/q38/tfbin/B8..B11; logs ~/q38/prof/... unchanged + runs ~/q38/runs/B9_*; S4 / S5 chain logs pulled to Mac results/rB/B9_S4, B9_S5.
RESUME: cd ~/.hermes/cache/scratch/q38/wt/B; ./push9.sh (kit9 -> box kit-B); bash shipB.sh NAME (build); box scripts kit-B/{lib9.sh,runB9.sh,w9*.sh,tab9.sh}; lease via ../../h.sh.
- 06:18 run logs of every round-9 run pulled to Mac results/rB/r9_runlogs/ (run.log a run, window logs, sampled jsonl). Handoff NOTE has the DEEP_MANY dead note. Skill reference qwen38-fastest-build.md updated (round-9 draft-side section). END.

# Round 10 (builder B, started 2026-11-14 06:54 UTC; lever area: workload-aware drafts: prediction field, suffix store, grammar-forced drafts, DEEP check)
- start: reading state. No lease held. Base = integ9 7061396 (wt/I9), worktree wt/B10 branch r10-B.
- 07:01 lease B acquired (120 min, window already open by C/D); B12 = integ9 7061396 built on 3d24 (sha 57821d885c0ad2f9). Workloads written: kit10/mkagent.py (16 bug-fix repos with real tools, pred sets plain/numbered/toolwrite, struct sets schema/call) + kit10/ar.py (driver, real agent loop, tests run in a network-less container). FOUND in the engine: (1) --recall = a cross-request record of prompts+replies that drafts copies (core/lanes/recall*.zig, bands 16/32/64), (2) grammar drafts already exist (windows.constrain: xgrammar's forced tokens appended as drafts when response_format / tool_choice required), (3) no prediction field. w10a.sh running on 3d24: L1 (token-id log of all workloads), RC (--recall test), DEEP check ref/on12/on15/ref on fresh-123 + loop + struct. Log ~/q38/runs/B10_w10a.log
- 07:22 (box clock) L1 DONE (3d24, B12 = integ9, round-9 defaults, TF_STEP_LOG=2, one user, temp 0, thinking off; 295 requests, 92,465 reply tokens; raw Mac results/rB/B10_L1; MEASURED, logging on):
  | set (what it is) | n | reply tokens | decode tok/s | wall tok/s | tokens a round | accept | prefill share of wall |
  | loop_fix (real agent loop: 16 repos with a seeded bug, tools list/read/write/str_replace/run_tests really executed, 5.6 turns a session) | 90 | 8,579 | 104.0 | 76.9 | 7.9 | 0.70 | 26 % |
  | loop_follow (second session on the repaired repo: rewrite the file with docstring examples, extend the tests) | 97 | 27,531 | 107.5 | 92.7 | 8.7 | 0.67 | 14 % |
  | plain (rewrite a file that sits in the prompt as it must come back) | 24 | 19,505 | 203.5 | 180.0 | 19.4 | 0.89 | 12 % |
  | numbered (the prompt holds `cat -n` output; the reply is the plain file) | 16 | 13,244 | 99.2 | 92.1 | 7.9 | 0.64 | 7 % |
  | toolwrite (reply = a write_file tool call with the whole file) | 16 | 5,472 | 183.8 | 141.2 | 17.2 | 0.89 | 23 % |
  | regen (same ask + a small change; the first reply is NOT in the prompt) | 12 | 8,120 | 86.8 | 85.2 | 5.5 | 0.71 | 2 % |
  | schema (response_format json_schema) | 12 | 2,068 | 89.7 | 80.1 | 6.3 | 0.76 | 11 % |
  | call_required / call_auto (one short tool call) | 8 / 8 | 481 / 490 | 114.9 / 110.0 | 82.8 / 97.3 | 9.1 / 8.5 | 0.89 / 0.78 | 28 / 12 % |
  FACTS: (1) this model's tool calls are XML-like (`<tool_call>\n<function=N>\n<parameter=K>\nVALUE\n</parameter>\n</function>\n</tool_call>`), values are RAW text, so a write_file call copies from the prompt
  like a plain rewrite; (2) tool-call replies are sent in one chunk at the end (no streaming): wall, not first-chunk time, is the ruler; (3) in a real agent loop a quarter of the wall is the prompt pass
  (short replies, ~500 new prompt tokens a turn): decode levers move agent loops by at most ~3/4 of their decode gain; (4) step-log rounds commit tokens past the end token (dropped by the engine).
- 07:27 REPLAY on L1 (kit10/parse10.py, sim10.py, forced10.py; SIMULATED; cost fitted on the run: round = 32.3 ms + 5.20 a head draft row (chain steps included) + 2.91 a copy row; the fit reproduces
  measured decode tok/s of every set within 4 %; tokens outside a replayed round keep their logged round's cost share):
  | set | measured base | (c) prediction, match 4, always | (c) match 8, always | (c) priced in hindsight | (b) other sessions' replies, match 8 | (b) match 16 | (b) same session beyond the prompt | retry (perfect store) |
  | loop_fix | 105 | 123 (+17 %) | 123 | 126 | 115 (+9 %) | 116 | 100 (-5 %) | 220 |
  | loop_follow | 109 | 116 (+6 %) | 121 (+12 %) | 134 | 134 (+23 %) | 138 | 107 (-1 %) | 246 |
  | numbered | 101 | 215 (+113 %) | 213 | 223 | (232: the plain set had produced the same files) | | 101 | 256 |
  | plain | 198 | 206 (+4 %) | 205 | 217 | 194 | 198 | 198 | 256 |
  | toolwrite | 185 | 190 (+3 %) | 189 | 194 | 189 | 190 | 185 | 249 |
  | regen | 86 | 84 (-2 %; 11 wasted rows a prediction round) | 91 (+6 %) | 100 | 90 | 90 | 86 | 255 |
  | schema | 91 | - | - | - | 99 (+8 %) | 102 | 91 | 238 |
  | call_required | 114 | - | - | - | 91 (-20 %: 10 wasted rows a round) | 114 | 114 | 209 |
  Ceiling of perfect drafts at 31 rows a window: 261 tok/s (122 ms a round); at 63 rows 297, at 127 rows 319 IF a row keeps costing 2.91 ms (to be measured: prompt passes read a row at ~0.43 ms).
  READING: (c) a client prediction pays where the prompt does not hold the reply's exact tokens (numbered +113 %, agent loop +17 %, follow-up +12 %); where the prompt already holds them
  the engine's own copies have taken the gain (plain +4 %, toolwrite +3 %). Match length 8 beats 4 (fewer broken copies); DECISION: default match 8. (b) earlier turns of the SAME session beyond the
  prompt give nothing (they are in the prompt already: zero, dropped); OTHER sessions' replies give +9..23 % on agent loops here, but my repos share functions across sessions (= sessions on one
  repository), and a short match loses on short tool calls (-20 %): the engine's own --recall prices copies by band (16/32/64): measured in run RC, not rebuilt.
  (d) grammar-forced tokens, COUNTED on the logged ids (forced10.py): in tool calls 41 % of tokens are forced whatever the tool, 27 % more with the tool's schema; in a real agent loop 11 % + 8 %.
  The MTP chain already drafts them: a forced token is a round's own (paid) token 6.6-8.7 % of the time against 10-14 % for free tokens; schema tokens (`<parameter=K`, `</function>`) 19-25 %.
  Ceiling if every forced token came free: 9 % fewer passes on agent-loop tool turns (3 % follow-up, 22-30 % on one-line tool calls); with schema tokens too 26 % / 11 % / 58-64 %.
  With response_format / tool_choice=required the engine ALREADY appends xgrammar's forced tokens as drafts (19 grammar rounds in L1). DECISION: no grammar build for tool_choice=auto in the engine;
  the skeleton can be sent as `prediction` parts by the client: tested through the new API if time allows.
  (e) per-round choice: "priced in hindsight" over "always" is +2..16 % (regen, follow-up): the cheap part of it is the match length and the engine's adaptive width; a calibrated chooser is not built.
- 07:15 B13 built (sha 86d0f552d8446d04) = base + verify windows up to 128 rows (mtp.max_rows 32 -> 128, TF_FLASHNEXT_Q38_WINDOW_ROWS 16..128); B14 (sha 751684058d9644ac) = B13 + `prediction` field:
  server/chat.zig predictionIds (OpenAI shape {type: content, content: str | [{type: text, text}]}; ids, parts split by 0xFFFFFFFF; TF_Q38_PREDICTION=0 ignores it, TF_Q38_PRED_MAX caps tokens),
  api.Request.prediction, core/lanes/proposer.zig SuffixLookup.predict / sideCopy (second copy corpus; reply start anchored to the prediction's start; suffix match >= TF_Q38_PRED_MATCH; ties prefer the
  first match after the last one used), core/lane_stream.zig. Drafts only. Branch r10-B in Mac q38/wt/B10.
- 07:35 RC DONE: the engine's OWN cross-request store (`--recall --recall-dir DIR`, upstream TensorFold: core/lanes/recall*.zig; a record of prompts + replies, copies priced by match band 16/32/64) on flashnext,
  3d24, B12, round-9 defaults, no logging (MEASURED; reference = E1 below: B14 without recall, same window, 20 min later; NOT first-and-last paired yet):
  | pass | reference wall tok/s | with --recall | texts vs L1 |
  | agent loop, store cold at start (16 fix sessions; repos share functions) | 77.3 | 82.7 (+6.9 %) | 90/90 |
  | follow-up sessions (store holds the fix sessions) | 93.9 | 104.8 (+11.6 %) | 97/97 |
  | the 16 fix sessions again (a retry: the store holds the same replies) | 77.3 | 110.8 (+43 %) | 90/90 |
  | numbered, held half (its plain twin ran just before) | ~93 | 140.8 | 28/28 (with plain + toolwrite) |
  /v1/recall: 140,691 ids held (0.56 MB disk, 16 MB RAM), 1,433 copies proposed 28,458 ids, 22,552 landed; band landing rates 0.92 / 0.94 / 0.996. 0 error lines.
  => source (b) EXISTS in the engine and pays on agent work; nothing to build. Open: paired chat/held check (no slow-down), then a recipe option (single-user box: the record is shared by every
  request of the server and is written to disk: opt-in server flag, say so). LESSON: ar.py cmp must ignore tool-call ids (random a request).
- 07:53 E1 (B14, 32-row windows, no prediction, no logging; MEASURED wall tok/s): numbered 93.1, plain 180.8, toolwrite 143.5, loop_fix 77.3, loop_follow 93.9 (logging in L1 cost < 1 %).
- 08:10 E2 DONE: `prediction` FIELD, first measurement (3d24, B14 sha 751684058d9644ac, 32-row windows, TF_Q38_PRED_MATCH=8, round-9 defaults; E1 = same binary, field not sent, the run before; ref2 = E6 pending; MEASURED wall tok/s):
  | set | E1 no prediction | E2 with prediction | change | replay had said (decode) | texts E2 vs E1 |
  | numbered (prompt = `cat -n` output, prediction = the plain file) | 93.1 | 184.2 | +98 % | +111 % | 16/16 |
  | plain (file already in the prompt) | 180.8 | 184.7 | +2.2 % | +3..4 % | 24/24 |
  | toolwrite | 143.5 | 143.9 | +0.2 % | +2..3 % | 16/16 |
  | loop_fix (prediction = the files the session has read or written so far, sent by the harness every turn) | 77.3 | 85.3 | +10.4 % (first-chunk-to-end 115.8 -> 135.6 = +17 %) | +17 % | 90/90 |
  | loop_follow | 93.9 | 104.4 | +11.2 % | +12 % | 97/97 |
  | regen (prediction = the earlier reply, which is not in the prompt) | ~85.4 | 91.1 | +6.5 % | +6 % | 24/24 (with regen_first) |
  Copy rounds: 21.6 -> 24.3 drafts a round, accept 0.666 -> 0.809. 0 error lines. The replay predicted every set within 2 points. E3 (prediction + 128-row windows) running.
- 08:25 E3 DONE: prediction + 128-ROW VERIFY WINDOWS (B14, TF_FLASHNEXT_Q38_WINDOW_ROWS=128 + WIDE_ROWS=128; MEASURED wall tok/s; texts vs E1: pred 56/56, loop 90/90, follow 97/97, regen 24/24; 0 error lines;
  start line still "5 streams at once" at 262K: 5.13 GiB a stream against 4.85):
  | set | E1 (32 rows, no prediction) | E2 (32 rows + prediction) | E3 (128 rows + prediction) |
  | numbered | 93.1 | 184.2 | 233.6 (+151 %) |
  | plain | 180.8 | 184.7 | 228.3 (+26 %) |
  | toolwrite | 143.5 | 143.9 | 178.7 (+25 %) |
  | loop_fix | 77.3 | 85.3 | 91.0 (+18 %) |
  | loop_follow | 93.9 | 104.4 | 105.2 (+12 %) |
  | regen | ~85.4 | 91.1 | 91.2 (+7 %) |
  ROW COST IN WIDE WINDOWS (MEASURED from the engine's per-request lines in E3, requests grouped by drafted rows a round): 21 rows a round: 92-98 ms a round, 175-197 tok/s;
  39 rows: 118 ms, 290 tok/s; 56 rows: 131 ms, 416 tok/s; 69 rows: 151 ms, 456 tok/s; 88 rows: 170 ms, 524 tok/s; 97 rows: 170 ms, 575 tok/s.
  => a copy round costs about 32 + min(3.1 x rows, 42 + 1.03 x rows) ms: past ~20 rows a verified row costs ~1 ms (the experts are already read), not 3. That is why 128-row windows pay.
  Replay with that curve (SIMULATED): the engine's doubling rule at 128 rows gives numbered 312 / plain 285 / toolwrite 271 decode tok/s (measured first-chunk-to-end 284 / 268 / 272);
  halving the width after a break instead of resetting to 15: +5 % on rewrites, 0 elsewhere (flag TF_Q38_COPY_RESET=half built, to be measured); starting at 63: +13 % on rewrites, -2..5 % on
  regen / follow-up (not built); 256- or 512-row windows: nothing on these sets (edits break a copy every 40-80 tokens); an online landing-rate rule for the width: zero (dropped).
- 08:26 B15 building (= B14 + prediction match default 8 + TF_Q38_COPY_RESET). w10f.sh queued on 3d24 behind E4 (it stops w10b: E5 / E6 dropped; every pair in w10f has its own references):
  F_ref1 / F_win (128 rows) / F_pred (128 rows + prediction, a WRONG prediction on ident + held) / F_half / F_rc (--recall) / F_ref2. Lease B renewed to 10:25.
- 08:33 E4 (B14, 128-row windows, NO prediction; MEASURED wall tok/s vs E1): plain 180.8 -> 217.8 (+20.5 %), toolwrite 143.5 -> 175.7 (+22.4 %), numbered 93.0 (no change: its prompt does not hold the reply's tokens).
  => wide windows alone lift every reply that copies from its prompt; the prediction adds the cases where the prompt cannot serve.
  B15 built (sha 678b52d2f9ae7d0c; tree 05f9ccd2a6df; patch in Mac q38/handoff/B/r10-workload-drafts/, sha a3219916...). Proposer unit tests (3, incl. the new prediction test) pass on the box (zig test -lc proposer.zig).
  SIMULATED and DROPPED: re-sync near the break point with a 2-4 token match (+3 % rewrites, -2..6 % follow-up / regen = zero net); landing-rate width rule (zero).
  Queue on 3d24: w10f (paired handoff window, B15) -> w10g (4 users at once with predictions; 16-lane start line with 128-row buffers) -> w10c (sampled requests with a prediction, seeded; DEEP check on A's fresh set).
- 09:13 INCIDENT (mine, report to Pico / integrator): the shared window S140401 (opened 04:01:30 by A) hit its 300-minute cap: the window's own watchdog began the restore at 08:58 (leases removed,
  stop-all killed my engine in the middle of F_ref1: ident + held/3 had finished, the workload part had not). My queue on 3d24 (w10f -> w10g) only knew MY lease end (10:23) through the deadline file,
  NOT the window cap, so w10g started two engines OUTSIDE any window: B10_M_ref 08:58:34-09:05:14 and B10_M_pred 09:05:14-09:11:35 (77 GiB of GPU memory on 3d24 while DeepSeek's restore was
  running), and a third was loading when I found it at 09:12 and stopped everything (no q38 container or script of mine left on 3d24 at 09:12:54; memory free). The window unit ended 09:02:40 without
  writing its "prod restored" line; the guard unit (fired 09:01:33) was still working through its own stop-all at 09:14 and restarts DeepSeek itself when health is not 200. I did NOT touch DeepSeek.
  CAUSE: I planned a 4-hour queue without checking the window's cap (A's round-9 lesson says to check `systemctl --user list-timers | grep q38-guard`; I did not). FIX in my kit before any new run:
  the deadline file is min(lease end, window OPEN + 300 min - 5 min), and run10.sh does not start an engine unless that deadline is > 7 min away. The two out-of-window runs' numbers are NOT used.
  Data kept from before the cap: E1-E4 (in window), F_ref1 ident + held/3 (B15: held ALL 66.59, chat 56.20 / code 94.25 / math 88.37, ident median 66.3).
  LESSON: `pkill -f PATTERN` over ssh killed my own ssh shell first (the pattern was in its command line), so the first stop attempt did nothing: write the pattern as "w10[a-z][.]sh".
- 09:26 DeepSeek production was restored and verified by the window's guard at 09:22:54 UTC (head ~/q38/guard.log: health=200, 391, session store 12.00 GiB on every rank, disk tier 28.2 GiB, image OK).
  A normal restore ends ~4.5 min after a window closes (08:58 -> ~09:03); this one ended 09:22:54: my two out-of-window engines on 3d24 (until 09:12:54) are the likely reason for the ~20 extra minutes
  of DeepSeek outage. Builder D opened a new window at ~09:23.
- 09:27 kit fixed: lease10.sh writes deadline = min(lease end - 2 min, window OPEN + 300 min - 7 min); run10.sh refuses a start under 7 min before it and its watchdog removes the engine at it.
  Lease B acquired (120 min) in D's window; w10h.sh started on 3d24 (B15; log ~/q38/runs/B10_w10h.log), in order of value: G_ref1 / G_win (128 rows) / G_pred (128 rows + prediction; a WRONG
  prediction on ident + held) / G_ref2 / G_rc (--recall) / 4 users without and with predictions / seeded sampled without and with / DEEP check on the fresh set (B12: ref, 12, 15, ref) / G_half / 16-lane start lines.
- 10:26 G_ref1 + G_win DONE (3d24, B15 sha 678b52d2f9ae7d0c, in D's window, one user, temp 0; MEASURED; ref2 pending). 128-ROW WINDOWS ALONE (TF_FLASHNEXT_Q38_WINDOW_ROWS=128 + WIDE_ROWS=128, no prediction):
  | set | G_ref1 (32 rows) | G_win (128 rows) | change |
  | held-104 all (chat / code / math), decode tok/s | 66.66 (56.20 / 94.45 / 88.82) | 66.66 (56.23 / 94.74 / 88.24) | 0.0 % |
  | ident median | 66.4 | 66.2 | |
  | plain rewrite, wall | 179.5 | 216.7 | +20.7 % |
  | toolwrite, wall | 141.1 | 172.0 | +21.9 % |
  | loop_fix / loop_follow, wall | 77.4 / 94.4 | 79.1 / 94.3 | +2.2 % / 0 |
  | numbered / regen / schema / call_required / call_auto | 92.4 / 86.1 / 81.0 / 84.0 / 98.0 | 92.2 / 86.1 / 81.0 / 84.4 / 98.1 | 0 |
  Texts G_win vs G_ref1: ident 51/51, held 107/107, pred 56/56, loop 90/90, follow 97/97, regen 24/24, struct 28/28. 0 error lines. 5 streams at 262K still fit (5.13 GiB a stream).
  Copy rounds 21.4 -> 28.4 drafts a round. => chat is NOT slower; rewrites +21 %. Lease B renewed to 12:26 (deadline file 12:24; window cap deadline 14:16).
- 10:55 G_pred DONE (B15, 128-row windows + `prediction` on every workload request; ident + held/3 sent with a deliberately WRONG prediction text; MEASURED; ref2 pending):
  | set | G_ref1 | G_win (128 rows) | G_pred (128 rows + prediction) | G_pred vs G_ref1 |
  | held-104 all (chat / code / math), WRONG prediction | 66.66 (56.20 / 94.45 / 88.82) | 66.66 | 66.52 (56.02 / 94.87 / 88.47) | -0.2 % (chat -0.3 %) |
  | numbered, wall | 92.4 | 92.2 | 230.1 | +149 % |
  | plain rewrite | 179.5 | 216.7 | 226.1 | +26 % |
  | toolwrite | 141.1 | 172.0 | 175.4 | +24 % |
  | loop_fix (real agent loop) | 77.4 | 79.1 | 91.0 | +17.5 % |
  | loop_follow | 94.4 | 94.3 | 105.6 | +12 % |
  | regen | 86.1 | 86.1 | 91.2 | +6 % |
  | schema / call_required / call_auto (no prediction sent) | 81.0 / 84.0 / 98.0 | | 80.8 / 84.4 / 97.8 | 0 |
  Texts G_pred vs G_ref1: ident 51/51 and held 107/107 (with the wrong prediction), pred 56/56, loop 90/90, follow 97/97, regen 24/24, struct 28/28. 0 error lines.
  Copy rounds 35.6 drafts a round, accept 0.72. => a wrong prediction costs chat 0.3 % and changes nothing; a right one is the table above.
- 11:15 G_ref2 DONE (32 rows, no prediction): held 66.62 (56.17 / 94.37 / 88.78), numbered 92.7, plain 180.3, toolwrite 143.0, loop_fix 76.5, loop_follow 94.4; texts = G_ref1 (107/107, 56/56, 90/90, 97/97).
  PAIRED RESULT, references first and last (mean of the two): 128-row windows alone: held 0.0 %, plain +20.5 %, toolwrite +21.0 %, loop_fix +2.9 %, others 0;
  128 rows + prediction: numbered +149 %, plain +25.7 %, toolwrite +23.4 %, loop_fix +18.3 %, loop_follow +12.0 %, regen +6.0 %, held with a WRONG prediction -0.2 %.
  Handoff NOTE written: Mac q38/handoff/B/r10-workload-drafts/NOTE.md (patch sha a3219916, B15). Running next: G_rc (--recall), 4 users, sampled, DEEP check.
- 11:43 G_rc DONE: the engine's `--recall` paired in the same window (B15, 128-row windows, no prediction, record empty at start; references G_ref1 / G_ref2; MEASURED wall tok/s):
  held-104 66.65 (refs 66.66 / 66.62: 0.0 %; the record was filling with the held prompts meanwhile), texts 107/107; agent loop with a cold record 83.2 (refs 77.4 / 76.5: +8.2 %; 128 rows alone 79.1);
  follow-up sessions 105.7 (94.4: +12.1 %); the same 16 sessions again (retry) 114.5 (+49 %). Texts 90/90, 97/97, 90/90. 0 error lines. /v1/recall: 151,528 ids, 0.6 MB disk, 16 MB RAM,
  1,102 copies, 24,198 ids proposed, 17,518 landed. => `--recall` is worth a recipe OPTION for agent boxes (server flag; the record is shared by all requests and written to disk).
- 11:58 FOUR USERS AT ONCE with predictions (B15, 128-row windows, the 56 predicted-output requests through a queue with 4 in flight; in window; MEASURED): no prediction 38,221 reply tokens in 201.7 s
  = 189.5 tok/s; with prediction 162.2 s = 235.7 tok/s (+24.4 %); texts against the one-user reference 56/56 in both. Per request the numbered ones go 32.5 -> 59.7 tok/s; plain 67.1 -> 63.1 and
  toolwrite 57.5 -> 50.0 (the pass's rows go to the streams that copy; shared rounds keep 15-token copies). Lease B renewed: deadline 13:50 (cap 14:16).
- 12:03 SAMPLED REQUESTS with a prediction (B15, 128-row windows, temperature 0.7, top-p 0.95, seed 4242, held half of the sets; MEASURED wall tok/s, no prediction -> prediction):
  numbered 89.8 -> 292.9; plain 281.3 -> 303.4 (+7.9 %); toolwrite 212.2 -> 218.9 (+3.2 %); agent loop 76.9 -> 89.7 (+16.6 %). Same seed, same texts: 28/28 and 43/43. 0 error lines.
  (The held half of plain copies longer stretches than the whole set: 281 against 217 for all 24.)
  w10i.sh replaced the tail of w10h (DEEP check on the fresh set: ref / on 12 / ref on B12, then the 16-lane start lines); COPY_RESET=half and DEEP at 15 are dropped from this window (tool-call budget).
- 12:35 DEEP / STOP2 OUT OF FIT (3d24, B12 = integ9, A's fresh set: 123 prompts x 384 that no threshold was set on; MEASURED decode tok/s; ref2 pending):
  | run | fresh ALL | chat | code | math | struct | agent | non-English |
  | d_ref1: DEEP off, STOP2 off, 6 drafts | 55.81 | 48.63 | 76.16 | 82.08 | 81.39 | 83.02 | 38.32 |
  | DRAFTS=12 + DEEP=0.9,6 + STOP2=0.5 (the recipe) | 57.20 | 49.10 | 79.27 | 87.61 | 86.41 | 90.14 | 38.74 |
  => +2.5 % (chat +1.0, code +4.1, math +6.7, struct +6.2, agent +8.6, non-English +1.1); texts 123/123 identical. On the held mix the same pair was +2.8 %: the gain does NOT collapse
  outside the prompts the 0.9 line was set on; the line stands, no re-fit.
- 12:58 d_ref2 55.92 => DEEP + STOP2 on the fresh set: +2.4 % against the mean reference (chat +0.9, code +4.0, math +6.6, struct +5.9, agent +8.5, non-English +1.0); texts 123/123. The line holds.
- 12:58 16-LANE START LINES (MEASURED): B12 (base) "16 streams at once, 1.49 GiB each at a 65536-token window"; B15 REFUSES: "--parallel 16 needs 29.69 GiB ... budget leaves 29.50 GiB".
  => the 128-row buffers cost +0.37 GiB a stream whatever the knob (compile-time mtp.max_rows): the patch is for the default profile; USERS16 needs 15 lanes / a ~60K pool window / or
  run-time sizing of the window buffers from the knob (follow-up described in the handoff NOTE; NOT built: found at the end of the window, no budget left for a build + test).
- 12:58 all runs in; no q38 container or script of mine on 3d24; deadline file 0; LEASE B RELEASED (window stays open on A / C / D leases).

## BUILDER B ROUND 10 SUMMARY (2026-11-14 06:30 -> 13:00 UTC; leases B 07:01-08:58 (cut by the window cap), 09:26-12:58 released)
| item | replay first (SIMULATED) | measured (3d24, one user, wall tok/s unless said, references first and last) | identity | status |
|---|---|---|---|---|
| 128-row verify windows (`TF_FLASHNEXT_Q38_WINDOW_ROWS=128` + `WIDE_ROWS=128`) | row cost curve from E3: ceiling ~620 tok/s at 127 rows against 261 at 31 | held 0.0 %; plain rewrite 179.9 -> 216.7 (+20.5 %); toolwrite +21.0 %; agent loop +2.9 % | 51/51, 107/107, 56/56, 90/90, 97/97, 24/24, 28/28 | in handoff; default profile only (USERS16 does not start at 16 x 64K) |
| `prediction` field (with 128 rows) | numbered +111 %, agent loop +17 %, follow-up +12 %, regen +6 %, plain +4 % | numbered 92.5 -> 230.1 (+149 %); plain +25.7 %; toolwrite +23.4 %; agent loop 76.9 -> 91.0 (+18.3 %); follow-up +12.0 %; regen +6.0 %; wrong prediction on held -0.2 %; 4 users 189.5 -> 235.7; sampled same-seed texts equal | same sets + wrong prediction 51/51, 107/107; 4 users 56/56; sampled 28/28, 43/43 | in handoff |
| cross-request store | other sessions' replies +9..23 % on agent loops | the engine's own `--recall`: cold agent loop +8.2 %, follow-up +12.1 %, retry +49 %, held 0.0 % | 107/107, 90/90, 97/97, 90/90 | exists upstream; recipe option proposed |
| same-session history beyond the prompt | -5..0 % | - | | dropped |
| grammar-forced drafts for tool calls | chain already keeps ~92 % of forced tokens; ceiling 9 % fewer passes on agent tool turns (26 % with schema) | - | | not built; engine already drafts forced tokens when a grammar is on |
| per-round chooser, width rules, near-break re-sync, 256 rows | 0 or mixed | - | | dropped; `TF_Q38_COPY_RESET=half` built, unmeasured, off |
| DEEP / STOP2 on the fresh set | | +2.4 % (code +4.0, math +6.6, struct +5.9, agent +8.5, chat +0.9) | 123/123 | holds, no re-fit |
Handoff: Mac q38/handoff/B/r10-workload-drafts/ (r10-B-workload-drafts.patch sha a3219916 = diff 7061396..0c56877, tree 05f9ccd2a6df, binary B15 678b52d2f9ae7d0c; NOTE.md; tools/).
What remains (in order): (1) run-time sizing of the window buffers so USERS16 keeps 16 x 64K (then 16 users with 128-row copies); (2) `--recall` + prediction together and with 4-16 agents;
(3) COPY_RESET=half and 64-row windows measured; (4) a calibrated copy-versus-chain chooser (hindsight worth +2..16 %); (5) tool-call skeleton as prediction parts (ceiling 9-26 % of passes on tool turns);
(6) prediction on /v1/responses and /v1/messages; (7) the agent loop's prompt pass is 26 % of its wall (builder C's area).
State on 3d24: no container, no script; binaries ~/q38/tfbin/B12 (integ9), B13, B14, B15 (= handoff); runs ~/q38/runs/B10_*; sets ~/q38/kit-B/sets/r10b; recall dirs under ~/q38/tf/kcache-default/recall{B,F}.
RESUME: cd ~/.hermes/cache/scratch/q38/wt/B; bash push10.sh (kit10 -> box; never while a queue script runs); bash ship10.sh NAME (build from wt/B10); bash kit10/lease10.sh acquire 120; box scripts kit-B/{lib10.sh,run10.sh,w10h.sh}.

# Round 11 (builder B, started 2026-11-14 13:02 UTC; levers: run-time window buffers + USERS16 with 128 rows, recall + prediction, chooser, prediction on other routes, COPY_RESET=half / 64 rows, prompt-cache check)
- 13:02 read round-10 log + handoff NOTE; worktree Mac q38/wt/B11 (branch r11-B from r10-B HEAD 0c56877). Plan: first build = run-time sizing (B16), start lines of BOTH profiles before any measurement.
- 13:08 B16 built (sha 5767b0772d6604fd; tree = r10-B + run-time window buffers mtp.bufRows() + prediction/recall fields on /v1/responses + /v1/messages + per-request "recall": false). Lease B acquired 13:08 (window open since 09:23:48, restore at 14:03:48 => my deadline 13:59:48). Queue w11a.sh started on 3d24: B11_u16_w128, B11_u16_w32, B11_def_w128.
- 13:12 CHOOSER REPLAY (SIMULATED; Mac wt/B/kit11/choose11.py on results/rB/B10_L1 = round-10 TF_STEP_LOG=2 log, 295 requests; prediction as the copy source, W=127, cost curve 32.3 + min(2.91 w, 42 + 1.03 w)):
  calibration table (fit half; P(kept>=1) / >=4 / >=15 / >=31 / mean kept): match 16-31 fresh 1.00/.97/.85/.75/64.5; 32-63 fresh .99/.98/.87/.60/58.7; 64+ fresh 1.00/.99/.88/.82/80.2;
  match 4-7 after a LATE break (last copy kept >= 4) .84/.63/.42/.35/33.8 (n 1549); match 4-7 after an early break .82/.50/.28/.22/21.8; match 4-7 fresh .77/.33/.08/0/4.1.
  | policy (fit on one half, evaluated on the other; held / fit halves, vs the fixed rule) | loop_fix | loop_follow | numbered | plain | regen | toolwrite |
  | bucket-calibrated chooser (survival by match class x last copy, best width, vs the stream's own EMA rate) | -1.2 / -6.4 | -11.9 / -13.8 | +2.6 / +18.1 | +7.2 / +9.0 | -14.6 / -21.2 | +0.3 / +9.2 |
  | same + per-stream correction factor | -1.4 / -3.2 | -8.9 / -12.0 | -46.8 / -0.9 | -26.8 / +3.1 | -6.8 / -6.5 | -38.2 / +8.6 |
  | rule A: match >= 16 and no break pending => first width 63 | +0.2 / +0.1 | +3.1 / +2.9 | 0.0 / +4.0 | +1.8 / +3.5 | -0.9 / -0.4 | +0.3 / -1.4 |
  | A + B (a 4-token match is enough after a late break) + C (match >= 32 => 127) | +0.5 / +0.2 | +1.7 / +1.6 | +5.6 / +10.7 | +3.9 / +8.4 | -0.1 / -1.8 | +2.1 / +1.9 |
  | hindsight priced (fixed widths) | +1.7 / +1.5 | +8.1 / +7.2 | +1.4 / +4.2 | -4.2 / +1.8 | +5.3 / +5.2 | 0.0 / 0.0 |
  DECISION: the pooled calibrated chooser is NOT shippable (the survival of a short match differs by workload: rewrites gain, follow-ups / regen lose 12-21 %, below "no prediction"); the
  part that holds on both halves is two threshold rules read off the table: (A/C) first copy width from the match length, (B) re-entry on a 4-token match after a late break. Build those behind
  flags (TF_Q38_COPY_FIRST, TF_Q38_COPY_REENTER), measure paired in the engine on every set. Chain-side term (the chain's own probabilities) not used: at a copy round the chain has not run.
- 13:15 PROMPT-CACHE CHECK of the agent traces (item 6; MEASURED on the round-10 log results/rB/B10_L1: token ids of every turn's prompt + the engine's `prompt cache:` lines; analysis inline python, no new run):
  | trace | follow-up turns | previous prompt is an exact prefix of the next | engine resumed at (previous prompt length - 7) | previous reply re-appears token for token after it | prompt growth a turn | of which the previous reply | first-chunk time a follow-up turn |
  | loop_fix | 74 | 74/74 | 74/74 | 74/74 | 387 tokens | 102 (26 %) | 0.353 s of 1.392 s wall (25 %) |
  | loop_follow | 81 | 81/81 | 81/81 | 81/81 | 595 tokens | 326 (55 %) | 0.469 s of 3.519 s wall (13 %) |
  So every turn hits the cache; the 7 tokens are the chat template's generation prompt (the cache mark sits before it). What is prefilled a turn = the previous reply (which the engine itself just
  generated: its decode state is not kept) + the tool result. From 5 turns of the engine log: first-chunk time ~= 0.16 s fixed + 0.55 ms a new prompt token (126 tokens 0.225 s, 389 -> 0.357, 592 -> 0.477).
  The fixed 0.16 s a turn (state restore, lane opening, render) is ~11 % of an agent turn; the reply's re-prefill is ~40-130 ms a turn (3-4 %): both are builder C's area (ARENA / LANE_REUSE / a kept state at the
  reply's end; the server has a background "warm turn" pass, warm.zig, but no such pass shows in these logs: unexplored).
  What a client must do to keep it this way: append only (never rewrite or re-order earlier messages), keep system prompt + tool list byte-identical every turn, send the assistant's tool call back exactly as received
  (same arguments string), put volatile text (timestamps, cwd listings) in the newest message, not the system prompt.
- 13:19 harness bug found and fixed (ar.py): run_tests used ONE docker name (q38-sbx-B), so with AR_PAR>1 two sessions' test runs collided and the tool result was a docker error => sessions took other paths (loop4 80/90, loop8 81/90 in B11_u16_w128 are NOT an engine difference; held 107/107 with 16 in flight is). Name is now per process + thread; loops re-run.
- 13:19 B11_u16_w128 DONE (3d24, B16, USERS16 profile = --parallel 16, pool 65536, WIDE_ROWS=128, LANE_REUSE=16, round-9 defaults, WINDOW_ROWS=128; MEASURED):
  START LINE: "16 streams at once, 1.86 GiB each at a 65536-token window, of 29.8 GiB left after a 9.0 GiB reserve" => it starts (16 x 1.86 = 29.76 of 29.8: by a hair; the budget was 29.5 in round 10).
  distinct 1 / 4 / 8 / 16 / 16 users: 86.4 / 161.0 / 211.9 / 276.5 / 280.6 tok/s; held prompts 16 in flight 198.6 tok/s, texts 107/107 vs the one-user reference;
  16 predicted-output requests in flight (with predictions): 317.2 tok/s total, texts 56/56. Agent loops 4 / 8 in flight: 108.0 / 126.6 tok/s but VOID for identity (harness bug above).
- 13:25 added to the tree (commit in wt/B11, for binary B17): core/lanes/q38_copy.zig knobs: TF_Q38_COPY_FIRST=1 (first copy width from the match: 63 from 16 tokens, 127 from 32),
  TF_Q38_COPY_REENTER=N (N-token match after a late break), TF_Q38_COPY_SHARED=1 (copy ramp also in shared rounds, capped by the pass's rows a live stream; today a stream in a shared round
  copies 15 tokens at most: reason 4 agents rewriting reach only ~236 tok/s total), TF_Q38_COPY_OFF, TF_Q38_COPY_MATCH (measuring aids), QCOPY log line. Queue w11b.sh (restartable) waits for the B17 build.
- 13:31 B11_u16_w32 DONE (same binary B16, same profile, WINDOW_ROWS=32 = the recipe's value; MEASURED). RUN-TIME BUFFER SIZING WORKS: "16 streams at once, 1.49 GiB each ... a 5.3 GiB prompt cache"
  (= the base's 1.49; with 128 rows 1.86 GiB each and only 0.1 GiB left for the prompt cache).
  | USERS16 profile, B16 | 1 user | 4 | 8 | 16 | 16 | held 16 in flight | 16 predicted-output requests | loop 4 agents | loop 8 agents |
  | WINDOW_ROWS=128 | 86.4 | 161.0 | 211.9 | 276.5 | 280.6 | 198.6 (107/107) | 317.2 (56/56) | 108.0 (void) | 126.6 (void) |
  | WINDOW_ROWS=32 | 87.1 | 157.4 | 219.9 | 280.3 | 284.5 | 200.2 (107/107) | 319.4 (56/56) | 110.6 (90/90) | 126.2 (90/90) |
  => with several users 128-row windows are ZERO to slightly negative (-1..-3 %, inside the +-3 noise of this bench) and they take 5.9 GiB from the prompt cache: in a shared round a stream copies 15 tokens
  whatever the window. Keep WINDOW_ROWS=32 for USERS16 unless TF_Q38_COPY_SHARED (B17) changes that. Agent loops with 4 / 8 agents in flight: texts 90/90 identical to the one-user reference.
- 13:43 B17 built (sha 7cff5d58a8cf9c37 = B16 + q38_copy.zig knobs); B11_def_w128 (B16, default profile, 128 rows): 5 streams at 5.13 GiB each, 4.2 GiB prompt cache, ident 51/51. w11b part 0 started (s4_ref1).
- 13:56 PART 0 first pair (3d24, B17, default profile 5 x 262K, 128-row windows, predictions sent, 4 requests / 4 agents in flight; MEASURED; second reference pending in the next window):
  | | 56 predicted-output requests, 4 in flight | agent loop, 4 agents | copy rounds: drafts a round / kept | texts vs one user |
  | B11_s4_ref1 (copies stay 15 tokens in a shared round) | 235.0 tok/s | 108.2 | 14.9 / 13.6 | 56/56, 90/90 |
  | B11_s4_on TF_Q38_COPY_SHARED=1 (ramp kept, cap = pass rows / live streams - 1 = 31) | 277.3 (+18.0 %) | 117.8 (+8.9 %) | 27.4 / 24.5 | 56/56, 90/90 |
- 13:57 lease B RELEASED (window S09:23 reaches its restore at 14:03:48; my deadline was 13:59:48; no engine of mine is up). Next: new window after the restore; meanwhile build B18 (COPY_SHARED=2: cap by free rows).
- 14:08 window S09:23 closed at its cap (restore began 14:03:48; DeepSeek health 200 at 14:08). B18 built during the turnover (sha 5acf227aba9ee51d = B17 + COPY_SHARED=2). Waiting for CLOSING to clear, then acquire; queue w11c.sh (restartable, B18) is on the box.
- 14:14 lease B acquired (new window, opened 14:10:33 by another builder; restore at 18:50:33); w11c.sh started (B18).
- 14:34 w11c part 0 (B18, default profile, 128 rows, predictions, 4 in flight; MEASURED): t4_ref1 pred 235.6 / loop 109.8 / follow 151.8 (56/56, 90/90, 97/97); t4_m1 (COPY_SHARED=1) pred 278.2 / loop 116.5 (56/56, 90/90), follow pending; m2 + ref2 running.
- 14:44 t4_m1 follow 158.7 (97/97), copy rounds 25.5 drafts / 20.8 kept; t4_m2 (COPY_SHARED=2, cap = free rows): pred 301.2 (+27.8 % vs ref1) / loop 120.6 (+9.8 %) / follow 157.8 (+4.0 %), texts 56/56, 90/90, 97/97.
- 15:10 PART 0 TABLE (3d24, ONE binary B18 sha 5acf227aba9ee51d, default profile 5 x 262K, 128-row windows, predictions sent, total wall tok/s; runs in this order; MEASURED):
  | run | 56 predicted-output requests, 4 in flight | agent loop (fix), 4 agents | follow-up sessions, 4 agents | 56 requests, 2 in flight | held chat mix, 4 in flight | texts vs the one-user reference |
  | t4_ref1 (copy = 15 tokens in a shared round) | 235.6 | 109.8 | 151.8 | - | - | 56/56, 90/90, 97/97 |
  | t4_m1 TF_Q38_COPY_SHARED=1 (cap = pass rows / streams) | 278.2 | 116.5 | 158.7 | - | - | 56/56, 90/90, 97/97 |
  | t4_m2 TF_Q38_COPY_SHARED=2 (cap = free rows) | 301.2 | 120.6 | 157.8 | 315.6 | 117.8 (107/107) | 56/56, 90/90, 97/97, 56/56 |
  | t4_ref2 | 235.6 | 110.1 | 153.1 | 208.8 | (below) | 56/56, 90/90, 97/97, 56/56 |
  => COPY_SHARED=2 vs the mean of the references: 4 rewrite requests in flight +27.8 %, 2 in flight +51.1 %, 4 agents in a bug-fix loop +9.7 %, follow-up sessions +3.5 %. Identity holds in flight.
  (The second "follow4" figure in m2 / ref2 run logs is an artefact: the file was appended to twice. Ignore 323.7 / 313.8.)
- 15:18 held chat mix 4 in flight: COPY_SHARED=2 117.8 vs ref 117.4 (+0.3 % = zero), 107/107 both. Lease renewed to 17:18 (deadline file 17:16:22). Part 1 (one-user copy rules) running.
- 15:54 part 1 so far (B18, one user, default profile, 128 rows + predictions; wall tok/s; MEASURED): c_ref1 numbered 232.8 / plain 227.5 / toolwrite 178.1 / loop 90.9 / follow 104.7 / regen 90.9 (first 85.4); held 66.64 (56.14 / 94.88 / 88.60); identity 56/56, 90/90, 97/97, 24/24, 51/51, 107/107. c_first (TF_Q38_COPY_FIRST=1): 243.9 / 235.7 / 182.6 / 92.1 / 105.7 (regen pending) = +4.8 / +3.6 / +2.5 / +1.3 / +1.0 %.
- 16:09 c_first regen 89.6 (-1.4 %; 24/24), copy rounds 48.7 drafts / 31.0 kept (ref 35.9 / 26.0). c_fr (FIRST + REENTER=4): numbered 244.1 / plain 238.4 / toolwrite 182.2 / loop 91.6 / follow 104.6: REENTER adds ~0 (+1.2 % plain, -0.6..-1.0 % loops).
- 16:24 c_fr regen 89.0 (-2.1 %), held 66.53 (chat 56.17 / code 94.02 / math 88.27: -0.2 % = zero), identity 56/56, 90/90, 97/97, 24/24, 51/51, 107/107.
- 16:39 c_half (TF_Q38_COPY_RESET=half): numbered 241.4 / plain 235.6 / toolwrite 178.6 / loop 91.8 / follow 105.0 / regen 91.1 = +3.7 / +3.5 / +0.3 / +1.0 / +0.3 / +0.2 % vs c_ref1; 56/56, 90/90, 97/97, 24/24. Lease renewed to 18:39 (deadline 18:37:08; window restore 18:50:41).
- 16:55 c_w64 (WINDOW_ROWS=64, 5 x 4.89 GiB a stream; 128 rows: 5.13; 32 rows: 4.85): numbered 215.1 / plain 214.6 / toolwrite 164.8 / loop 89.5 (see next line) vs 128 rows 232.8 / 227.5 / 178.1 / 90.9.
  B11_r4_ref skipped on purpose (marker file): t4_ref1 / t4_ref2 are the same configuration (4 agents, no recall).
- 17:10 PART 1 TABLE (3d24, ONE binary B18, default profile, ONE user, 128-row windows + predictions = the round-10 handoff state as reference; wall tok/s; runs in this order; MEASURED):
  | run | numbered | plain rewrite | toolwrite | agent loop (fix) | follow-up | regen | held chat mix | identity |
  | c_ref1 | 232.8 | 227.5 | 178.1 | 90.9 | 104.7 | 90.9 | 66.64 | 56/56, 90/90, 97/97, 24/24, 51/51, 107/107 |
  | c_first TF_Q38_COPY_FIRST=1 | 243.9 | 235.7 | 182.6 | 92.1 | 105.7 | 89.6 | - | 56/56, 90/90, 97/97, 24/24 |
  | c_fr FIRST + TF_Q38_COPY_REENTER=4 | 244.1 | 238.4 | 182.2 | 91.6 | 104.6 | 89.0 | 66.53 | same + 51/51, 107/107 |
  | c_half TF_Q38_COPY_RESET=half | 241.4 | 235.6 | 178.6 | 91.8 | 105.0 | 91.1 | - | 56/56, 90/90, 97/97, 24/24 |
  | c_w64 WINDOW_ROWS=64 (4.89 GiB a stream; 128: 5.13) | 215.1 | 214.6 | 164.8 | 89.5 | 105.1 | 91.2 | - | 56/56, 90/90, 97/97, 24/24 |
  | c_ref2 | 233.1 | 227.3 | 178.0 | 90.7 | 104.9 | 91.1 | - | 56/56, 90/90, 97/97, 24/24 |
  vs the mean of the references: FIRST +4.7 / +3.6 / +2.6 / +1.4 / +0.9 / -1.5 %; FIRST+REENTER +4.8 / +4.8 / +2.3 / +0.9 / -0.1 / -2.2 %, held -0.2 %; RESET=half +3.6 / +3.6 / +0.3 / +1.1 / +0.2 / +0.1 %;
  64 rows -7.7 / -5.6 / -7.4 / -1.4 / +0.3 / +0.2 %.
  DECISIONS: COPY_RESET=half is never negative => propose as default with 128-row windows. COPY_FIRST pays on rewrites and loops, costs 1.5 % on regen => option (propose on for agent / edit boxes).
  COPY_REENTER adds nothing consistent (replay said +2..5 % on numbered: wrong, the engine's cursor preference already re-enters) => dropped (flag stays, off). 64 rows keep ~70 % of the 128-row gain: only if memory is short.
  ROUTES (rt11.py in c_ref2, 4 `numbered` items, non-streaming): /v1/chat/completions 105.2 -> 426.3 tok/s with the prediction, /v1/responses 104.1 -> 426.8, texts identical 4/4 each (messages + malformed pending).
- 17:14 ROUTES done: /v1/messages 104.2 -> 426.5 tok/s, 4/4; malformed predictions (number, bare string, no content, content number, mixed list, empty list, null, empty text, unknown type, bool, nested, wrong text, odd unicode): chat 13/13 and responses 13/13 answered 200 with the reference text; "recall": false accepted (messages line pending).
- 17:25 handoff patch written: handoff/B/r11-agent-drafts/r11-B-agent-drafts.patch sha256 66c5134e71e356609e648fca4ca762e8b3bbe0e6816a5f62ffe8b79e7c60393b = diff 7061396..70fb440 (branch r11-B, tree 119c8aed72d5, applies cleanly to 7061396, 16 files +317/-41) = binary B18 5acf227aba9ee51d. It REPLACES r10-B's patch.
- 17:35 RECALL + PREDICTION, ONE AGENT (B11_r_one: B18, default profile, 128 rows, predictions sent, `--recall --recall-dir` on an EMPTY dir; references = c_ref1 / c_ref2, same binary and config without --recall; MEASURED wall tok/s):
  | | agent loop (fix) | follow-up sessions | texts |
  | prediction only (c_ref1 / c_ref2) | 90.9 / 90.7 | 104.7 / 104.9 | |
  | + recall, record cold | 92.6 (+1.9 %) | 110.6 (+5.6 %) | 90/90, 97/97 |
  | + recall, the same sessions again (warm) | 128.7 (+41.7 %) | 156.9 (+49.8 %) | 90/90, 97/97 |
  Store after cold / warm: 93,992 / 133,926 token ids, disk 376 KB / 536 KB (cap 62.9 MB = --recall-gib default 1/16), ram_bytes 16.8 MiB (its own figure), copies proposed 31,985 kept 25,016,
  landing rate by match band 16 / 32 / 64: 0.93 / 0.97 / 0.997. With predictions already sent, a cold record adds little on the fix loop (+1.9 %; round 10 without predictions: +8.2 %).
- 17:49 RECALL + PREDICTION, 4 AGENTS IN FLIGHT (B11_r4: B18, default profile, 128 rows, COPY_SHARED off, empty record at start; references t4_ref1 / t4_ref2 = same config without --recall; total wall tok/s; MEASURED):
  | | agent loop (fix), 4 agents | follow-up sessions, 4 agents | texts |
  | prediction only | 109.8 / 110.1 | 151.8 / 153.1 | |
  | + recall, cold | 112.6 (+2.4 %) | 159.6 (+4.7 %) | 90/90, 97/97 |
  | + recall, same sessions again | 141.2 (+28.4 %) | 199.7 (+31.0 %) | 90/90, 97/97 |
  Store identical in size to the one-agent run (133,926 ids, 536 KB disk, 16.8 MiB RAM figure). Engine RSS 2.2 GB; MemAvailable at the end of a default-profile run ~10-11 GiB with or without recall
  (the elastic prompt cache; known since round 9), so recall's own memory is its 16.8 MiB + 2 x cap at most (cap 62.9 MB default).
- 17:49 w11d.sh written and pushed (third queue: proposed defaults together, references first and last, one user + 4 in flight; WIDE_ROWS=256; 16-user leftovers).
- 17:54 PER-REQUEST SWITCH (B11_r_off: --recall on, every request sent with "recall": false): the record stayed EMPTY (ids 0, disk 0, proposals 0), agent loop 91.4 tok/s (= no-recall reference 90.8), texts 90/90. PART 2 done.
- 18:13 harness slip: the MIX step used a bare `wait`, which also waits for run11.sh's watchdog subshell => B11_v16_w32 sat idle 18:07-18:13 until I killed that watchdog by hand (engine untouched). Fixed in w11d.sh (wait $HPID). Lesson: in a script that keeps a background watchdog, wait on an explicit PID.
- 18:17 B11_v16_w32 (B18, USERS16 profile, WINDOW_ROWS=32, predictions sent; reference for COPY_SHARED; MEASURED): agent loop 4 agents 111.8, 8 agents 196.8 (prompts warm from the 4-agent pass), 16 predicted-output
  requests in flight 314.0, mix of 12 held chat + 4 predicted-output in flight 117.7 + 162.5, distinct 16 users 278.2 / 285.3, held 16 in flight 207.5; texts 90/90, 90/90, 56/56, 107/107, 56/56, 107/107.
- 18:31 PART 3 (3d24, B18, USERS16 profile = 16 x 64K, WIDE_ROWS=128, WINDOW_ROWS=32, predictions sent; same order of steps in both runs; total wall tok/s; MEASURED; one reference only, the second was refused by the deadline):
  | run | loop 4 agents | loop 8 agents | 16 predicted-output requests | 12 held chat + 4 predicted-output in flight | distinct 16 users | held 16 in flight | texts |
  | v16_w32 (reference) | 111.8 | 196.8 | 314.0 | 117.7 + 162.5 | 278.2 / 285.3 | 207.5 | 90/90, 90/90, 56/56, 107/107 + 56/56, 107/107 |
  | v16_w32s TF_Q38_COPY_SHARED=2 | 116.3 (+4.0 %) | 208.0 (+5.7 %) | 317.0 (+1.0 %) | 117.8 + 165.1 | 282.0 / 282.5 | 209.3 | same counts, all identical |
  => with 16 users COPY_SHARED is zero (cap = 15 = today's width) and does not slow chat users sharing the pass; with 4-8 agents +4..6 % at 32-row windows.
- 18:32 lease B RELEASED; no engine of mine up; W11C_DONE 18:30:37. Window (opened 14:10:41) restores at 18:50:41.
- 18:40 handoff NOTE.md + API-NOTE.md written (handoff/B/r11-agent-drafts/). Next: third window for w11d.sh (proposed defaults together: RESET=half + FIRST + SHARED=2 vs references; WIDE_ROWS=256).
