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
