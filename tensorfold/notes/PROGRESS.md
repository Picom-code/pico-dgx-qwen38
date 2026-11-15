# q38: fastest single-Spark Qwen3.8-Flash-Next recipe — progress log (resumable)

All times UTC. Box under test: spark-923b (picom@spark-923b via head pico@spark-head). Work dir there: ~/q38. Head: ~/q38.
Mac: ~/.hermes/cache/scratch/q38 (h.sh = run on head, r3.sh = run on spark-923b, kit/, harness/, ext/ = upstream clones).

## Phase 0 (no downtime) — started 2026-11-12 05:10
- spark-923b has NO internet (wifi down, no default route). No host change made: a user-level CONNECT proxy runs on the head
  (unit q38-proxy, ~/q38/cproxy.py, spark-head:18399, allow-list of hosts); containers on spark-923b use HTTPS_PROXY.
  Stop it when downloads are done: `systemctl --user stop q38-proxy` on the head.
- Downloads (capped containers q38-dl-*, ~/q38/dl/hfdl.py: stdlib, resumable, sha256-verified, DONE line):
  - nvidia/Qwen3.8-Flash-Next-FP4-Local @ b9ce18a6887e18c089fdea8509a33e63429a3300 (102.26 GB) -> ~/q38/hf/hub/models--nvidia--.../snapshots/b9ce... (symlink ~/q38/models/nvidia-fp4-local)
  - turboderp/Qwen3.8-Flash-Next-exl3 @ 55a732e0c4c3d4614bc42b68493bb930d9b02c0a (branch 4.05bpw_h6_ng6, 107.46 GB) -> ~/q38/models/turboderp-exl3-4.05bpw (custom engine R's checkpoint)
  - local-inference-lab/Qwen3.8-Flash-Next-FLARE @ df4a4faf92cd8afc0f9821ac3cbaafb2e2b122ca (gb10 prepared core, 105.6 GB) -> ~/q38/models/flare-gb10 (a third engine)
  - Already on the box (read-only use): ~/models/qwen38-flash-next-nvfp4 = RadixArk/NVFP4 rev 7b719225 (135 GB; NOT FP4-Local).
- Images (crane via proxy -> docker load): MiaAI tensorfold v1.1.4 digest 5c6c42b9 (tag tensorfold-qwen38:v1.1.4-eab8582), flare release-972e28e-main-20261107-arm64, another vLLM recipe d55a2740; custom engine R release v0.3.13 tarball (sha256 verified) in ~/q38/custom engine R, runs inside vllm/vllm-openai:nightly-aarch64 (torch 2.13.0+cu130).
- Kit on spark-923b ~/q38/kit: qk.py (bench/sd/held/ident/cmp/prefill/needle/ttft/img), suite.sh, qeval_q.py, sdw/ (sparkDash 1.8.8 + sdrun.mjs), evict.py, qstop.sh.
- Window harness on head: ~/q38/qw.sh LABEL GUARD_MIN SCRIPT (trap + q38-guard unit; qstop + evict on spark-923b; prod_up; verify_prod.sh). Window log: head ~/q38/windows.log.

## Windows (DeepSeek outages)
| label | start | end | outage | what |
|---|---|---|---|---|
| smoke_tf | 2026-11-12T05:45:08Z | 05:53:24Z | 496 s | MiaAI TF default quick suite; restore verified (health 200, 391, session store 12.00 GiB, image ok) |
| A | 2026-11-12T06:01:04Z | (see head ~/q38/windows.log) | | 923b: TF default full suite, then NVIDIA model-card vLLM+b12x; 3d24: TF variants on a temp copy |

## Pico's update at ~05:50 (multi-box)
- Workers usable in parallel: spark-923b, spark-3d24 (pico@spark-3d24), spark-4e5c (pico@spark-4e5c). Head excluded.
- MEASURED disk: 3d24 has 106 GiB free (not 133), 4e5c 32 GiB. DeepSeek's session-store disk tier is sized at engine start as
  min(32, free - 32 GiB) per rank (rank.0.log line "session store's disk tier"), and each rank keeps a ~28 GiB spill file.
  => a ~100 GB checkpoint left on 3d24 would shrink DeepSeek's disk tier from 28 to ~7 GiB. DECISION: checkpoints on 3d24 are
  TEMPORARY: copied 923b -> 3d24:~/q38/tmpmodels inside the window over the fabric (kit/xfer.py, raw TCP, 0.7 GB/s, 147 s for 102 GB)
  and deleted by kit/qstop.sh in the trap and the guard before DeepSeek restarts. 4e5c (32 GiB) is left out: nothing fits, and it is
  the rank that already limits the disk tier.
- qw.sh now cleans + evicts all three workers in trap and guard, and logs GPU temp/power of all four boxes every 30 s (runs/LABEL/temps.log).
- Box difference measured (same TF default config): 923b chat 47.2 / code 77.8 / sd prose 57.4 / sd code 93.1; 3d24 45.8 / 74.4 / 54.4 / 88.6
  (3d24 ran the same binary bind-mounted into the older image, no vision). Final tables use 923b only.

## Facts learned (measured unless noted)
- TensorFold (MiaAI 1.1.4+17) on 923b: start 202-220 s, 78.2 GiB GPU after load, MemAvailable 37 GiB, TTFT 170-178 ms.
  no-draft decode 24.4 tok/s (41 ms per 1-row step). fp8 KV is the README default and is LOSSY vs bf16 KV (their note: ~98.8 % top-1);
  bf16 KV at 262,144 fits only 3 streams (39 GiB for 5).
- Official checkpoint bytes by group (safetensors headers, kit/bytes.py): experts 59.94 GiB (2.44 MiB per expert; 10 x 48 per row = 1.15 GiB),
  n-gram 26.88, GDN attn 2.005 (MXFP8), hyper-connection 1.195 (BF16), lm_head 1.184 (BF16), embed 1.184, QSA 0.593, MTP experts 1.318,
  shared expert 0.227, MTP dense 0.156 (BF16), router 0.117 (BF16), vision 0.304. Total 95.12 GiB.
  1-row step reads ~ dense 4.13 + head 1.18 + experts 1.15 = 6.46 GiB -> 25.4 ms at 273 GB/s (estimate) vs 41 ms measured (62 % of the limit).
- NVIDIA's model card serves this checkpoint with vLLM + b12x (a-vllm-image) and FORCE_A16 decode / A4 prefill and an MXFP8 LM head:
  "the official checkpoint" does not define one bit-exact output; identity is defined per engine (tricks on vs off).
- a third engine has a prepared gb10 core of the same checkpoint (local-inference-lab/Qwen3.8-Flash-Next-FLARE@gb10, 2026-11-11), runtime image
  ghcr.io/local-inference-lab/flare:release-972e28e-main-20261107-arm64 (184 MB).
- another vLLM recipe recipe serves nvidia/Qwen3.8-Flash-Next-NVFP4 (135 GB, a different NVIDIA checkpoint), not FP4-Local.

## Parent note, 2026-11-12 14:10 UTC: the first child died at 11:01 UTC (account limit) while launching window E
- State verified by the parent: DeepSeek up (health 200, 391 local + public, image ok, session store 12.00 GiB). No q38 window/guard unit;
  q38-proxy still running on the head. No q38 containers running on any box.
- Windows that ran (head ~/q38/windows.log): smoke_tf 496 s, A 3687 s, B 6233 s, C 3445 s, D 3223 s. Window E (TensorFold draft-gate sweep on three
  boxes, script head ~/q38/w_E.sh) did not run.
- Bake-off table, all on spark-923b: Mac results/bakeoff_923b.txt (built by kit/table.py from the suite logs under results/r923*/ and r3d24*/).
  Headline: custom engine R v0.3.13 (EXL3 re-quant) chat 54.6 / code 88.6, sparkDash 67.7 / 111.5, held-320 mean 65.66, 8 slots 173 distinct;
  TensorFold MiaAI 1.1.4+17 (official FP4-Local) 47.2 / 77.8, 57.0 / 92.7, 58.78, 5 slots 163; a third engine gb10 core 39.3 / 64.0, 47.7 / 63.3, 42.83,
  8 slots 158; TF 1.2.7 stock 42.5 / 68.4 (identity on/off 45/51). NVIDIA model-card vLLM+b12x (A_nvref) and another vLLM recipe: see the suite logs; not in the table.
- NOT yet written up: window D (nsys one-stream kernel profiles of TensorFold on 923b and custom engine R on 3d24; gate runs D_gate_base, D_gate_c25,
  D_tf_prof) and the Phase 2 step budget. Raw data: 923b ~/q38/runs/D_* and ~/q38/prof, 3d24 ~/q38/prof and ~/q38/nsys, Mac results/r*d.
- Disk changed: Pico had every model except the live DeepSeek deleted. Free now: head ~995 GB, 3d24 2.1 TB, 4e5c 2.2 TB, 923b 1.9 TB. Checkpoint copies
  may now stay on all three workers (3d24 and 4e5c each hold 96 GB under ~/q38/models from window E's setup). 4e5c is usable.
- Network: at 10:56 UTC the Wi-Fi uplink dropped on the head and 4e5c and did not come back by itself (tailnet + public endpoint down ~3 h).
  Fixed by root `nmcli con up pico` through the nsenter container. If the head is unreachable on tailnet, go through spark-3d24 (Mac hp.sh).

## Round 2 (second agent), started 2026-11-12 ~13:30 UTC

### Window D written up from the raw data (was unrecorded)
Raw: Mac results/rD/{923,3d24,4e5c}/ (copies of the workers' ~/q38/runs/D*), head ~/q38/runs/D.
- nsys "kernel profiles" FAILED on both engines, no per-kernel GPU times exist from window D (measured fact, not an estimate):
  - TensorFold (923b, D_tf_prof): the .nsys-rep is 60 KB and holds no CUDA data at all. tensorfold-native is a Zig binary that opens
    libcuda.so.1 itself, so `nsys launch`'s injection sees nothing.
  - custom engine R (3d24, D_fly_prof): 12 MB report with host-side API calls only (cudaLaunchKernel 51,373, cudaGraphLaunch 7,421,
    cudaStreamSynchronize 4,943 calls = 53.3 s host wait, 517K graph-node events with start == end), and NO GPU kernel table.
  - Speeds under the profiler: TF held-12 chat 48.06 tok/s, custom engine R 54.83 tok/s (same 12 prompts).
  => per-kernel times had to come from an engine-side profiler (TF_KPROF, built this round; see window F).
- TensorFold draft-gate sweep (lossless: only how many MTP drafts get verified changes). Same binary, same day, each box vs its own base.
  ident 51/51 and held 107/107 texts identical in all three pairs (the gate is output-neutral on these sets).

  | box  | gate (running product), max drafts | chat c1 | code c1 | sdm prose | sdm code | held chat | held code | held math | held ALL (104) | drafts/round | accepted/round | tok/round | ms/round |
  |------|------------------------------------|---------|---------|-----------|----------|-----------|-----------|-----------|----------------|--------------|----------------|-----------|----------|
  | 923b | 0.40, 6 (README default)           | 46.4    | 75.0    | 54.6      | 89.0     | 47.66     | 76.28     | 71.68     | 55.71          | 3.89         | 2.51           | 3.51      | 61.0     |
  | 923b | 0.25, 6                            | 42.4    | 74.5    | 54.5      | 90.3     | 46.77     | 75.06     | 70.58     | 54.72 (-1.8 %) | 4.49         | 2.60           | 3.60      | 63.9     |
  | 3d24 | 0.40, 6                            | 44.6    | 74.6    | 54.1      | 88.2     | 47.47     | 75.49     | 70.83     | 55.37          | 3.87         | 2.49           | 3.49      | 61.4     |
  | 3d24 | 0.20, 8                            | 41.3    | 68.6    | 50.0      | 88.1     | 45.09     | 75.86     | 70.18     | 53.39 (-3.6 %) | 5.53         | 2.81           | 3.80      | 69.0     |
  | 4e5c | 0.40, 6                            | 44.5    | 75.4    | 54.5      | 88.9     | 47.77     | 76.18     | 71.42     | 55.76          | 3.92         | 2.52           | 3.52      | 61.2     |
  | 4e5c | 0.12, 8                            | 40.4    | 68.9    | 49.0      | 91.0     | 43.71     | 75.76     | 70.63     | 52.25 (-6.3 %) | 6.15         | 2.85           | 3.84      | 71.0     |
  Verdict: loosening the gate loses on every box. Window E (tighter gate 0.5, 8 drafts) never ran.
- TensorFold round cost, fitted on the 190 drafted one-stream requests of each base run (kit/tfsteps.py on the engine's "done" lines):
  ms/round = 44.5 + 4.25 x drafts verified (923b), 44.6 + 4.35 (3d24), 44.5 + 4.27 (4e5c). Average round: 4.89 rows, 3.51 tokens, 61.0 ms.
  The 4.25 ms per draft contains BOTH the draft's own MTP head step and its verify row.
- The three boxes are equal within 1 % (held ALL 55.71 / 55.37 / 55.76). The "923b is 4-5 % faster than 3d24" note above was wrong:
  it compared launcher `tf` (MiaAI start.sh, 923b) with launcher `tfb` (binary bind-mounted, 3d24). On 923b itself `tf` gave sdm prose
  57.4 / code 93.1 (window A) and `tfb` gave 54.6 / 89.0 (window D). The launchers differ in: --vision on (tf), TENSORFOLD_CUDA_KERNELS=native
  set (tfb), explicit sampling defaults (tfb). OPEN: which of these costs 5 %. Until known, table rows say which launcher.
- A_nvref (NVIDIA model-card vLLM + b12x command): the container exited during start in window A ("FAILED nvref/default: container exited");
  the suite log holds only connection errors. No numbers. another vLLM recipe image was pulled but never run through the suite. Neither is in the table.
- held set sizes: window A/B rows used all 320 prompts, window D rows used every 3rd (104 answered). Do not compare 58.78 (320) with 55.71 (104).

### Bytes a 1-row step reads, both checkpoints (safetensors headers, kit/bytes.py; no weights read)
| group | official FP4-Local (TensorFold) | EXL3 4.05 bpw (custom engine R) |
|---|---|---|
| GDN linear attention | 2.005 GiB (MXFP8) | 1.471 (EXL3 ~4 bpw) |
| hyper-connection | 1.195 (BF16) | 1.193 (F16) |
| QSA attention | 0.593 (MXFP8) | 0.427 |
| shared expert | 0.227 (MXFP8) | 0.166 |
| router | 0.117 | 0.117 |
| LM head | 1.184 (BF16) | 0.444 (6 bpw) |
| routed experts, 10 x 48 per row | 1.15 (NVFP4) | 1.11 |
| total | 6.47 GiB | 4.93 GiB (24 % fewer) |
custom engine R's checkpoint is not only "the same weights in another container": attention, shared expert and LM head are stored at lower precision
than NVIDIA's (4 / 4 / 6 bits against 8 / 8 / 16). A quarter of its speed lead is bytes that the official checkpoint does not let an engine skip.

## Round 3 (third agent), started 2026-11-12 14:49 UTC
- 14:49: window F STILL RUNNING (q38-win + q38-guard.timer active, guard 15:39). 3d24 F2_* done 14:38, 4e5c F3_* done 14:44,
  923b F_src_base done 14:35 (ident 51/51 + held 107/107 identical: source build kp1 vs shipped binary), F_graph in progress.
  Not interrupted. Raw preview (head ~/q38/runs/F/f_*.log): 3d24 no-draft plain 22.00 vs graph 22.00 tok/s (held n 11) => decode-graph flag: no gain (measured, short set).

### Window F results so far (written 14:58 UTC, window still open for 923b F_graph)
Raw copies: Mac results/rF/3d24 (F2_*), results/rF/4e5c (F3_*). Tools added: kit/kprof.py (per-kernel table), kit/gatesim.py (gate simulation).
- F_src_base (923b): source build kp1 == shipped MiaAI binary: ident 51/51 and held 107/107 texts identical (measured).
- Decode-graph flag TF_FLASHNEXT_DECODE_GRAPH=1, no drafts, 3d24: 22.00 vs 22.00 tok/s (held 12). NO GAIN (measured). 923b drafted run pending.
- Per-kernel profile, no drafts, 1 row (3d24, TF_KPROF serialised launches; kernel sum 45.3 ms vs 45.5 ms plain wall => GPU-bound, host is not the limit):
  | part | ms / step | share | bytes | rate |
  | qmmf (MMA tile GEMM used as 1-row GEMV: MXFP8 attention/QSA/shared + BF16 hyper-connection + BF16 LM head) | 29.4 | 65 % | 5.31 GiB | 194 GB/s = 71 % of 273 |
  | nvfp4 expert kernels (2 per layer) | 7.9 | 17.5 % | 1.15 GiB | 156 GB/s = 57 % |
  | glue: ~800 tiny launches (hc_normalize2 / dense_mix2 / reduce_act 97 each, writebacks, router, gdn, attention) at 6-25 us each | 8.0 | 17.6 % | ~0.12 GiB | launch-latency bound |
  qmmf by instance (ms/step): BF16 '000' 8.90 (97.75 calls, 91 us), MXFP8 '000' 8.90 (47.7 calls, 187 us), BF16 f32+cluster 4.84, MXFP8 f32+cluster 4.01, MXFP8 cluster 1.75, MXFP8 f32 0.79.
  Launcher note: tfb no-draft = 22.0 tok/s (45.5 ms); the older "24.4 tok/s = 41 ms" was launcher tf in window A. Still OPEN which is right (window G: tf/tfb/tf A-B-A).
  Where TensorFold loses against the limit (estimate): at ~235 GB/s (what DeepSeek's GEMV kernels reach on this box) qmmf would be 24.3 ms (-5.1), experts 5.3 (-2.6),
  glue fused maybe 5 (-3): step 45.5 -> ~35 ms (+30 % no-draft, ~+22 % drafted). All three are exact if the arithmetic order is kept.
- Step log (4e5c): F3_steplog_long = 8 drafts always verified, 160 held prompts, 10,821 rounds: 78.2 ms/round, 3.71 tok/round, kept 2.69 of 7.85 drafts.
  F3_steplog_base (gate 0.40, 6): 61.1 ms/round, 3.40 tok/round, fit 45.0 + 4.22 x drafts. Cost model 44.5 + 4.25 x d reproduces both (78.5 predicted for 8).
- Gate simulation on the 10,821 always-8 rounds (SIMULATED, iid rounds, cost 44.5 + 4.25 d; default reproduces the measured 55.5):
  | policy | tok/s | tok/round | drafts/round |
  | product >= 0.40, max 6 (default) | 55.11 | 3.29 | 3.56 |
  | product >= 0.50, max 6 | 55.03 | 3.18 | 3.14 |
  | product >= 0.40, max 8 | 56.89 (+3.2 %) | 3.52 | 4.08 |
  | product >= 0.50, max 8 | 57.00 (+3.4 %) | 3.40 | 3.57 |
  | product >= 0.25, max 6 | 54.02 (-2.0 %; measured -1.8 %) | | |
  | product >= 0.20, max 8 | 54.47 (-1.2 %; measured -3.6 % => sim is ~2 % optimistic for 8 drafts) | | |
  | logistic gate on (product, last p, min p, depth), 2-fold | 57.07 best | | |  => learned gate on these inputs: +0.1 % over the product gate, DEAD
  | oracle: verify exactly the accepted drafts | 66.0 (+20 %) | 3.69 | 2.69 |
  | oracle: accepted + 1 | 61.9 (+12 %) | | |
  DECISION: test product 0.5 / max 8 and 0.45 / max 8 in window G (config only, expected +1.5..3 %).
- 15:00 UTC: window F CLOSED by itself. DeepSeek verified: health 200, 391, session store 12.00 GiB all ranks, disk tier 28.2 GiB, image ok. Outage 2727 s
  (14:14:10 -> 14:59:37). No q38 containers on any worker.
- F on 923b (launcher tfb, held 104): shipped binary 55.71, source build kp1 55.98, kp1 + TF_FLASHNEXT_DECODE_GRAPH=1 55.86; ident 51/51 + held 107/107 identical
  in both comparisons. Decode graph = NO GAIN (dead). Source build = shipped binary (same speed, same texts). Round fit kp1: 44.28 + 4.27 x drafts.
  Raw: Mac results/rF/923/.
- Dense tensor shapes (per layer): hyper-connection mix_up BF16 (10240 x 320) and mix_down BF16 (320 x 10240), two of each; QSA q_proj MXFP8 (12288 x 2560),
  o_proj (2560 x 6144), k/v (512 x 2560), indexer (640 x 2560); shared expert gate/up (640 x 2560), down (2560 x 640); head BF16 (248320 x 2560).
  mix_down runs as 5 column tiles x 8 K-slices = 40 blocks only (49.5 us for 6.55 MB = 132 GB/s): under-parallel.
- PLAN (15:00): exact kernel-throughput levers first (qmmf pipeline depth / block geometry, expert-kernel prefetch depth + grid), measured with a
  standalone CUDA microbench (kit/mb) run inside a window without loading the model, bit-compared against the stock kernel on the same inputs;
  then gate 0.5/max 8; then launcher A-B-A. Certified head only after those (largest effort).

### Window G launched 15:14 UTC (unit q38-win, guard 75 min => ends by itself <= 16:29; script head ~/q38/w_G.sh, Mac harness/w_G.sh)
- Build kp2 (4e5c ~/q38/build/out/kp2, sha 319d905ba671c73f, copied to ~/q38/tfbin/kp2 on all 3 workers) = kp1 + exact kernel knobs, all off unless set:
  TF_Q38_EXP_D=N (gate-up expert kernel prefetch depth 3/4/6/8; stock 2), TF_Q38_EXP_D1=N (down kernel 3/4/6/8/12),
  TF_Q38_QMMF_ST=N (qmmf 16-row MXFP8/BF16 tile cp.async stages 3/6/8/12; stock 4), TF_Q38_QMMF_ST_CL=N (cluster instances only).
  Source: Mac work/tf (git diff vs 6a8e032), files zig/kernels/cuda/{nvfp4_experts.cu,qmmf.cu}, zig/src/cuda/{experts.zig,qmmf.zig}.
  Why exact: the knobs change only how many weight stages are in flight; each accumulator sees the same mma/fma sequence.
- Microbench kit/mb (mb.cu, mbbuild.sh, mbrun.sh; on 4e5c ~/q38/mb): stock qmmf vs stage variants on decode shapes + raw read ceiling, bit-compared.
- Lanes: 923b g_923b.log (launcher tf/tfb/tf A-B-A; gate 0.5/8 and 0.4/8 vs tfb base), 3d24 g_3d24.log (qmmf stages, no drafts, 6 held prompts),
  4e5c g_4e5c.log (microbench -> ~/q38/runs/G3_mb.txt; expert prefetch depths; TF_NVFP4_DECODE_WARPS=1). Results: head ~/q38/runs/G/*.log, workers ~/q38/runs/G*.
- 15:17 MICROBENCH (measured, 4e5c GB10: 48 SMs, 100 KB smem/SM, L2 24 MB; raw: 4e5c ~/q38/runs/G3_mb.txt). Stock qmmf kernel alone, M=1 and M=5 equal:
  | shape | us | GB/s | stage variants 3/6/8/12 |
  | head BF16 248320x2560 | 5292 | 240 | no gain (239-241), bits same |
  | q_proj MXFP8 12288x2560 | 140 | 232 | no gain |
  | hc mix_up BF16 10240x320 | 30.4 | 215 | no gain |
  | hc mix_down BF16 320x10240 (8 slices, cluster) | 36.2 | 217 | st3 222, st6/8 208 |
  | o_proj MXFP8 2560x6144 (cluster) | 79.3 | 205 | no gain, st12 170 |
  | shared 640x2560 (cluster) | 12.0 | 141 | no gain |
  VERDICT: qmmf already runs at 205-240 GB/s (the practical ceiling of this box, 75-88 % of 273). The pipeline-depth lever is DEAD (all variants bit-identical, none faster).
  Correction to the profile above: TF_KPROF's per-launch synchronisation adds ~5-13 us to every launch (mix_down 36 us alone vs 49.5 in the engine table),
  so the dense bytes cost ~25 ms of the 45.5 ms step, the experts ~7.5 ms, and ~12 ms (26 %) is launch count: ~1370 kernel launches per token (28 per layer),
  ~950 of them tiny glue kernels. Floor for all bytes at 225 GB/s = 30.8 ms. Remaining exact levers: expert kernels (57 % of the limit) and fusing glue launches.
- FOUND in old raw data (Mac results/r3d24c/runs/B2_tf_p16, window C on 3d24, launcher tfb variant p16 = --parallel 16, TF_FLASHNEXT_PRODUCT_STREAMS=16,
  --context 65536, fp8 KV; never written up): TensorFold DOES serve 16 users at a 64K window. Measured: distinct 4 = 142.8, 8 = 191.1, 16 = 206.4 / 212.5,
  copies 8 = 212.4, 16 = 231.5 tok/s; max TTFT at 16 = 7 s. (p16 at 262,144 failed to start: memory.) So "nobody serves 16 users" holds only for README defaults.
- PDL idea (15:25): ~1370 launches/token. Programmatic dependent launch lets the next kernel spin up and prefetch its first weight stages while the previous
  kernel finishes. Exact (no arithmetic change). qmmf.cu patched in Mac work/tf (grid_launch at entry, weight stages before grid_wait, inputs after).
  Microbench kit/mb/mb2.cu (hc chain tiny->mix_down->tiny->mix_up->tiny, plain vs PDL on qmmf vs PDL on all) built on 4e5c ~/q38/mb/mb2; runs in window H.
- Certified LM head (ESTIMATE, not simulated: no logit dump exists): the head costs 5.29 ms alone at M=1..5 (measured) = 8.7 % of a 61 ms drafted round, 11.6 % of a
  no-draft step. An MXFP8 coarse copy (reusing the qmmf MXFP8 instance) would read half the bytes: ~-2.5 ms/round = ~+4 % if FP8 prunes as it did on DeepSeek. Not built this round.
- 15:38 window G lanes on 3d24/4e5c DONE (no drafts, 6 held prompts, texts identical 6/6 in every pair; measured):
  | box | variant | tok/s | vs base |
  | 3d24 | base kp2 | 22.00 | |
  | 3d24 | TF_Q38_QMMF_ST=8 / 6 / 3 | 21.77 / 21.93 / 21.93 | -1.0 / -0.3 / -0.3 % |
  | 3d24 | ST=12, ST_CL=12 | engine did not start (BF16 stage x 12 = 122 KB > 99 KB shared memory) | |
  | 4e5c | base kp2 | 22.35 | |
  | 4e5c | TF_Q38_EXP_D=4,D1=8 / 6,12 / 3,4 / 8,8 | 21.85 / 21.84 / 22.04 / 21.77 | -2.2 / -2.3 / -1.4 / -2.6 % |
  | 4e5c | TF_NVFP4_DECODE_WARPS=1 (existing upstream knob) | 22.40 | +0.2 % (noise) |
  VERDICT: expert prefetch depth, qmmf pipeline depth and the 1-warp expert grid are all DEAD (exact, but slower or equal). Upstream's kernel geometry is already tuned.
- Launcher (923b, same window, held 104, drafts on): tf (MiaAI start.sh) 58.41 vs tfb 55.75 tok/s => tf is +4.8 % faster. Image has no captured kernel set
  (both run native kernels), same env list; the remaining differences are --vision (+3 flags) and TENSORFOLD_CUDA_KERNELS=native being set explicitly.
  Side runs started 15:37 inside the open window: 4e5c G3_vis (tfb + --vision flags), 3d24 G2_kern (tfb with TENSORFOLD_CUDA_KERNELS unset). eng.sh gained TFARGS and TFKERN.
- PDL microbench (4e5c ~/q38/runs/G3_mb2.txt, measured): hyper-connection chain tiny->mix_down->tiny->mix_up->tiny: plain 72.5 us, PDL on qmmf only 69.2,
  PDL on every launch 61.0 (-16 %), bits same. 10 tiny kernels: 20.5 us plain (2.05 us per launch) -> 16.0 with PDL.
  Scaled to the step (ESTIMATE): ~290 qmmf launches x ~3.5 us + ~1080 other launches x ~0.45 us = ~1.5 ms of 45.5 = ~+3 %.
  NEW QUESTION: kernels alone add up to ~34 ms (dense 25.3 + experts ~7 + ~1080 tiny x 2 us), yet the step is 45.5 ms => ~10 ms per token may be HOST-side
  (per-layer synchronisation, n-gram gather pread, downloads). TF_KPROF hid this because it synchronises every launch. Check F2_prof_nd (TF_FLASHNEXT_PROFILE=1).
- 15:42 KEY FINDING (measured, F2_prof_nd on 3d24, TF_FLASHNEXT_GATHER_LOG=1, no drafts): every 1-row pass WAITS 4.3-4.6 ms for the host n-gram gather
  ("gather 4.2 ms, join wait 4.56 ms" per pass; ~16 rows read per pass at 100-300 us each; row-cache hit rate only 10-30 %). MiaAI's default
  TF_FLASHNEXT_GATHER_READ=pread reads every n-gram row from the NVMe with O_DIRECT (the 26.9 GiB table is not in memory). That is ~10 % of a no-draft step.
  LEVER "n-gram rows from memory" (lossless: same file bytes): (a) GATHER_READ unset = upstream's mmap path + page cache pre-warmed with the table
  (fits: MemAvailable 37-40 GiB idle; cache is reclaimable so the KV pool can still grow), (b) bigger row cache TF_FLASHNEXT_GATHER_CACHE_MIB, (c) buffered pread.
  Drafted rounds: unknown how much of the wait is hidden by DRAFT_AHEAD; the round constant 44.5 ms ~ the no-draft step, so up to ~7 % (estimate). Test in window H.
- 15:52 launcher gap LOCATED (measured, 923b): tf 58.41 vs tfb 55.75 tok/s (held 104), texts identical 107/107; round fit tf = 41.8 + 4.27 d, tfb = 44.5 + 4.27 d:
  tf saves 2.7 ms of fixed cost per round. Not vision (4e5c tfb + --vision flags: 55.52, no change), not captured kernels (none in the image).
  docker inspect of the running tf container: it starts through the image's /opt/nvidia/nvidia_entrypoint.sh; eng.sh tfb overrides the entrypoint with the
  engine binary and so skips that script (CUDA forward-compat setup). HYPOTHESIS: the entrypoint enables a newer CUDA userland => ~2 us less per launch
  (1370 launches per pass). Test in window H: tfb with TFENTRY=/opt/nvidia/nvidia_entrypoint.sh. All earlier tfb numbers (windows D-G) carry this ~4.8 % handicap;
  comparisons between tfb runs stay valid.
- 16:02 A-B-A on 923b complete: tf 58.41 / tfb 55.75 / tf 58.06 (held 104). The launcher gap is real: +4.4..4.8 %.
- Builds ready on all 3 workers (~/q38/tfbin): kp3 (sha cf6fbcae2b29e24b) = kp2 + TF_Q38_NGRAM_BUFFERED=1 (n-gram reader without O_DIRECT) + PDL-ready qmmf
  prologue; kp4 (sha 7fd0e1ed4dd06351) = kp3 + TF_Q38_PDL=1 (PDL attribute on 57 glue kernels + qmmf + nvfp4 expert + plan kernels; each waits at entry:
  kernels/cuda/q38_pdl.cuh, src/cuda/q38_pdl_names.zig, src/cuda/launch.zig Q38Pdl). Source: Mac work/tf working tree (git diff vs 6a8e032).
- kit additions: eng.sh TFENTRY / TFARGS / TFKERN; kit/warm.sh (cat the 7 n-gram shards, list in ~/q38/kit/ple_files.txt on each worker); kit/gx.sh.
- NEXT: window H (head ~/q38/w_H.sh, Mac harness/w_H.sh) right after G closes: entry base, buffered/mmap n-gram + warm, PDL, gate, 16 slots.
  Launch: ./h.sh 'systemctl --user reset-failed q38-win; systemd-run --user --unit=q38-win /bin/bash ~/q38/qw.sh H 75 ~/q38/w_H.sh'
- custom engine R does and TensorFold (MiaAI) does not (from its README): PDL launches (ported: kp4), CUDA graphs for every decode shape (TF's flag measured: no gain),
  one fused persistent MoE decode kernel, NVMe prefix cache + multi-turn prompt checkpoints, YaRN to 1M context, 8 batched sequences;
  and lower-precision weights (EXL3 4.05 bpw, head 6 bpw), which is not lossless against the official checkpoint.
- 16:14 window G CLOSED (ended early by SIGTERM to qw.sh after the gate 0.5 run; the 0.4/max 8 run was dropped). DeepSeek verified: health 200, 391,
  session store 12.00 GiB, disk tier 28.2 GiB, image ok. Outage 3669 s (15:12:57 -> 16:14:06).
- Gate result (923b, tfb, held 104): product 0.5 / max 8 drafts = 56.24 vs base 55.75 tok/s = +0.9 % (simulated +3.4 %); ident 51/51, held 107/107 identical.
  Small but free; included as a candidate in window H on top of the n-gram lever.
- 16:15 window H LAUNCHED (guard 75 min => closes by itself <= 17:30). Lanes: head ~/q38/runs/H/h_923b.log, h_3d24.log, h_4e5c.log.
- 16:33 window H so far (measured):
  * ENTRYPOINT LEVER CONFIRMED. 923b drafts on, held 104: tfb + TFENTRY=/opt/nvidia/nvidia_entrypoint.sh = 58.21 tok/s (tfb without: 55.75; MiaAI start.sh: 58.41/58.06),
    ident 51/51 + held 107/107 identical to the no-entry run; round fit 42.1 + 4.21 d (was 44.5 + 4.27 d). 4e5c: 58.09 (was 55.5-55.8). 3d24 no drafts: 23.03 (was 22.00), 6/6 identical.
    MECHANISM (verified in the running container): the entrypoint's 50-gpu-driver-check.sh links /usr/local/cuda/compat/lib -> lib.real, i.e. the container's
    CUDA forward-compat userland (libcuda built for driver 580.178.04) replaces the host's 580.173.02 libcuda; launches get ~2 us cheaper (x ~1370 per pass).
    No host change. Any launcher that overrides the image entrypoint loses 4.5 %. NEXT-ROUND IDEA: try the compat libs of a newer NGC image.
  * n-gram rows from memory: no drafts, 3d24: buffered pread + warm 23.05 (+0.1 %), mmap + warm 23.25 (+1.0 %), texts identical. In drafted rounds the gather
    is already done ahead (no gather log lines at all: passes are "prepared"). VERDICT: the 4.5 ms join wait seen under TF_FLASHNEXT_PROFILE is not a real
    stall worth more than 1 % without drafts and nothing with drafts. Lever DEAD for the default (drafted) config; mmap+warm is a +1 % option for --no-drafts.
- 16:45 CORRECTION: TF_FLASHNEXT_DECODE_GRAPH=1 never engaged in windows F: the code path is `if (e.collective != null and round_graph.enabled() ...)`
  (cuda_prompt.zig:454), i.e. only in tensor-parallel rank mode, and only for rounds of one row per stream. On one GPU the flag is a NO-OP. So the earlier
  "decode graph: no gain" is void; graph capture of the product path is UNTESTED and is not a flag flip (the server's lane rounds do not go through that function).
- PDL in the engine (kp4, TF_Q38_PDL=1), 3d24 no drafts, 6 held: 23.18 vs 23.03 tok/s = +0.65 %, texts identical 6/6; kp4 with the flag off = kp3 (23.03).
  The microbench's -16 % did not transfer => the GPU is not waiting on kernel start-up; it is waiting on the HOST to submit launches
  (consistent with the entrypoint finding: a newer libcuda userland alone gave -2.7 ms per round). ~1370 cuLaunchKernel calls per pass are the bottleneck
  between the big kernels. The lever that removes it is CUDA-graph capture of the drafted window (one submission per pass; custom engine R does this) or fusing glue kernels.
  ESTIMATE of the prize: step 43.4 ms now vs ~34 ms of kernel work => up to ~9 ms (20 %) no-draft; drafted rounds proportionally (est. +10-15 %).
- 16:52 MISTAKE FOUND AND RECORDED: builds kp3 and kp4 did NOT contain the buffered n-gram patch (the source tarball never reached the build box; the
  build box's direct_io.zig was unpatched). So every "TF_Q38_NGRAM_BUFFERED=1" run in window H (H2_buf, H2_buf_cold, H3_buf, H_buf, H2_k4_pdl_buf) was a
  NO-OP for the engine: those rows measure only run-to-run noise plus a warm page cache that O_DIRECT reads never use (23.05, 23.04, 23.55 no-draft; 57.88, 58.19 drafted).
  The buffered-pread lever is UNTESTED. Only the mmap variant (real: upstream path) was measured: +1.0 % without drafts.
  kp5 (building on 3d24 ~/q38/build, all files from `git diff --name-only 6a8e032` shipped this time) = kp4 + buffered patch for real + TF_HOSTPROF aid,
  and the q38 knobs renamed TF_FLASHNEXT_Q38_* (NGRAM_BUFFERED, PDL, EXP_D, EXP_D1, QMMF_ST, QMMF_ST_CL) so MiaAI's start.sh forwards them.
  Lesson: after copying sources to a build box, grep one new symbol there before building.
- 17:08 window H CLOSED (ended early by SIGTERM; its last two runs are repeated in window I). DeepSeek verified: health 200, 391, session store 12.00 GiB,
  disk tier 28.2 GiB, image ok. Outage 3216 s (16:14:23 -> 17:07:59).
- H results added (measured): PDL with drafts, 4e5c held 104: 58.05 vs 58.09 = no gain (ident 51/51, held 107/107 identical). PDL = exact but ~0: stays off.
  mmap n-gram with drafts, 923b: 54.68 vs 58.21 = -6.1 % (it turns off the pread read-ahead) => mmap is only for --no-drafts (+1 %).
  16-slot run and gate+entry run were cut; both are in window I.
- BEST lossless config so far = MiaAI's start.sh path (image entrypoint kept) + gate product 0.5 / max 8 drafts. Our patches add flags that measure ~0 with drafts.
- 17:09 window I LAUNCHED (guard 90 min => closes by itself <= 18:39; script head ~/q38/w_I.sh, Mac harness/w_I.sh; build kp5 sha d370c291395c1676 on all 3 workers).
  923b: I_base (no q38 flags; c1, sd, ident, held 104) then I_best (gate 0.5/8; FULL suite) -> head ~/q38/runs/I/i_923b.log, 923b ~/q38/runs/I_base, I_best.
  3d24: buffered n-gram for real (no drafts x4, drafts x3 incl. PDL) -> i_3d24.log. 4e5c: TF_HOSTPROF host launch cost; 16 slots at 64K best vs base -> i_4e5c.log.
- Step budget under many users (measured, from the 16-slot run of window C on 3d24: launcher tfb without the entrypoint, code prompts, kit/mustep.py on the
  engine's "done" lines; a shared round verifies every user's row + drafts in one pass):
  | users | tok/s total | ms / round | rows / round | drafts verified / user | tokens / user / round | ms per row |
  | 1 (fit) | 55.7 held | 44.5 + 4.27 d = 61 | 4.9 | 3.9 | 3.4-3.5 | 4.3 marginal |
  | 4 | 144-148 | 118-121 | 24 | 5.0 | 4.8-4.9 | 5.0 |
  | 8 | 195-215 | 157-176 | 49-50 | 5.1-5.3 | 4.8-5.0 | 3.2-3.5 |
  | 16 | 206-212 distinct / 232-250 copies | ~211 | 68 | 3.2 (the engine's cost model trims drafts) | 3.8 | 3.1 |
  Reading: one user pays ~42-44 ms of fixed dense + head + launches per round and ~4.2 ms per extra row; with many users the fixed part is shared and the
  cost is rows x ~3.1-3.5 ms, i.e. expert bytes (a row's own experts are 1.15 GiB = 5.4 ms at 230 GB/s if nothing were shared). Byte budget per round:
  dense 4.13 GiB + head 1.18 GiB once, + up to 1.15 GiB per row (less when rows share experts).
- 17:14 HOST LAUNCH COST MEASURED (kp5, TF_HOSTPROF, 4e5c, no drafts, 23.58 tok/s): 548,000 launches = 1377 per token; time inside the driver's launch
  calls 1154 ms of 16,892 ms wall = 2.1 us per launch = 2.9 ms of a 42.4 ms step = 6.8 %. So host submission is NOT a 20 % stall either: a perfect CUDA-graph
  capture could save at most ~7 % (if the GPU waits for all of it). Revised picture of a 42.4 ms no-draft step (measured parts + estimate):
  dense + head bytes ~25 ms at 205-240 GB/s (at the ceiling), experts ~7 ms (177 GB/s), ~1080 small kernels ~2-5 ms, launch calls 2.9 ms, n-gram wait <= 0.5 ms.
  Floor if everything ran at 225 GB/s with no glue: ~31 ms. TensorFold + MiaAI's patches are within ~25 % of that floor; no single large exact lever is left
  in the launch path. Remaining exact levers by estimate: certified LM head ~6 %, graph/fused glue <= 7 %, expert kernel to 225 GB/s ~3.5 %.
- 17:19 window I so far (measured, kp5):
  * buffered n-gram reader FOR REAL (TF_FLASHNEXT_Q38_NGRAM_BUFFERED=1 + warm), 3d24 no drafts: 23.17 vs 23.22 = no gain, 6/6 identical. Lever DEAD
    (the "join wait" in the profile is not a real stall: the read-ahead threads already hide it).
  * host launch cost WITH drafts (4e5c, 58.96 tok/s on 27 held prompts): 2,575,600 launches, 12,374 ms inside the driver's launch calls of 99,130 ms wall
    = 4.8 us per launch (2.1 us without drafts), ~1520 launches per round = 7.3 ms of a 58.4 ms round = 12.5 %.
    => upper bound for CUDA-graph capture / glue fusion in the default drafted config: ~12 % (only if the GPU idles during all of it). OPEN: why a launch
    costs 2.3x more with drafts on (second stream for the shared expert? MTP kernels? contention with the 7 host draw threads?).
- 17:28 16 slots at 64K (4e5c, kp5, entrypoint, launcher tfb p16; distinct code prompts, 512 tokens; measured):
  | config | 1 user | 4 | 8 | 16 distinct (two runs) | 16 copies |
  | default gate 0.4 / 6 | 76.1 | 141.8 | 195.6 | 214.8 / 221.0 | 251.1 |
  | gate 0.5 / 8 | 73.6 | 143.8 | 197.5 | 218.4 / 215.1 | 241.4 |
  The gate makes no difference for many users (within run-to-run noise). 16 users at 64K = ~215-221 tok/s total, max TTFT 6.7 s.
- 17:35 I_base (923b, kp5, entrypoint, MiaAI default gate 0.4/6, launcher tfb; measured): chat 47.0 / code 78.1 (3-prompt bench, 9 runs); sparkDash-method
  prose 57.2 / code 92.5 / structured 106.7; held-104 58.33 (chat 49.99 / code 79.59 / math 74.67); TTFT 172 ms; round fit 42.0 + 4.18 d.
  = MiaAI's own launcher in window A (47.2 / 77.8, 57.4 / 93.1): our source build with all q38 flags off is MiaAI's recipe, same speed.
- no drafts, 3d24 (kp5): buffered + PDL 23.14 (0 %), mmap + warm + PDL 23.99 vs 23.22 = +3.3 %, texts identical 6/6 => best --no-drafts config, not the default.
- 17:57 window I: 923b held-104 paired: I_base 58.33 -> I_best (gate 0.5/8) 59.00 tok/s = +1.15 % (chat 49.99 -> 50.33, code 79.59 -> 81.68, math 74.67 -> 76.00).
  3d24 drafts on: buffered n-gram + warm 57.52 vs base 57.39 = no gain (confirms DEAD for real this time). I_best full suite still running (sparkDash, qeval, image, needle).
- 18:10 window I CLOSED (ended by SIGTERM after I_best finished; the last 3d24 run, PDL with drafts, was cut: already measured 0 % on 4e5c).
  DeepSeek verified: health 200, 391, session store 12.00 GiB, disk tier 28.2 GiB, image ok. Outage 3704 s (17:08:14 -> 18:09:58). No q38 containers on any worker.
  I_best vs I_base: ident 51/51, held 107/107 identical; also identical to H_ent.
- 18:15 FINAL TABLE: Mac results/final_923b.txt (kit/table.py). Same box 923b. Paired in window I (our build kp5, entrypoint, launcher tfb):
  | bench | MiaAI default gate 0.4/6 (I_base) | gate 0.5/8 (I_best) | change |
  | held-104 all / chat / code / math | 58.33 / 49.99 / 79.59 / 74.67 | 59.00 / 50.33 / 81.68 / 76.00 | +1.15 % / +0.7 / +2.6 / +1.8 |
  | 3-prompt bench chat / code (9 runs) | 47.0 / 78.1 | 46.7 / 75.1 | -0.6 % / -3.8 % |
  | sparkDash-method prose / code / structured | 57.2 / 92.5 / 106.7 | 55.6 / 94.0 / 123.9 | -2.8 % / +1.6 % / +16 % |
  | 5 users distinct / copies | (MiaAI start.sh, window A: 162.9 / 166.7) | 153.2 / 163.5 | lower (different window and launcher) |
  ident 51/51 and held 107/107 identical. VERDICT: the 0.5/8 gate is a TRADE, not a win: better on the held average and on structured output, worse on the
  3-prompt bench and prose. DECISION: the recipe default stays MiaAI's 0.4/6; 0.5/8 is documented as an option for structured-heavy use.
  I_best also: prefill 2400 / 2462 / 2308 tok/s, needle 200K 3/3, 227,621-token prompt 3/3, qeval 33/35, TTFT 180 ms. Image check FAILED only because the test
  launcher (eng.sh tfb) starts without --vision; MiaAI's start.sh (vision on) passed it in window A.
- BOTTOM LINE of round 3: no lossless lever beat MiaAI-Lab's shipped configuration across benches. Best lossless config on the official checkpoint =
  MiaAI's recipe as shipped (entrypoint kept): chat 47 / code 78, sparkDash 57 / 93, held 58.3-58.8, 5 users 163, and 16 users 215-221 at a 64K window.

## Round 3 END STATE (18:16 UTC 2026-11-12)
- No window open (q38-win, q38-guard inactive), q38-proxy STOPPED, no Qwen engine container on any worker, page cache evicted (3-5 GiB cached per worker),
  DeepSeek verified locally (health 200, 391, session store 12.00 GiB, disk tier 28.2 GiB, image ok) and through the public URL.
- Outages this round: F 2727 s, G 3669 s, H 3216 s, I 3704 s (all ended by themselves; G/H/I were ended early by SIGTERM to qw.sh once the needed runs were in).
- Deliverables: private repo Picom-code/pico-dgx-qwen38 branch tensorfold-fp4-local commit d568ff5 (tensorfold/: README, recipe.sh, patches 0101-0104, kit, notes);
  skill reference dsgx10-spark-deepseek-engine/references/qwen38-fastest-build.md; final table Mac results/final_923b.txt.
- Source: Mac work/tf (git: 6a8e032 MiaAI base, then "q38 working tree (kp4)", "kp5"); work/tfA = aids-only tree; patches Mac repo/tensorfold/patches.
- RESUME COMMANDS
  cd ~/.hermes/cache/scratch/q38; cat PROGRESS.md | tail -80
  ./h.sh 'systemctl --user is-active q38-win q38-guard.timer; tail -3 ~/q38/windows.log; curl -s -o /dev/null -w "%{http_code}\n" 127.0.0.1:18900/health'
  build:  ./h.sh 'ssh -n pico@spark-3d24 "cd ~/q38/build/tf && tar xzf /tmp/SRC.tgz && grep -c NEWSYMBOL FILE; nohup bash ~/q38/kit/tfbuild.sh NAME 8 12g > ~/q38/build/NAME.log 2>&1 &"'
  window: ./h.sh 'systemctl --user reset-failed q38-win; systemd-run --user --unit=q38-win /bin/bash ~/q38/qw.sh LABEL 75 ~/q38/w_LABEL.sh'
  end a window early: ./h.sh 'pkill -TERM -f "^/bin/bash /home/pico/q38/qw.sh"'   (never systemctl stop)
  engine for a lane: TFBIN=$HOME/q38/tfbin/kp5 TFMODEL=$HOME/q38/models/nvidia-fp4-local TFENTRY=/opt/nvidia/nvidia_entrypoint.sh bash ~/q38/kit/eng.sh tfb default|nodraft|p16
- NEXT ROUND FIRST: (1) ask Pico whether a labelled re-quantised checkpoint is acceptable for "fastest" (official one is within ~25 % of its byte floor);
  (2) why launches cost 4.8 us with drafts vs 2.1 us without (TF_HOSTPROF per kernel / per stream; try TF_FLASHNEXT_SHARED_SIDE=0, fewer host draw threads);
  (3) logit dump -> does an MXFP8 coarse head prune -> certified LM head (est. +4-6 %); (4) graph capture / glue fusion of the drafted window (bound 12 %);
  (5) compat libcuda from a newer NGC image (same mechanism as the entrypoint gain).

## Round 4 (fourth agent), started 2026-11-12 18:16 UTC
- 18:16: state verified clean (no window, health 200). Reading harness + source for lever 1 (launch cost with drafts).
- 18:30 build kp6 (sha b41d891e8c384c73, 3d24 ~/q38/build, on all workers ~/q38/tfbin/kp6) = kp5 + per-kernel host launch timer (TF_HOSTPROF=N + TF_HOSTPROF_K=1: "hostprofk name n ns slow"), stream-sync timer, TF_FLASHNEXT_Q38_SCHED=spin|yield|blocking (context wait policy; nothing in the engine sets it, so it is the driver default), TF_FLASHNEXT_Q38_HDUMP=PREFIX (LM-head input rows to PREFIX.x/.n).
  Mac helpers added: ship.sh NAME BOX (tar changed files, verify sha256 on the box, build), dist.sh NAME BUILDBOX (copy build + kit + harness to all workers).
- Lever 2 prep: the MiaAI image compat libcuda is 610.43.02 (CUDA 13.3), not 580.178 as round 3 wrote. NVIDIA apt repo (ubuntu2404/sbsa) has cuda-compat-13-4 = 615.71.10 and cuda-compat-13-5 = 620.70.06;
  both debs downloaded on each worker (sha256 checked against the repo Packages file), extracted to ~/q38/compat/x13-4, x13-5 (no host change; bind-mounted over /usr/local/cuda/compat/lib.real).
- 18:33 window J LAUNCHED (guard 95 min; head ~/q38/w_J.sh): 4e5c lever 1 (hostprof per kernel, SCHED spin/blocking, SHARED_SIDE=0, HOST_DRAW_THREADS=0), 3d24 lever 2 (compat 610/615/620 A-B-A), 923b head-row dump + offline pruning study (kit/hstudy.py). Logs head ~/q38/runs/J/j_*.log.
- 18:41 LEVER 1 first data (measured, 4e5c, kp6, drafts on, 27 held prompts, 59.04 tok/s; raw Mac results/rJ/J3_hp_dr.log, table by kit/hpk.py):
  2,584,000 launches, 12.40 s inside launch calls (4.80 us each), wall 99.4 s; 61,380 stream syncs (36 per round, one per 42 launches) in which the
  host waits 83.6 s = 84 % of wall. One thread launches (0 thread switches); it spins (top: 91 % CPU), so the context schedule is already "spin".
  Per kernel: ~90 % of launches cost 1.7-2.2 us (same as without drafts). The excess is in a few launch sites that block now and then:
  tf_route_topk 52.8 us mean (9.9 % of its launches > 20 us; ~5 per round = 4.5 s of the 12.4 s), fn_gdn_rows_late 16.4 us, fn_attn_chunks_kv8 27 us,
  fn_hc_reduce_act 5.0, tf_route_logits_few 5.1, MXFP8 plain qmmf 4.7, fn_hc_writeback_slices 3.7. So "4.8 us per launch" is NOT a higher per-launch
  price: it is ~8500 x ~0.5 ms stalls inside specific launch calls (7.5 s = 4.4 ms per round). OPEN: is the GPU idle during those stalls (a real loss) or
  is the call blocked because the queue is full while the GPU works (no loss)? Test: sensitivity run with an artificial per-launch delay (kp8).
- 18:44 build kp7 (sha 7a1f6bbd75a119a0, built on 923b ~/q38/build; build tree copied there from 3d24) = kp6 + certified LM head (TF_FLASHNEXT_Q38_CERT=1, =2 checks every call against the whole head; new src/families/flashnext/cuda_cert.zig, kernels fn_cert_select / fn_cert_check in flashnext_batch.cu, qmmf.Lane.matmulTiles). kp8 = kp7 + TF_FLASHNEXT_Q38_LAUNCH_DELAY_NS + sync call sites in hostprof (building).
  Cert design: MXFP8 copy of the head made at load (E4M3 + power-of-two scale per 32, +0.61 GiB), fp32 coarse scores, bound c[v]*||x|| (measured residual norm + 1e-3 of the row norm for float rounding), survivors = entries whose upper bound reaches the best lower bound less two bf16 steps, flagged 64-row tiles recomputed by the SAME BF16 kernel with shifted pointers (bit-identical by construction), rest -inf; one extra host sync per verify pass.
  Gate: only verify windows whose streams are all greedy, unmasked, no logprobs, <= 64 rows, one GPU; everything else takes the whole head.
- FOUND (code reading): TensorFold already has a priced cross-session draft allocator (core/lanes alloc.allocatePriced) that MiaAI leaves unconfigured: TF_FLASHNEXT_ROUND_MS="a+b" (shared verify ms by rows) and TF_FLASHNEXT_DRAFT_ROW_MS. Unpriced = every draft the gate lets through is verified up to 64 rows a pass. Lever 7 = configure it from our measured fit (42 + 2.5..3.3 ms per row).
- no-drafts hostprof (4e5c): 1,411,100 launches at 2.13 us, 51,333 syncs = one per layer (the host waits 79.5 % of wall).
- 18:59 window J partial (measured, held every 6th prompt = 54, drafts on, texts identical 54/54 in every pair):
  LEVER 2 (3d24): image compat libcuda 610.43.02: 58.95 tok/s (58.76 ms/round); NVIDIA cuda-compat 13.4 (615.71.10, verified loaded from /proc maps): 58.71;
  cuda-compat 13.5 (620.70.06, verified loaded): 58.64. => newer CUDA userland than the image's: NO GAIN (-0.4 / -0.5 %, inside run-to-run noise). The entrypoint gain was
  580-host-lib -> 610; nothing further from 615 / 620.
  LEVER 1 (4e5c): base 58.82 (58.88 ms/round); TF_FLASHNEXT_Q38_SCHED=spin 59.10 (58.57 ms/round) = +0.5 %, inside noise: the driver default already spins (top showed 91 % CPU on the launch thread).
  Head-row dump (923b, 16 in flight, 640 prompts): 219,691 rows in 4,321 head calls (50.8 rows a call) -> ~/q38/prof/hd16.x.
- 19:04 LEVER 4 OFFLINE STUDY (measured on real rows; GPU study in window J on 923b, kit/hstudy.py; raw Mac results/rJ/study.txt). Rows: the LM head's input rows
  dumped from the engine: 15,475 rows of 1-user drafted runs (4.7 rows a head call) and 109,104 of the 219,691 rows from 16 users in flight (50.8 rows a call).
  Head facts: top logit median 21.7; top1-top2 gap median 1.47 (10 % of rows under 0.235); bf16 output ties top1 == top2 in 1.4 % of rows (so the
  certified path must keep every entry that can tie after bf16 rounding); ||w|| ||h|| ~ 99.
  | coarse code of the 248,320 x 2,560 head | bits | residual / row norm | bound (logits) | survivors per row mean / p99 | 64-row tiles per 1-user window mean / p99 | tiles per 16-user pass mean / p99 |
  | MXFP8 (E4M3, scale per 32; the engine's own lane format) | 8.25 | 2.70 % | 3.2 | 13.9 / 205 | 47 / 355 | 293 / 843 |
  | int8, scale per row | 8 | 0.98 % | 1.7 | 2.7 / 17 | 9.7 / 31 | 67 / 135 |
  | int8, scale per 32 | 8.25 | 0.60 % | 0.8 | 1.6 / 7 | 6.4 / 17 | 46 / 82 |
  | int6, scale per 32 | 6.25 | 2.32 % | 3.0 | 9.4 / 118 | 32 / 218 | 210 / 573 |
  | int5, scale per 32 | 5.25 | 4.74 % | 6.2 | 410 / 6686 | 533 / 3180 (51 % of windows > 256 tiles) | 1672 / 3874 |
  | int4, scale per 32 or 16 | 4.25-4.5 | 8.8-10.1 % | 11-13 | 50,000-100,000 (prunes nothing) | ~3700 of 3880 | ~3800 |
  Per-64-column Cauchy-Schwarz instead of whole-row: 47 -> 32 tiles (not worth 40 MB more to read). The true best was inside the survivors in 100 % of rows for every code.
  VERDICT: MXFP8 prunes: a 1-user window needs ~47 exact tiles = 15 MB of the 1.18 GiB head. Certified head reads 0.61 GiB + ~15 MB instead of 1.18 GiB:
  ESTIMATE -2.3 ms of a 58.8 ms round (+3.9 %). 4-bit / 5-bit codes and a 4-bit + sketch do not prune (as on DeepSeek). int6 would save another ~0.15 GiB
  (ESTIMATE +1.1 %) but needs a new bit-unpacking kernel: not built (follow-up).
- window J: TF_FLASHNEXT_SHARED_SIDE=0 (4e5c): 57.92 vs 58.82 = -1.5 %: the side stream is a gain, stays on. Lever 2 second base (3d24): 58.58 (so base 58.95 / 58.58, compat 13.4 58.71, 13.5 58.64).
- kp9 (building on 923b) = kp8 + MTP chain-step timer (mtpstat lines under TF_HOSTPROF) + cert limits 600 runs / 1500 tiles (16-user passes leave ~300 scattered tiles).
- 19:13 window J CLOSED (ended early by SIGTERM; the last informational run SCHED=blocking was cut). DeepSeek verified by the harness: health 200, 17*23 = 391,
  session store 12.00 GiB on every rank, disk tier 28.2 GiB, image ok. Outage 2448 s (18:32:50 -> 19:13:38).
  More J results (measured): TF_FLASHNEXT_HOST_DRAW_THREADS=0: 58.84 vs 58.82 (no change for greedy); no-drafts launch cost with compat 13.5: 2.121 us (610: 2.125): LEVER 2 DEAD.
  LEVER 1 hypotheses from the brief, all tested: wait policy (already spin, +0.5 % = noise), second stream (turning it off costs 1.5 %), host draw threads (0),
  thread contention (one launching thread, 0 switches). What remains is the stall pattern inside ~8500 launch calls; the delay-sensitivity run in window K decides whether it costs anything.
- 19:14 window K LAUNCHED (guard 95 min; head ~/q38/w_K.sh; build kp9 sha 01117c237d3bdc09 on all workers): 923b cert off / on / checked; 4e5c launch-delay 0 / 2000 / 4000 ns,
  hostprof with sync sites + MTP step timer, 16 x 96K; 3d24 16 slots unpriced vs TF_FLASHNEXT_ROUND_MS=46+2.45 / 40+3.2, 12 x 128K. Logs head ~/q38/runs/K/k_*.log.
- 19:30 LEVER 1 / LEVER 3 DECIDED BY MEASUREMENT (4e5c, kp9, held-54, texts identical 54/54): a 2000 ns spin before EVERY launch (1520 launches = +3.0 ms of host time per round)
  moved the round from 58.63 to 58.85 ms (+0.22 ms; 59.08 -> 58.86 tok/s = -0.4 %, inside noise). So at most ~7 % of launch-call time is on the critical path: the GPU does not
  wait for the host; the "4.8 us per launch" is the host blocking inside launch calls while the GPU works through its queue (the host waits 84 % of wall anyway).
  => fixing launch cost (lever 1) or CUDA-graph capture of the drafted round (lever 3, removes host submission only) can win at most ~0.07 x 7.3 ms = 0.5 ms = ~0.9 % (ESTIMATE from the
  measured sensitivity), not the 7-12 % bound. Both DEAD as levers; not built. True kernel fusion (fewer GPU-side tiny kernels) is bounded by ~430 mergeable launches x <= 2 us = <= 1.5 %
  (estimate from the microbench's 2.05 us per tiny kernel) for a large exactness risk: not built.
- lever 7 first data (3d24, 16 slots at 64K, distinct code prompts): unpriced 1 / 4 / 8 / 16 users = 74.5 / 149.3 / 191.8 / 222.2 + 222.0, copies 246.2;
  TF_FLASHNEXT_ROUND_MS=46+2.45 (priced draft allocation): 74.8 / 140.0 / 200.2 / 214.9 + 213.2, copies 241.1 => pricing does not help at 16 (-3.6 %).
  Observation: max TTFT at 16 users is 6.7 s: the 16 prompts are prefilled one after another between rounds, so the 512-token bench spends ~15 % of its wall ramping up;
  decode with all 16 running is ~288 tok/s (3.8 tok/user/round, 211 ms rounds).
- 19:37 launch-delay 4000 ns: 59.02 ms/round vs 58.63 (+0.39 ms for +6.1 ms of host delay): sensitivity 0.064, linear with the 2000 ns point. Levers 1 and 3 closed as above.
- lever 7: TF_FLASHNEXT_ROUND_MS=40+3.2: 16 users 211.0 / 214.9 (unpriced 222.2 / 222.0): priced allocation DEAD (-3 to -5 %). WHY (measured from the same runs): trimming 4.5 rows at
  8 users (50.3 -> 45.8 rows a round) saved only 2.2 ms (169.3 -> 167.1 ms): a marginal draft row costs ~0.5 ms with many streams, not the 2.5-3.3 ms average, because rows share experts.
  => the lever is the OPPOSITE: let a shared verify pass hold MORE rows. TensorFold caps a pass at 64 rows (facts.batch_rows = 4 x 16, window cap 64), so at 16 users each
  stream gets only 3.07 drafts (5.3 at 8 users). New knob TF_FLASHNEXT_Q38_WIDE_ROWS=N (64..256; our DeepSeek "wide drafted step") in kp10. ESTIMATE at 16 users, 100 rows:
  ~230 ms rounds x ~78 tokens = ~340 tok/s with all 16 running (now 259-288).
- LEVER 4 first engine result (923b, kp9, TF_FLASHNEXT_Q38_CERT=1, running): first 33 held prompts base 47.78 -> cert 49.44 tok/s (+3.5 %), 33/33 texts identical. Engine line:
  "MXFP8 copy of the head made (625 MiB), mean residual 0.0266 of a row's norm"; GPU memory +686 MiB.
- kp10 (building on 4e5c) = kp9 + TF_FLASHNEXT_Q38_WIDE_ROWS + certified head for passes up to 256 rows (exact kernel 64 rows at a time).
- 19:44 LEVER 4 MEASURED (923b, kp9, launcher tfb + entrypoint, held every 3rd prompt = 104 answered, paired in window K):
  | | base (cert off) | TF_FLASHNEXT_Q38_CERT=1 | change |
  | held all / chat / code / math tok/s | 58.30 / 49.99 / 79.46 / 74.54 | 60.58 / 52.01 / 82.46 / 77.03 | +3.9 % / +4.0 / +3.8 / +3.3 |
  | ms per round, fit | 58.20 = 42.43 + 4.14 d | 56.00 = 40.70 + 4.01 d | -2.2 ms |
  Same rounds (3.399 tok/round, 3.813 drafts) in both, i.e. identical steps. Texts + ident comparison and the checked run follow below.
- sync call sites (4e5c K3_hp, kit/syncsites.py with nm): per round the host waits in 3.9 draft picks (cuda_lanes draftFn: 1.70 ms each = the MTP chain step's GPU time, 11.2 % of wall),
  1 verify pass (Pass.chunk 43.8 ms), 1 familyRound (0.15 ms). The "36 syncs per round" was load-time syncs (50,176 in loadLayerAtRank) counted in.
  => slope 4.07 ms per draft = 1.70 ms MTP step + ~2.4 ms verify row.
- LEVER 5 status: (a) lower-precision / shortlisted draft head: ALREADY UPSTREAM. TensorFold's draft head is the 79,591-row shortlist of lm_head re-coded to 4 bits
  ("affine4", TF_FLASHNEXT_DRAFT_HEAD; their note: 1.5 ms of the 2.9 ms bf16 step) = ~127 MB = ~0.55 ms of the 1.70 ms step. Nothing left to take but a shorter list (not tested;
  on DeepSeek it simulated +0.3 %). (b) cheaper verify rows by expert union: already how the engine works (a draft row costs 2.4 ms at one user, ~0.5 ms with many streams).
  (c) prompt-lookup / suffix drafts: NOT simulated: the step logs carry probabilities and accept counts but no token ids. Open.
- LEVER 6 (expert kernel toward 225 GB/s): SKIPPED by the brief's rule: this round's profiles show no concrete cause (alignment / gather order / decode); two geometry variants were already dead.
- 16 slots x 96K window does not fit: "needs 31.57 GiB for its streams, budget 30.77 GiB" (1.97 GiB a stream at 98,304). 12 x 128K fits: 1 / 4 / 8 / 12 users = 75.4 / 142.9 / 198.1 / 216.0.
- 19:46 LEVER 4 identity (923b, window K): cert on vs off: held 107/107 texts identical, ident 51/51 identical.
- 19:55 `recipe.sh prepare` RUN END TO END on 923b (WORK=~/q38/recipe_work, HF_CACHE=~/q38/hf; log ~/q38/recipe_work/prepare.log): clones MiaAI-Lab's recipe at 20e15e4, adds patches 0101-0105,
  docker-builds the image (TensorFold eab8582 + 0001-0017 + 0101-0105, Zig 0.17.0), verifies the checkpoint: "Done" in ~5 min (base image and checkpoint already on the box).
  recipe.sh change: prepare no longer forces --rebuild (prepare.sh rebuilds when the patch set changes; REBUILD=1 forces it). Patches 0101-0105 applied to the MiaAI base reproduce
  the build tree exactly (git write-tree equal). Final-suite launcher: kit/rs.sh = recipe.sh start (MiaAI's start.sh, vision on) on that image.
- 19:55 cert CHECK mode (TF_FLASHNEXT_Q38_CERT=2, 923b, 5 in flight, running): 2000 head calls, 35,915 rows, 3 whole-head fallbacks, 35,860 rows compared with the whole head:
  best-mismatch 0, value-mismatch 0 (828,951,104 entries compared bit for bit). 363 tiles and 95 exact launches a call at 17.5 rows a call (joined runs): kp11 joins only across one empty tile.
- kp11 (building on 4e5c; patch 0105 regenerated, sha256 6546dfa58ec04b3a; prepare re-run on 923b) = kp10 + that join rule + MTP step timer every 2000 picks.
- 20:06 window K CLOSED by itself. DeepSeek verified by the harness: health 200, 17*23 = 391, session store 12.00 GiB, disk tier 28.2 GiB, image ok. Outage 3096 s (19:14:15 -> 20:05:51).
  K_cert2 final print: 5000 calls, 107,384 rows, 5 whole-head fallbacks, 107,289 rows compared: best-mismatch 0, value-mismatch 0.
- 20:07 window L LAUNCHED (guard 200 min; head ~/q38/w_L.sh; build kp11 sha 6537c027ae242a6c on all workers; 923b runs the image recipe.sh prepare built):
  923b: levers off (c1, sd, ident, held-320) -> stack on (CERT=1) FULL suite, vision on -> compare -> 16 slots + wide rows (value read from head ~/q38/wide.env when it gets there, default 128) + cert -> MiaAI default many users -> MiaAI config with 16 slots.
  3d24: wide rows 128 / 64 / 192 / 96 at 16 slots (benches + 107 held prompts 16 in flight, texts compared) -> CERT=2 checked run 5 in flight.
  4e5c: wide 128 without / with cert (texts compared) -> CERT=2 checked run 16 in flight (wide) -> 16 x 80K slots.
  Logs head ~/q38/runs/L/l_*.log.
- 20:15 LEVER 7 "wide drafted verify pass" FIRST RESULT (measured, 16 slots at 64K, kp11, distinct code prompts, 512 tokens):
  | pass rows cap | box | 16 users distinct (two runs) | 16 copies | rows / round | ms / round | tok / user / round | tok/s while all 16 run |
  | 64 (TensorFold / MiaAI) | 3d24, window K | 222.2 / 222.0 | 246.2 | 65 | 195.5 | 3.70 | 259 |
  | 128 (TF_FLASHNEXT_Q38_WIDE_ROWS=128) | 3d24 | 236.2 / 234.2 | 259.3 | 96.6 | 230.5 | 4.74 | 277 |
  | 128 | 4e5c | 232.4 / 230.9 | 278.8 | 95.9 | 228.4 | 4.79 | 284 |
  => +6 % on the 16-user bench (+7-9 % while all 16 run); 1 / 4 / 8 users unchanged (their rounds never reached 64 rows). The 31 extra rows cost 1.1 ms each.
  Identity under load: pending (107 held prompts 16 in flight, wide 64 vs 128 texts).
- 20:29 window L so far (measured, kp11):
  wide rows, same window on 3d24: cap 64: 16 users 216.4 / 221.0, copies 244.5 (267 tok/s while all 16 run, 67.6 rows, 198.6 ms rounds);
  cap 128: 236.2 / 234.2, copies 259.3 => +7.6 % distinct at 16 users. IDENTITY UNDER LOAD: 107 held prompts with 16 in flight, cap 64 vs cap 128: 107/107 texts identical.
  cert at 16 users (4e5c, wide 128): 234.8 / 234.0 vs 232.4 / 230.9 without (+1 %); 107 held prompts 16 in flight, cert on vs off: 107/107 texts identical.
  Spare-capacity extras started on 4e5c (kit/extra_mu.sh, after its lane): wide 160 + 8 drafts, gate 0.5 / 0.3, wide 192 + 10 drafts -> 4e5c ~/q38/runs/X_extra.txt.
- 20:58 window L, 923b, DOCUMENTED PATH (recipe.sh start on the prepared image, vision on, MiaAI defaults), measured:
  L_off (CERT=0 = MiaAI's configuration on our build): ttft 178 ms; 3-prompt chat 47.4 / code 79.0 (9 runs); sparkDash-method prose 56.9 / code 92.2 / structured 106.1;
  held-320: all 57.39, chat 50.01, code 74.65, math 72.70 (315 answered). On the 104-prompt subset L_off = 58.21 vs K_base 58.30 (tfb launcher), 104/104 texts identical.
  L_on (default recipe = CERT=1), first benches: chat 48.9 / code 82.0 (+3.2 % / +3.8 %); sparkDash-method prose 59.1 / code 95.9 / structured 110.1 (+3.9 / +4.0 / +3.8 %).
  wide rows sweep done (3d24, 16 users distinct, two runs each): cap 64: 216.4 / 221.0; 96: 226.0 / 227.5; 128: 236.2 / 234.2; 192: 228.3 / 232.0 (no more rows to add at 6 drafts: ~97 rows a round).
  Texts with 16 in flight identical 107/107 for 64 vs 128, 64 vs 192, 64 vs 96. DECISION: WIDE_ROWS=128 for the 16-user profile.
  cert check, wide passes 16 in flight (4e5c, running): 349,004 rows, 348,406 compared, 0 mismatches.
- 21:14 window L: cert CHECK totals so far (every head call compared with the whole head, best value + id and every recomputed entry bit for bit), all 0 mismatches:
  K 923b 5 in flight 107,289 rows; L 4e5c 16 in flight wide-128 passes 408,747 rows (11 whole-head fallbacks of 6000 calls); L 3d24 5 in flight (kp11) 215,076 rows
  (6 fallbacks of 10,000 calls, 226 tiles and 94 exact launches a call at 21.5 rows a call). Sum 731,112; two more checked runs queued (kit/extra_chk.sh on 3d24 and 4e5c).
  16 slots x 80K window (4e5c, wide 128): 16 users 231.9 / 228.7 distinct, 265.0 copies: fits and runs like 64K.
- 21:29 window L, 923b documented path, 320 held prompts (315 answered), PAIRED: levers off 57.39 (chat 50.01 / code 74.65 / math 72.70) -> stack (CERT=1) 59.50 (51.91 / 77.05 / 75.36)
  = +3.7 % all, +3.8 % chat, +3.2 % code, +3.7 % math.
  many-user extras (4e5c, 16 slots 64K, cert on; 16 users distinct, two runs): wide 160 + 8 drafts 230.4 / 232.6 (110.9 rows, 248.9 ms rounds); wide 160 + 8 drafts + gate 0.5: 234.2 / 235.8;
  wide 128 + gate 0.3: 233.1. None beats wide 128 with MiaAI's gate (234-236): deeper or looser drafting adds rows and time in step. DEAD.
- 21:45 window L, 923b: STACK FULL SUITE DONE (documented path, vision on; table Mac results/final_r4_923b.txt; raw results/rL/):
  chat 48.9 / code 82.0; sparkDash-method prose 59.1 / code 95.9 / structured 110.1; sparkDash 1.8.8 real prose 58.8 / code 95.6, catalogs 45.2 / 68.7 / 84.2; held-320 59.50;
  4 users 150.0 distinct / 154.9 copies; 5 users 162.4 / 167.3; 16 queued on 5 slots 158.9; prefill 2388 / 2464 / 2305; needle 200K 3/3; 227,621-token prompt 3/3; image PASS; qeval 33/35; ttft 180 ms.
  IDENTITY stack vs levers off (same image, same window): ident 51/51, held 320/320 texts identical. During the suite the certified head served 40,000 calls / 217,101 rows,
  3 whole-head fallbacks, 52.9 tiles and 30.6 exact launches a call.
  cert CHECK total now 1,079,719 rows compared (3d24 spare run: 348,607 rows, 16 in flight, wide passes), 0 mismatches; one more run on 4e5c in progress.
  16-user profile on 923b (USERS16=1 WIDE_ROWS=128 + cert, documented path): 1 / 4 / 8 / 16 users distinct 78.7 / 140.5 / 199.1 / 229.9 + 228.3; copies 8 220.2, 16 269.9.
- 21:55 MiaAI default (5 slots at 262K, CERT=0), 923b, same window, documented path: distinct 1 / 4 / 5 / 8 / 16 users = 76.5 / 147.7 / 163.2 / 151.2 / 155.4
  (8 and 16 wait in the queue: last first token after 15.7 s / 38.4 s); copies 4 / 5 / 8 / 16 = 148.6 / 166.5 / 156.5 / 159.8.
  MTP chain step by depth (3d24, hostprof, 79,591 draft-head rows): 1.94 ms first, 1.73, 1.72, 1.70, 1.70, 1.69 ms; one-user held-54 with cert under the profiler 61.01 tok/s.
- 22:05 window L CLOSED by itself. DeepSeek verified by the harness: health 200, 17*23 = 391, session store 12.00 GiB on every rank, disk tier 28.2 GiB, image ok.
  Outage 7044 s (20:07:06 -> 22:04:30). No q38 containers or helper scripts on any worker; workers' memory 116 GiB free before DeepSeek restarted.
- window L last results (923b, documented path, one window): MiaAI config with 16 slots x 64K and 64-row passes (CERT=0): 1 / 4 / 8 / 16 = 74.1 / 145.9 / 194.9 / 219.0 + 221.4, copies 8 213.4, 16 243.7.
  pico USERS16 (128-row passes + cert): 78.7 / 140.5 / 199.1 / 229.9 + 228.3, copies 8 220.2, 16 269.9 => 16 users +4 % on 923b (+7.6 % on 3d24, 232-236 on 4e5c).
  extras (4e5c): wide 192 + 10 drafts 223.2 (worse). cert CHECK final total: 107,289 + 408,747 + 215,076 + 348,607 + 270,893 = 1,350,612 rows compared with the whole head, 0 best mismatches,
  0 value mismatches (one-user windows, 5 and 16 in flight, 64- and 128-row passes; counts are the last 1000-call prints of each run).
- another vLLM recipe recipe (optional item): NOT run: its checkpoint (RadixArk NVFP4, 135 GB) was deleted from the boxes on Pico's order and spare capacity went to the checked runs.

## Round 4 SUMMARY (2026-11-12 18:16 -> 22:10 UTC)
| lever (brief order) | result | status |
| 1 launch cost 4.8 us with drafts | not a price per launch: ~8,500 launch calls block ~0.5 ms while the GPU drains its queue; 2000 / 4000 ns spin per launch costs 0.22 / 0.39 ms a round; wait policy already spin; side stream off -1.5 % | DEAD (bound ~0.9 %), nothing to fix |
| 2 newer CUDA userland | NVIDIA cuda-compat 13.4 (615.71.10) / 13.5 (620.70.06) vs the image's 610.43.02: 58.71 / 58.64 vs 58.95 + 58.58; launch 2.121 vs 2.125 us | DEAD |
| 3 graph capture / glue fusion | bounded by lever 1's probe: <= ~0.9 % (host side), <= ~1.5 % (GPU-side tiny kernels) | not built |
| 4 certified exact LM head | one user: held-320 57.39 -> 59.50 (+3.7 %), chat 47.4 -> 48.9, code 79.0 -> 82.0, sparkDash-method prose 56.9 -> 59.1, code 92.2 -> 95.9; ident 51/51, held 320/320, 1,350,612 checked rows 0 mismatches | SHIPPED (recipe default) |
| 5 draft side | 4-bit shortlisted draft head is already upstream; verify rows already share experts; suffix drafts not simulated (no token ids in logs) | nothing built |
| 6 expert kernel | no concrete cause in the profiles | skipped by rule |
| 7 many users | priced allocation -3..-5 %; 128-row drafted passes +4..+8 % at 16 users (texts identical 107/107 with 16 in flight); 16 x 80K and 12 x 128K start, 16 x 96K does not | SHIPPED (USERS16 profile) |
| 8 stack + suite + recipe | full suite on 923b through recipe.sh (prepare run end to end), repo updated | done |
Honest bottom line: the lossless stack beats MiaAI-Lab's shipped configuration by 3-4 % for one user and 4-8 % at 16 users (same box, same window, same image). It does NOT reach custom engine R's
re-quantised numbers (54.6 / 88.6 chat / code, held 65.66): 48.9 / 82.0, held 59.50.

## Round 4 END STATE (22:10 UTC 2026-11-12)
- No window open (q38-win, q38-guard inactive), no Qwen container on any worker, page cache evicted by the harness, DeepSeek verified after window L (health 200, 391, session store 12.00 GiB, disk tier 28.2 GiB, image ok).
- Outages this round: J 2448 s, K 3096 s, L 7044 s (J ended early by SIGTERM; K and L ended by themselves).
- Builds on all workers ~/q38/tfbin: kp6..kp11 (kp11 sha 6537c027ae242a6c = the shipped patch set 0101-0105). Image on 923b: tensorfold-qwen38:v1.1.4-eab8582 rebuilt by recipe.sh prepare
  (MiaAI's own image is still there as ghcr.io/miaai-lab/...:i-was-a-digest). Build trees: 923b / 3d24 / 4e5c ~/q38/build. cuda-compat debs + extracts: ~/q38/compat on each worker (190 MB, can go).
  Head-row dumps on 923b ~/q38/prof/hd16.x (1.1 GB), hd1.x: evicted from cache; delete when not needed.
- Source: Mac work/tf (git: b0a7eb1 kp5 -> d47ea4a kp10 -> kp11 HEAD); patch 0105 = git diff b0a7eb1 HEAD.
- RESUME COMMANDS
  cd ~/.hermes/cache/scratch/q38; tail -120 PROGRESS.md
  ./h.sh 'systemctl --user is-active q38-win q38-guard.timer; tail -3 ~/q38/windows.log; curl -s -o /dev/null -w "%{http_code}\n" 127.0.0.1:18900/health'
  build:   ./ship.sh NAME pico@spark-4e5c      (tars every changed file, verifies sha256 on the box, starts tfbuild; log ~/q38/build/NAME.log)
  spread:  ./dist.sh NAME pico@spark-4e5c SYMBOL   (copies the build + kit + harness to all workers, prints sha and a strings count)
  window:  ./h.sh 'systemctl --user reset-failed q38-win; systemd-run --user --unit=q38-win /bin/bash ~/q38/qw.sh LABEL 95 ~/q38/w_LABEL.sh'
  end early: ./h.sh 'pkill -TERM -f "^/bin/bash /home/pico/q38/qw.sh"'   (never systemctl stop; kill any kit/extra_*.sh on the workers first)
  documented path on 923b: bash ~/q38/kit/rs.sh [CERT=0] [USERS16=1 WIDE_ROWS=128]   (recipe.sh start on the prepared image; API 127.0.0.1:18301)
  test launcher: TFBIN=$HOME/q38/tfbin/kp11 TFMODEL=$HOME/q38/models/nvidia-fp4-local TFENTRY=/opt/nvidia/nvidia_entrypoint.sh TFENV='-e TF_FLASHNEXT_Q38_CERT=1' bash ~/q38/kit/eng.sh tfb default|p16|pc_16_81920
- NEXT ROUND FIRST: (1) batch the prompt passes of users who arrive together (16-user bench +10 % est., first token 6.7 s -> < 1 s); (2) certified head on plain one-row rounds and the first token;
  (3) a 6-bit coarse copy (+1 % est.) only with a kernel that keeps ~230 GB/s; (4) ask Pico about a labelled re-quantised checkpoint: the official one is at its byte floor within ~20 %.
- 22:11 DELIVERED: private repo Picom-code/pico-dgx-qwen38 branch tensorfold-fp4-local commit 2a0ebc8 (author Pico noreply; gitleaks: no leaks; visibility PRIVATE): README (final table,
  paired proof, many-user table, how the certified head works), recipe.sh (CERT default on, USERS16 = 16 slots + 128-row passes, prepare without forced rebuild), patch 0105, kit, harness w_J/K/L, notes + raw logs.
  Skill reference dsgx10-spark-deepseek-engine/references/qwen38-fastest-build.md rewritten for rounds 1-4. Tables: Mac results/final_r4_923b.txt, results/final_r4_manyuser.txt.
  Final state re-checked 22:10 UTC: no window / guard / proxy unit, no q38 containers or helper scripts on any worker, DeepSeek health 200, 17*23 = 391, session store 12.00 GiB, disk tier 28.2 GiB, image ok.
  Caveat to carry: sampled, grammar and logprobs requests take the whole head by the gate in cuda_lanes_verify.zig (not separately identity-tested with a seed); the suite's sampled benches and qeval ran clean.

## Round 5 (fifth agent), started 2026-11-12 22:13 UTC
- 22:13: state verified clean (no window/guard, health 200, last window L closed 22:04). Reading the lane scheduler for lever 1 (batched prompt passes).
- 22:50 FOUND by code reading (TensorFold lane core, zig/src/core): (a) the host already batches admissions (lane_host_admit.many -> Engine.addStreams -> backend vtable
  prefill_many / many_rows); GLM implements it, Flash Next does NOT (its vtable has prefill + prefill_step only) => lever 1 = implement prefill_many for Flash Next.
  (b) prompt-lookup drafts ALREADY EXIST upstream: every drafting stream gets a SuffixLookup proposer (core/lanes/proposer.zig, "Python SuffixLookupProposer",
  core/lane_stream.zig:35), copy plans up to exact_width-1 = 15 tokens when >= 8 (copy_match) history tokens match, copy_rate 0.94 (core/lanes/config.zig, windows.zig copyProposal).
  So lever 3 is not "add suffix drafts" but "measure what the existing ones do on edit / agent sets and whether the 16-row window cap binds".
  (c) a stream's caches are allocated at ITS OWN size (cuda_lanes.open: cap = prompt + max_new + 16), only the start-up check charges every lane the full window
  => lever 2 (mixed slot sizes) = admission by bytes against a pool, no fixed slot classes needed.
- 22:55 build kp12 (4e5c ~/q38/build/out/kp12, sha 77937339f7b2d427, BUILD_OK first try) = kp11 + lever 1: TF_FLASHNEXT_Q38_PREFILL_MANY=N (<= 256): waiting fresh text prompts of <= N tokens
  go in ONE prompt pass. New: families/flashnext/cuda_lanes_many.zig, Engine.prompts (cuda_prompt_windows.zig), Ops.prefillChunkMany (cuda_mtp_ops.zig), Pass.lane_last
  (cuda_batch.zig, cuda_batch_head.zig: each lane's last row scored, lane i at row i), vtable many_table in cuda_lanes.zig; aid TF_FLASHNEXT_Q38_LHASH=1 (hash of a stream's
  scores after its prompt pass and at release: "q38 lhash first|end pid= plen= fed= h="). Design for exactness: the pass runs with prompt=true so each row takes the lone
  chunk's kernels (incl. the relaxed MX128 prompt tile), per-lane GDN/attention/PLE on the stream's own caches, shared-expert path as the lone chunk (not direct),
  head rows read out one at a time. Only drafting text streams without a kept prefix or cache marks; anything else falls back to one-by-one.
- 22:30 build kp13 (4e5c, sha daac44f2d9608853, on all workers) = kp12 + lever 2 TF_FLASHNEXT_Q38_POOL_CTX=N (pool admission by bytes: Cuda.need/q38Pool/q38Fits in cuda_lanes.zig, api.Admit.always, start-up charge in cuda_lanes_load.zig; a request that does not fit returns error.LaneMemory = the lane host's own requeue) + TF_FLASHNEXT_Q38_CERT_PLAIN=1 (certified head on one-row rounds of --no-drafts streams). Kit: waves.py, lhcmp.py, mkedit.py, poolt.py.
- 22:30 window M LAUNCHED (guard 80 min; head ~/q38/w_M.sh, Mac harness/w_M.sh): 4e5c lever 1 off/on (benches, waves of 16, texts + score hashes), 3d24 lever 2 fixed vs pool + mixed-size scenarios, 923b copy-draft measurement on edit / agent / chat sets + no-drafts certified plain rounds. Logs head ~/q38/runs/M/m_*.log.
- 22:40 LEVER 5a MEASURED OFFLINE (kit/bf16stat.py on the checkpoint files, no GPU; 4 MiB samples of 4 tensors a group): is the BF16 data (hyper-connection 1.18 GiB,
  LM head 1.18, router 0.12, MTP dense 0.17) exactly re-encodable in fewer bytes?
  | group | distinct 16-bit values | zero-order entropy | exponent entropy | mantissa entropy (of 7) | always-zero low mantissa bits | exponent spread in 32-value blocks p50 / p99 / max |
  | hyper-connection mix_down | 4300-4500 | 10.7-10.8 bits | 2.7-2.9 | 6.98 | 0 | 7 / 13 / 20-23 |
  | lm_head | 3993 | 10.51 | 2.58 | 6.97 | 0 | 7 / 13 / 22 |
  | router gate | 3800-4200 | 10.6 | 2.6-2.7 | 6.98 | 0 | 7 / 13 / 20-25 |
  | MTP dense | 4000-4350 | 10.6-10.75 | 2.7-2.8 | 6.97 | 0 | 7 / 13 / 20-23 |
  Reading: the BF16 weights are real 16-bit floats (full 7-bit mantissa, no padding bits). An EXACT code needs sign + 7 mantissa bits + the exponent: 10.6 bits with an entropy
  coder (-34 % bytes, but variable-length decode in the GEMV: measured 10-48 % SLOWER on DeepSeek, stack5), 13 bits fixed (-19 %), or 12 bits fixed with a 4-bit block-relative
  exponent and an exact escape list for ~0.5 % of blocks (-25 %). Best case for the hyper-connection: 1.195 -> 0.90 GiB = -1.3 ms of a 56 ms round = +2.3 % (ESTIMATE, upper bound,
  only if a kernel that unpacks 12-bit fields and rebuilds exponents still reads ~215 GB/s; on DeepSeek the fixed-length coded FP8 GEMV was "about par", i.e. the saving was eaten).
  A coarse-then-exact scheme does not apply to the hyper-connection: its output feeds every layer, there is no argmax to certify. DECISION: not built (needs a new MMA tile
  kernel with exponent rebuild; expected gain 0..2 %, exactness risk in the escape path). Recorded as the largest remaining "bytes" idea for the dense side.
- 22:44 window M, 923b, LEVER 3 MEASURED (the copy drafts TensorFold already ships; kp13, cert on, default 5 x 262K, one stream; sets from kit/mkedit.py: 24 rewrite-a-file tasks and
  16 agent-style turns built from Python stdlib sources, chat = every 8th held prompt; kit/tfsteps.py on the engine's done lines):
  | set | tok/s (decode) | tokens / round | drafts verified / round | accept rate | ms / round | round fit |
  | edit (rewrite a file, <= 1500 tokens) | 157.7 | 12.59 | 12.53 (window cap 15) | 92.4 % | 79.8 | 51.1 + 2.29 d |
  | agent-style (tool calls with file text / patches) | 74.0 | 4.64 | 5.53 | 66.0 % | 62.7 | 45.5 + 3.10 d |
  | chat (held every 8th) | 58.7 | 3.27 | 3.74 | 60.8 % | 55.7 | 40.0 + 4.19 d |
  Reading: prompt-lookup is ALREADY worth 2.7x on file rewrites in the stock engine (12.6 tokens a round), and the 16-row window cap binds there (12.5 of 15 possible drafts a round,
  per-token landing ~0.987). SIMULATED (chain with p = 0.987, cost 51.1 + 2.29 d): 31 drafts -> 26.3 tok / 122 ms = 215 tok/s (+36 %); 63 drafts -> 43.6 / 195 = 223 (+41 %).
  Chat: ~0 (no copies). => BUILD: 32-row windows (knob TF_FLASHNEXT_Q38_WINDOW_ROWS=32; mtp.max_rows buffers 32, lane core told the knob; default 16 = stock).
- 22:44 window M, 3d24, LEVER 2 first results (pool: 16 lanes, 23,382 MiB = 16 x 1,461 MiB (64K windows); a full 262,144-token stream takes 4,815 MiB; kp13, wide 128 + cert):
  benches with pool: 1 / 4 / 8 / 16 users distinct 77.7 / 149.7 / 197.2 / 230.6 (same as fixed 16 x 64K slots).
  mixed sizes, all sent together, 0 errors: one 136,607-token prompt + 15 short: needle 3/3, long first token 77.1 s, the 15 short streams ran meanwhile (48 tok/s total, last first token 8.1 s);
  three ~91K prompts + 13 short: 3/3 x 3, 0 errors; one 182,082-token prompt + 8 short: 3/3. Lowest MemAvailable 21-27 GiB.
- 22:44 window M, 4e5c, lever 1 baseline (many off): 16 users distinct 233.4 / 227.5, copies 266.6; waves of 16 held prompts (256 tokens, 12 waves): 126.9 tok/s total, last first token
  of a wave 7.09 s mean / 7.48 max; waves of 4: 89.6 tok/s, 1.27 s.
- 23:00 window M results (measured, kp13):
  LEVER 1 DID NOT ENGAGE in window M: 16 users 228.0, last first token 6.78 s, no "q38 many" lines. Cause (code reading): requests that arrive together are admitted ONE BY ONE into the
  host's fill list (admission does no work, so the queue never holds two), and the host's batch path (lane_host_admit.many) only fires when two wait in the queue. Fix in kp14:
  core/lane_host_fill.many: fillOne batches the fill-list jobs whose prompt pass has not begun. (The two 4e5c runs of window M are therefore two "off" runs: waves of 16 126.9 / 125.7 tok/s,
  last first token 7.09 / 7.23 s; texts 192/192 + 64/64 identical, first-score hashes 218/218 identical. The "end" hash differs between identical runs, 182 of 218: the last
  verify window's last row depends on round timing; only the "first" hash is a valid invariant. lhcmp's "end" line is to be ignored.)
  LEVER 2 (pool admission, TF_FLASHNEXT_Q38_POOL_CTX=65536, 16 lanes, --context 262144; 3d24): benches 1 / 4 / 8 / 16 distinct 77.7 / 149.7 / 197.2 / 230.6 + 236.5, copies 16 267.6
  (fixed 16 x 64K on 4e5c in the same window: 79.1 / 151.1 / 195.2 / 233.4 + 227.5, copies 266.6: no cost). Mixed sizes sent together, 0 errors in every scenario, needle 3/3 for every long prompt:
  | scenario | result |
  | 1 x 136,607-token prompt + 15 short | long first token 77 s; shorts ran meanwhile, last first token 8.1 s; lowest MemAvailable 24 GiB |
  | 3 x ~91K + 13 short | all served; 21 GiB |
  | 1 x 182,082 + 8 short | all served; 27 GiB |
  | 8 x ~54.6K + 8 short | all served; 18 GiB |
  | 5 x ~182K at once (more than the pool holds: 4 fit, 1 waits) | all five served, 3/3 each, 418 s wall; 13 GiB |
  | largest request | 227,621-token prompt answered 3/3 (prefill 2124 tok/s) in the 16-lane configuration |
  So 16 lanes now keep the full 262,144-token window: memory is a pool of 23.4 GiB (16 x 1.46 GiB); a full-window stream takes 4.8 GiB, so e.g. 4 full windows + 2 x 64K, or 1 full + 12 x 64K.
  LEVER 4a (certified head on plain one-row rounds, --no-drafts, TF_FLASHNEXT_Q38_CERT_PLAIN=1; 923b, 27 held prompts): 23.5 -> 24.8 tok/s (+5.5 %), 42.73 -> 40.48 ms a step;
  texts 27/27 identical; CERT=2 check with 5 in flight: 14,828 rows compared with the whole head, 0 mismatches. Matters only for --no-drafts (the default drafted config already used it).
- 23:00 build kp14 (4e5c, sha 3555aeba61ca13b1, on all workers) = kp13 + fill-list batch + TF_FLASHNEXT_Q38_WINDOW_ROWS=N (16..32; buffers sized 32, lane core told N) + TF_FLASHNEXT_Q38_GPROF=1
  (GPU ms by group from CUDA events) + refusal prints in cuda_lanes_many.zig.
- 23:06 window M CLOSED (ended early by SIGTERM after the 923b lane finished; 3d24's last two runs were cut: not needed). DeepSeek verified by the harness: health 200, 17*23 = 391,
  session store 12.00 GiB on every rank, disk tier 28.2 GiB, image ok. Outage 2113 s (22:30:36 -> 23:05:49).
- 23:07 window N LAUNCHED (guard 70 min; head ~/q38/w_N.sh; kp14): 4e5c lever 1 (many on vs off: benches, waves 16/4/8, texts + first-score hashes); 3d24 lever 3 (window rows 16 / 32 / 24 on
  edit / agent / chat sets, texts compared; 16-user bench with 32); 923b GPU time by group (drafts on / off / 16 users) + candidate stack (pool + many + rows 32) benches and mixed sizes.
  Logs head ~/q38/runs/N/n_*.log.
- 23:20 window N: lever 1 STILL not engaged with kp14 (16 users 230.6 / 231.1, last first token 6.9 s). Cause found in the engine log: every prompt "resumed at 2" (or 6): MiaAI's config
  keeps prompt states, the chat template's opening tokens are a kept prefix, and both the host rule (batchable: nothing kept to resume from) and my backend check refuse such streams.
  Fix in kp15 (4e5c, sha 59f9ca26ff1549eb): a kept prefix of <= 16 tokens is not restored when prompts are batched (the prompt runs from token 0; the kept state stays in the cache,
  the host reports it untouched; the short prompt's own cache marks are skipped: a cache-population change, not an output change). The 4e5c lane of window N was stopped to build.
- 23:20 LEVER 5 MEASURED: GPU time by group from CUDA events (TF_FLASHNEXT_Q38_GPROF=1; kp14, 923b, one user; a wait for the host counts to the group after it):
  | group | bytes read | 1-row step, no drafts, whole head | GB/s | drafted round, 4.64 rows, certified head |
  | GDN linear attention (36 layers) | 2.005 GiB MXFP8 | 10.78 ms | 200 | 11.47 ms |
  | QSA attention (12 layers) | 0.593 GiB MXFP8 | 3.18 | 200 | 3.29 |
  | hyper-connection, both sides (97 read-outs + write-backs) | 1.195 GiB BF16 | 4.24 + 4.81 = 9.05 | 142 | 4.22 + 4.89 = 9.11 |
  | MoE: router + shared + routed experts | 1.49 GiB (1 row) | 9.56 | 168 | 20.26 |
  | PLE (n-gram) | 0.06 GiB | 4.55 (4 ms of it = waiting for the host's n-gram reads) | - | 0.54 (rows are read ahead) |
  | LM head | 1.184 GiB BF16 (0.61 + tiles certified) | 5.39 | 236 | 3.54 |
  | embed | - | 0.07 | - | 0.04 |
  | total main pass | 6.47 GiB | 42.57 ms (163 GB/s overall) | | 48.24 ms (+ ~6.4 ms of MTP chain steps = the 56 ms round) |
  16 users (54-99 rows a pass, 111-144 ms): MoE 71-95 ms = 64-66 % of the pass; dense + hc ~36-45 ms; head 1.2-1.4 ms.
  Where the 1-row step is below ~225 GB/s (the box's practical ceiling), in ms: n-gram wait 4.0 (no-drafts only; drafts-on reads ahead), hyper-connection 3.3 (9.05 vs 5.7),
  MoE 2.5 (9.56 vs 7.1), GDN 1.2, QSA 0.35. Drafted round: hyper-connection 3.4, GDN 1.9, QSA 0.5, MoE ~3.5 (20.3 vs ~16.7 for ~3.5 GiB): ~9 ms of 56 (16 %).
  Hyper-connection detail: each of the 97 read-outs is 5 launches (normalize, down GEMV 6.5 MB, reduce+act, up GEMV 6.5 MB, mix) + a write-back; the two GEMVs alone are 61 us (215 GB/s,
  microbench), the measured read-out is ~93 us: ~32 us of small kernels a read-out = 3.1 ms a step. Upstream's fused read-out (TF_FLASHNEXT_PROMPT_HC2, same bytes by their statement)
  is a prompt tile: its grid is 2 CTAs for <= 48 rows, so it cannot serve decode rows. A decode-geometry fused read-out (norm in the down kernel's prologue, mix in the up kernel's
  epilogue, same op order) is the largest concrete exact lever left: ESTIMATE ~2 ms of a 56 ms round = +3.5 %. NOT built this round (two new MMA kernels + a bit-compare microbench;
  no time left to prove it exact). MTP chain step 1.70 ms reads ~0.24 GiB (draft head 0.10, q/o proj 0.05, hc 0.026, experts 0.024, fc 0.026, ...) = 1.07 ms at 225 GB/s: ~0.6 ms of small
  kernels; chain steps are sequential (each needs the previous token), so they cannot share reads. A verify row's 2.4 ms = its own experts (~0.5 GiB new a row at 168 GB/s) + ~0.1 ms dense.
- 23:30 window N, 3d24, LEVER 3 MEASURED (kp14, one user, cert on; every reply text identical to the 16-row run: edit 24/24, agent 16/16, chat 40/40):
  | set | 16-row windows (stock) | 32-row windows, every copy up to 31 tokens | change |
  | edit (rewrite a file) | 157.2 tok/s, 12.59 tok/round, accept 92.4 % | 197.6 tok/s, 19.33 tok/round, accept 86.6 %, 97.8 ms/round | +25.7 % |
  | agent-style | 73.5 tok/s, 5.53 drafts/round, accept 66.0 % | 71.3 tok/s, 6.69 drafts/round, accept 55.1 % | -3.0 % |
  | chat | 58.3 tok/s | 57.9 tok/s (3.91 drafts/round vs 3.74) | -0.7 % |
  Reading: wide copies win big where text is repeated and lose a little where a copy breaks early (each wasted row ~2.4 ms). FIX (kp17): the copy window starts at the stock 15 tokens
  (facts.first_copy_rows = 16) and the lane core's own rule doubles it to 31 only after a copy that landed whole, back to 15 when one breaks; shared rounds keep 15. Re-measure in window O.
  First candidate-stack run on 923b (kp14: pool + rows 32 without that rule): 16 users 209.3 / 224.3 (lower than 230: 31-token copies crowd the 128-row pass) => the rule above also fixes this.
- 23:30 kp15 lever 1 first engagement CRASHED (segfault in prefillChunkMany): pointers into the lane hash table were taken while later streams were still being opened (the table moves
  when it grows). Fixed in kp16 (all streams opened first, pointers taken after). kp17 = kp16 + the copy-width rule. Lesson: never hold a pointer into a hash map across an insert.

## Round 6 (sixth agent), started 2026-11-13 06:55 UTC
- 06:57 state verified clean (no window / guard, DeepSeek health 200, no Qwen container, kp17 sha 73aeabcb6fa3b3b8 on all three workers, kit hashes equal Mac = workers,
  head w_O.sh hash equal to the Mac copy). recipe.sh + patch 0106 HAD arrived on 923b ~/q38/repo_tf (hashes equal), but `recipe.sh prepare` had NOT been re-run: 923b's image was
  still the round-4 one (0101-0105). DECISION: w_O.sh's 923b lane now runs `recipe.sh prepare` first (log ~/q38/recipe_work/prepare3.log), inside the window.
- 06:58 window O LAUNCHED (guard 150 min; unit q38-win; logs head ~/q38/runs/O/o_*.log). Ends by itself when the three lanes finish.
- 07:10 PICO UPDATE (mid-round): "keep on going do not stop looking for performance gains, use the sparks to test different things". Rules added: three workers run DIFFERENT
  experiments in every window; keep a lever queue here; commit + push the private branch after each lever that ships; head stays excluded.
  LEVER QUEUE (each exact, behind a flag, microbench or simulate first):
  Q1 fused decode-geometry hyper-connection read-out (est +3.5 %)      Q2 MoE 168 -> 225 GB/s (gather order, expert sharing inside a round, NVFP4 decode)
  Q3 GDN 200 -> 225 GB/s                                               Q4 MTP chain-step small kernels (~0.6 ms a step)
  Q5 shorter draft shortlist                                           Q6 draft depth per workload
  Q7 6-bit coarse head                                                 Q8 exact fixed-width BF16 re-encoding (only if a microbench keeps GB/s)
  Q9 prefill speed 2.3-2.5K tok/s vs a third engine 2.8K: find why; TTFT   Q10 anything exact in custom engine R's / a third engine's source that TensorFold lacks
- 07:05 window O, 923b: `recipe.sh prepare` re-run END TO END with patches 0101-0106 (prepare rc 0, "Done", image tensorfold-qwen38:v1.1.4-eab8582 rebuilt, ~5 min).
- 07:08 build kp18 (3d24 ~/q38/build/out/kp18, sha 8267bbfc0a55e941, BUILD_OK first try, built with 3 CPUs beside the running lane) = kp17 + Q1 FUSED DECODE READ-OUT:
  TF_FLASHNEXT_Q38_HCF=1 (up + mix in one launch: upstream's hcn::upmix_body at a 16-row tile, new instance fn_hcn_up1_k5) or =3 (also fn_q38_hcdn: the BF16 lane
  kernel's down block with fn_hc_normalize2's expression forming its input stage in shared memory and fn_hc_reduce_act's arithmetic in its epilogue): 5 launches -> 2.
  Same mma order, same cluster slice order, same rounding points. TF_FLASHNEXT_Q38_HCF_MAX (default 16 rows), _ST=3|4|6, and _CHECK=1 = run BOTH chains at every read-out
  and compare all five outputs word for word on the device ("q38 hcf check" lines). Files: kernels/cuda/flashnext_hcf.cu (end), cuda_tiles.zig readoutQ38 / q38Compare,
  cuda_tile_kernels.zig, cuda_knobs.zig, cuda_batch.zig readoutKeep. Mac work/tf commit "kp18".
  Test script head ~/q38/x_hcf.sh WIN BOX BUILD (spare lane inside an open window): ref / check / HCF=3 / HCF=1 / ref / stages 3 / 6 / 16-in-flight check.
- 07:18 window O, 4e5c, LEVER 1 (batched prompt pass, kp17, 16 x 64K, wide 128 + cert; MEASURED, speed part):
  | 16 users | PREFILL_MANY off | PREFILL_MANY=256 |
  | distinct, two runs | 225.9 / 233.8 tok/s, last first token 6.9 / 6.8 s | 257.7 / 265.5 tok/s (+14 %), last first token 1.6 / 1.1 s |
  | copies | 271.3 (6.4 s) | 333.4 (1.1 s) |
  | 8 / 4 users distinct (off numbers from 3d24 all-off: 191.2 / 147.3, ttft 2.9 / 1.2 s) | | 212.8 (0.75 s) / 146.9 (0.52 s) |
  off waves: 16: 126.8 tok/s, last first token 7.16 s mean; 8: 104.3, 3.21 s; 4: 88.1, 1.37 s. Identity (waves texts + first-score hashes) pending in the same lane.
  3d24 whole many-user stack (pool + many + rows 32 adaptive + wide 128 + cert) vs all off, 1 / 4 / 8 users: 79.5 / 151.2 / 209.8 vs 75.4 / 147.3 / 191.2 (ttft 0.21 / 0.58 / 0.79 vs 0.21 / 1.18 / 2.93 s).
- 07:22 window O, 3d24: WHOLE MANY-USER STACK (pool 65536 + PREFILL_MANY=256 + WINDOW_ROWS=32 adaptive + wide 128 + cert + cert-plain) vs ALL OFF, kp17, MEASURED:
  identity with 16 in flight: held 107/107 texts identical; edit set 4 in flight 24/24 identical. 16 users 264.2 tok/s while all run (all-off 16 distinct 215.6, copies 251.7).
  held 16 in flight 175.3 vs 166.6 tok/s (+5.2 %). "q38 many" engaged (e.g. 3 prompts 128 rows in one pass 303 ms). Mixed sizes with the stack, 0 errors:
  1 x 136,607-token prompt + 15 short: recall 3/3, long first token 72 s, shorts' last first token 1.34 s (was 8.1 s without batching); 3 x ~91K + 13 short: 3/3 x 3, shorts' last first token 1.03 s.
  DECISION: the last two 3d24 runs of w_O (no-drafts certified plain rounds: already measured in window M, +5.5 %, 27/27) were cut (lane subshell killed) to free 3d24
  for the fused read-out test: head ~/q38/x_hcf.sh O w2 kp18 started 07:23 (log head ~/q38/runs/O/x_hcf_w2.log).
- 07:25 RESEARCH INPUT (Mac q38/research/ideas.md, tf_changelog_since_1.1.4.md, src/tf-main = TensorFold main 223da76, custom engine R, a third engine). Queue additions, cheapest first:
  A  free env tests: 16 users with TF_FLASHNEXT_PROMPT_EXPERT_TILE_ROWS=64 / 96 / 128 (texts under load); prefill with TF_FLASHNEXT_CHUNK=4096 / 8192 at one user (identity, memory)
  B  platform facts (read only): desktop session / second CUDA context on the workers, CPU governor / idle states
  C  cherry-pick upstream exact items one at a time (no rebase): #1351 equal-wave NVFP4 experts, #1702 tiled MTP head norm, #1925 (check if absent), #1171 half tiles (check),
     #1739 attention lanes; prefill: CHUNK_SOLO, BF16 128x128 tile, KEEP_INPASS, PREFIX_WAIT. NOT READOUT_ROWS=128 / the 64-row switches (non-native path).
  D  microbench dense + hc kernel form at 54-128 rows; run-length GB/s microbench (decides Q2 / Q3)
  E  drafts: 3-token lookup confirmed by the MTP head's first draft (custom engine R), cross-request history ring (a third engine): simulate first. Parallel public drafter: last.
  Checked by the research agent and not worth building: certified router, activation sparsity, huge pages, draft trees.
- 07:26 QUEUE B, PLATFORM FACTS (read only, all three workers alike; nothing changed): default target graphical; Xorg (18 MiB) and gnome-shell (6 MiB) hold GPU contexts
  beside the engine (an idle desktop session = a second GPU client; stopping it would be a host change: NOT done, reported for Pico); CPU governor "performance";
  cpuidle states LPI-0..3 all enabled (exit latency 0 / 42 / 231 / 433 us); 20 cores. Round 4 measured host-side sensitivity ~0.07, so idle states are unlikely to matter.
- 07:26 window O, 4e5c, LEVER 1 IDENTITY + WAVES (kp17, PREFILL_MANY=256 vs off, 16 x 64K, wide 128 + cert), MEASURED:
  | waves of held prompts | off: tok/s total, last first token of a wave (mean) | on | texts on vs off |
  | 16 at once, 12 waves | 126.8, 7.16 s | 147.3 (+16 %), 1.80 s | 192/192 identical |
  | 8 at once | 104.3, 3.21 s | 125.4 (+20 %), 1.12 s | 160/160 identical |
  | 4 at once | 88.1, 1.37 s | 96.3 (+9 %), 0.67 s | 64/64 identical |
  first-score hashes (TF_FLASHNEXT_Q38_LHASH): 270 prompts in both runs, 270 identical, 0 differ. 58 batched passes ("q38 many: 15 prompts, 625 rows in one pass, 763 ms").
  => LEVER 1 SHIPS (exact, +14 % on the 16-user bench, first token 6.8 s -> 1.1-1.6 s). Remaining check: prompt-cache behaviour of single requests (suite ttft on 923b).
- 07:47 Q1 FUSED READ-OUT, EXACTNESS (3d24, kp18, TF_FLASHNEXT_Q38_HCF=3 + _CHECK=1, default one-user config with drafts, real requests = ident set + held prompts; every read-out of
  the main pass AND of the MTP chain steps runs both chains): after 160,000 read-outs / 674,931 rows: 2,067,939,468 16-bit words compared (mixed rows, their sums, activation,
  its sums, inject gates), 0 differ. First run of the kernel, no fix needed. Texts / speed pairs follow in the same lane.
  NOTE: run XH_ref of that lane is INVALID as a reference (the cut lane's no-drafts kp17 engine was still starting and answered it: "drafts: off", 23 tok/s). XH_ref2 is the valid one.
  kp19 (Mac work/tf commit, NOT built yet) = kp18 + upstream #1171 port: TF_FLASHNEXT_Q38_HALF_BELOW=N (default 0; upstream 96): MXFP8 / BF16 decode lane calls whose 64-column grid has
  fewer than N blocks take 32-column half tiles (twice the blocks; same sums a column). Candidate cause of "GDN 200 vs 225 GB/s" (40 blocks on 48 SMs). Test script head ~/q38/x_b.sh.
  Checked: upstream #1351 / #1702 / #1925 / #1171 / #1739 markers are all ABSENT from our tree (so they are real cherry-pick candidates); #1351 is a 365-line kernel diff for +0.6 % upstream: last.
- 07:54 Q1 FUSED READ-OUT, EXACTNESS FINAL for the one-user run (3d24, kp18, HCF=3 + CHECK): 720,000 read-outs, 3,284,280 rows, 10,062,798,848 words compared, 0 differ;
  ident 51/51 and held 54/54 texts identical to the reference run.
- 07:53 window O, 4e5c, LEVER 3 WITH THE ADAPTIVE RULE (kp17, one user, default 5 x 262K, cert on; WINDOW_ROWS=32 + PREFILL_MANY=256 vs both off), MEASURED:
  | set | off (16-row windows) | on (15 copy tokens, 31 after a copy landed whole) | change | texts |
  | edit (rewrite a file) | 156.5 tok/s decode, 12.59 tok/round, 80.5 ms/round | 193.9, 18.38 tok/round, 94.8 ms/round | +23.9 % | 24/24 identical |
  | agent-style | 73.5 | 73.3 | -0.3 % (noise; was -3.0 % without the rule) | 16/16 |
  | chat | 58.4 | 58.3 | -0.2 % (noise) | 40/40 |
  | waves of 5 prompts at once (5 slots) | 97.4 tok/s, last first token 1.83 s | 105.8, 0.87 s | +8.6 % | 160/160 |
  first-score hashes 200/200 identical. => LEVER 3 SHIPS with the adaptive rule (COPY_ROWS=32 default): the edit gain is kept, the agent / chat loss is gone.
- 07:56 build kp19 started on 4e5c (ship.sh, tree verified 40 files) while its lane runs the CERT=2 checked run (not speed-sensitive).
- 08:19 Q1 FUSED READ-OUT + Q3 HALF TILES, PAIRED SPEED (window O spare lanes; one user, default config, cert on; 54 held prompts x 256 tokens; tfsteps over the same 49 drafted requests,
  3.527 tok/round and 3.928 drafts/round in every run = identical steps), MEASURED:
  | box | run | ms / round | wall for the 54 prompts | texts vs ref |
  | 3d24 kp18 | ref (XH_ref2) | 56.94 | 175.8 s | - |
  | 3d24 kp18 | HCF=1 (up + mix fused, 4 launches) | 56.67 (-0.27 ms, +0.5 %) | 174.8 s | 54/54 |
  | 3d24 kp18 | HCF=3 (2 launches) | n/a on the same sample (run also held the ident set); wall ratio => ~56.2 (-0.8 ms, +1.4 %) | 173.4 s | 54/54, ident 51/51 |
  | 4e5c kp19 | ref (XB_ref) | 56.89 | 175.6 s | - |
  | 4e5c kp19 | HALF_BELOW=96 (upstream #1171 port) | 56.49 (-0.40 ms, +0.7 %) | 174.4 s | 54/54 |
  READING: the fused read-out is exact and positive but ~+1.4 %, not the +3.5 % estimate: it removes 3 x 109 = 327 launches a round for ~0.8 ms = ~2.4 us a removed launch,
  i.e. the round-4 microbench figure (2.05 us per tiny kernel). The "32 us of small kernels a read-out" from GPROF was mostly waiting attributed to the group, not GPU work.
  RULE OF THUMB from here: a removed launch is worth ~2.4 us; ~1,200 launches remain a round (~2.9 ms = 5 % if ALL went away), so kernel fusion is a 1-2 % lever at most now.
  whole many-user stack CHECKED with 16 in flight (4e5c, kp17, CERT=2, pool + many + rows 32 + wide 128): 83,690 rows compared with the whole head, 0 mismatches.
- 08:24 window O, 923b, DOCUMENTED PATH (recipe.sh start on the image prepare built with 0101-0106 = kp17; vision on), levers off (CERT=0 MANY=0 COPY_ROWS=16) vs recipe default, MEASURED:
  | bench | levers off | recipe default (cert + cert-plain + MANY=256 + COPY_ROWS=32 adaptive) |
  | ttft short prompt | 186 ms | 184 ms (prompt cache of single requests unchanged) |
  | 3-prompt chat / code, 9 runs | 47.8 / 78.1 | 48.1 / 80.4 |
  | sparkDash-method prose / code / structured | 57.0 / 92.5 / 106.4 | 59.0 / 95.7 / 109.8 |
  | held (off: every 3rd = 104; on: all 315 answered) all / chat / code / math | 58.18 / 49.92 / 79.08 / 74.36 | 59.50 / 51.83 / 77.53 / 75.32 |
  | 4 / 5 users distinct (copies) | 149.8 (151.0) / 160.4 (172.7), last first token 1.2 / 1.6 s | 148.9 (159.6) / 170.6 (184.6), 0.60 / 0.59 s |
  | edit set (24 file rewrites), wall tok/s | 137.8 | 169.1 (+22.7 %; decode 193.3) |
  | agent set, wall | 54.5 | 55.8 |
  | sparkDash 1.8.8 real prose / code / structured; catalogs | | 58.7 / 95.2 / 109.3; 45.1 / 68.6 / 84.0 |
  | prefill 8K / 32K / 128K; needle 200K; image; qeval | | 2409 / 2463 / 2309; 3/3 (2206 tok/s); PASS; (see suite log) |
  IDENTITY default vs off (same image, same window): ident 51/51, held 107/107, edit 24/24, agent 16/16 texts identical.
- 08:30 window O spare lanes, MEASURED: fn_q38_hcdn stages (3d24, same 49 requests): ref 56.94 ms/round; HCF=3 with 3 stages 55.94 (-1.00 ms, +1.8 %), 4 stages ~56.2 (wall ratio),
  6 stages 56.96 (no gain): FEWER stages are better => 2-stage instances added in kp20; default to be set after measuring 2 vs 3. Texts 54/54 in every run.
  4e5c kp19: HALF_BELOW=96 + HCF=3 (4 stages): 56.09 vs ref 56.89 (-0.80 ms; half alone -0.40): pairs are at the noise floor (~0.2 ms) => window P re-measures on 923b with more prompts.
  FUSED READ-OUT CHECK WITH 16 IN FLIGHT (4e5c, kp19, p16, wide 128 + many + HALF_BELOW=96, HCF=3 with HCF_MAX=128, CHECK): 60,000 read-outs, 3,086,875 rows, 9,457,982,868 words, 0 differ.
  Total checked so far: 13.35 M rows / 19.5 billion words, 0 differ.
- 08:31 build kp20 started on 3d24 = kp19 + TF_FLASHNEXT_Q38_MTP_LAST=1 (upstream #1925 port: a lone stream's MTP head call runs its experts on the call's last row only)
  + TF_FLASHNEXT_Q38_HCF_HALF=1 (fn_q38_hcdn on 32-column tiles) + 2-stage fn_q38_hcdn (TF_FLASHNEXT_Q38_HCF_ST=2).
- 08:38 window O, 923b, USERS16 PROFILE through recipe.sh (USERS16=1 = 16 lanes + pool 65536 + wide 128 + many + copy rows 32 + cert; image = 0101-0106), MEASURED:
  distinct 1 / 4 / 8 / 16 users: 78.2 / 154.4 / 213.9 / 257.3 + 258.9 tok/s (last first token 0.27 / 0.59 / 0.85 / 1.42 + 1.18 s); copies 4 / 8 / 16: 153.2 / 236.9 / 320.1;
  waves of 16: 147.6 tok/s, last first token 1.71 s mean; waves of 4: 95.4, 0.77 s; mixed 1 x 136,607-token + 15 short: 0 errors, recall 3/3, shorts' last first token 1.59 s;
  227,621-token prompt: 3/3 (2134 tok/s) in the 16-lane configuration. (MiaAI config with 16 x 64K slots: running now for the same-window "off" column.)
  kp20 BUILD_OK on 3d24 (sha d9d926b28783a2d2), copied to all three workers' ~/q38/tfbin.
  queue A first data (4e5c spare lane, kp19, 16 users, wide 128 + many + cert): TF_FLASHNEXT_PROMPT_EXPERT_TILE_ROWS=64: 265.7 / 263.3 vs 262.4 / 262.5 at the default 256 (+0.8 %, inside noise); 8 users 212.3 vs 212.1.
- 08:50 window O, 923b, MiaAI CONFIG WITH 16 x 64K SLOTS (CERT=0 MANY=0 COPY_ROWS=16, 64-row passes) through recipe.sh, same window as the USERS16 run, MEASURED:
  | users (distinct code prompts, 512 tokens) | MiaAI config 16 x 64K | USERS16 (ours) | change |
  | 1 | 76.2 | 78.2 | +2.6 % |
  | 4 | 147.3 (last first token 1.15 s) | 154.4 (0.59 s) | +4.8 % |
  | 8 | 190.7 (2.83 s) | 213.9 (0.85 s) | +12.2 % |
  | 16, two runs | 218.2 / 221.2 (6.8 s) | 257.3 / 258.9 (1.4 / 1.2 s) | +17.5 % |
  | 16 copies | 246.1 | 320.1 | +30 % |
  | waves of 16 held prompts | 127.6 tok/s, last first token 7.17 s | 147.6, 1.71 s | +15.7 % |
  | waves of 4 | 87.8, 1.32 s | 95.4, 0.77 s | +8.7 % |
  identity USERS16 vs MiaAI 16: waves-16 texts 128/128, waves-4 64/64 identical.
- 08:49 spare lanes, MEASURED: MTP_LAST (3d24, kp20): 56.48 ms/round vs ref 56.94 (-0.46 ms, +0.8 %), same 3.928 drafts/round, texts 54/54 => positive, exact.
  HCF=3 with 2 stages: 56.11 (3 stages 55.94): 3 stages stays the default. QUEUE A expert tile rows at 16 users: 64 -> 265.7 / 263.3, 128 -> 258.3 / 258.3 vs 262.4 / 262.5 at 256;
  held 16 in flight 172.4 / 174.4 vs 172.9; texts 107/107 both => ZERO (noise), DEAD, stays 256.
- 08:53 window O CLOSED by itself (work done 08:49:30). DeepSeek verified by the harness AND by me at 08:56: health 200, 17*23 = 391, session store 12.00 GiB on every rank,
  disk tier 28.2 GiB, image ok; all four ranks started 08:53:23. Outage 6939 s (06:58:02 -> 08:53:41).
  INCIDENT (no damage, fixed by hand in ~1 min): the spare-lane script x_b.sh started a new Qwen engine on 4e5c at 08:49:33, AFTER the window's stop-all, because its "window
  open" test (unit active and no DONE file) is still true while the restore runs. I killed the spare-lane scripts and ran qstop.sh on 4e5c at 08:53:15; DeepSeek's ranks started
  08:53:23 with normal memory (free 28.3 GiB on that rank). LESSON: no spare-lane scripts next to a window that is about to end: everything goes INTO the window script;
  if one must run, its open test must also read "restore begin" in run.log, and it must be killed before the last lane ends. Also: `pgrep -f PATTERN | kill` inside an ssh
  command kills that ssh's own shell when PATTERN is in its command line (use a [x] bracket in the pattern).
- 08:58 window P LAUNCHED (guard 110 min; head ~/q38/w_P.sh; kp20 on all workers): 923b fused read-out variants paired (107 held x 256, ref first + last); 3d24 half tiles 96 / 192, MTP_LAST, all together + ident, ref first + last; 4e5c prompt chunk 3072 / 4096 / 8192, 16 users: expert tile 128, fused stack with HCF_MAX 16 / 128. Logs head ~/q38/runs/P/p_*.log. No spare-lane scripts.
- 08:59 COMMITTED + PUSHED private repo: 92a94e4 on tensorfold-fp4-local (author Pico noreply; gitleaks no leaks; key values not present; visibility PRIVATE): patch 0106, recipe defaults (MANY, COPY_ROWS, pool), window M-O logs + notes.
- 09:12 window P, 4e5c, QUEUE A PROMPT CHUNK (kp20, one user, default config; cold prefill tok/s = prompt tokens / time to first token), MEASURED:
  | TF_FLASHNEXT_CHUNK | 7.3K prompt | 29K | 117K | 91K needle prompt | GPU MiB after load |
  | 3072 (default) | 2226 | 2518 | 2355 | 2397, 3/3 | 78,0xx |
  | 4096 | 2168 | 2492 | 2358 | 2409, 3/3 | 78,044 |
  | 8192 | 2144 | 2539 | 2399 | 2436, 3/3 | 81,732 |
  => chunk size is NOT the prefill limit here (+0..+2 % at 8192 for +3.7 GiB): ZERO, stays 3072. (MiaAI's build already keeps the GPU busy across chunks.) Q9 stays open:
  next step is GPU time by group of a prompt pass (TF_FLASHNEXT_Q38_GPROF on a 32K prefill) to see which group is below its byte rate.
- 09:13 kp21 (Mac work/tf commit 1c06bc2; build started on 4e5c with 3 CPUs) = kp20 + QUEUE E first form: TF_FLASHNEXT_Q38_SHORT_COPY=M[,C]: a lone stream's head chain whose drafts
  ALL agree with the continuation of a context match of >= M tokens (below the 8 a copy of its own needs) gets up to C (default 7) more tokens of that continuation appended
  (core/lanes/context_copy.zig extendChain, windows.zig plan; the backend now accepts a chain written as parents). Drafts only. UNMEASURED.
- 09:40 window P so far (kp20; one user, 107 held prompts x 256 tokens, same 96 drafted requests, 3.456 tok/round in every run; texts 107/107 identical in every pair), MEASURED:
  | 923b | ms / round | vs ref |
  | ref (cert + many + copy rows 32) | 56.24 | - |
  | + HCF=3, 3 stages | 55.44 | -0.80 ms, +1.4 % |
  | + HCF=3, 2 stages | 55.91 | -0.33 ms |
  | + HCF=3, 2 stages, fused down kernel on half tiles (HCF_HALF=1) | 54.99 | -1.25 ms, +2.3 % |
  3d24: lane half tiles (HALF_BELOW=96): wall 352.9 s vs ref 353.9 s (+0.3 %); MTP_LAST: 56.34 vs 56.17 ms/round on the same 144 requests (no gain here; spare-lane run had
  -0.46 ms against a kp18 reference): MTP_LAST is 0 +- 0.3 ms by this ruler => window Q measures the chain step directly (TF_HOSTPROF mtpstat, MTP_LAST 0 vs 1).
  4e5c, 16 users (wide 128 + many + rows 32 + cert; distinct 16 twice / 8 / held 16 in flight): base 254.4 / 254.2 / 210.7 / 173.1; expert tile rows 128: 256.2 / 251.1 / 213.5 / 172.9;
  + fused read-out + half tiles + MTP_LAST (HCF_MAX 16): 258.1 / 252.6 / 213.6 / 173.4; same with HCF_MAX=128: 259.0 / 253.4 / 212.4 / 174.2; texts 107/107 in every pair.
  => at 16 users the one-user levers are exact and flat (+0.2..0.6 %, inside noise): passes are expert-bound. HCF_MAX stays 16.
  OPEN QUESTION for window Q: 16 users read ~254 here with WINDOW_ROWS=32 against ~262 in window O without it: copy rows 16 vs 32 at 16 users, same box, same window.
- 09:41 4e5c lane finished early => spare-lane script x_d.sh (fixed open test: also stops on "restore begin" / "work done" and after a deadline of 09:52 for engine starts):
  kp21 short context copies, ref vs SHORT_COPY=3,7 on the edit / agent / chat sets.
- 09:54 window P, 923b lane DONE (kp20, 107 held x 256, same 96 requests; texts 107/107 in every pair), MEASURED:
  | run | ms / round | vs mean ref 56.33 |
  | ref first / last | 56.24 / 56.41 | - |
  | HCF=3, 3 stages | 55.44 | -0.89 ms, +1.6 % |
  | HCF=3, 2 stages | 55.91 | -0.42 ms |
  | HCF=3, 2 stages + HCF_HALF | 54.99 | -1.34 ms, +2.4 % |
  | HCF=3, 3 stages + HCF_HALF | 54.96 | -1.37 ms, +2.5 % |
  => SHIP: FUSED=3, 3 stages, fused down kernel on half tiles (recipe defaults TF_FLASHNEXT_Q38_HCF=3, _HCF_ST=3, _HCF_HALF=1). One user +2.5 % (measured), exact (19.5 billion words checked).
- 09:55 kp21 BUILD_OK (4e5c, sha df6d09e276739442), on all three workers. Patch 0107 generated (git diff kp17..kp21, 724 lines, sha256 e4ed898688625fed); patches 0101-0107 applied to the
  MiaAI base reproduce the build tree exactly (git write-tree 6dbc4df8ff4e = HEAD tree of Mac work/tf 1c06bc2). recipe.sh (FUSED / FUSED_ST / FUSED_HALF / HALF / MTP_LAST) + 0107 are on
  923b ~/q38/repo_tf (hashes equal); head ~/q38/w_Q.sh = Mac harness/w_Q.sh (hash equal).
  4e5c spare lane, kp21 SHORT_COPY=3,7 vs off (one user, new stack), first data: edit 198.4 -> 198.5 tok/s (18.38 -> 18.61 tok/round), agent 75.1 -> 75.6 (+0.7 %, 4.647 -> 4.715 tok/round);
  no crash, chain-as-parents windows verified. Small; window Q tries 5,7 / 3,15 on 3d24.
- 10:10 window P CLOSED by itself. DeepSeek verified by the harness: health 200, 17*23 = 391, session store 12.00 GiB on every rank, disk tier 28.2 GiB, image ok. Outage 4372 s
  (08:57:02 -> 10:09:54). No Qwen containers on any worker (checked by me at 10:10); the spare-lane script had ended at 09:57.
  3d24 lane (kp20, 107 held x 256, n = 96 requests): ref last 56.66; HALF_BELOW=96 56.12; =192 56.20; (first ref on a 144-request sample 56.17; MTP_LAST 56.34; all three levers
  without HCF_HALF 55.67, ident 51/51 + held 107/107 identical). Lane half tiles: -0.5 ms against the last ref, +0.3 % by wall against the first: small positive, exact => kept (HALF=96).
  SHORT_COPY=3,7 (4e5c, kp21): edit 198.4 -> 198.5, agent 75.1 -> 75.6, chat 59.9 -> 59.9; texts 24/24, 16/16, 40/40 identical.
- 10:12 window Q LAUNCHED (FINAL; guard 150 min; head ~/q38/w_Q.sh; kp21 = patches 0101-0107 on all workers):
  923b: recipe.sh prepare (0101-0107) -> levers off -> recipe default FULL suite -> identity -> USERS16 -> USERS16 with COPY_ROWS=16 -> MiaAI config 16 x 64K.
  3d24: short copies 3,7 / 5,7 / 3,15 on edit / agent / chat; MTP chain-step timer with MTP_LAST 0 / 1; no-drafts config stack vs cert only.
  4e5c: 16 users copy rows 16 / 32 / 16; whole final stack CHECKED (cert + fused read-out) with 16 in flight; GPU time by group of prompt passes; mixed sizes; held every 2nd prompt off vs stack.
  Logs head ~/q38/runs/Q/q_*.log.
- 10:16 window Q, 923b: `recipe.sh prepare` RE-RUN END TO END with patches 0101-0107 (rc 0, "Done", image tensorfold-qwen38:v1.1.4-eab8582 rebuilt; log ~/q38/recipe_work/prepare4.log).
- 12:12 window Q CLOSED by itself (work done 12:07:38). DeepSeek verified by the harness: health 200, 17*23 = 391, session store 12.00 GiB on every rank, disk tier 28.2 GiB, image ok.
  Outage 7270 s (10:11:20 -> 12:12:30). The round-6 agent died at ~10:22 (account limit) before any Q result was written.

## Round 7 (seventh agent), started 2026-11-13 13:38 UTC
- 13:38 state verified: no window / guard unit, DeepSeek health 200, windows.log last line Q 7270 s. Raw Q data pulled to Mac results/rQ (+ rQ.tgz).
### WINDOW Q WRITTEN UP (tables: Mac results/final_r6_923b.txt, final_r6_manyuser.txt, final_r6_levers.txt). All MEASURED, kp21 = patches 0101-0107.
- 923b documented path (recipe.sh prepare rc 0 with 0101-0107, then recipe.sh start, vision on), levers off vs recipe default, same image + window:
  | bench | levers off | recipe default | change |
  | ttft | 182 ms | 182 ms | 0 |
  | 3-prompt chat / code (9 runs) | 47.1 / 78.4 | 51.1 / 82.7 | +8.5 % / +5.5 % |
  | sparkDash-method prose / code / structured | 56.2 / 90.8 / 106.4 | 60.8 / 98.8 / 113.3 | +8.2 / +8.8 / +6.5 % |
  | held, same 107 prompts: all / chat / code / math | 58.40 / 50.13 / 79.39 / 74.45 | 62.26 / 53.36 / 85.39 / 79.01 | +6.6 / +6.4 / +7.6 / +6.1 % |
  | held-320 (315 answered) | not run | 61.33 (chat 53.45 / code 79.91 / math 77.40) | (window O default: 59.50) |
  | edit set wall (decode) | 137.9 (153.1) | 173.4 (197.9) | +25.7 % |
  | agent set wall (decode) | 54.5 (71.4) | 57.3 (75.8) | +5.1 % |
  | 4 users distinct / copies; last first token | 149.5 / 151.3; 1.22 s | 149.9 / 168.6; 0.58 s | |
  | 5 users distinct / copies; last first token | 167.1 / 166.9; 1.61 s | 171.0 / 181.4; 0.61 s | +2.3 / +8.7 % |
  | sparkDash 1.8.8 real prose / code / structured; catalogs | | 59.8 / 97.2 / 110.4; 46.5 / 70.7 / 86.7 | |
  | prefill 7.3K / 29K / 117K; needle 182K; 227,621-token prompt; image; qeval | | 2450 / 2521 / 2354; 3/3; 3/3 (2160 tok/s); PASS; 33/35 | |
  IDENTITY default vs off: ident 51/51, held 107/107, edit 24/24, agent 16/16 texts identical. One-user stack is now +6.6 % paired (cert +3.7, fused read-out + half tiles +2.5).
- 923b many users (distinct, 512 tokens; last first token): MiaAI config 16 x 64K: 1 / 4 / 8 / 16 = 75.1 / 136.8 / 190.7 / 214.0 + 219.5 (6.9 s), copies 16 245.4;
  USERS16: 77.6 / 154.4 / 213.8 / 260.6 + 262.7 (1.29 / 1.13 s), copies 16 317.8 => 16 users +20.7 %, 8 users +12 %. USERS16 with COPY_ROWS=16: 81.3 / 154.4 / 211.2 / 260.6 + 262.9, copies 326.1.
  waves of 16: 128.1 tok/s, 7.16 s -> 148.4, 1.65 s; waves of 4: 87.9, 1.34 s -> 96.5, 0.76 s; texts 128/128 + 64/64 identical. Mixed 136,607-token + 15 short: 3/3, shorts 1.55 s; 227,621-token prompt 3/3 in the 16-lane config.
- 4e5c: copy rows 16 / 32 / 16 at 16 users: 248.7 257.3 267.7 / 257.7 264.7 263.4 / 254.4 264.5 256.1; held 16 in flight 174.1 / 174.6 / 174.1; texts 107/107 => copy rows 32 is free at 16 users (the P question was noise).
  Final stack CHECKED with 16 in flight (CERT=2 + HCF check, HCF_MAX=128): 19,511,807,096 words / 6,368,221 rows, 0 differ. One user 160 held: off 58.0 -> stack 62.1 decode tok/s (+7.0 %), 160/160 identical.
  Mixed sizes: 1 x 137K + 15, 3 x 91K + 13, 5 x 182K at once: all 3/3, 0 errors.
  GPU time by group of a PROMPT pass (2653 rows, 1032 ms = 0.389 ms a row): MoE 396.9 (38.5 %), hyper-connection 239.8 (23.2 %), GDN 231.3 (22.4 %), QSA 150.9 (14.6 %), PLE 12.8.
- 3d24: SHORT_COPY 3,7 / 5,7 / 3,15: edit 198.5 -> 198.7 / 198.2 / 201.2, agent 75.6 -> 76.2 / 75.9 / 76.3, chat 60.3 -> 60.3 / 60.3 / 60.2; texts identical => +0..1.4 %, noise-level: stays OFF (flag kept).
  MTP chain step: MTP_LAST=1 cuts the first step of a round 1.952 -> 1.688 ms, later steps -0.02 ms => ~0.3 ms a round (+0.5 %): real, small, stays on.
  no-drafts: cert only 23.3 -> whole stack 25.4 tok/s (+9.2 %), 40/40 identical.
- 13:45 COMMITTED + PUSHED private repo: 539d70b on tensorfold-fp4-local (author Pico noreply; gitleaks no leaks; key values absent; PRIVATE): patch 0107 (tree hash check TREE_EQUAL 6dbc4df8), recipe.sh defaults (FUSED / HALF / MTP_LAST), README rewritten for round 6, window Q tables + raw logs.
- 13:50 ANALYSIS BEFORE BUILDING (arithmetic on measured numbers; ESTIMATE where said):
  * Prefill (Q9) is at the tensor-core ceiling, not a byte or chunk problem: a prompt row costs 0.389 ms for 11.6 GFLOP of multiply-adds (dense 6.9 + experts 4.7)
    = ~30 TFLOP/s of bf16 m16n8k16 mma, the same rate in the dense groups (29) and the expert tile (31). Chunk 8192's +0..2 % = fewer half-empty row tiles. So upstream's
    chunk / tile items cannot give much here; only fewer wasted mma rows or cheaper weight decode can. Q9 closed unless the microbench shows decode ALU matters.
  * Dense at 16 users (ideas.md #1b) is dead by the same arithmetic: 54-99 rows cost 36-45 ms = the 1-row byte cost (23 ms) + ~0.2 ms a row, which IS the prompt-tile rate (0.234 ms a row).
  * Experts: mma per byte is 3.5x the BF16 head's and 1.8x MXFP8's (72 B of NVFP4 per m16n8k16 against 128 / 256), and GB/s falls in that order (head 240, MXFP8 232, experts ~177).
    Round 3's prefetch-depth test was confounded (deeper stages = more registers = fewer resident warps). MiaAI's shared-memory tile has the decode kernel's bits at any row count
    and is switched by env (TF_FLASHNEXT_PROMPT_EXPERT_TILE_ROWS=1): a free test of "better overlap of memory wait and compute".
  * Native FP4 / FP8 decode (PTX ISA 9.2 cvt.rn.bf16x2.e2m1x2 / e4m3x2; upstream TensorFold -Dnative-fp-decode uses it for other kernels, not for NVFP4 experts): added behind
    Q38_NATIVE_FP in kernels/cuda/q38_fp.cuh (work/tf, uncommitted), compiled for sm_121a in the microbench. Exact except E4M3 NaN bytes (checked exhaustively in the microbench).
  * Lookup drafts replacing the chain (ideas.md #4a): our SHORT_COPY variant (chain extended) found matches in few rounds (+0.5..1.5 % tokens a round on agent / edit, 0 on chat);
    replacing the chain saves 1.7 ms a skipped step only in those rounds => bound ~1 %: not built. Platform item #7 (cpu_dma_latency / idle states) needs a host PM setting: not done.
- 13:53 window R LAUNCHED (guard 100 min; head ~/q38/w_R.sh; kp21): 4e5c expert microbench kit/mb/mbx.cu (stock + native binaries) then one user tile from 2 / 6 rows;
  923b one user tile at every row count in 3 shapes + gate/up only, ref first + last; 3d24 16 users same shapes + no-drafts GPROF. Logs head ~/q38/runs/R/r_*.log.
- 13:58 window R, 4e5c, EXPERT MICROBENCH (kit/mb/mbx.cu; the engine's own kernels on synthetic weights, 48 layers a step, experts drawn from a pool of 1024 so nothing is cached;
  every variant's outputs hashed and equal to the stock decode kernel's; raw Mac results/rR/R3_mbx.txt). MEASURED, kernels alone (no router / shared expert / glue):
  | rows a pass, distinct experts a layer | stock gate/up GB/s | stock down GB/s | both, ms a step (GB/s) | best exact alternative |
  | 1 row, 10 experts | 196 | 211 | 6.60 (201) | gate on the shared-memory tile 220 GB/s: 6.09 ms (+8 %) |
  | 5 rows, 25.5 (a drafted round) | 227 | 218 | 15.14 (224) | +0.8 % |
  | 16 rows, 82 | 222 | 230 | 48.4 (224) | gate with 3 stages or tile +4 %: 47.1 (+2.7 %) |
  | 97 rows, 120 (16 users) | 223 | 214 | 72.4 (220) | +1.6 % |
  | 97 rows, 245 | 232 | 219 | 143.0 (227) | +0.6 % |
  | 1024 rows, 512 (prompt-sized) | 216 | 122 (decode form) / 180 (tile) | 395 / 333 | the tile, which MiaAI already uses from 256 rows |
  Resident warps an SM: decode gate 16 (4 CTAs), 3-4 stages 12 (so round 3's prefetch-depth test also cut warps), tile 8.
  VERDICT (Q2 "MoE 168 -> 225 GB/s"): the routed-expert kernels ALREADY read at 220-230 GB/s from 5 rows up. The 168 GB/s of the GPROF group is router + shared expert + glue + waits,
  not the expert kernels. At 16 users the pass is at the box's byte rate on experts: no kernel lever left there (DEAD, with the cause now measured). Only 1-row passes have a gap
  (gate/up 196 GB/s: 220 units on 768 warp slots): the shared-memory tile closes it (+0.5 ms a step = ~+1.2 % for --no-drafts; drafted rounds are 2+ rows). Engine A/B is running in this window.
  NOTE: the first "native FP decode" binary was NOT native (the macro needs -gencode=arch=compute_121a,code=sm_121a, not -arch=sm_121a); rebuilt, runs in the next window.
- 14:18 window R, ENGINE A/B of the expert tile at low row counts (kp21, whole stack; one user: 107 held x 256, same 96 drafted requests, 3.456 tok/round in every run; texts 107/107 identical in every pair). MEASURED:
  | box | run | ms / round |
  | 923b | ref | 54.64 |
  | 923b | TF_FLASHNEXT_PROMPT_EXPERT_TILE_ROWS=1 (tile for every pass) | 54.58 (-0.06 ms) |
  | 4e5c | ref | 54.63 |
  | 4e5c | tile from 2 rows | 54.67 |
  | 3d24, 16 users (three runs; held 16 in flight) | ref 257.0 / 260.7 / 259.4; 174.5 | tile at every row count 261.5 / 264.9 / 263.9; 175.1 | tile with 32-pair items 257.1 / 265.9 / 262.5; 173.5 |
  => as the microbench said: ZERO for one user (drafted rounds are 2+ rows, where the decode kernel already reads at 220+ GB/s) and inside noise at 16 users. DEAD; MiaAI's 256 stays.
  Window R ended early by SIGTERM (the remaining runs were shape variants of a dead lever).
- 14:21 window R CLOSED. DeepSeek verified by the harness: health 200, 17*23 = 391, session store 12.00 GiB on every rank, disk tier 28.2 GiB, image ok. Outage 1699 s (13:53:16 -> 14:21:35).
- 14:20 build kp22 started on 3d24 (3 CPUs, 8 GiB; Mac work/tf 2d9130f, tree verified 48 files) = kp21 + TF_FLASHNEXT_Q38_NATIVE_FP=1: the NVFP4 expert kernels (decode form and tile)
  as second images built for sm_120a / sm_121a with the one-instruction E2M1 / E4M3 decodes (kernels/cuda/q38_fp.cuh; build/cuda_kernels.zig units nvfp4_experts_nat, nvfp4_experts_tile_nat).
- 14:23 window S LAUNCHED (guard 100 min; head ~/q38/w_S.sh; kp21): 4e5c native-decode microbench, then 16 users under the host profiler (sync call sites, MTP step timer) and GPROF;
  3d24 per-kernel GPU time (TF_KPROF) one user with drafts + 16 users + chain depth log; 923b draft gate re-fit (0.4/8, 0.35/8, 0.3/6, 0.5/8 vs 0.4/6, ref first + last).
  QUESTION for S: a 16-user round is ~210 ms but its verify pass ~145 ms: what is the other ~65 ms (MTP chain passes, n-gram reads, host)?
- 14:25 kp22 BUILD_OK (3d24, sha 0dcf5847c5ebf08f, 99 fatbins), copied to all three workers' ~/q38/tfbin.
- 14:30 window S, 4e5c, NATIVE FP DECODE MICROBENCH (mbxn built with -gencode=arch=compute_121a,code=sm_121a -DQ38_NATIVE_FP; its PTX holds 2,684 cvt.rn.bf16x2.e2m1x2 and
  1,089 e4m3x2 instructions, cuobjdump shows them in the binary; the line "native FP decode compiled: 0" is printed by host code and is wrong). MEASURED (raw Mac results/rR/S3_mbxn.txt):
  exactness: all 16,777,216 E2M1 pair decodes and all E4M3 scale pairs without a NaN byte equal the stock arithmetic; every kernel's output hash equals the stock binary's.
  | scenario | stock ms a step (two runs) | native ms a step |
  | 1 row | 6.599 / 6.529 | 6.528 |
  | 5 rows, 25.5 experts | 15.137 / 15.188 | 15.128 |
  | 97 rows, 120 experts | 72.40 / 72.16 | 72.19 |
  | 1024 rows, 512 experts: gate decode / down tile | 209.8 / 126.0 | 209.4 / 126.5 |
  => hardware FP4 / FP8 decode: exact and ZERO, also on the compute-heavy prompt-sized tile. Weight decode is not what limits these kernels. DEAD: kp22 is not shipped
  (flag TF_FLASHNEXT_Q38_NATIVE_FP stays out of the recipe; no engine A/B spent on it).
- 14:40 window S, ROUND ANATOMY (kp21, final stack). MEASURED:
  * one user, per-kernel GPU time (TF_KPROF, serialised; 3d24): a round = ~57 ms: routed experts 20.4 ms (35 %; gate/up 13.6 + down 6.8, at ~200 GB/s = bytes of ~32 distinct experts a layer),
    MXFP8 projections (GDN / QSA / shared) ~12, fused hyper-connection read-out 8.1 (85 us a read-out under the profiler, ~69 real = 190 GB/s), certified head coarse pass 2.4,
    4-bit draft head 2.2 (556 us a chain step), other BF16 2.8, ~1000 small kernels ~4-9. Nothing stands out: bytes at 190-225 GB/s. One-user engine work is at its floor (est. within ~15 %).
  * 16 users, GPU time by group (GPROF, 4e5c): 78.7 rows a pass, 200.7 ms a pass: MoE 128.9 (64 %), GDN 39.4 (20 %), hyper-connection 15.2, QSA 8.5, head 7.5, PLE 1.2.
    The pass IS the round (round ~210 ms): there is no large cost outside it (the "65 ms outside the pass" guess was wrong: older pass figures were for fewer rows).
    MoE: 159 distinct experts a layer at ~220 GB/s (KPROF: gate/up 1.30 ms + down 0.68 ms a layer): byte-bound, as the microbench said.
  * FOUND (KPROF 16 users + code): GDN costs 39 ms at 79 rows but 10.9 ms at one row. TensorFold's lane-batched DeltaNet path (TF_FLASHNEXT_GDN_LANES=1, MiaAI default: 3 launches a layer
    for all lanes) holds at most 8 lanes (gdn.lanes_max = 8, the kernel's by-value lane table). With 9-16 lanes every layer falls back to 5 launches A LANE: 16 x 36 x 5 = 2,880 launches a pass;
    fn_gdn_rows_late alone is 53.5 calls a profiled pass at 39 us. The batched kernel does 8 lanes in 74 us.
    LEVER (kp23, building on 3d24): TF_FLASHNEXT_Q38_GDN_LANES_MANY=1 = more than 8 lanes go through the same batched kernels in groups of 8 (each block indexes its own lane's
    pointers: L.lane[blockIdx.z]; same per-lane program and bytes). ESTIMATE: -15..20 ms of a 200 ms pass = +8..10 % at 9-16 users. Flag, default off until measured.
- 14:47 kp23 BUILD_OK (3d24, sha c0ce3d45c3735f1e; build tree checked against the Mac tree by Mac treecheck.sh: 0 of 7,352 files differ), on all workers. Spare lanes head ~/q38/x_e.sh inside window S (engine starts only before 15:08, stop on 'restore begin'): 4e5c 16 users ref / GDN_LANES_MANY=1 / ref + texts; 3d24 GPROF with the flag, 12 users and waves of 16 on / off.
  LESSON: ship.sh sends only files that differ from the MiaAI base in the CURRENT tree; files an earlier build changed and a revert restored stay stale on the build box. Run treecheck.sh BOX before trusting a build.
  window S, 923b gate re-fit so far: 0.4/8: 63.4 vs 63.2 tok/s decode (3.595 vs 3.456 tok/round, 56.70 vs 54.66 ms/round): zero. chain depth at 16 users (3d24, DEPTH log, 5+ streams drafting): 0:113 1:1657 2:933 3:1299 4:887 5:728 6:2807.
- 15:07 LEVER "GDN lanes for more than 8 lanes" MEASURED (window S spare lanes, kp23, 16 lanes pool + wide 128 + whole stack; TF_FLASHNEXT_Q38_GDN_LANES_MANY=1 vs off):
  | 4e5c, 16 users distinct, three runs | off 262.5 / 259.5 / 258.4 (mean 260.1) | on 266.4 / 272.2 / 268.9 (mean 269.2) | +3.5 % |
  | 4e5c, held 107 with 16 in flight | 174.2 | 177.3 (+1.8 %) | texts 107/107 identical |
  | 3d24, GPROF 16 users, GDN group / whole pass | 39.4 / 200.7 ms (window S, kp21) | 34.8 / 197.9 ms | -4.6 ms |
  | 3d24, 12 users distinct; waves of 16 | off - ; 147.1 tok/s | on 245.2 / 240.7; 149.8 | waves texts 96/96 identical |
  8 users unchanged by construction (<= 8 lanes already took the batched kernels). Smaller than the estimate (+8..10 %): the batched chain is 2 launches of 8 lanes a layer, and most of the
  GDN excess turned out to be something else (next item). SHIPS as a recipe default once the documented path is re-run (exact: per-lane program on its own pointers; texts identical under load).
- 15:08 FOUND + MICROBENCHED (kit/mb/mb3.cu on 3d24, raw Mac results/rR/S2_mb3.txt): TensorFold's decode lane kernel takes passes of more than 64 rows as 64-row tiles SIDE BY SIDE
  (qmmf.zig bucket(): 16 / 32 / 64), and every row tile is a block that loads the weight tile itself: at 65-128 rows every dense weight is read TWICE. A 128-row tile (the same template,
  BM = 128, one block a weight tile) gives the same bits (compared with stock for every case) and, at 79 / 97 / 128 rows:
  | projection | stock two 64-row tiles | one 128-row tile, 2 stages | change |
  | q_proj MXFP8 12288 x 2560 | 285 / 290 / 298 us (114 GB/s of weights) | 155 / 158 / 163 us (209 GB/s) | +83 % |
  | GDN-like MXFP8 6144 x 2560 (cluster, 2 K slices) | 129 / 127 / 137 | 89.5 / 90.9 / 93.7 | +44 % |
  | hyper-connection mix_up BF16 10240 x 320 | 40.1 / 40.9 / 41.5 | 32.4 / 32.4 / 32.8 | +24 % |
  | o_proj MXFP8 2560 x 6144 (cluster, 8 slices) | 218 / 221 / 225 | 199 / 201 / 204 (4 or 3 stages: 332, worse) | +10 % |
  | mix_down BF16 320 x 10240, shared 640 x 2560 | 78 / 23 | 77 / 25 | 0 / -6 %: keep stock |
  At 48 rows (one 64-row tile) the 128-row tile loses 1-37 %: only for more than 64 rows, and only for outputs of >= 2048 columns.
  This is ideas.md #1b ("kernel form by row count"), real after all: my 13:50 arithmetic that called it dead compared per-row costs and missed the second weight read.
  ESTIMATE at 16 users (79 rows a pass): q/o proj, GDN projections, mix_up and the certified head's coarse pass (0.61 GiB read twice today) = ~8 ms of a 198 ms pass = ~+4 %.
  NEXT: kp24 = kp23 + TF_FLASHNEXT_Q38_ROWS128=1 (128-row instances, 2 stages) + GDN lanes in one launch of 16.
- window S, 923b, DRAFT GATE RE-FIT (kp21, one user, 107 held x 256; texts 107/107 identical in every run; ref 54.66 ms/round, 3.456 tok/round, 63.2 tok/s decode):
  0.4 / 8 drafts: 63.4 tok/s (3.595 tok/round, 56.70 ms); 0.35 / 8: 62.9 (3.644, 57.95); 0.3 / 6: 62.3 (3.531, 56.71). => MiaAI's 0.4 / 6 is still the optimum after the stack: DEAD.
- 15:14 kp24 BUILD_OK (3d24, sha 855982293097cf65; Mac work/tf fb01d8c; treecheck 0 of 7,352 files differ) = kp23 + TF_FLASHNEXT_Q38_ROWS128=N (128-row lane tiles, 2 stages, MXFP8 + BF16,
  for calls of more than 64 rows on weights of >= N output columns) + DeltaNet lane table of 16 (TF_FLASHNEXT_Q38_GDN_LANES_MANY=2; 1 = groups of 8). On all three workers.
  Patch 0108 generated (git diff 1c06bc2..fb01d8c, 261 lines, sha256 9f3c7ee4d8c335ce): patches 0101-0108 on the MiaAI base reproduce the build tree (TREE_EQUAL c3e5e34e).
  recipe.sh: GDN_MANY=2 and ROWS128=2048 as defaults (to be confirmed by window T; 0 = stock). recipe.sh + 0108 are on 923b ~/q38/repo_tf (hashes equal).
  Window 4e5c second reference: 255.2 / 257.4 / 257.2 => GDN lanes (groups of 8): on mean 269.2 vs refs 260.1 and 256.6 = +4.2 % at 16 users.
- 15:22 window S CLOSED by itself. DeepSeek verified by the harness: health 200, 17*23 = 391, session store 12.00 GiB on every rank, disk tier 28.2 GiB, image ok. Outage 3632 s (14:22:12 -> 15:22:44).
  Gate re-fit last runs: 0.5 / 8: fit 40.64 + 3.70 d (trade as before); second reference 54.96 ms/round (first 54.66).
- 15:15 COMMITTED + PUSHED private repo a7d55d8 (patch 0108, recipe defaults GDN_MANY / ROWS128, microbenches, windows R-S notes; gitleaks clean, PRIVATE).
- 15:22 window T LAUNCHED (FINAL of round 7; guard 110 min; head ~/q38/w_T.sh; kp24 = patches 0101-0108 on all workers):
  923b documented path: recipe.sh prepare (0101-0108) -> one user default (c1, ttft, held 107, edit, agent; texts against window Q's) -> USERS16 default -> MiaAI config 16 x 64K -> USERS16 with GDN_MANY=0 ROWS128=0.
  4e5c 16 users: ref / both levers / ROWS128 alone / GDN table of 16 alone / GDN groups of 8 / ref / ROWS128=1024; texts 16 in flight each.
  3d24: everything on under both checkers with 16 in flight; on vs off at 12 / 9 users, waves, edit set 4 in flight, one user (texts); GPROF; KPROF.
  Logs head ~/q38/runs/T/t_*.log.
- 16:15 window T so far (kp24; MEASURED):
  * CORRECTION: the 128-row lane tile (TF_FLASHNEXT_Q38_ROWS128) is ZERO in the engine. GPROF 16 users with both levers on (3d24, 78.1 rows a pass): pass 196.0 ms, GDN 34.3, QSA 8.60,
    hyper-connection 15.2, head 7.59, MoE 129.0; with the GDN lever alone (window S): 197.9, GDN 34.8; stock: 200.7, GDN 39.4, QSA 8.48, head 7.50. The 128-row instances do run
    (KPROF: qmmf_kernel<2,128,...> 4.38 calls a pass) but save nothing. CAUSE: my microbench kit/mb/mb3.cu launched the stock kernel with band = 1; the engine passes band = min(row tiles, ..)
    = 2, which orders the blocks so the two 64-row tiles of one weight tile run together and the second finds the weights in the GPU cache. The "weights read twice" claim was a
    microbench artefact. LESSON: copy the launcher's every argument into a microbench (here the `group` / band argument), not only grid and shared memory. ROWS128 stays OFF (default 0).
  * GDN lanes for 9-16 streams is the real part: GPROF GDN 39.4 -> 34.3..34.8 ms (-2.5 % of the pass).
    4e5c 16 users distinct (three runs; held 107 with 16 in flight), texts 107/107 identical in every run:
    | ref first / last | 254.2 260.8 255.3 ; 174.5 | (last: see t_4e5c.log) |
    | table of 16 (GDN_MANY=2) | 257.0 266.2 266.1 (mean 263.1) ; 177.4 |
    | groups of 8 (GDN_MANY=1) | 261.2 263.0 265.8 (mean 263.3) ; 175.9 |
    | GDN_MANY=2 + ROWS128=2048 | 262.3 273.4 262.4 (mean 266.0) ; 177.9 |
    | ROWS128=2048 alone | 262.1 263.9 263.6 ; 174.0 |
    3d24 on vs off: waves of 16 150.4 vs 147.1 tok/s; 12 users 240.9 vs -; texts waves 96/96, edit set 4 in flight 24/24, one user 54/54 identical.
    Checked run with everything on, 16 in flight (CERT=2 + fused read-out check): 22,920,022,376 words, 0 differ.
  * 923b DOCUMENTED PATH (recipe.sh prepare rc 0 with 0101-0108; one window):
    one user default: chat 51.1 / code 81.6 (9 runs), ttft 182 ms, held-107 54.80 ms/round (63.1 tok/s decode), edit 173.3 wall, agent 57.1; texts against window Q (kp21):
    edit 24/24, agent 16/16 identical; held 107/107 equal or a prefix (Q ran 384 tokens, T 256).
    USERS16 with GDN_MANY=2 ROWS128=2048: 1 / 4 / 8 / 16 = 82.6 / 147.8 / 208.7 / 260.8 + 262.1, 12 users 247.2, copies 16 332.6, copies 8 233.6; waves of 16 151.4 (1.64 s), waves of 4 96.4;
      137K prompt + 15 short: 3/3, shorts' last first token 1.56 s; 227,621-token prompt 3/3 (2145 tok/s).
    USERS16 with GDN_MANY=0 ROWS128=0 (same window): 82.1 / 154.6 / 213.8 / 254.9 + 259.7, 12 users 243.3, copies 16 321.7 => 923b: +1.6 % distinct 16, +3.4 % copies, +1.6 % at 12 users.
    MiaAI config 16 x 64K: 74.3 / 148.4 / 194.0 / 218.3 + 215.0 (6.8 s), 12 users 221.7, copies 16 244.8; waves of 16 127.4 (7.20 s), waves of 4 87.5; texts vs USERS16 128/128 + 64/64 identical.
- 16:14 kp25 BUILD_OK (3d24, sha fb1cd91c2a139b17; work/tf 52b32e1; treecheck 0 mismatches) = kp24 + TF_FLASHNEXT_Q38_BEST_LANES=1: a shared round's draft step read every lane's best
  draft with ONE fn_row_best launch a LANE (16 x ~50 us a chain step, one block each: KPROF 0.49 ms a profiled pass = ~3 ms a 16-user round); now one launch, a block a lane, the same
  program a lane (fn_row_best_lanes). ESTIMATE +1.5 % at 16 users. Patch 0109 (TREE_EQUAL). recipe.sh: ROWS128 default 0, BEST_LANES default 1 (to be confirmed in window U).
- 16:22 window T CLOSED by itself. DeepSeek verified by the harness: health 200, 17*23 = 391, session store 12.00 GiB on every rank, disk tier 28.2 GiB, image ok. Outage 3561 s (15:22:46 -> 16:22:07).
  T last numbers: 4e5c refs 256.8 / 262.2 (held 16 in flight 174.5 / 172.9), GDN table of 16 263.1 (177.4), groups of 8 263.3 (175.9), + ROWS128 266.0 (177.9), ROWS128=1024 267.0 (177.0),
  ROWS128 alone 263.2 (174.0): texts 107/107 in every run. 923b USERS16 GDN_MANY=0: waves of 16 147.9 (on: 151.4), waves of 4 96.2.
- 16:22 window U LAUNCHED (guard 70 min; head ~/q38/w_U.sh; kp25 = patches 0101-0109): 923b recipe.sh prepare (0101-0109, rc 0) -> USERS16 BEST_LANES=0 -> USERS16 default;
  4e5c ref / BEST_LANES=1 / ref; 3d24 KPROF with it on, on / off at 12 / 9 / 4 users, one user and edit set texts.
- 16:50 window U RESULTS (MEASURED):
  * BEST_LANES (one launch for every lane's best draft), 4e5c 16 users distinct, three runs each: ref 261.5 261.7 268.9 (264.0) / on 268.2 269.6 270.6 (269.5) / ref 262.2 263.0 263.8 (263.0)
    => +2.3 %; held 107 with 16 in flight 178.1 / 178.2 / 176.3 (0..+1 %); texts 107/107. 3d24: 12 / 9 / 4 users on 243.4 / 220.0 / 153.2, off 243.2 / 213.8 / 157.3 (noise);
    one user 54/54 and edit set 4 in flight 24/24 texts identical. KPROF: fn_row_best 0.487 ms a profiled pass -> fn_row_best_lanes 0.028 + other call sites 0.225.
    SHIPS (exact: the same program a lane; drafts only), default on.
  * 923b DOCUMENTED PATH, FINAL (image = 0101-0109): USERS16 default (GDN_MANY=2, BEST_LANES=1, ROWS128 off): 1 / 4 / 8 / 12 / 16 users = 81.2 / 151.5 / 212.3 / 247.9 /
    269.4 + 266.1 + 272.7 (last first token 1.30 / 1.13 / 1.12 s), copies 16 336.8; waves of 16 151.5 tok/s (1.63 s), waves of 4 97.2 (0.71 s);
    137K prompt + 15 short 3/3 (shorts' last first token 1.46 s); 227,621-token prompt 3/3 (2148 tok/s).
    Same window with BEST_LANES=0: 81.6 / 156.5 / 207.0 / 242.2 / 265.2 + 267.7 + 268.6, copies 331.3, waves of 16 151.5.
    Texts: waves-16 128/128 identical to BEST_LANES=0 and 128/128 identical to MiaAI's 16 x 64K configuration (window T's run).
- 16:54 window U CLOSED by itself. DeepSeek verified by the harness: health 200, 17*23 = 391, session store 12.00 GiB on every rank, disk tier 28.2 GiB, image ok. Outage 1924 s (16:22:11 -> 16:54:15).

## Round 7 SUMMARY (2026-11-13 13:38 -> 17:00 UTC). Tables: Mac results/final_r6_*.txt (window Q), final_r7_manyuser.txt, final_r7_levers.txt
| item | result | status |
| window Q written up (kp21, patches 0101-0107) | one user +6.6 % paired on held-107 (58.40 -> 62.26), chat 47.1 -> 51.1, code 78.4 -> 82.7, edit set 137.9 -> 173.4; 16 users 216.7 -> 261.6 | done, repo 539d70b |
| GDN lane batching for 9-16 streams (patch 0108, GDN_MANY=2) | 16 users +1.4..+4.2 % by window; GPROF GDN 39.4 -> 34.3 ms of a 200 ms pass | SHIPPED |
| one launch for every lane's best draft (patch 0109, BEST_LANES=1) | 16 users +0.8..+2.3 % | SHIPPED |
| 128-row lane tiles (patch 0108, ROWS128) | zero in the engine; the microbench was wrong (band argument) | off |
| expert kernels to 225 GB/s; expert tile at low rows; native FP decode; gate re-fit; prefill | all measured: no gap / zero (see final_r7_levers.txt) | DEAD |
| final 16 users, documented path | MiaAI 16 x 64K 218.3 / 215.0 (6.8 s) -> ours 269.4 / 266.1 / 272.7 (1.2 s) = +24 % (windows T and U); 12 users 221.7 -> 247.9; 8 users 194.0 -> 212.3 | done |
Honest bottom line: the one-user path is at this engine's byte floor (round = bytes at 190-225 GB/s plus ~1000 small launches); round 7 found nothing for one user and +2..5 % at 9-16 users.
We still trail custom engine R's re-quantised numbers for one user (54.6 / 88.6, held 65.66 against 51.1 / 82.7, 61.33).

## Round 7 END STATE (17:00 UTC 2026-11-13)
- No window open (q38-win, q38-guard inactive), no Qwen container or helper script on any worker, DeepSeek verified after window U.
- Outages this round: R 1699 s, S 3632 s, T 3561 s, U 1924 s (R ended early by SIGTERM; S, T, U ended by themselves).
- Builds on all workers ~/q38/tfbin: kp22 (native FP images, not shipped), kp23, kp24, kp25 (sha fb1cd91c2a139b17 = patches 0101-0109 = Mac work/tf 52b32e1; branch q38-native-fp holds kp22's source).
  923b image tensorfold-qwen38:v1.1.4-eab8582 = 0101-0109 (prepare6.log). Old builds kp1..kp20 are still there (tidy: keep kp11, kp21, kp25).
  NOTE: 4e5c's build tree ~/q38/build/tf still has the native-FP kernel files of the microbench (q38_fp.cuh + two .cu): run Mac treecheck.sh pico@spark-4e5c and push the Mac files before building there.
- LEVER QUEUE for the next round (each exact, flag, measure with GPROF / KPROF at the user count it targets):
  U1 per-lane QSA attention launches at 9-16 streams (12 layers x 16 lanes x ~5 kernels): port upstream #1739 TF_FLASHNEXT_ATTN_LANES; est +1.5 % at 16 users
  U2 other per-lane loops in a shared pass (PLE conv + window shift a lane; the MTP chain pass's per-lane kernels; fn_row_best's other call sites): KPROF 16 users lists them
  U3 first token of short new prompts: port upstream #1866 KEEP_INPASS (needs their fused DeltaNet rows form) ; upstream figure -19 %
  U4 a public parallel drafter (incoai DFlash2 / DSpark) with an 8-bit body: measure accepted length at temperature 0 on our held prompts FIRST (vLLM recipe on a worker)
  U5 two-tier draft head (frequent rows first, the rest only when the top probability is low): <= 1.8 % one user (estimate)
  U6 6-bit coarse head copy (+1 % est.) only with a kernel that keeps ~230 GB/s
  Not worth repeating: anything in final_r7_levers.txt marked DEAD.
- RESUME COMMANDS
  cd ~/.hermes/cache/scratch/q38; tail -80 PROGRESS.md
  ./h.sh 'systemctl --user is-active q38-win q38-guard.timer; tail -3 ~/q38/windows.log; curl -s -o /dev/null -w "%{http_code}\n" 127.0.0.1:18900/health'
  build:   ./ship.sh NAME pico@spark-3d24 3 8g ; ./treecheck.sh pico@spark-3d24 ; then ./dist.sh NAME pico@spark-3d24 SYMBOL
  patch:   bash mkpatch.sh FROM_COMMIT 01NN-name.patch   (prints TREE_EQUAL) ; ./push.sh picom@spark-923b '~/q38/repo_tf/patches' repo/tensorfold/patches/01NN-...
  window:  ./h.sh 'systemctl --user reset-failed q38-win; systemd-run --user --unit=q38-win /bin/bash ~/q38/qw.sh LABEL 100 ~/q38/w_LABEL.sh'
  end early: ./h.sh 'pkill -TERM -f "^/bin/bash /home/pico/q38/q[w].sh"'
  spare lane in an open window: head ~/q38/x_e.sh WIN DEADLINE_EPOCH w2|w3   (stops on "restore begin"; no engine start after the deadline)
  documented path on 923b: bash ~/q38/kit/rs.sh [USERS16=1] [GDN_MANY=0 BEST_LANES=0 ...]
  microbench (inside a window, no model): bash ~/q38/mb/mbxbuild.sh mbx ; bash ~/q38/mb/mbrun.sh mbx 3
