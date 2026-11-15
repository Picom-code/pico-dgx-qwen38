# Builder C handoff: first-token levers (spark-4e5c, 2026-11-13)

One cumulative patch against the base (work/tf 52b32e1 = patches 0101-0109): `C-stack.patch`
(`mkhandoffC.sh` regenerates it from Mac `q38/wt/C`, branch builder-C, and proves base + patch = the built tree).
Every lever is behind its own flag and OFF unless the flag is set, so the patch can be integrated whole.
Build: `./shipC.sh NAME` (whole tree to 4e5c `~/q38/build-C/tf`, sha256 of all 7,353 files checked, zig build in the
NVIDIA PyTorch container, ~3 min) -> `~/q38/tfbin/NAME`.
Patch: `C-stack.patch` sha256 fa62a400a3293ca8, 947 lines, 13 files; wt/C HEAD 4c8eedb; `mkhandoffC.sh` printed TREE_EQUAL (base + patch = that tree).
Binary built from exactly that tree: C9, sha d14ca644ac3405d6 (the build is not bit-reproducible: C7, sha eba8f7bb30b2eeb5, came from the same source).
C9 with the three flags as recommended passed the identity set against the flags-off reference (window 9, below).
Earlier binaries in the tables: C5 5812074a53e5de5e, C6 30d28bb25f0310b0, C8 640b02a74dedb731 (= C7 + a flag that was measured zero and removed).
All numbers MEASURED on spark-4e5c, recipe one-user configuration (5 x 262,144, fp8 KV, vision on, recipe lever
environment of patches 0101-0109), unless a line says otherwise. Raw logs: box `~/q38/runs/C_w*`, Mac `q38/results/rC`.

## Lever 1: `TF_FLASHNEXT_Q38_KEEP_INPASS=4096`  (recommended recipe default)
What: a prompt-cache mark inside a lone prompt pass of at most N rows no longer cuts the pass in two. The pass runs
through; the mark's kept state is filled on the way. Each DeltaNet layer runs the rows before the mark and the rows
after it as the cut pass's two chunks would (the same kernels on the same inputs; the state between them is copied to
the kept state), the conv windows and the n-gram tail of the kept state are the pre-pass windows moved on by the
mark's rows, and the attention rows are copied after the pass. The cache gets the same state it got before.
Why it pays: the chat template puts a mark 7 tokens before the end of EVERY prompt (end of the user text), so every
request ran a second 7-row pass that costs 58-72 ms (a pass is byte-bound: 7 rows cost 58 ms, 16 rows 78 ms).
Credit: the idea is upstream TensorFold #1866 (TF_FLASHNEXT_KEEP_INPASS, inside its fused DeltaNet kernel, which this
tree's base lacks); this is our own implementation on the base's kernels.
Files: cuda_batch.zig (markRow, gdnSegment, pleBlock), cuda_batch_state.zig (Hooks.mark), cuda_snap.zig (prepare /
finish / layout), cuda_lanes.zig (step, snapSave), cuda_mtp_ops.zig (prefillChunkMark), cuda_prompt.zig, cuda_knobs.zig.
Limits: one mark a pass (a second one cuts as before); one GPU, no images in the stream (those cut as before).

## Lever 2: `TF_FLASHNEXT_Q38_LANE_REUSE=N`  (recommended: N = --parallel; 5 for the one-user profile, 16 for USERS16)
What: a released stream's fixed-size device buffers are kept for the next stream instead of freed and allocated again:
the sequence's 73 buffers (36 conv windows, 36 DeltaNet states, n-gram tail, 110 MiB) and the MTP draft head's ~45
buffers plus its host array. Up to N idle sets; every reused buffer is zero-filled on the compute stream before the
stream's first pass. Measured with PLOG: a lane's opening is 22.5 ms a request (sequence 5.9 + draft head 16.5) and
0.7 ms with the flag.
Cost: up to N x (110 MiB + ~0.5 GiB) stays allocated while idle (memory the lanes take anyway when busy).
At process exit the idle host arrays are reported as leaks by the debug allocator (harmless; no shutdown hook yet).
Credit: upstream TensorFold has the idea for a lane's fixed parts as TF_FLASHNEXT_LANE_PARTS (#1763); own form here.

## Lever 3: `TENSORFOLD_PREFIX_WAIT_MS=60000`  (port of upstream #1321; see the results table for the verdict)
What: a queued request whose prompt shares a cut another prompt is still filling waits for that fill to keep the cut,
then resumes it, instead of prefilling the shared prefix itself. Ported from TensorFold main (lane_host_prefix.zig,
Store.wouldServe, Job.reached / prefix, admit.pick). Unset = off here (upstream: on, 60 s).

## Measuring aids in the patch (values never change)
`TF_FLASHNEXT_Q38_PLOG=1` one line a prompt chunk / lane opening / first token / host fill; `TF_FLASHNEXT_Q38_GPROF=2`
also prints every prompt chunk's GPU ms by group; `TF_FLASHNEXT_Q38_KNOBFILE=/path` re-reads `keep_inpass=`,
`lane_reuse=`, `prefix_wait_ms=` from a file at every use (paired timing in one running engine).

## For the integrator (builder A)
- recipe.sh defaults proposed: `TF_FLASHNEXT_Q38_KEEP_INPASS=4096`, `TF_FLASHNEXT_Q38_LANE_REUSE=$PARALLEL` (5; 16 with USERS16),
  `TENSORFOLD_PREFIX_WAIT_MS=60000`.
- a new profile for agents (several long conversations at once), no code needed: `PARALLEL=5` + `TF_FLASHNEXT_Q38_POOL_CTX=131072`
  (the five lanes share a pool of 5 x 131,072 tokens; one stream can still take the whole 262,144 window, all five together 655,360 tokens;
  the prompt cache gets 13-16 GiB instead of 4). Follow-up turns of 5 agents at 56K: 85 s -> 1.0 s. On the default profile a multi-agent
  load re-prefills every turn: say so in the README next to the one-user numbers.
- everything here was measured on spark-4e5c only; re-run the documented path on 923b (`qk.py ttft`, held set, ident) before quoting.
- the prompt-cache budget is computed from free memory at start: a build or any other job beside a starting engine shrinks it
  (4.3 GiB -> 1.2 GiB seen once), and then even short chats miss the cache.
- test kit: Mac `q38/kitC` (box `~/q38/kit-C`): `tt.py` (short / prefill / needle / agent / burst / cmp, deterministic prompts), `gp.py`
  (GPROF=2 chunk table), `kp.py` (KPROF tables), `engC.sh` (variants one / u16 / agN), `runW*.sh` (the windows as run).

RESULTS: RESULTS.txt in this directory. Not built, with the design: NEXT-qsa-certified-select.md.
