# 0111 two-stage draft head (builder B) : one user +1.6 % (measured, paired), texts identical

Lossless by construction: it changes only which tokens the MTP chain PROPOSES and the probability the draft gate reads. Every token is
still the main model's own greedy pick (verify pass unchanged). Idea credit: SVD-softmax (Shim et al., NeurIPS 2017: preview in a rotated
basis, exact scores for a short list); here the rotation is the PCA of the head's INPUTS, not the SVD of the head (that one fails, below).

## What it does
The MTP chain's draft head scores 79,591 shortlist rows of 4-bit weights every chain step: 102 MB, 558 us (KPROF), ~4.2 steps a round.
With `TF_FLASHNEXT_Q38_HEAD2=FILE` a lone greedy chain instead
1. rotates the head's input row into a fixed basis and keeps 512 coordinates (`fn_q38_head2_rot`, 5 MB),
2. scores every row from the rotated rows' first 512 coordinates, 4-bit again: 25.5 MB (`fn_q38_head2_scan`),
3. re-scores the 3 best rows of each 256-row block (933 rows) with the head's own 4-bit rows (`fn_q38_head2_exact2`),
4. takes best + log-sum-exp from those and the blocks' tails (`fn_q38_head2_fin`); `fn_row_best` is not launched.
Sampled requests, shared rounds (2+ users drafting together) and prompt passes keep the full head.

## Flags
| variable | default | meaning |
|---|---|---|
| `TF_FLASHNEXT_Q38_HEAD2=FILE` | unset = off | the two-stage file (made by `mkhead2.py`); a bad or missing file prints `q38 head2: OFF (...)` and the full head runs |
| `TF_FLASHNEXT_Q38_HEAD2_MODE` | `block` | `topk`: re-score the preview's global best K through `fn_row_cand` + `fn_row_best` (first version, same gain, slower step) |
| `TF_FLASHNEXT_Q38_HEAD2_C` | 1.02 block / 1.06 topk | odds factor on the gate's probability (the preview under-counts tail mass; measured by `calB2.py`) |
| `TF_FLASHNEXT_Q38_HEAD2_K` | 32 | top-k form only (64 was used in tests) |
| `TF_FLASHNEXT_Q38_HEAD2_ROWS` | 8 | top-k form only: rows a warp of the preview kernel |
Start-up line to check: `q38 head2: on, rank 512, block form, re-score 933 of 79591 rows`. Costs +151 MiB device memory.

## Build and file
- Patches on base 52b32e1 (0101-0109): `0110-q38-draft-study-logging.patch` (handoff/B/draft-study-logging) then
  `0111-q38-two-stage-draft-head.patch`. Checked: base + 0110 + 0111 = git tree 7cd70d95 = build B7 (sha 25683938f4de8020,
  spark-3d24 `~/q38/tfbin/B7`). Source: Mac `q38/wt/B/tf`, branch `q38-B`, commits 5436541, c1ed232, 305e4f2.
- File (30.7 MB, derived from the checkpoint's own lm_head rows, so made locally, not shipped):
  `python3 mkhead2.py MODELDIR draft_vocab_default.txt head2_rot_r512.rot OUT.bin 512` in a CUDA pytorch container
  (`nvcr.io/nvidia/pytorch:26.07-py3 --gpus all`, ~1 min). `head2_rot_r512.rot` (5.2 MB, sha256 127645d7...) is in this folder: the
  rotation only (statistics of head inputs from 104 held prompts, no weights). On the GB10 GPU this reproduces the TESTED file bit for
  bit (sha256 c1f89b71...; checked). A CPU run gives a slightly different file (float order; 1.8 M bytes differ): fine as drafts, but untested.
  `draft_vocab_default.txt` = `zig/src/families/flashnext/draft_vocab_default.txt`. A new rotation: run with `TF_STEP_LOG=2
  TF_FLASHNEXT_Q38_MTPDUMP2=/prof/x.mix` (patch 0110), then `mkhead2.py MODELDIR VOCAB x.mix OUT.bin 512 ENGINE.log`.
- Engine: mount the file and add `-e TF_FLASHNEXT_Q38_HEAD2=/path/in/container/head2_r512.bin`.

## Measured (spark-3d24, one window each, references first and last, 104 held prompts x 384 tokens, temp 0, COPY_LOG on everywhere)
| test (binary) | ref first | lever on | ref last | gain vs mean ref | chat / code / math on |
|---|---|---|---|---|---|
| T1 (B4, top-k K64, no calibration) | 61.75 | 62.38 | 61.78 | +1.0 % | 53.34 / 86.04 / 79.65 |
| T5 (B6, top-k K64, c 1.06) | 62.20 | 63.05 | 61.90 | +1.6 % | 53.93 / 86.85 / 80.53 |
| T5 (B6, top-k K64, c 1.00) | 62.20 | 62.82 | 61.90 | +1.2 % | 53.72 / 86.58 / 80.27 |
| T6 (B7, block form, c 1.02) = handoff default | 61.85 | 63.02 | 62.15 | +1.6 % (+1.9 / +1.4 vs each ref) | 53.93 / 86.65 / 80.44 |
T6 references by category: chat 53.01 / 53.26, code 84.83 / 85.24, math 78.68 / 79.09 => chat +1.5 %, code +1.9 %, math +2.0 %.
Chain step (TF_HOSTPROF mtpstat, identity set): 1.67-1.71 ms off -> 1.43-1.46 top-k form -> 1.35-1.39 block form.
Round statistics (held): off 3.625 drafts / 2.293 kept; block form 3.630 / 2.279 (0.6 % fewer kept drafts: the price of the preview).
The references drift ~0.5 % across an hour on this box; all comparisons are within one binary and one window.

## Identity proof
Each lever-on run and each last reference against the first reference of its test: identity set 51/51, held 107/107 texts identical
(T1, T5 c 1.06, T5 c 1.00, T6 block). 0 panic / segfault / error lines. Raw: Mac `q38/results/rB/B_T1_*, B_T5_*, B_T6_*`.
NOT yet run by builder B (for the integrator): 3-prompt bench x9, sparkDash method, edit set, agent set, 16 users (the lever is off in
shared rounds by design; only lone chains take it), image request, long prompts.

## Offline evidence (handoff/B/study/headstudy.txt, kit/mkhead2.py, calB2.py), replay of 30,870 real head inputs
| variant | pick == full 4-bit head |
|---|---|
| shipped 4-bit head vs exact BF16 rows (context) | 96.96 % |
| rotation = SVD of the head rows, rank 512, best 64 | 76.5 % (dead) |
| rotation = PCA of head inputs, rank 256, best 64 | 97.6 % (in engine: kept drafts -4.9 %, slower than off: dead) |
| rank 512, best 64 (rotation fitted on other requests) | 99.0 % |
| rank 512, 3 best of each 256-row block | 99.46 % |

## Limits / honest notes
- The rotation was fitted on our 104 held prompts (English chat / code / math). On very different text the preview may agree less:
  drafts get worse (slower), never wrong. Re-fit with broader prompts before making it a recipe default for everyone.
- What is left in the head step (KPROF): preview ~200 us for 25.5 MB (120 GB/s: a warp-a-row kernel, not a cp.async tile kernel),
  rotation 34 us (fp32, could be bf16). A tuned preview kernel could give another ~0.1 ms a step (~+0.7 %), estimate.
