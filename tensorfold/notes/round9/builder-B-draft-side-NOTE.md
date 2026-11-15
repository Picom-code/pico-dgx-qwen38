# r9 draft side, one user (builder B): four flags, all default OFF, all drafts-only (texts identical)

UPDATE 03:57 UTC (for the integrator; the patch is UNCHANGED, sha256 c850019bddac1789..., only settings and measurements are new):
1. Depth: `TF_FLASHNEXT_DRAFTS=15` (a 16-row window) is better than 12: DEEP=0.9,6 at 15 drafts +2.47 % against +2.19 % at 12; 24 drafts = 15 (+2.38 %).
   Recommended one-user setting: `TF_FLASHNEXT_DRAFTS=15 TF_FLASHNEXT_Q38_DEEP=0.9,6 TF_FLASHNEXT_Q38_STOP2=0.5` (+ `HEAD2_FAST=3` with HEAD2).
2. `TF_FLASHNEXT_Q38_HEAD2_SAMPLED=1` is now MEASURED with builder D's seeded ruler: +1.83 % on sampled requests, texts identical 30/30.
3. All test windows are done (tables below); lease released.
4. UPDATE 05:45: P = 0.95 at 15 drafts measured +1.9 % (below 0.9: +2.47 %), so 0.9 stays. Sampled requests (D's ruler, HEAD2 off) with
   DEEP + STOP2: chat +0.5 %, code +3.85 %, texts identical 30/30 and 20/20. A sequence-aware gate simulator (`seqsim.py`, `seqsim_S5.txt`)
   finds no schedule materially better than the measured setting.

Patch: `r9-B-draft-side.patch` = `git diff 0b5cf9b HEAD` on Mac `q38/wt/B/tf` branch `q38-B9` (commits 6c46c35 B8, b8956b7 B9, 2708a5b B10).
Checked: base 0b5cf9b + patch = git tree 2168edda641ae9c7dc83ded225888617f9c9c896 = HEAD. sha256 of the patch starts c850019bddac1789.
Binaries on spark-3d24: `~/q38/tfbin/B8` (sha 8d74625319d7e60e: STOP2 + HEAD2_FAST), `B9` (49164345a523c1ad: + DEEP), `B10` (da979a99fddd03c8 = this patch).
Files touched: zig/kernels/cuda/flashnext_mtp.cu (2 new kernels, nothing existing changed), cuda_knobs.zig, cuda_mtp.zig, cuda_mtp_ops.zig, cuda_lanes_draft.zig.

Why lossless: every flag changes only which draft tokens the MTP chain PROPOSES (how many, or how the draft head scores them).
The verify pass and the accept rule are untouched: a token is kept only if it is the main model's own greedy token.

## Flags
| flag | what | measured (held-104 x 384, temp 0, 3d24, paired in one window) |
|---|---|---|
| `TF_FLASHNEXT_DRAFTS=15` + `TF_FLASHNEXT_Q38_DEEP=0.9,6` | a lone chain goes past 6 drafts (up to DRAFTS) only while the running product of its drafts' probabilities is >= 0.9; chains of a shared round stay at 6 | +2.47 % at 15 drafts (chat +0.5, code +6.3, math +9.1); +2.19 % at 12 |
| `TF_FLASHNEXT_Q38_STOP2=0.5` | a lone chain does not compute its next step once the product is below 0.5 (that step's draft would cross the 0.4 gate more often than not and be thrown away) | +0.56 % HEAD2 off; +0.86 % on top of HEAD2 + FAST |
| `TF_FLASHNEXT_Q38_HEAD2_FAST=3` (needs HEAD2) | bit 1: the two-stage head's preview with the rotated input as int8 and dp4a (`fn_q38_head2_scan2`); bit 2: the rotation as bf16 (`fn_q38_head2_rot2`) | +0.94 % over HEAD2 |
| `TF_FLASHNEXT_Q38_HEAD2_SAMPLED=1` (needs HEAD2) | a lone SAMPLED chain drafts with the two-stage head too (its keyed draw takes its top_k candidates from the preview row with the 933 re-scored rows in place) | +1.83 % on sampled requests (temperature 0.7, seeded), texts identical 30/30 |
Start-up line for FAST: `q38 head2: fast 3 (1 = integer preview, 2 = bf16 rotation)`.

## Measured tables (all runs: identity set 51 + held 107, COPY_LOG on; texts compared with the first reference of the window)
Early stop, HEAD2 off (B8), in run order:
| run | held ALL | chat | code | math | drafts / kept a round | texts |
|---|---|---|---|---|---|---|
| ref1 | 62.49 | 53.59 | 85.59 | 79.30 | 3.625 / 2.293 | |
| STOP2=0.50 | 62.72 | 53.88 | 85.64 | 79.26 | 3.483 / 2.293 | ident 51/51, held 107/107 |
| STOP2=0.55 | 62.57 | 53.72 | 85.39 | 79.29 | 3.371 / 2.249 | ident 51/51, held 107/107 |
| ref2 | 62.12 | 53.28 | 85.09 | 78.78 | 3.625 / 2.293 | ident 51/51, held 107/107 |
The references drifted -0.6 % in that hour; against the reference interpolated to the run's time STOP2=0.5 is +0.56 %, 0.55 +0.53 %.

HEAD2 on (B8), in run order:
| run | held ALL | chat | code | math | drafts / kept | texts vs HEAD2-off ref1 |
|---|---|---|---|---|---|---|
| HEAD2 ref1 | 63.27 | 54.20 | 86.89 | 80.49 | 3.630 / 2.279 | 51/51, 107/107 |
| + FAST=3 | 63.87 | 54.69 | 87.90 | 81.21 | 3.630 / 2.279 | 51/51, 107/107 |
| + FAST=3 + STOP2=0.5 | 64.42 | 55.23 | 88.23 | 81.91 | 3.484 / 2.281 | 51/51, 107/107 |
| HEAD2 ref2 | 63.28 | 54.21 | 86.87 | 80.54 | 3.630 / 2.279 | 51/51, 107/107 |

Deep chains (B9, HEAD2 off), in run order:
| run | held ALL | chat | code | math | drafts / kept | texts vs d_ref1 |
|---|---|---|---|---|---|---|
| d_ref1 | 62.19 | 53.33 | 85.24 | 78.89 | 3.625 / 2.293 | |
| DRAFTS=12, DEEP=0.85,6 | 63.28 | 53.41 | 89.48 | 84.05 | 3.984 / 2.469 | 51/51, 107/107 |
| DRAFTS=12, DEEP=0.9,6 | 63.46 | 53.51 | 90.05 | 84.37 | 3.919 / 2.444 | 51/51, 107/107 |
| DRAFTS=8, DEEP=0.8,6 | 62.52 | 53.25 | 86.90 | 80.76 | 3.883 / 2.449 | 51/51, 107/107 |
| d_ref2 | 62.01 | 53.16 | 84.99 | 78.74 | 3.625 / 2.293 | 51/51, 107/107 |
Against the reference interpolated to each run's time: 0.85 at 12 +1.83 %, 0.9 at 12 +2.19 %, 0.8 at 8 +0.75 %.

Deeper maxima and the HEAD2-off stack on B10 (= this patch), in run order:
| run | held ALL | chat | code | math | drafts / kept | texts vs e_ref1 |
|---|---|---|---|---|---|---|
| e_ref1 | 62.06 | 53.21 | 85.08 | 78.71 | 3.625 / 2.293 | |
| DRAFTS=15, DEEP=0.9,6 | 63.66 | 53.52 | 90.51 | 85.94 | 3.977 / 2.470 | 51/51, 107/107 |
| DRAFTS=24, DEEP=0.9,6 | 63.67 | 53.57 | 90.16 | 85.98 | 4.032 / 2.494 | 51/51, 107/107 |
| STOP2=0.5 + DRAFTS=12, DEEP=0.9,6 | 63.99 | 54.05 | 90.54 | 84.62 | 3.688 / 2.357 | 51/51, 107/107 |
| e_ref2 | 62.32 | 53.45 | 85.37 | 79.02 | 3.625 / 2.293 | 51/51, 107/107 |
Interpolated: DEEP at 15 +2.47 % (chat +0.5, code +6.3, math +9.1), at 24 +2.38 %, STOP2 + DEEP at 12 +2.79 % (chat +1.2, code +6.1, math +7.2).
The start line stays "5 streams at once, 4.76 GiB each at a 262144-token window" with DRAFTS=24.

Sampled chain (B10, HEAD2 + FAST=3 in both; `sq.py run OUT s07off 30 256 chat`, seed 9000 + id), `sq.py cmp`:
HEAD2_SAMPLED off 47.07 tok/s -> on 47.93 (+1.83 %; per-prompt ratio +1.94 % +- 0.37); identical 30/30; 0 error lines.
(Builder D's note proposed candidates from the re-scored rows only; this version leaves the preview scores in the row as well: same idea, no new kernel.)

0 panic / segfault / error lines in every run.

## Microbench of the preview kernels (3d24 GPU, real head2 file, 1,500 real head inputs, engine launch shapes, cache flushed before each launch)
| kernel | us |
|---|---|
| `fn_q38_head2_scan` (stock, 25.5 MB) | 236 |
| `fn_q38_head2_scan2` (int8 input, dp4a) | 116 (220 GB/s: the bus; 65 when cached) |
| `fn_q38_head2_rot` fp32 / `fn_q38_head2_rot2` bf16 | 26 / 15 |
97.2 % of 256-row blocks keep the same 3 candidates; the stock preview's best row is among the candidates for all 1,500 inputs.
Source: `mbh2.cu`, `mbh2build.sh`, `mbh2run.sh` (run only inside a window).

## Why DEEP works (round-8 always-8 log, `gsim.py` + PROGRESS-B.md)
Going past 6 drafts pays only when the product after 6 drafts is still >= ~0.8: 11.8 % of rounds have >= 0.9 and keep 82 % of their first
6 drafts; they are 43 % of code rounds, 55 % of math rounds, 9 % of chat rounds. Raising the depth for every chain (0.4/8, round 7) was
+0.3 % because the chains between 0.4 and 0.6 lose what the confident ones win.
In an easy stretch a round of D drafts costs (38.6 + 4.1 D) ms for D + 1 tokens: 9.0 ms a token at 6, 6.3 at 15. The head's product is
over-confident as a whole-chain probability (measured at drafts 6-8: product 0.9-0.95 -> 59 % of chains whole, 0.95-0.98 -> 71 %, >= 0.99 -> 92 %;
a step with its own p >= 0.98 is right 93-97 % of the time), so real chains decay ~4 % a step and nothing is gained past ~15.
P = 0.95 at 15 drafts (B10, between references 62.22 / 62.11): 63.35 (53.34 / 89.75 / 85.09) = +1.9 %; ident 51/51, held 107/107.
Sampled requests, HEAD2 off, default -> DRAFTS=15 + DEEP=0.9,6 + STOP2=0.5 (`sq.py`, temperature 0.7, seeded): 30 chat prompts 46.73 -> 46.97
(+0.5 %), identical 30/30; 20 code prompts 65.99 -> 68.53 (+3.85 %), identical 20/20.
Simulated only (`seqsim.py` on an always-31 log, generous to deeper policies by 0.4-1 point): BASE 3-8, P 0.8-0.98, per-depth schedules, a
recalibrated product, gate 0.3-0.5: nothing more than +0.5 point above the measured setting.

## Not in the recipe yet (integrator)
- These flags act on lone chains. Not run by builder B: 3-prompt bench, sparkDash method, edit set, agent set, 16 users (DEEP caps shared
  rounds at BASE; `TF_FLASHNEXT_DRAFTS=12` also sizes each stream's draft buffers: check the start line "N streams at once" stays 5 at 262K).
- A request that drafts alone inside a round of 2-5 streams also follows the DEEP rule (product rule active up to PRODUCT_STREAMS).

## Measured dead after the handoff (not in the patch)
`TF_FLASHNEXT_Q38_DEEP_MANY=1` (branch q38-B9, build B11): shared rounds under the product rule following DEEP + STOP2. 4 users: 154-159 -> 154-156 on distinct
code, held with 4 in flight 113.0 -> 113.4; 16 users 274-279 -> 272-274, held with 16 in flight 186.7 -> 186.6; texts 107/107 identical in flight. Zero: do not port.
The lone-chain settings (DRAFTS=15, DEEP=0.9,6, STOP2=0.5) leave 16 users unchanged (274.1 / 279.4 / 274.5 on 3d24).
