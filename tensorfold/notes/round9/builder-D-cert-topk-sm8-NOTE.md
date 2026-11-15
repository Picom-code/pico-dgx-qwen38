# r9-cert-topk-sm8 (builder D, round 9, head spark-3caa): certified LM head for SAMPLED passes

Patch: `D9c-cumulative.patch` = `git diff 0b5cf9b f5209dd` (Mac q38/wt/D9, branch r9-D). sha256 ac9f022ac811cd86... = the tree that was BUILT AND MEASURED
last (head binary ~/q38/tfbin/D9c, sha256 3afe212327473c17). Applies cleanly on BASE 0b5cf9b (`git apply --check`).
D9c = D9b (commit 0c6dfb9, binary 771c2b0de0db2f91, on which most one-user numbers below were measured) + the row limit `CERT_TOPK_ROWS`; D9c one user re-measured:
sampled 48.73 / greedy 51.24, texts 40/40 + 40/40 identical to D9b's.

## What it is
Two flags, both off by default (flags off = base: texts identical, speed equal, see "inert" below):

| flag | what |
|---|---|
| `TF_FLASHNEXT_Q38_CERT_CODE=sm8` | the certified head's coarse copy holds sign-magnitude 7-bit integer codes with a power-of-two scale per 32 inputs (new qmmf mode SM8: MXFP8's container and the MXFP8 lane kernel's loop, only the code -> bf16 step differs) instead of E4M3 codes. Same 625 MiB. Residual 0.87 % of a row's norm (MXFP8 2.66 %), so the bound is about a third and a greedy pass needs ~10 exact tiles instead of ~47. |
| `TF_FLASHNEXT_Q38_CERT_TOPK=20` (0..64) | sampled streams whose keyed draw reads only their row's top_k best scores (0 < top_k <= K, no grammar mask, no logprobs) take the certified head too. The pass keeps every entry that can be among, or tie with, a row's top_k best (fn_cert_select_k). Needs CERT=1. Use it WITH `CERT_CODE=sm8` (with the MXFP8 copy 63 % of sampled windows would fall back: measured offline, table below). |

| `TF_FLASHNEXT_Q38_CERT_TOPK_ROWS=8` (code default 16: SET 8) | a top-k pass of more rows than this goes straight to the whole head. Needed: rows of DIFFERENT sampled streams share few exact tiles (~80 a row), so several sampled users in one pass would pay for the coarse copy and then most of the head (measured -3..-4 % at 16 sampled users without it). |

Proposed recipe defaults: `CERT_CODE=sm8`, `CERT_TOPK=20`, `CERT_TOPK_ROWS=8` (server default top_k is 20; a request with top_k above 20 or 0 takes the whole head as today).
Please change the code default of CERT_TOPK_ROWS from 16 to 8 when integrating, or export it in recipe.sh (8 was measured through the environment on D9c).

## Why it is exact
- The sampler is a keyed Gumbel-max draw (core/lanes/sampling.zig): the token at a position depends only on (seed, position, the row's top_k BF16 scores and
  their ids); temperature / top_p / min_p act inside those top_k. Nothing reads the rest of a sampled row (masks and logprobs are excluded by the gate).
- theta = a value with at least top_k entries whose LOWER bound (coarse - c h) is >= theta. So at least top_k exact scores are >= theta, and an entry whose UPPER
  bound is under theta less two bf16 steps is strictly below the top_k-th largest rounded score: not a candidate, not a tie. All other entries are computed
  by the head's own BF16 kernel on the BF16 weights (the round-4 mechanism, unchanged), the rest of the row is -inf. fn_row_cand / the host scan then select the
  same top_k values and columns as from the whole row, so the keyed draw returns the same token for the same seed.
- The bound itself: c[v] = residual norm of row v of the coarse copy (+1e-3 of the row norm for float rounding), as before; only the code changed.

## Measured (head, binary D9b, one user 3 x 262K, one shared window, 2026-11-14 00:00-02:00 UTC; ruler = head ~/q38/kit-D/sq.py: 40 held chat prompts x 256 tokens, request "seed" 9000 + id, server defaults top_k 20 / top_p 0.95)
Check mode (`CERT=2`, sampled + greedy, 40 + 40 prompts): 3,254 top-k passes, 0 fallbacks; 11,462 sampled rows: the 20 best (values + columns) of the certified
row equal the whole head's in all of them (topk-mismatch 0); the bound |coarse - exact| <= c h tested on EVERY entry of every checked row: 0 violations of
3.23 billion; best-mismatch 0, value-mismatch 0 of 315 M recomputed entries.

| pair, same seeds (reference = flags off, first / last) | texts | reference | sm8 + topk | change |
|---|---|---|---|---|
| T 0.7, thinking off | 40/40 identical | 47.12 / 46.76 | 49.00 (48.92 second engine) | +4.0 % / +4.8 % (per prompt +3.83 +- 0.18) |
| T 0.7, thinking on (reasoning + answer compared) | 40/40 identical | 50.25 / 49.76 | 51.70 (52.22) | +2.9 % / +3.9 % |
| greedy, thinking off | 40/40 identical | 50.55 / 50.56 | 51.43 | +1.7 % |
| sparkDash-method prose T0.7 think-on, 6 seeded rounds | 6/6 identical | 53.00 | 55.13 | +4.0 % (+4.03 +- 0.25) |
| SM8 copy alone (no TOPK): greedy / sampled | 40/40 + 40/40 | 50.55 / 47.12 | 51.31 / 47.03 | +1.5 % / 0 |

Levers off (MiaAI configuration) on the same ruler: 45.55 / 48.55 / 46.71 (sampled off / on / greedy); recipe default at base: 47.17 / 50.25 / 50.57.
Engine counters, sampled passes: 359 exact tiles of 3,880 a pass (9.3 %), 57 launches a pass, 0 whole-head fallbacks in 10,558 passes.

## Final-suite identity and speed (flags off vs CERT_CODE=sm8 CERT_TOPK=20; binary D9b unless said)
| check | result |
|---|---|
| ident set, greedy | 51/51 identical |
| held 107 x 256, greedy, one user | 107/107 identical; 63.41 -> 64.18 tok/s (+1.2 %) |
| sampled T0.7 think off, 40 seeded (three engine pairs) | 40/40 identical each time; +3.7..+4.8 % |
| sampled T0.7 think on, 40 seeded | 40/40 identical; +2.9..+4.9 % |
| 16 users greedy (16 x 32K): held 107 with 16 in flight | 107/107 identical (also to the one-user texts); 189.2 -> 191.9; distinct 16: 278.0 / 287.6 -> 283.8 / 287.0 / 284.2 |
| 16 users SAMPLED (T0.7, seed 4242, held 107, 16 in flight; D9c), run a (cold) / run b | reference 189.7 / 190.2; no row limit 181.3 / 185.1 (NEGATIVE); TOPK_ROWS=16: 182.5 / 188.9; TOPK_ROWS=8: 188.9 / 190.2 (zero); texts 107/107 identical in all |
| sparkDash 1.8.8 itself (greedy protocol), levers off -> this, 2 runs x 5 rounds | prose 56.61, 56.21 -> 61.43, 61.14; code 91.96, 91.61 -> 99.81, 98.72; structured 105.46, 105.04 -> 114.19, 113.08 |
Not run by me: edit / agent sets, waves, needle, image, qeval, held-320 (integrator's suite). The change touches only the LM head of verify passes.

## Offline study behind it (head ~/q38/kit-D/hstudyk.py, 6,025 dumped head rows of sampled passes, 7-row windows)
| coarse copy | bound | exact tiles a window at k = 1 | k = 20 | share of head bytes at k = 20 | windows over the fallback line |
|---|---|---|---|---|---|
| MXFP8 (base) | +-3.1 logits | 180 | 1,662 | 0.93 | 63 % |
| int8, scale per 32 (ideal scale) | +-0.74 | 15 | 205 | 0.55 | 0 |
The shipped SM8 code uses a power-of-two scale (so that MXFP8's scale bytes and kernel loop serve unchanged): 359 tiles a real pass at k = 20.

## Also in the patch (all default OFF; measured zero or negative: do NOT turn on; kept only because the measured binary contains them)
- `TF_FLASHNEXT_Q38_EXP_WAVES=1` (upstream #1351's wave form for the NVFP4 decode experts) + `TF_FLASHNEXT_Q38_RMS_TILES=10` (port of upstream #1702, tiled
  10,240-wide MTP norm): texts identical 40/40 + 40/40; greedy +0.14 % against the first reference and -0.2 % against the last, sampled 0.0 %: ZERO.
- `TF_FLASHNEXT_Q38_SGATE=M,A,B,C` (+ knob-file words sgate / sgate_c; QSAMP log lines under TF_STEP_LOG): another stop-rule input for sampled chains.
  Measured in one engine: head's top probability at 0.35: -1.3 % (thinking off), +0.05 % (on): ZERO to negative. MiaAI's rule stays.
You may drop those hunks (nvfp4_experts.cu, experts.zig, norms.cu, norms.zig, cuda_knobs.zig SGate, cuda_mtp.zig drawStats / sampledGate, the three call
sites in cuda_mtp_ops.zig); the lever itself is: qmmf.cu (SM8), qmmf.zig (Weight.sm8, sm8_fns), flashnext_batch.cu (fn_cert_select_k, fn_cert_bound),
cuda_batch_state.zig, cuda_cert.zig, cuda_lanes_verify.zig. A tree without the dead hunks has NOT been built by me.

## Limits / what stays on the whole head
top_k = 0 or > CERT_TOPK, grammar masks, logprobs, first token of a request, passes over 1,500 exact tiles (many sampled users in one wide pass: falls back
cleanly, counted in the `q38 cert topk:` line as whole-head).

## Tools (copies here)
sq.py (seeded sampled ruler: run / sd / cmp), gsim.py (stop-rule study from TF_STEP_LOG logs), hstudyk.py (top-k survivor study), run9.sh / w9a.sh / w9e.sh
(head run scripts with the head's memory rules).
