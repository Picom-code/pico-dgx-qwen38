# Builder D, round 10 handoff: one-launch gather for the certified head, half-unit decode experts, top-k row limit, knob words

Patch: `D10e-cumulative.patch` (ONE cumulative patch against integ9 7061396; `git apply --check` passes on Mac q38/wt/I9).
Source: Mac q38/wt/D10, branch r10-D, commit f0552a0. Built on the head as ~/q38/tfbin/D10e, binary sha256 a8f8b3f767ca9356
(tensorfold-native). Patch sha256: see `SHA256` in this directory. Everything below is MEASURED on the head (spark-3caa), 2026-11-14,
official FP4-Local checkpoint, recipe round-9 defaults, unless a line says otherwise.

## What is in the patch

```
+-----------------------------------+---------+-------------------------------------------------------------------------------+
| flag                              | default | what                                                                          |
+-----------------------------------+---------+-------------------------------------------------------------------------------+
| TF_FLASHNEXT_Q38_CERT_GATHER=1    | off     | the certified head's exact tiles in ONE launch a row tile (the head's own     |
|  (recommend: recipe default ON)   |         | BF16 kernel takes its tile index from a list; only the flagged tiles) instead |
|                                   |         | of one launch a run of neighbouring tiles (54-57 launches a sampled pass)     |
| TF_FLASHNEXT_Q38_EXP_HALF=1       | off     | port of upstream TensorFold #1351's nvfp4_expert_half: two warps a unit for   |
|  (recommend: stay off, or on for  |         | decode expert calls whose half units all fit the resident warps (one-row      |
|   --no-drafts deployments)        |         | gate-up: 220 units -> 440 on 768 warps). 2 = fp32 down too (never fits: inert)|
| TF_FLASHNEXT_Q38_CERT_TOPK_ROWS=N | 8       | a top-k (sampled) pass of more than N rows goes straight to the whole head    |
|  (already the code default)       |         | (integ9 has no limit: 16 sampled users lose 1.6 %). 0 = no limit.             |
| TF_FLASHNEXT_Q38_KNOBFILE words   | aid     | cert, cert_topk, cert_topk_rows, cert_gather, exp_half, hcf, hcf_half,        |
|  (measuring aid; off unless set)  |         | half_below, mtp_last, late_skip, head2, head2_sampled, deep_off, stop2,       |
|                                   |         | gdn_many, best_lanes, attn_lanes (+ the round-9 words). The file is now       |
|                                   |         | parsed once every 200 ms (a knob read = a clock read + a short scan).         |
+-----------------------------------+---------+-------------------------------------------------------------------------------+
```

## Measured (lever value = time with the lever off / reference time - 1; knob-file pairs inside one engine, reference twice)

```
+---------------------------------------------+------------------------+-----------------------+---------------------------------+
| bench (D10e, levers on in the environment)  | CERT_GATHER            | EXP_HALF              | ruler noise (ref2 vs ref, 95 %) |
+---------------------------------------------+------------------------+-----------------------+---------------------------------+
| T0.7 seeded, 40 chat prompts x 256          | +1.61 % (+1.56 +- 0.21)| -0.12 % (+-0.22)      | +-0.10                          |
|   same on D10c (form 4), 40 prompts         | +1.67 % (+1.59 +- 0.18)| -0.01 % (+-0.16)      | +-0.10                          |
| greedy held-104 x 256                       | +0.16 % (+-0.29)       | +0.05 % (+-0.25)      | +-0.16                          |
| no-draft streams, 40 chat x 256 (repeats)   | not measured           | +1.05 % +- 0.09       | +0.13 +- 0.12                   |
|   same on D10c, 24 x 128                    |                        | +1.01 % +- 0.42       | +0.08 +- 0.17                   |
| 16 users (16 x 32K), both on                | distinct-16 277.5 (cold) / 293.7, held 16 in flight 200.1 (references 198.5-200.0)     |
+---------------------------------------------+------------------------+-----------------------+---------------------------------+
```
First and last references are inside every row (each prompt runs reference, variants, reference again, order rotated by prompt).

Reading: the gather is a SAMPLED-request lever (+1.6 %; estimate before building was +0.4 %: the flagged-only list also drops the
tiles between joined runs). It is zero for greedy text (a greedy pass on the SM8 copy has ~10 exact tiles in a few launches).
EXP_HALF is +1.0 % for one-row passes only (no-draft streams); zero for drafted streams, greedy or sampled (upstream reports +0.6 %
one stream). Upstream #1351's wave form was measured zero in round 9.

## Identity proof (D10e with both flags on, against D10a = integ9 + knob words)

- identity set 51/51; held 107/107 (greedy, 256 tokens); seeded T0.7 40/40; every knob variant inside the pairs identical to its
  reference (39/39 sampled, 104/104 held, no-draft 24/24 and 40/40); 16 in flight: held 107/107 identical to D10a's 16-user reference.
- Device check (CERT=2 with the gather on): 3,000 certified calls, 10,627 rows: best-mismatch 0, value-mismatch 0 of 94.8 M
  recomputed entries; 1,907 top-k passes, 6,585 sampled rows: the 20 best values + columns equal the whole head's in all;
  bound violations 0 of 2.38 billion; whole-head fallbacks 0. 0 crash lines in every engine.
- 16 sampled users (knob run on D10a): CERT_TOPK_ROWS=8: 190.7; top-k off 190.6; no limit (integ9) 187.7. Texts 107/107.

## A trap worth keeping (cost me a window hour)

The first gather build copied the tile list with a SYNCHRONOUS cuMemcpyHtoD right before the launch and gave intermittently WRONG text
(D10b: identity set 2/51; bisect on D10c: sync copy 7/10, 3/8, 27/39 prompts right; the same list sent with cuMemcpyHtoDAsync on
the pass's stream: 10/10, 39/39, then everything above). The kernel and the flagged-only choice were right (old launches without
joining: identical, but 2.8 % SLOWER: the join exists for a reason). I did not find why the synchronous copy is not seen by the
following launch on this engine / GB10; the patch uses the stream copy plus a guard (a list last sent on another stream waits for that
stream before the host list is rewritten). Rule for the next kernel that reads a host-made list: queue the copy on the stream.

## Not in the patch / not done

- A recipe.sh change (integrator): CERT_GATHER=1, HALF=0, HEAD2_SAMPLED=1 as defaults; see ../r10-audit/AUDIT.md.
- The recommended sampled stack as ONE configuration (D10e, one engine, knob pairs, seeded, texts identical): today's default ->
  + CERT_GATHER -> + HALF=0 + HEAD2_SAMPLED=1: T0.7 thinking off 49.10 -> 49.93 -> 50.85 (+3.56 %), thinking on 51.97 -> 52.80 ->
  53.44 (+2.83 %), greedy 36 held prompts 64.50 -> 64.50 -> 64.68 (+0.27 %). Edit / agent / needle / image / qeval were not run on
  D10e, nor 16 users with the three changes together: the integrator's suite.
- New finding, not explained: a no-draft stream's FIRST request for a prompt decodes 9.8 % (+-0.5) slower than repeats of the same
  prompt (drafted streams show nothing like it). If the cause is a state that a cache-resumed stream has and a fresh one lacks, that
  is +10 % for --no-drafts deployments.
