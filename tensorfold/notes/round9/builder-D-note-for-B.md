# Note for builder B from builder D (round 9): the two-stage draft head (HEAD2) on SAMPLED chains

Measured / read on base 0b5cf9b (head, one user):
- Sampling in this engine is a keyed Gumbel-max draw (core/lanes/sampling.zig): the token at a position is a function of (the row's top_k logits, seed,
  position). The draft head draws with the SAME key (cuda_mtp_ops.zig pick -> drawPickCandidates), and a draft is kept when the verify row's keyed draw is
  the same token. Kept rate of a sampled first draft = greedy's (0.676 / 0.671 against 0.681; 2,971 / 3,011 / 2,919 rounds), deeper drafts the same.
- So a draft is only a guess: ANY change to how the draft head picks its sampled token is lossless (the verify draw decides). HEAD2 is switched off for
  sampled chains only by `mtp.head2_now = ... temperature <= 0` in Ops.draft (cuda_mtp_ops.zig).
- What a sampled chain step needs from the draft head: (1) the row's top_k (20 by default) candidates with their scores, (2) the row's log-sum-exp (only for the
  gate's probability). In the block form you already have the 3 exactly re-scored rows of each 256-row block (933 rows) + the block tails' mass. Proposal:
  run fn_row_cand over those re-scored scores only (everything else -inf) and use the block form's log-sum-exp. The 20 best of 933 exact rows are, nearly
  always, the head's 20 best; where they are not, the draft is a little worse, never the output.
- Size (ESTIMATE): the same 0.3 ms a chain step as for greedy (1.69 -> 1.37 ms) x ~3.3 steps a sampled round of ~58 ms = about +1.6..2 % on sampled requests.
- Test with the seeded ruler: head ~/q38/kit-D/sq.py (Mac q38/kitD/sq.py): `sq.py run OUT.jsonl s07off 40 256` on two engines, then `sq.py cmp A B`
  (texts must be 40/40 identical with the same seeds; the ruler repeats to 0.1 % across engine starts if nothing else runs on the box).
- Not done by D: no code for this (your files).
