# Tree verify (a second candidate row in the verify pass): simulated again with real costs, NOT built (builder B, round 9)

## Simulation on the real default rounds (round-8 run B_S2, 7,274 head rounds; `dsim3.py`, table `dsim3_S2.txt`)
A sibling row = the head's rank-2 token at one draft position, verified in the same pass. It costs one verify row (2.40 ms, measured
average of a draft row one user) and, when it lands (the chain's draft there is wrong and rank 2 is right), adds exactly one token.
No chain below the sibling (that would need the head's next step for the sibling: 1.4-1.7 ms + another row).
| shape | rows a round | land | one user, SIMULATED |
|---|---|---|---|
| oracle: the row exactly at the first miss | 0.57 | 43 % | +4.7 % |
| priced by the head (reach x p2 >= 0.15), any position, 1 row | 0.49 | 25.8 % | +1.6 % |
| same, only at the FIRST draft | 0.23 | 29.8 % | +1.0 % |
| same, only at the LAST verified draft | 0.24 | 25.4 % | +0.75 % |
Same numbers with the two-stage head on (step 1.37 ms). The priced sibling sits at the last draft in 47 % of cases, first 22 %, middle 30 %.
The estimate is inside what these add-on simulations can resolve (a landed-row share is measured, the row cost is an average).

## Why it was not built
A verify window in the flashnext backend is the prompt-pass forward over consecutive rows (`cuda_mtp_ops.zig verify` -> `Engine.forward`),
and everything that carries sequence state is prefix-shaped:
| mechanism | where | what a sibling row needs |
|---|---|---|
| attention K/V + indexer keys written into the stream's cache BY POSITION, block pooling, block scores, select, chunks, merge | `cuda_batch.zig attnRows`, 6 kernels x fp8 / bf16 / lane-batched (`ATTN_LANES`) forms | a second slot for the same position, a per-row "do not see slot X" in the dense chunk and in pooling, and a slot move when the sibling is kept |
| DeltaNet chain + late log (`KEEP_LATE`, `LATE_SKIP`) | `gdnBlock`, `fn_gdn_rows_late`, `fn_gdn_front_logged` | a state fork at the parent row; the log row of the kept sibling in the parent's next slot |
| conv taps, n-gram tail | `fn_gdn_front*`, `pleBlock`, `fn_shift_window`, `fn_conv_roll` | parent-indexed history instead of "the rows before me" |
| rollback | `cuda_mtp_ops.zig rollback / convRoll`, `cuda_lanes.zig keepFn / draftFn` | "keep a path", today "keep the first n rows" (both refuse with `CannotRewind` / `TreesNotBuilt`) |
| draft head | `cuda_mtp_ops.zig draft / pick` | rank 2 and its probability (free in the block form: `fn_q38_head2_fin` sees all re-scored rows) |
The core is ready and is NOT the work: `core/lanes/engine.zig:506` plans head trees and chain + grafts (`plan_lanes.pick`, `shape.Shape`,
`accept.acceptPath`, `DraftRequest.lanes`, `Window.parents`) behind `head_trees` / `head_lanes`; note `config.zig:107`: head trees and the
product gate (`TF_FLASHNEXT_CONFIDENCE < 0`) exclude each other today (`DraftConfidenceNeedsChain`), and the product gate is worth more than
the tree, so the graft would have to be planned inside the backend or that rule relaxed.

## Two designs for whoever builds it
(a) In-kernel fork: the general tree. A parent index a row in every kernel above. Largest change, any position.
(b) "Alias lane", no kernel change (the one I would try first): verify the sibling's whole path `[pending, d1 .. d(j-1), s]` as a SECOND
    LANE of the same pass on the SAME sequence (`verifyMany` / `Engine.windows` already run several windows in one forward).
    - Rows that repeat the chain's prefix are bit-equal to the chain's rows (TensorFold outputs do not depend on batch composition) and
      read the same experts, so they cost little (dense tiles are 16 rows wide); the sibling row itself is the 2.4 ms.
    - The lane needs its own keep data (a light `Draft`: saved tail / proj / hist / log, logits, streams; no head cache), its late log
      seeded with the waiting rows, and `skip_state` (no end-state write) or its own spare states.
    - Attention must run the two lanes one after the other (not `attnLanesBatched`: both write the same cache slots), sibling lane
      first, so the cache ends as after a plain chain window; if the sibling is the kept path, replay its K/V rows from the lane's
      kept projections (`attnPrep*` on one row a layer, 12 launches) and re-pool the block if the position closes one.
    - Keeping the sibling path = the ordinary prefix rollback on the sibling lane's keep data (it IS a chain window).
    - Cost model to re-check first: a duplicate prefix row (est. 0.3-0.5 ms) makes a first-position sibling ~2.8 ms and a last-position
      one up to ~5 ms, so the first-position shape (+1.0 % at 2.4 ms) is the one that survives: ~+0.8 % (estimate).
Either way: exactness tests = identity 51/51 + held 107/107 texts with the flag on, plus a check mode that re-verifies every kept sibling
token as a plain chain row in the next round for a few thousand rounds.
