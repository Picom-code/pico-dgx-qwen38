# Builder C, round 9: certified indexer select (long-prompt prefill) + elastic prompt cache (agents on the default profile)

Base: 0b5cf9b (handoff/BASE.txt, patches 0101-0115). ONE cumulative patch `C-r9.patch` = `git diff 0b5cf9b HEAD` of Mac `q38/wt/C`
(branch r9-C, HEAD e7a68cf), sha256 2a2a3370e1cc0580, 1,286 lines, 14 files; `q38/mkhandoffC9.sh` printed TREE_EQUAL (base + patch = tree 833c12e8).
Binary BUILT AND MEASURED from exactly this tree: C13, sha c11dfbcea69a2849 (spark-4e5c `~/q38/tfbin/C13`). Build: `./shipC.sh NAME`.
Every lever is behind a flag and OFF unless set. Everything MEASURED on spark-4e5c, recipe one-user profile (5 x 262,144, fp8 KV, vision on,
recipe environment of 0101-0115). Raw logs: Mac `q38/results/rC9.tgz`; full log `q38/PROGRESS-C.md` (Round 9).

## Lever 1: `TF_FLASHNEXT_Q38_QSA_CERT=1` + `TF_FLASHNEXT_Q38_QSA_HALF=1`  (proposed recipe default: both on)
What. The sparse-attention indexer scores every prompt row against every 4-token block (scalar fp32, fixed tree order) only to keep
each row's 512 best blocks. That list depends on the ORDER of the exact scores, not their values. So:
1. `fn_q38_qsa_coarse16`: the same sums on tensor cores (bf16 mma), 64-block key tiles staged once for all 256 rows of a group,
   stored as IEEE halves; also each tile's largest key norm.
2. `fn_q38_qsa_select_cert16`, one block a row: v = the 512th largest stored score. Bound per block:
   |exact - stored| <= eps + |stored| 2^-11 (x 1.01) + 2^-24, eps = SLACK x (sum of the 4 heads' |q|) x (largest |k|) / sqrt(128).
   A block whose interval lies wholly above the interval of v is certainly listed, wholly below certainly not. Only the blocks in
   between (21 a row on average, 76 at most, of up to 54K) get the exact sum: the scalar kernel's operations in its order, bit for
   bit; they are ranked exactly, ties to the lowest block as in `fn_qsa_select`. A row with a non-finite score or bound, or more
   than 1,024 in-between blocks, takes the exact sums for all its blocks and the stock selection inside the same kernel.
Files: `zig/kernels/cuda/q38_qsa_cert.cuh` (new), `flashnext_batch.cu` (include), `cuda_qsa_cert.zig` (new), `cuda_batch.zig` (attnRows),
`cuda_batch_state.zig`, `cuda_knobs.zig`, `cuda_lanes.zig` (stats line).
Flags: `_QSA_CERT=1` on, `=2` check mode (the stock kernels run as well; the device counts rows whose list differs, in-between sums
whose bits differ from the stock kernel's, and bound violations); `_QSA_HALF=1` the 16-bit form (0: fp32 scores, four radix passes);
`_QSA_CERT_MIN=64` fewest rows of a row group that take it (decode rows stay stock); `_QSA_SLACK=100` (millionths);
`_QSA_STATS=1` one counter line per request. Also in the patch, not recommended: `_QSA_LOOP=L` (the exact score kernel with L row tiles
a block: +2-3 % alone at L=8, 0 at 32; cert replaces that kernel).

Speed (first-token time of a cold prompt; identical prompts on both sides):

    FRESH ENGINES, env flags, no profiler (window X6)          flags off            CERT + HALF          gain
    109,617 tokens                                             46.55 s  2355 tok/s  44.75 s  2450 tok/s   +4.0 %
    213,604 tokens                                             98.26 s  2174 tok/s  90.40 s  2363 tok/s   +8.7 %
    27,502 tokens (window X5 B, with the elastic cache on)     10.84 s  2536        10.88 s  2527          0
    ONE ENGINE, knob file, reference first and last (X5 A)     off first / last     CERT      CERT+HALF
    27.3K                                                      11.28 / 11.07 s      10.67 s   10.63 s     +4.1 % vs last off
    109K                                                       47.12 / 47.04 s      44.56 s   44.36 s     +6.1 %
    214K                                                       100.36 / 100.17 s    91.51 s   91.08 s     +10.1 %
    a 3,072-row chunk at ~200K context                         1,661 / 1,649 ms     1,405 ms  1,405 ms

Quote the FRESH-ENGINE rows. In an engine started with the knob file the stock path is ~1.5-2 % slower at long context than in
a fresh env-flag engine (seen in four knob engines against two fresh flags-off ones; caching the knob file did not remove it; cause not
found), so the in-engine pairs overstate the gain by about that much. Decode speed does not depend on context length (the same
200-token task after 45 / 27K / 110K prompt tokens: 43.9-46.6 / 46.3-46.4 / 42.1-44.7 tok/s) and cert on decode rows gives nothing.

Identity proof.
- Check mode, device compare against the stock kernels, every prompt row: 16-bit form 4,478,408 row selections (27K prompt, 109K
  needle, 214K prompt): 0 lists differ, 0 exact-sum bit differences, 0 bound violations, 0 fallbacks; fp32 form 1,720,550 + 328,041
  selections: 0 / 0 / 0 / 0. Largest |stored - exact| / bound: 0.73 (16-bit, the half rounding itself), 0.002 (fp32).
- C13 with `QSA_CERT=1 QSA_HALF=1 CACHE_ELASTIC=1` against the flags-off engine, both fresh, same requests in the same order:
  ident set 51/51 texts identical; needle 109K 3/3, same answer text; short 24/24; 1 agent 4/4; prefill 3/3; 5 agents x 56K 15/15;
  16 agents x 20K 48/48; score hashes after the prompt pass 144/144 identical; release-time hashes 82 identical / 62 differ = the 63
  requests that ran concurrently (round 8: that hash is not reproducible between two identical concurrent runs). Fresh pair X6
  (one user): texts 2/2, score hashes first 3/3 and end 3/3.

## Lever 2: `TF_FLASHNEXT_Q38_CACHE_ELASTIC=1`  (proposed recipe default: on for the one-user profile; not with POOL_CTX)
Why the default profile lost every follow-up: its prompt cache is what five FULL 262K windows leave at start (3.3-3.7 GiB of 27.7),
while a kept state is a device copy of 110 MiB + ~17 KiB a token (1.09 GiB at 56K) and a request keeps several. No eviction order
fits 5 x 56K into 3.5 GiB, so upstream #2699's policy alone cannot fix it; page-shared KV (`TF_FLASHNEXT_KV_PAGES`) does not exist in
this tree's Flash Next family (a full port; not attempted).
What. A stream only allocates caches for its own prompt + max_tokens. With the flag the cache budget is
start-up budget + (lanes x full window - live streams' bytes - 640 MiB x LANE_REUSE). A stream that opens first takes its room back:
`Store.q38Resize` evicts in the usual order before the stream's caches are allocated; if states borrowed by running passes block it,
the request goes back to the queue through the existing LaneMemory path (0 times in every run). Five full-window streams still fit,
and then the cache is today's. The flag also turns on the round-8 sibling eviction order (a request's earlier marks go before any
conversation's newest state; `TF_FLASHNEXT_Q38_CACHE_SIBLINGS=0` turns that off): this is upstream #2699's first rule and it is what
makes 16 agents work (elastic alone: 118.7 s, unchanged).
Files: `core/prompt_cache.zig` (q38Resize), `core/prompt_cache_policy.zig`, `native/cuda.zig` (hook), `cuda_lanes.zig`, `cuda_lanes_load.zig`, `cuda_knobs.zig`.

    default one-user profile, 5 lanes; own system prompt per agent, +1.5K tokens a turn     flags off             ELASTIC (C13)
    5 agents x 56K x 3 turns: follow-up first token median / max; wall                      64.8 / 102.8 s; 324 s  0.99 / 12.2 s; 121 s
    16 agents x 20K x 3 turns: follow-up median / max; wall                                 123.5 / 185.9 s; 446 s 15.3 / 95.7 s; 170 s
       requests that resumed a kept state (of 48)                                           0                      33
    1 agent 20K x 4 turns: follow-up                                                        0.78 s                 0.84 s (C11 run 0.79 s)
    short prompt first token, median of 12                                                  101.0 ms               100.2 ms
    prefill 2.8K / 7.3K / 27.5K tok/s                                                       2498 / 2527 / 2536     2549 / 2519 / 2527
Texts identical in every row (see above). Engine line: "q38 elastic: the prompt cache takes 4146 MiB at start-up and up to 21607 MiB
more of the 5 lanes' room while streams leave it".
Known costs, as measured: a newly kept state is a fresh device allocation, ~0.15 ms a MiB: +0.09 s at the end of a 27.5K prompt,
+0.27 s at 109K, about +0.05 s on a one-agent follow-up turn (the static cache pays this only until its 3.5 GiB is cycling).
16 agents still wait on 5 lanes (turn 1 median 48 s, turn 2 13.4 s): that is queueing, the USERS16 profile is the answer there.

## Dropped, with the reason
- Select v2 (two 12-bit radix passes, each thread walking its own contiguous block range, one scan a phase): slower at 214K than
  v1 (+6.4 % against +9.5 % in the same kind of pair): per-thread ranges make every warp load touch 32 cache lines. Kept only as
  `kitC/q38_qsa_cert.v2.cuh.txt`.
- KPROF split of the remaining growth: not run (time). Under cert a chunk still grows ~1.2 ms per 1K tokens (stock 2.3).

## For the integrator
- recipe.sh: `TF_FLASHNEXT_Q38_QSA_CERT=1 TF_FLASHNEXT_Q38_QSA_HALF=1`; `TF_FLASHNEXT_Q38_CACHE_ELASTIC=1` when POOL_CTX is unset.
- Re-run on 923b before quoting: check mode once (`QSA_CERT=2 QSA_STATS=1`, a 30K and a 120K prompt, the "q38 qsacert" line must show
  list_diff=0 exact_bit_diff=0 bound_viol=0), then a fresh pair at 117K / 227K, needle, ident, and the 5-agent trace.
- The knob file is now cached for 200 ms (words: qsa_cert, qsa_half, qsa_cert_min, qsa_slack, qsa_stats, qsa_loop + round 8's).
  Do not use a knob-file engine for an off-vs-on speed claim (see above); use it for on-vs-on and for check mode.
- Test kit: Mac `q38/kitC` (box `~/q38/kit-C`): `runX5.sh` / `runX6.sh` (the final windows), `cw.py` (chunk walls from PLOG, no
  profiler), `tt.py decode` (decode speed after N prompt tokens), `engC.sh` (now with the 0101-0115 recipe defaults).
