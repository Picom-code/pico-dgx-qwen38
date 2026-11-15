# Not built: an exact, faster indexer for long prompts (the one prefill item left with a measured gap)

Status: DESIGN ONLY (builder C, 2026-11-13). Nothing here was run. Numbers marked MEASURED come from window 1
(spark-4e5c, build C1, `TF_FLASHNEXT_Q38_GPROF=2`); everything else is ARITHMETIC or ESTIMATE.

## What was measured
A 3,072-row prompt chunk costs the same in every group at any context length EXCEPT the QSA attention group:

    context   3K    12K   30K   49K   67K   86K   104K
    QSA ms    162   178   208   236   271   309   340      (of a 1,127 .. 1,335 ms chunk)

That is +2.1 ms a chunk per 1,000 tokens of context. At 117K tokens it is ~10 % of the whole prefill, at 227K ~18 %
(ESTIMATE, linear). Without the growth a 117K prompt would run at ~0.363 ms a row = ~2,700 tok/s (we measure 2,368;
a third engine measured 2,848 on the same box in round 1).

## Where the growth comes from (code reading, cuda_batch.zig attnRows)
For every 16 rows, for each of 12 attention layers, once the sequence has more than `top` complete 4-token blocks:
1. `fn_qsa_scores` (the 8-row tile form `qsa_tile::scores<8,4>`): every row against EVERY complete block:
   4 heads x 128 products, scalar fp32 fma in a fixed tree order (the bits are defined by that order). At 104K:
   3,072 rows x 26,000 blocks x 636 flops x 12 layers = 610 GFLOP a chunk on scalar units.
2. `fn_qsa_select`: per row, four radix passes over all block scores, then a placement scan.
3. the fused attention over the selected rows (constant).
MEASURED split (KPROF, serialised launches, a 117K prompt, ms per ~13 chunks, middle of the prompt -> its end; Mac
results/rC/C_w7_kprof): block scores 807 -> 1,547 (+740 = 74 % of the growth), select 323 -> 513 (+190 = 19 %),
fused attention 1,210 -> 1,282 (+72 = 7 %). Every other kernel is flat.

## Why the obvious fix is not exact, and what would be
The same sums as one bf16 tensor-core GEMM (IQ [rows x 4, 128] x POOLED^T) are ~15-30x cheaper (ARITHMETIC: 982 GFLOP
a chunk at ~30 TFLOP/s = ~33 ms against ~220 ms), but mma sums do not have the scalar tree's bits, and the selection
reads the scores, so a plain swap can change which blocks a row attends to. Not lossless.

The selection only needs the SET of the `top` best blocks (fn_qsa_select writes them in block order; ties at the cut
go to the lowest block numbers). So the certified-head pattern applies:
1. fast scores F (tensor cores), with a bound eps(row, block) on |F - exact|;
2. v = the top-th largest F of the row. The exact cut E* is within eps of v, so a block with F > v + 2 eps is
   certainly selected and one with F < v - 2 eps certainly not;
3. only the blocks in between get the exact scalar sums (the existing kernel's code on a short list);
4. write the row's score array as +FLT_MAX (certainly in), -FLT_MAX (certainly out), exact value (in between) and
   run the UNCHANGED fn_qsa_select: it returns the same ids as on the exact array (the tie rule only ever applies
   between exact values);
5. a check mode that also runs the exact kernel and compares the id lists row by row (as CERT=2 does for the head).
A row with too many in-between blocks falls back to the exact kernel.

## What it needs
- a GEMM with fp32 output over a "weight" that is a live cache (the pooled index keys), in row groups so the
  [rows x 4, blocks] fp32 scratch stays small (256 rows at 65K blocks = 266 MB);
- a rounding model for mma that the bound can lean on (round 4 used 1e-3 of the norm product for the head; here the
  gap between neighbouring order statistics among 26K blocks is ~1e-4 of the range, so that slack would make the
  in-between set large: measure the real |F - exact| distribution first, offline, on dumped IQ / POOLED rows);
- the top-th value of F per row (a radix select like fn_qsa_select's four passes: its cost is paid twice unless the
  classify step and the final select are merged).
ESTIMATE of the gain if scores are ~2/3 of the growth: -80 ms of a 1,335 ms chunk at 104K unmerged (-6 %),
-150 ms merged (-11 %); over a whole 117K prompt about -3 % / -6 %, over 227K about -6 % / -10 %.
Do the offline |F - exact| study and the KPROF split BEFORE building.
