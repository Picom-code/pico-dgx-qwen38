#!/bin/bash
# recipe.sh prepare | start | stop | bench | head2
# Qwen3.8-Flash-Next (NVIDIA's official FP4-Local checkpoint) on ONE DGX Spark, on TensorFold with MiaAI-Lab's patch set
# plus this repo's patches. Nothing here changes the host: everything runs in containers.
#
# Pinned inputs
#   engine      ashhart/TensorFold v1.1.4, commit eab8582819688325e371907e4706e1c602b8a273 (Apache-2.0)
#   recipe base MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold, commit 20e15e41f2f514a0d287710f4665ee1b5009ba41 (MIT):
#               its patches 0001-0017, scripts/prepare.sh (image build) and start.sh (launcher) are used as they are
#   ours        patches/0101-0129 of this directory, applied after MiaAI's by the same prepare.sh
#   base image  nvcr.io/nvidia/pytorch:26.07-py3 @ sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
#   Zig         0.17.0, sha256 9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8 (pinned by MiaAI's config.sh)
#   checkpoint  nvidia/Qwen3.8-Flash-Next-FP4-Local @ b9ce18a6887e18c089fdea8509a33e63429a3300 (pinned by MiaAI's config.sh)
#   host        DGX OS 7.2.3, kernel 6.14.0-1015-nvidia, driver 580.173.02 (what we measured on)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=${WORK:-$HOME/pico-qwen38-tf}
MIA_REPO=https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
MIA_REV=20e15e41f2f514a0d287710f4665ee1b5009ba41
SRC=$WORK/miaai-recipe

# Defaults = MiaAI-Lab's shipped configuration plus the lossless levers that measured positive here. Every one of them only changes HOW
# tokens are computed or which drafts get verified: the greedy output is the same as with the lever off (README: identity proof).
#   CERT=1 (default)   TF_FLASHNEXT_Q38_CERT=1: certified exact LM head (patch 0105). A greedy verify pass reads an MXFP8 copy of the head
#                      (half the bytes), bounds every score rigorously and recomputes only the entries that could be or tie with the best
#                      with the head's own BF16 kernel. +0.7 GiB GPU memory. Sampling, grammar and logprobs requests take the whole head.
#                      TF_FLASHNEXT_Q38_CERT_PLAIN=1 extends it to the one-row rounds of --no-drafts streams (patch 0106).
#   MANY=256 (default) TF_FLASHNEXT_Q38_PREFILL_MANY: fresh text prompts of up to 256 tokens that wait together are prefilled in ONE
#                      prompt pass instead of one pass each (patch 0106). 16 users arriving together: last first token 6.9 s -> under 1 s.
#                      MANY=0 turns it off.
#   COPY_ROWS=32 (default) TF_FLASHNEXT_Q38_WINDOW_ROWS: a verify window holds up to 32 rows instead of 16, so the copied continuation
#                      TensorFold's suffix-lookup proposer drafts when a reply repeats the prompt or itself (file rewrites, patches) can
#                      be 31 tokens instead of 15. A window starts at the stock 15 and doubles only after a copy landed whole.
#                      COPY_ROWS=16 is the stock behaviour.
#   FUSED=3 (default)  TF_FLASHNEXT_Q38_HCF: the hyper-connection read-out of decode rows (97 a pass, 3 more a draft step) in two launches
#                      instead of five: the normalisation is formed inside the down projection's input stage, the activation in its
#                      epilogue, and the up projection mixes in place (patch 0107). Same multiply-adds in the same order: every output
#                      word equal to the five-launch chain's (checked on the device, TF_FLASHNEXT_Q38_HCF_CHECK=1). FUSED=0 turns it off.
#                      FUSED_HALF=1 (default) runs the fused down kernel on 32-column tiles (96 blocks instead of 48).
#   HALF=96 (default)  TF_FLASHNEXT_Q38_HALF_BELOW: small decode projections (fewer than 96 blocks) run on 32-column tiles so that every
#                      SM has work (port of upstream TensorFold #1171). HALF=0 turns it off.
#   MTP_LAST=1 (default) TF_FLASHNEXT_Q38_MTP_LAST: the draft head runs its experts on the last row of a call only, the one row that
#                      reaches the draft (port of upstream TensorFold #1925). MTP_LAST=0 turns it off.
#   GDN_MANY=2 (default) TF_FLASHNEXT_Q38_GDN_LANES_MANY: rounds of 9-16 streams keep TensorFold's lane-batched DeltaNet launches
#                      (three a layer for all lanes; stock holds 8 lanes a launch and falls back to five launches A LANE above that:
#                      2,880 launches a pass at 16 streams). 1 = groups of 8, 2 = one table of 16 (patch 0108). GDN_MANY=0: stock.
#   BEST_LANES=1 (default) TF_FLASHNEXT_Q38_BEST_LANES: a shared round's draft step reads every stream's best draft with one launch
#                      instead of one launch a stream (patch 0109). Drafts only. BEST_LANES=0: stock.
#   ATTN_LANES=1 (default) TF_FLASHNEXT_Q38_ATTN_LANES: a shared round's attention (prep, pool, scores, selection, chunks, merge) takes
#                      every text stream in one launch a kernel, up to 16 streams a launch, instead of five or six launches A STREAM
#                      a layer (patch 0110; the idea is upstream TensorFold's TF_FLASHNEXT_ATTN_LANES, #1739). ATTN_LANES=0: stock.
#   LATE_SKIP=16 (default) TF_FLASHNEXT_Q38_LATE_SKIP: a drafted window whose rows plus the rows waiting in its stream's DeltaNet log are
#                      at most 16 writes no DeltaNet end states (36 layers x 3.1 MB a stream and pass); kept whole, its rows wait in
#                      the log like a partly kept window's and the next pass takes them first (patch 0111). LATE_SKIP=0: stock.
#   KEEP_INPASS=4096 (default) TF_FLASHNEXT_Q38_KEEP_INPASS: the prompt-cache mark 7 tokens before the end of every prompt no longer
#                      cuts a prompt pass of up to 4096 rows in two (patch 0112, builder C; the idea is upstream TensorFold #1866).
#                      A short new prompt's first token: 175 -> 136 ms. KEEP_INPASS=0: stock.
#   LANE_REUSE (default: the number of streams) TF_FLASHNEXT_Q38_LANE_REUSE=N: a finished stream's fixed-size device buffers are kept
#                      for the next stream instead of freed and allocated again (22 ms of every request; patch 0112, the idea is
#                      upstream #1763). LANE_REUSE=0: stock.
#   ROWS128=0 (default: off) TF_FLASHNEXT_Q38_ROWS128=2048 gives passes of more than 64 rows one 128-row tile a block (patch 0108).
#                      Exact, measured ZERO in the engine (the stock kernel already orders its 64-row tiles so the second one finds the
#                      weight tile in the GPU cache), so it stays off.
export TF_FLASHNEXT_Q38_CERT="${CERT:-1}"
export TF_FLASHNEXT_Q38_GDN_LANES_MANY="${GDN_MANY:-2}" TF_FLASHNEXT_Q38_ROWS128="${ROWS128:-0}" TF_FLASHNEXT_Q38_BEST_LANES="${BEST_LANES:-1}"
export TF_FLASHNEXT_Q38_ATTN_LANES="${ATTN_LANES:-1}" TF_FLASHNEXT_Q38_LATE_SKIP="${LATE_SKIP:-16}"
export TF_FLASHNEXT_Q38_KEEP_INPASS="${KEEP_INPASS:-4096}"
export TF_FLASHNEXT_Q38_HCF="${FUSED:-3}" TF_FLASHNEXT_Q38_HCF_ST="${FUSED_ST:-3}" TF_FLASHNEXT_Q38_HCF_HALF="${FUSED_HALF:-1}"
export TF_FLASHNEXT_Q38_HALF_BELOW="${HALF:-96}"
export TF_FLASHNEXT_Q38_MTP_LAST="${MTP_LAST:-1}"
export TF_FLASHNEXT_Q38_CERT_PLAIN="${CERT_PLAIN:-${CERT:-1}}"
export TF_FLASHNEXT_Q38_PREFILL_MANY="${MANY:-256}"
export TF_FLASHNEXT_Q38_WINDOW_ROWS="${COPY_ROWS:-32}"
# GATE=structured: verify a draft chain while the running product of its probabilities stays >= 0.5, at most 8 drafts
#   (default 0.4 and 6). Same texts; +16 % on structured output and +1 % on held prompts, -3 to -4 % on prose and code prompts.
if [ "${GATE:-}" = structured ]; then export TF_FLASHNEXT_CONFIDENCE=-0.5 TF_FLASHNEXT_DRAFTS=8; fi
# Round-11 settings (patches 0124-0129), defaults. R11=0 gives the round-10 defaults.
R11="${R11:-1}"
# KV (default bf16 in EVERY profile):
#   KV=bf16  the attention cache holds bf16 keys / values = the model's own precision: EXACT. Default profile: 3 streams at a
#            262,144-token window (8.36 GiB a stream). USERS16 and AGENTS: see their pools below. Every table at the top of the README
#            was measured with bf16 KV.
#   KV=fp8   MiaAI-Lab's default, here an explicit OPT-IN for more room: LOSSY against bf16 KV (their note: ~98.8 % top-1; measured here:
#            37 of the 51 identity-set replies differ from the bf16-KV replies). Buys 5 streams at 262K and a pool of 16 x 64K.
KV="${KV:-bf16}"
if [ "${KV_DTYPE:-$KV}" = bf16 ]; then export KV_DTYPE=bf16; [ "${USERS16:-}" = 1 ] || export PARALLEL="${PARALLEL:-$([ "${AGENTS:-}" = 1 ] && echo 5 || echo 3)}"; else export KV_DTYPE=fp8; fi
# USERS16=1: 16 users at once. Their caches come from a POOL of 16 x 64K-token windows (TF_FLASHNEXT_Q38_POOL_CTX, patch 0106) while the
#   largest request stays the full 262,144 tokens: each stream allocates what its own prompt + max_tokens need and a request waits in
#   the queue when the pool is full (e.g. 16 x 64K, or 1 full window + 12 x 64K, or 4 full windows + 2 x 64K). Drafted verify passes
#   hold up to WIDE_ROWS rows (TensorFold stops at 64, which leaves each of 16 users ~3 drafts a round). POOL_CTX=0 gives fixed slots
#   (then set CONTEXT=65536 yourself).
#   Pool size: KV=fp8 16 x 65536 tokens (patch 0127 sizes the window buffers at run time again, so the round-10 cut to 16 x 48K is gone);
#   KV=bf16 (default) 16 x POOL16_BF16 tokens: a bf16 token takes 30,784 bytes of cache against fp8's 17,888.
POOL16_BF16=40960
if [ "${USERS16:-}" = 1 ]; then
  export PARALLEL=16 TF_FLASHNEXT_PRODUCT_STREAMS=16 TF_FLASHNEXT_Q38_WIDE_ROWS="${WIDE_ROWS:-128}"
  if [ "$KV_DTYPE" = bf16 ]; then export TF_FLASHNEXT_Q38_POOL_CTX="${POOL_CTX:-$POOL16_BF16}"
  else export TF_FLASHNEXT_Q38_POOL_CTX="${POOL_CTX:-$([ "$R11" = 1 ] && echo 65536 || echo 49152)}"; fi
fi
# AGENTS=1: a handful of agents with long prompts of their own (measured by builder C on spark-4e5c: 5 agents, 56K-token system prompts,
#   follow-up first token 85 s -> 1 s). The 5 streams share a POOL of 5 x 131,072 tokens, which leaves the prompt cache ~13 GiB instead of
#   ~4 GiB, so every agent's kept states survive the others' requests. The largest request stays 262,144 tokens.
if [ "${AGENTS:-}" = 1 ]; then export TF_FLASHNEXT_Q38_POOL_CTX="${POOL_CTX:-131072}"; fi
# Round 11 (builder C): with a pool the elastic prompt cache gets what idle lane sets do not hold, so pool profiles keep 5 sets, not 16.
export TF_FLASHNEXT_Q38_LANE_REUSE="${LANE_REUSE:-$( [ "$R11" = 1 ] && [ -n "${TF_FLASHNEXT_Q38_POOL_CTX:-}" ] && [ "${PARALLEL:-5}" -gt 5 ] && echo 5 || echo "${PARALLEL:-5}")}"
# PREFIX_WAIT=60000 (default): a queued request whose prompt starts with a prefix another request is still prefilling waits (up to that
#   many ms) for it and resumes from the kept state instead of prefilling the same prefix itself (port of upstream TensorFold #1321,
#   patch 0112). Measured on spark-4e5c with 16 streams: 8 requests on a NEW 25.5K-token system prompt, last first token 86.7 -> 12.9 s;
#   texts identical. Round 9, 16 distinct users on spark-923b: 280.4 tok/s with it, first token 0.7-0.9 s with and without;
#   the two runs without it in the same window read 276.6 and 284.6: it costs nothing when prompts share only the chat template's opening. PREFIX_WAIT=0 turns it off.
if [ "${PREFIX_WAIT:-60000}" != 0 ]; then export TENSORFOLD_PREFIX_WAIT_MS="${PREFIX_WAIT:-60000}"; fi
# HEAD2 (default: see HEAD2_DEFAULT below): the two-stage draft head (patch 0115, builder B). A lone greedy chain scores its 79,591 draft
#   rows from a rank-512 4-bit preview in a rotated basis and re-scores the three best rows of each 256-row block with the head's own
#   rows. Drafts only: every token is still the main model's pick, texts identical. It needs a 30.7 MB file derived from the checkpoint's
#   own lm_head rows and a 5.2 MB rotation shipped here; `recipe.sh prepare` builds it (or `recipe.sh head2` alone).
#   HEAD2_ROT=r9 (default): rotation head2/head2_rot_r9.rot, fitted in round 9 on the head inputs of a 260-prompt corpus that shares no
#   prompt with any bench set (head2/corpus/fit.jsonl: other topics, code in ~30 languages, 16 natural languages, agent transcripts,
#   structured output, long-form). HEAD2_ROT=r512: the round-8 rotation, fitted on our own 104 held prompts (kept for comparison).
#   HEAD2=0 turns the lever off (the full 4-bit draft head, as before).
HEAD2_DEFAULT=1
HEAD2_ROT="${HEAD2_ROT:-r9}"
HEAD2_FILE="${HF_CACHE:-$HOME/.cache/huggingface}/q38/head2_${HEAD2_ROT}.bin"
case "$HEAD2_ROT" in r9) HEAD2_SHA=fce7c913060d45f2 ;; r512) HEAD2_SHA=c1f89b717e2288f1 ;; *) HEAD2_SHA= ;; esac
if [ "${HEAD2:-$HEAD2_DEFAULT}" = 1 ]; then
  if [ -s "$HEAD2_FILE" ]; then export TF_FLASHNEXT_Q38_HEAD2=/root/.cache/huggingface/q38/head2_${HEAD2_ROT}.bin
  elif [ "${1:-}" = start ]; then echo "recipe.sh: $HEAD2_FILE is missing (run ./recipe.sh prepare or ./recipe.sh head2): starting WITHOUT the two-stage draft head" >&2; fi
fi
mkhead2() {  # the two-stage draft head's file from the checkpoint's own lm_head rows and the shipped rotation; GPU, ~1 min, engine stopped.
  # On a GB10 GPU the file is bit-identical to the tested one (sha256 prefix $HEAD2_SHA); another GPU or a CPU gives a file that differs in
  # float rounding: fine as drafts (they can never change a reply), but not the measured file, so it is reported.
  local M; M=$(ls -d "${HF_CACHE:-$HOME/.cache/huggingface}"/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/* | head -1)
  mkdir -p "$(dirname "$HEAD2_FILE")"
  # the whole cache is mounted so that the snapshot's links into blobs/ resolve
  docker run --rm --gpus all --ipc=host -v "${HF_CACHE:-$HOME/.cache/huggingface}":/hf:ro -v "$HERE/head2":/h:ro \
    -v "$(dirname "$HEAD2_FILE")":/out --entrypoint python3 nvcr.io/nvidia/pytorch:26.07-py3 \
    /h/mkhead2.py "/hf/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/$(basename "$M")" /h/draft_vocab_default.txt "/h/head2_rot_${HEAD2_ROT}.rot" "/out/head2_${HEAD2_ROT}.bin" 512
  local got; got=$(sha256sum "$HEAD2_FILE" | cut -c1-16)
  if [ "$got" = "$HEAD2_SHA" ]; then echo "head2 file $HEAD2_FILE sha256 $got = the measured file"
  else echo "head2 file $HEAD2_FILE sha256 $got DIFFERS from the measured file ($HEAD2_SHA): usable (drafts only), but not the file the tables were measured with" >&2; fi
}
# Round-9 levers (patches 0116-0118), all defaults. R9=0 turns all of them off at once (= the round-8 defaults plus HEAD2).
R9="${R9:-1}"
# Round-10 settings (patches 0119+), defaults. R10=0 gives the round-9 defaults.
R10="${R10:-1}"
#   QSA_CERT=1 (builder C, patch 0116) TF_FLASHNEXT_Q38_QSA_CERT + _QSA_HALF: the sparse-attention indexer picks each prompt row's 512
#                      blocks from coarse scores computed on tensor cores with a rigorous bound; only the blocks the bound cannot place
#                      (about 21 a row) get the exact scalar sum, bit for bit as before, so the lists are the same. Long prompts only:
#                      first token of a cold 110K prompt -4 %, 214K -8 % (spark-4e5c); nothing below ~30K. QSA_CERT=0: stock.
export TF_FLASHNEXT_Q38_QSA_CERT="${QSA_CERT:-$R9}" TF_FLASHNEXT_Q38_QSA_HALF="${QSA_HALF:-${QSA_CERT:-$R9}}"
#   Round 10, builder C (patch 0121), long prompts; R10=0 turns them off:
#   QSA_LIST=1 / QSA_ROWS=1  TF_FLASHNEXT_Q38_QSA_LIST / _QSA_ROWS: the certified select as a histogram pass + a list pass (rows of >= 16K
#                      tokens of context), and its coarse scores' launch cut by rows. Same lists.
#   QSA_SLACK (unset)  TF_FLASHNEXT_Q38_QSA_SLACK (millionths): the float part of the select's bound; the code default is the designed 100.
#                      Patch 0120 fixes a round-9 bug that made the engine run slack 1. Builder C proposed 32 (2x the worst-case fp32
#                      summation error) for ~0.5 % on 100K+ prompts; paired on spark-923b 32 and 100 read the same (116K: 2534 / 2533
#                      tok/s), so the recipe keeps the larger margin.
#   HEAD_KV=512        TF_FLASHNEXT_Q38_HEAD_KV: the draft head's prompt rows, except each call's last 512, stop after their cache writes
#                      (nothing read their output). Needs MTP_LAST.
#   WARM=3072          TF_FLASHNEXT_Q38_WARM: one made-up 3,072-row prompt pass at start-up (1.3 s), so the first real long prompt does not pay it.
#   ARENA=0 (option)   TF_FLASHNEXT_Q38_ARENA=1: the lanes' + prompt cache's room taken once at start-up (29 GiB) and sub-allocated. Faster lane
#                      opens and kept-state writes on long prompts, but the IDLE engine then leaves only 7-8 GiB MemAvailable. Off by default.
if [ "${TF_FLASHNEXT_Q38_QSA_CERT}" != 0 ]; then
  export TF_FLASHNEXT_Q38_QSA_LIST="${QSA_LIST:-$R10}" TF_FLASHNEXT_Q38_QSA_ROWS="${QSA_ROWS:-$R10}"
  [ -n "${QSA_SLACK:-}" ] && export TF_FLASHNEXT_Q38_QSA_SLACK="${QSA_SLACK}"
fi
K10="${HEAD_KV:-$([ "$R10" = 1 ] && echo 512 || echo 0)}"; [ "$K10" != 0 ] && [ "${TF_FLASHNEXT_Q38_MTP_LAST}" != 0 ] && export TF_FLASHNEXT_Q38_HEAD_KV="$K10"
W10="${WARM:-$([ "$R10" = 1 ] && echo 3072 || echo 0)}"; [ "$W10" != 0 ] && export TF_FLASHNEXT_Q38_WARM="$W10"
#   ELASTIC=1 (builder C, patch 0116) TF_FLASHNEXT_Q38_CACHE_ELASTIC: the prompt cache may also use the part of the streams' windows
#                      that live requests are not using (start-up budget ~4 GiB, up to ~21 GiB more); a stream that opens takes its
#                      room back first. 5 agents with 56K-token prompts of their own: follow-up first token 65 s -> 1 s on the
#                      default profile. Only without a pool (USERS16 / AGENTS set POOL_CTX and keep their own budget).
#                      Round 11 (patch 0126): ELASTIC=2 is the same cache WITH a pool (USERS16 / AGENTS): it borrows the part of the pool
#                      that live streams are not using. 16 agents on USERS16: follow-ups resumed from the cache 7 of 48 -> 48 of 48.
if [ -z "${TF_FLASHNEXT_Q38_POOL_CTX:-}" ]; then export TF_FLASHNEXT_Q38_CACHE_ELASTIC="${ELASTIC:-$R9}"
elif [ "$R11" = 1 ] || [ -n "${ELASTIC:-}" ]; then export TF_FLASHNEXT_Q38_CACHE_ELASTIC="${ELASTIC:-2}"; fi
#   COPY_ROWS=128 (builder B, patch 0122; default profile) TF_FLASHNEXT_Q38_WINDOW_ROWS=128 + _WIDE_ROWS=128: a copied continuation that
#                      keeps landing whole is verified up to 127 tokens a window (15, 31, 63, 127; back to 15 when a copy breaks).
#                      File rewrites / write-file tool calls +20 %; zero on text that does not repeat. USERS16 keeps 32.
#                      The same patch adds a `prediction` field to /v1/chat/completions (OpenAI's predicted-outputs shape): the text is a
#                      second copy source for drafts; a wrong prediction cannot change the reply. TF_Q38_PREDICTION=0 ignores the field.
if [ "$R10" = 1 ] && [ -z "${COPY_ROWS:-}" ] && [ "${USERS16:-}" != 1 ]; then export TF_FLASHNEXT_Q38_WINDOW_ROWS=128; fi
if [ "${TF_FLASHNEXT_Q38_WINDOW_ROWS}" -gt 32 ] && [ -z "${TF_FLASHNEXT_Q38_WIDE_ROWS:-}" ]; then export TF_FLASHNEXT_Q38_WIDE_ROWS="${WIDE_ROWS:-128}"; fi
if [ "${ARENA:-0}" = 1 ] && [ "${TF_FLASHNEXT_Q38_CACHE_ELASTIC:-0}" != 0 ]; then export TF_FLASHNEXT_Q38_ARENA=1; fi
#   CERT_TOPK=20 + CERT_CODE=sm8 (builder D, patch 0117) TF_FLASHNEXT_Q38_CERT_TOPK / _CERT_CODE: the certified head's coarse copy holds
#                      7-bit sign-magnitude codes (residual a third of MXFP8's, so fewer exact tiles), and SAMPLED requests whose draw
#                      reads only their row's top_k <= 20 best scores (the server default top_k is 20) take the certified head too: every
#                      entry that can be among or tie with the 20 best is computed by the head's own BF16 kernel, so a seeded request
#                      returns the same text. Sampled (T 0.7) +4 %, greedy +1.5 % (head box). CERT_TOPK=0 CERT_CODE=mxfp8: round-8 head.
#                      USERS16 keeps CERT_TOPK=0: builder D measured 16 SAMPLED users 3-4 % slower with it (many sampled rows in one wide
#                      pass need too many exact tiles); the row limit that removes the loss (TF_FLASHNEXT_Q38_CERT_TOPK_ROWS, D's later
#                      patch 0119) arrived in round 10: see below. Greedy passes are not affected by CERT_TOPK either way.
if [ "${TF_FLASHNEXT_Q38_CERT}" != 0 ]; then
  if [ "${CERT_CODE:-$([ "$R9" = 1 ] && echo sm8 || echo mxfp8)}" = sm8 ]; then export TF_FLASHNEXT_Q38_CERT_CODE=sm8; fi
  # Round 10: patch 0119 adds the row limit TF_FLASHNEXT_Q38_CERT_TOPK_ROWS (default 8: a sampled pass of more than 8 rows takes the whole
  # head at once), so USERS16 can keep CERT_TOPK=20 (R10=0: USERS16 keeps CERT_TOPK=0 as in round 9).
  export TF_FLASHNEXT_Q38_CERT_TOPK="${CERT_TOPK:-$([ "$R9" = 1 ] && { [ "${USERS16:-}" != 1 ] || [ "$R10" = 1 ]; } && echo 20 || echo 0)}"
  export TF_FLASHNEXT_Q38_CERT_TOPK_ROWS="${CERT_TOPK_ROWS:-8}"
  # CERT_GATHER=1 (builder D, patch 0123): the certified head's exact tiles in one launch a row tile, read from a list, instead of one
  # launch a run of tiles (54-57 launches a sampled pass). Sampled requests +1.6 % (builder D, inside one engine); zero for greedy text.
  [ "${CERT_GATHER:-$R10}" = 1 ] && export TF_FLASHNEXT_Q38_CERT_GATHER=1
fi
#   DEEP=0.9 (builder B, patch 0118) TF_FLASHNEXT_DRAFTS=12 + TF_FLASHNEXT_Q38_DEEP=0.9,6: a lone chain drafts past 6 tokens (up to 12)
#                      only while the running product of its drafts' probabilities is still >= 0.9; shared rounds stay at 6. DEEP=0: off.
#   STOP2=0.5 (builder B, patch 0118) TF_FLASHNEXT_Q38_STOP2: a lone chain does not compute its next draft step once the product is
#                      below 0.5 (that draft would mostly cross the 0.4 gate and be thrown away). STOP2=0: off.
#   HEAD2_FAST=3 (builder B, patch 0118; needs HEAD2) TF_FLASHNEXT_Q38_HEAD2_FAST: the two-stage head's preview on int8 inputs (dp4a)
#                      and its rotation in bf16. HEAD2_FAST=0: the round-8 kernels.
#   All three are drafts-only: every token is still the main model's own pick.
if [ "${GATE:-}" != structured ]; then
  D9="${DEEP:-$([ "$R9" = 1 ] && echo 0.9 || echo 0)}"
  if [ "$D9" != 0 ]; then export TF_FLASHNEXT_DRAFTS="${DEEP_MAX:-12}" TF_FLASHNEXT_Q38_DEEP="$D9,6"; fi
fi
S9="${STOP2:-$([ "$R9" = 1 ] && echo 0.5 || echo 0)}"; [ "$S9" != 0 ] && export TF_FLASHNEXT_Q38_STOP2="$S9"
F9="${HEAD2_FAST:-$([ "$R9" = 1 ] && echo 3 || echo 0)}"; [ "$F9" != 0 ] && [ -n "${TF_FLASHNEXT_Q38_HEAD2:-}" ] && export TF_FLASHNEXT_Q38_HEAD2_FAST="$F9"
#   HEAD2_SAMPLED=1 (builder B, patch 0118; needs HEAD2) TF_FLASHNEXT_Q38_HEAD2_SAMPLED: a lone SAMPLED chain drafts with the two-stage
#                      head too. Drafts only.
if [ "${HEAD2_SAMPLED:-$R10}" = 1 ] && [ -n "${TF_FLASHNEXT_Q38_HEAD2:-}" ]; then export TF_FLASHNEXT_Q38_HEAD2_SAMPLED=1; fi
# Round 11 (patches 0124-0129). R11=0 turns all of them off.
#   Prompt passes (builder C, patches 0124 / 0125; measured on spark-4e5c, re-measured here: README):
#   GDN_PROD=2         TF_FLASHNEXT_Q38_GDN_PROD: the prompt DeltaNet kernel with two producer groups (fewer registers a thread, more threads).
#   ROUTE_BAND=2048    TF_FLASHNEXT_Q38_ROUTE_BAND: router logits of a prompt pass launched in bands of 2,048 rows.
#   SCRATCH_TRIM=3072 + CHUNK=4608 (bf16 default profile) / 6144 (fp8, pools)  TF_FLASHNEXT_Q38_SCRATCH_TRIM + TF_FLASHNEXT_CHUNK: a prompt pass takes more rows (stock 3,072; expert
#                      weights are read once a pass) and its scratch keeps the K-slice partials only for the rows that use them, so the
#                      wider pass needs LESS memory than the stock one did. Guarded: a user that could exceed a trimmed buffer errors.
#   QSA_CERT_MIN=1     TF_FLASHNEXT_Q38_QSA_CERT_MIN: decode rows take the certified select too (same lists; +2 % decode at 100K+ context).
#   HC_DN_TALL=3072    TF_FLASHNEXT_Q38_HC_DN_TALL: prompt passes of 3,072+ rows take the 64-row tile of the read-out's down projection.
#   Prompt cache (builder C, patch 0126):
#   SNAP_DELTA=1       TF_FLASHNEXT_Q38_SNAP_DELTA: a kept prompt state shares the attention rows of the state it grew from (append-only
#                      rows, reference-counted ranges) instead of being a whole copy: a kept turn costs 110 MiB + its NEW tokens.
#   Copy drafts (builder B, patch 0127; drafts only):
#   COPY_SHARED=2      TF_FLASHNEXT_Q38_COPY_SHARED: with several streams in flight a copy window keeps the lone stream's ramp.
#   COPY_RESET=half    TF_FLASHNEXT_Q38_COPY_RESET (with 128-row windows): a broken copy halves the next copy's width instead of 15.
#   COPY_FIRST=1 (OPTION, off) TF_FLASHNEXT_Q38_COPY_FIRST: first copy width from the match length: rewrites +3..5 %, "regenerate with a
#                      change" -1.5 %.
#   NGRAM_SPLIT=256 (builder D, patch 0128) TF_FLASHNEXT_Q38_NGRAM_SPLIT: a --no-drafts stream's first pass over new text reads its
#                      missing n-gram rows with two threads woken together (+7.9 % there; zero for drafted streams).
#   RECALL=1 (OPTION, off; single-user boxes only) serve --recall: TensorFold's record of earlier prompts and replies as a third copy
#                      source. The record is shared by every request and written to disk: see README (privacy note).
if [ "$R11" = 1 ]; then
  export TF_FLASHNEXT_Q38_GDN_PROD="${GDN_PROD:-2}" TF_FLASHNEXT_Q38_ROUTE_BAND="${ROUTE_BAND:-2048}" TF_FLASHNEXT_Q38_HC_DN_TALL="${HC_DN_TALL:-3072}"
  export TF_FLASHNEXT_Q38_SCRATCH_TRIM="${SCRATCH_TRIM:-3072}"
  # 3 bf16 streams at 262,144 tokens leave room for 4,608-row passes (6,144: "--parallel 3 needs 25.25 GiB, the budget leaves 25.03"); fp8 and pools take 6,144
  C11=$( [ "$KV_DTYPE" = bf16 ] && [ -z "${TF_FLASHNEXT_Q38_POOL_CTX:-}" ] && echo 4608 || echo 6144 )
  if [ "${TF_FLASHNEXT_Q38_SCRATCH_TRIM}" != 0 ]; then export TF_FLASHNEXT_CHUNK="${CHUNK:-$C11}"; elif [ -n "${CHUNK:-}" ]; then export TF_FLASHNEXT_CHUNK="$CHUNK"; fi
  [ "${TF_FLASHNEXT_Q38_QSA_CERT}" != 0 ] && export TF_FLASHNEXT_Q38_QSA_CERT_MIN="${QSA_CERT_MIN:-1}"
  export TF_FLASHNEXT_Q38_SNAP_DELTA="${SNAP_DELTA:-1}"
  export TF_FLASHNEXT_Q38_COPY_SHARED="${COPY_SHARED:-2}"
  [ "${TF_FLASHNEXT_Q38_WINDOW_ROWS}" -gt 32 ] && export TF_FLASHNEXT_Q38_COPY_RESET="${COPY_RESET:-half}"
  export TF_FLASHNEXT_Q38_NGRAM_SPLIT="${NGRAM_SPLIT:-256}"
fi
[ "${COPY_FIRST:-0}" = 1 ] && export TF_FLASHNEXT_Q38_COPY_FIRST=1
# Other flags our patches add (all off unless set; exact; measured as no gain with drafts on, so they stay off):
#   TF_FLASHNEXT_Q38_PDL=1, TF_FLASHNEXT_Q38_NGRAM_BUFFERED=1, TF_FLASHNEXT_Q38_SCHED=spin|yield|blocking, and the measuring aids
#   TF_HOSTPROF / TF_HOSTPROF_K / TF_KPROF / TF_FLASHNEXT_Q38_HDUMP / TF_FLASHNEXT_Q38_LAUNCH_DELAY_NS / TF_FLASHNEXT_Q38_GPROF (GPU ms by
#   group) / TF_FLASHNEXT_Q38_LHASH (hash of a stream's first scores), see notes/.

case "${1:-}" in
prepare)
  mkdir -p "$WORK"
  [ -d "$SRC/.git" ] || git clone "$MIA_REPO" "$SRC"
  git -C "$SRC" fetch -q origin && git -C "$SRC" checkout -q "$MIA_REV"
  cp "$HERE"/patches/01*.patch "$SRC/patches/"
  # builds the image with patches 0001-0017 + 0101-0129 (prepare.sh rebuilds by itself when the patch set changed) and downloads
  # the checkpoint unless HF_CACHE already holds it; REBUILD=1 forces a clean rebuild and a fresh pull of the base image
  ( cd "$SRC" && ./scripts/prepare.sh ${REBUILD:+--rebuild} )
  # the two-stage draft head's file (skipped when the measured file is already there)
  if [ -s "$HEAD2_FILE" ] && [ "$(sha256sum "$HEAD2_FILE" | cut -c1-16)" = "$HEAD2_SHA" ]; then echo "head2 file present: $HEAD2_FILE"; else mkhead2; fi
  ;;
start)
  # MiaAI's start.sh keeps the image's own entrypoint (/opt/nvidia/nvidia_entrypoint.sh). Keep it: that script switches the
  # container to its CUDA forward-compat userland, which makes every kernel launch cheaper (+4.5 % tok/s measured against a
  # launcher that overrides the entrypoint with the engine binary).
  # RECALL=1: the record lives beside the kernel cache (the container's /cache, the only writable mount of MiaAI's start.sh)
  ( cd "$SRC" && HOST=${HOST:-127.0.0.1} PORT=${PORT:-8888} ./start.sh restart ${RECALL:+--recall --recall-dir /cache/recall} )
  ;;
head2)
  mkhead2
  ;;
warm)   # only useful with TF_FLASHNEXT_Q38_NGRAM_BUFFERED=1: read the checkpoint shards that hold the n-gram table into the page cache
  M=$(ls -d "${HF_CACHE:-$HOME/.cache/huggingface}"/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/*/tensors | head -1)
  cat "$M"/model-000{15,16,17,18,19,20,21}-of-00021.safetensors > /dev/null && echo warmed
  ;;
stop)
  ( cd "$SRC" && ./stop.sh )
  ;;
bench)  # the same kit every number in README.md was measured with (stdlib Python only)
  export QK_URL=http://${HOST:-127.0.0.1}:${PORT:-8888}/v1/chat/completions QK_MODEL=Qwen3.8-Flash-Next
  python3 -I "$HERE/kit/qk.py" c1 9                      # 3 chat + 3 code prompts, 512 tokens, temperature 0, thinking off
  python3 -I "$HERE/kit/qk.py" sd 5                      # sparkDash's decode method (prose / code / structured)
  QK_HELD_STRIDE=3 python3 -I "$HERE/kit/qk.py" held /tmp/held.jsonl   # 104 of the 320 held prompts, per-category tok/s, texts saved
  python3 -I "$HERE/kit/qk.py" ident /tmp/ident.json     # identity set: compare two runs with: qk.py cmp A.json B.json
  for n in 4 5; do python3 -I "$HERE/kit/qk.py" distinct $n; done
  ;;
*) sed -n 2,5p "$0"; exit 2 ;;
esac
