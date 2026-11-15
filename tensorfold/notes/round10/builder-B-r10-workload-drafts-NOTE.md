# Builder B, round 10: workload-aware drafts (128-row verify windows + a `prediction` request field)

Base: integ9 7061396 (patches 0101-0118). ONE cumulative patch: `r10-B-workload-drafts.patch`
(sha256 a32199166daed97b77723ab38206215aa6ac5e7673ab8536b218316813059478; `git write-tree` after apply 05f9ccd2a6df; branch r10-B in Mac q38/wt/B10, HEAD 0c56877).
Built and measured version: binary B15 sha 678b52d2f9ae7d0c on spark-3d24 (= this patch). 8 files, +158 / -6.
Everything is drafts only: a drafted token is kept only where it equals the model's own token (the engine's normal verify window). Credits: OpenAI's predicted-outputs API shape;
SuffixDecoding (arXiv 2411.04975) and Hybrid Verified Decoding (2606.01019) for the idea of copy sources beyond the prompt; TensorFold's own SuffixLookup proposer, copy-width rule and Recall.

## What is in the patch
| piece | flag / API | default in the patch | what it does |
|---|---|---|---|
| 128-row verify windows | `TF_FLASHNEXT_Q38_WINDOW_ROWS=16..128` (was 16..32; `mtp.max_rows` 32 -> 128). Set `TF_FLASHNEXT_Q38_WIDE_ROWS=128` with it (a lone window is also limited by the pass's rows) | knob default unchanged (16); the recipe sets 32 today (`COPY_ROWS`) | a copied continuation that keeps landing is verified up to 127 tokens a window. Width rule = the lane core's own: 15, then 31, 63, 127 after each copy that landed whole; back to 15 when one breaks |
| `prediction` request field | `/v1/chat/completions`: `"prediction": {"type": "content", "content": "text"}` or `"content": [{"type": "text", "text": "..."}, ...]` (OpenAI's predicted-outputs shape) | used whenever the field is present; `TF_Q38_PREDICTION=0` ignores it; `TF_Q38_PRED_MAX` caps its tokens (65536); `TF_Q38_PRED_MATCH` = matching tokens a copy needs (8) | the text is tokenised and becomes a second copy corpus of the stream's suffix-lookup proposer: the reply's first tokens are compared with the prediction's start; afterwards 8 matching tokens start a copy (again after every divergence); parts are kept apart. A wrong, malformed or absent prediction cannot change the reply: unknown shapes are ignored, not refused. `"draft": false` turns it off with the other drafts. Not wired into `/v1/responses` or `/v1/messages` |
| copy width after a break | `TF_Q38_COPY_RESET=half` | off | a broken copy halves the next copy's width instead of going back to 15 (replay: +5 % on rewrites; measurement: see the end) |

Recommended recipe change: `COPY_ROWS=128` + `WIDE_ROWS=128` for the default (5-stream) profile ONLY. The USERS16 profile does not start with this patch at 16 x 64K
(see BLOCKER below).

Not in the patch because the engine already has it (upstream TensorFold, found by reading the lane core):
- a cross-request suffix store: `serve ... --recall --recall-dir DIR [--recall-gib G] [--recall-read-only]` (a record of prompts + replies of earlier requests on disk, copies priced by
  match band 16 / 32 / 64, `GET /v1/recall` shows its state). Measured below. It is a SERVER flag: the record is shared by every request the server gets and written to disk, so on a
  box with more than one user it leaks text between users by design; opt-in only.
- grammar-forced drafts: with `response_format` / `tool_choice: required` xgrammar's forced tokens are appended to a window as drafts (`core/lanes/windows.zig constrain`).

## Files
`zig/src/server/openai.zig` (field passed on), `server/chat.zig` (predictionIds), `core/engine_api.zig` (Request.prediction), `core/lane_stream.zig` (hands it to the proposer),
`core/lanes/proposer.zig` (SuffixLookup.predict / sideCopy + a unit test: `zig test -lc proposer.zig` 3/3 pass), `core/lanes/engine.zig` (copy reset knob),
`families/flashnext/cuda_mtp.zig` (max_rows), `families/flashnext/cuda_knobs.zig` (knob range).

## Paired result (spark-3d24, ONE binary B15, one window, one user, temp 0, thinking off, round-9 recipe defaults underneath; runs in this order; MEASURED)
Workload sets (Mac q38/wt/B/kit10/mkagent.py + ar.py; written for this round, no fetched text): `plain` = rewrite a file that is in the prompt (A's edit set + the prediction);
`numbered` = the prompt holds `cat -n` output, the reply is the plain file; `toolwrite` = the reply is a write_file tool call with the whole file; `loop_fix` = a REAL agent loop
(16 small repos with a seeded bug; the model calls list/read/write/str_replace/run_tests, the tools really run; 90 turns; the harness sends the files read or written so far as
the prediction); `loop_follow` = a second session on the repaired repo (97 turns); `regen` = the same ask with a small change, prediction = the earlier reply (not in the prompt).
Wall tok/s = reply tokens / request wall time (tool-call replies are not streamed, so wall is the ruler); held = decode tok/s of qk.py.

| set | G_ref1 (32 rows) | G_win (128 rows) | G_pred (128 rows + prediction) | G_ref2 (32 rows) | 128 rows vs refs | + prediction vs refs |
|---|---|---|---|---|---|---|
| held-104 all (chat / code / math) | 66.66 (56.20 / 94.45 / 88.82) | 66.66 (56.23 / 94.74 / 88.24) | 66.52 (56.02 / 94.87 / 88.47), sent with a WRONG prediction | 66.62 (56.17 / 94.37 / 88.78) | 0.0 % | -0.2 % |
| plain rewrite | 179.5 | 216.7 | 226.1 | 180.3 | +20.5 % | +25.7 % |
| numbered | 92.4 | 92.2 | 230.1 | 92.7 | 0 | +149 % |
| toolwrite | 141.1 | 172.0 | 175.4 | 143.0 | +21.0 % | +23.4 % |
| loop_fix (real agent loop) | 77.4 | 79.1 | 91.0 | 76.5 | +2.9 % | +18.3 % |
| loop_follow | 94.4 | 94.3 | 105.6 | 94.4 | 0 | +12.0 % |
| regen | 86.1 | 86.1 | 91.2 | - | 0 | +6.0 % |
| schema / call_required / call_auto (no prediction sent) | 81.0 / 84.0 / 98.0 | 81.0 / 84.4 / 98.1 | 80.8 / 84.4 / 97.8 | - | 0 | 0 |

Identity (texts; tool calls compared by name + arguments, their ids are random): G_win, G_pred and G_ref2 against G_ref1: ident 51/51, held 107/107 (G_pred: with the wrong
prediction), pred sets 56/56, loop 90/90, follow 97/97, regen 24/24, struct 28/28. 0 panic / error lines in every run. Start line unchanged: 5 streams at 262K.
The first measurements on B14 in the earlier window (E1-E4, same code, match passed by env) gave the same numbers within 1 % and the same identities.

Where it is ZERO, plainly: chat and anything that does not repeat text (held 0.0 %); a prediction where the prompt already holds the reply's exact tokens adds only +2..5 % over
the wider window; structured outputs and one-line tool calls 0.

Why 128 rows pay (MEASURED from the engine's per-request lines, requests grouped by drafted rows a round): 21 rows a round = 92-98 ms; 39 rows 118 ms; 56 rows 131 ms; 69 rows
151 ms; 88 rows 170 ms; 97 rows 170 ms (575 tok/s). A copy round costs about 32 + min(3.1 x rows, 42 + 1.03 x rows) ms: past ~20 rows the experts are already read and a row
costs ~1 ms, not 3.

## Replay first (SIMULATED on token ids logged with TF_STEP_LOG=2; tools kit10/parse10.py, sim10.py, forced10.py; cost fitted on the run; it predicted every measured gain within ~2 points)
| source of drafts | replay result | decision |
|---|---|---|
| (a) the request's own prompt | already in the engine (baseline) | - |
| (b) earlier turns of the SAME session beyond the prompt | 0 (they are in the prompt): -5..0 % | dropped |
| (b) other sessions' replies, match 8 / 16 | agent loop +9 %, follow-up +23..27 %, schema +8 %, one-line tool calls -20 % at match 8 (0 at 16) | the engine's `--recall` does this with band pricing: measured, not rebuilt |
| (c) client prediction, match 4 / 8 | numbered +113 / +111 %, agent loop +17 / +17 %, follow-up +6 / +12 %, regen -2 / +6 %, plain +4 % | built, match 8 |
| (d) grammar-forced tokens in tool calls (this model: `<tool_call>\n<function=N>\n<parameter=K>\nVALUE\n</parameter>\n</function>\n</tool_call>`) | 41 % of a tool call's tokens are forced whatever the tool (+27 % with the schema); 11 % + 8 % of a real agent loop's reply tokens. The MTP chain already drafts them: a forced token is a round's own token 6.6-8.7 % of the time (free tokens 10-14 %, schema tokens 19-25 %). Ceiling if all came free: 9 % fewer passes on agent tool turns (26 % with schema), 3 % on follow-ups, 22-30 % (58-64 %) on one-line calls | not built (tool_choice auto has no grammar; a client can send the skeleton as prediction parts) |
| (e) per-round choice copy vs chain | hindsight pricing over "always copy": +2..16 %; online landing-rate width rule: 0; re-sync near the break with a 2-4 token match: +3 % rewrites, -2..6 % others | not built (zero) |
| 256- / 512-row windows | nothing on edit tasks (an edit breaks a copy every 40-80 tokens); would pay only for long verbatim replies (ceiling ~760 / ~850 tok/s against ~620 at 127) | not built |
Ceilings (perfect drafts): 261 tok/s at 31 rows, ~620 at 127 rows.

## BLOCKER for the many-user profile (read before integrating)
The window buffers are sized by the compile-time `mtp.max_rows`, not by the knob: with 128 a stream costs +0.37 GiB whatever `WINDOW_ROWS` says.
- Default profile (5 x 262K): fits (4.85 -> 5.13 GiB a stream, start line unchanged). MEASURED.
- USERS16 profile (`--parallel 16`, pool 65536, WIDE_ROWS=128): B15 REFUSES TO START on 3d24: "--parallel 16 needs 29.69 GiB for its streams at a 65536-token window, and the CUDA
  memory budget leaves 29.50 GiB: serve --parallel 15, or lower --context". The base (B12) starts with 16 streams at 1.49 GiB each. MEASURED (start lines only).
So as it stands the patch is for the default profile; USERS16 needs one of: 15 lanes, a smaller pool window (~60K), or the follow-up below. Not fixed here (found at the end of the window).
Follow-up (not built): size per-stream buffers from `cuda_knobs.windowRows()` at run time. The sizes that follow max_rows are in `cuda_mtp_draft.zig`: `logits` (max_rows x vocab x 2 =
63.5 MB at 128), `host` (the same on the host), `saved` (`proj_at` / `tail_at` / `nrow_at` / `saved_bytes`: linear_n x max_rows x proj_bytes + max_rows x wide x 2), the late log
(`late_cap = 2 x max_rows` rows a DeltaNet layer) and `bytesFor`; `tail_at` / `nrow_at` / `proj_at` are also read in `cuda_mtp_ops.zig` (lines ~652-688, 815-816, 876-877). Keep the
small `[max_rows]` arrays. Then USERS16 with WINDOW_ROWS=32 costs what it costs today.

## More results from the same window (B15 unless said; MEASURED)
| test | result | identity |
|---|---|---|
| the engine's `--recall` (128-row windows, no prediction, record empty at start) | held-104 66.65 (refs 66.66 / 66.62: 0.0 %); agent loop with a cold record 83.2 (refs 77.4 / 76.5: +8.2 %; 128 rows alone 79.1); follow-up sessions 105.7 (+12.1 %); the same 16 sessions again 114.5 (+49 %) | held 107/107, loop 90/90, follow 97/97, retry 90/90 |
| 4 users at once on the 56 predicted-output requests (128 rows) | 189.5 tok/s total without predictions, 235.7 with (+24.4 %). Per request: numbered 32.5 -> 59.7, plain 67.1 -> 63.1, toolwrite 57.5 -> 50.0 (the pass's rows go to the streams that copy) | 56/56 against the one-user reference, both runs |
| seeded sampled requests (temperature 0.7, top-p 0.95, seed 4242; held half of the sets; 128 rows) without -> with prediction | numbered 89.8 -> 292.9; plain 281.3 -> 303.4; toolwrite 212.2 -> 218.9; agent loop 76.9 -> 89.7 | same seed, same text: 28/28, 43/43 |
| DEEP / STOP2 outside the fit (B12 = integ9; A's fresh set, 123 prompts): ref / DRAFTS=12 + DEEP=0.9,6 + STOP2=0.5 / ref | 55.81 / 57.20 / 55.92 = +2.4 % (chat +0.9, code +4.0, math +6.6, struct +5.9, agent +8.5, non-English +1.0) | 123/123 |
=> the round-9 DEEP line holds on text it was not set on (held mix was +2.8 %): no re-fit.
NOT measured (dropped for time): `TF_Q38_COPY_RESET=half` (replay +5 % on rewrites; stays off), DEEP at 15 drafts on the fresh set, 64-row windows, 16 users.

## For the integrator
1. Default profile: `COPY_ROWS=128` (TF_FLASHNEXT_Q38_WINDOW_ROWS) + `WIDE_ROWS=128`. Free on chat, +20 % on rewrites. Mind the blocker above for USERS16.
2. `prediction` needs no recipe change (request field). README text: what it is, that it is drafts only, the shape, and that an agent harness can send the files it has read.
3. `--recall` as a documented OPTION for single-user agent boxes (e.g. `RECALL=1` -> `--recall --recall-dir <state dir>/recall`), with the sharing / disk note.
4. Bench additions worth keeping: kit10/mkagent.py + ar.py (real agent loop, predicted outputs, structured sets) beside the edit / agent sets.

## Incident (mine)
My queue on 3d24 knew only my lease end, not the shared window's 300-minute cap: two engines of mine ran 08:58:34-09:11:35 UTC after the window had started its restore, and
DeepSeek came back at 09:22:54 instead of ~09:03 (verified by the guard: health 200, 391, session store, image). Those two runs are not used. Fixed in kit10/lease10.sh + run10.sh
(deadline = min(lease end, window OPEN + 300 min - 7 min)).
