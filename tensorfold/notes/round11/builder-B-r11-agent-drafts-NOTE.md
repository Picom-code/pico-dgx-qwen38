# Builder B, round 11: agent-workload drafts continued

Base: integ9 7061396 (patches 0101-0118). ONE cumulative patch `r11-B-agent-drafts.patch`
(sha256 66c5134e71e356609e648fca4ca762e8b3bbe0e6816a5f62ffe8b79e7c60393b; = `git diff 7061396..70fb440`, branch r11-B in Mac q38/wt/B11;
`git write-tree` after apply 119c8aed72d5; 16 files, +317 / -41). **It REPLACES r10-B's patch** (`handoff/B/r10-workload-drafts/`): it contains all of it.
Built and measured version: binary B18 sha 5acf227aba9ee51d on spark-3d24 (= this patch). B16 (5767b0772d6604fd) and B17 (7cff5d58a8cf9c37) are earlier
states of the same branch; the tables say which binary.

Everything here is drafts only: a drafted token is kept only where it equals the model's own token. Identity was checked on texts for every flag
(counts in each table), also with other requests in flight. Credits: OpenAI's predicted-outputs API shape; TensorFold's own SuffixLookup proposer,
copy-width rule and Recall store (upstream, MiaAI-Lab patches MIT); SuffixDecoding (arXiv 2411.04975) and Hybrid Verified Decoding (2606.01019) for copy
sources beyond the prompt. No claim of novelty.

## What is in the patch (new in round 11; the round-10 pieces are unchanged)
| piece | flag / API | default in the patch | proposed recipe default | what it does |
|---|---|---|---|---|
| window buffers sized at run time | none (follows `TF_FLASHNEXT_Q38_WINDOW_ROWS`) | on | - | `mtp.bufRows()` = max(32, WINDOW_ROWS) sizes a stream's score buffer, host copy, rollback copies and late log. `mtp.max_rows` (128) stays the knob's ceiling and sizes only small host arrays. With WINDOW_ROWS=32 a stream costs what it cost before round 10 |
| copy ramp in shared rounds | `TF_Q38_COPY_SHARED=2` (1 = even share) | off | **on** (both profiles) | with several streams in flight a stream's copy window keeps the lone stream's ramp (15, then doubled after each copy that landed whole, back to the start after a break) up to the rows the pass has left when every other live stream takes a 16-row window. Before: 15 tokens whatever happened. The round's allocation still trims every plan to the pass |
| copy width after a break | `TF_Q38_COPY_RESET=half` (from round 10, now measured) | off | **on** with 128-row windows | a broken copy halves the next copy's width instead of going back to 15 |
| first copy width from the match | `TF_Q38_COPY_FIRST=1` | off | option (on for edit / agent boxes) | a lone stream's copy starts at 63 tokens when the match is >= 16 tokens and no broken copy is pending, at 127 when >= 32 |
| short match after a late break | `TF_Q38_COPY_REENTER=N` | off | leave off (zero) | - |
| `prediction` on the other routes | `/v1/responses`, `/v1/messages` (same field and semantics as `/v1/chat/completions`) | on | - | the field is passed through the two translators |
| per-request recall switch | `"recall": false` (all three routes) | - | - | the request neither drafts from the server's `--recall` record nor is added to it |
| measuring aids | `TF_Q38_COPY_OFF=1`, `TF_Q38_COPY_MATCH=N`, `QCOPY` line with `TF_STEP_LOG` | off | - | chain-only logs; a copy window's source and match length |

Knobs of the copy drafts live in a file of their own (`core/lanes/q38_copy.zig`), not in `cuda_knobs.zig` (no merge conflict there).
API text for the README: `API-NOTE.md` in this directory.

## 1. Run-time buffer sizing and the 16-user profile (MEASURED, spark-3d24, start lines of both profiles before any number)
| profile | WINDOW_ROWS | start line | prompt cache |
|---|---|---|---|
| USERS16 (`--parallel 16`, pool 65536, WIDE_ROWS=128) | 32 | 16 streams at once, 1.49 GiB each (= the base) | 5.3 GiB |
| USERS16 | 128 | 16 streams at once, 1.86 GiB each, of 29.8 GiB (fits by 0.04 GiB; refused in round 10 at a 29.5 budget) | 0.1 GiB |
| default (5 x 262K) | 32 / 64 / 128 | 5 streams at once, 4.85 (round 10) / 4.89 / 5.13 GiB each | 3.8-4.2 GiB at 128 |

USERS16, one binary (B16), references = the 32-row run; total wall tok/s:
| WINDOW_ROWS | 1 user | 4 | 8 | 16 | 16 | held chat 16 in flight | 16 predicted-output requests | texts |
|---|---|---|---|---|---|---|---|---|
| 128 | 86.4 | 161.0 | 211.9 | 276.5 | 280.6 | 198.6 | 317.2 | 107/107, 56/56 |
| 32 | 87.1 | 157.4 | 219.9 | 280.3 | 284.5 | 200.2 | 319.4 | 107/107, 56/56, loops 90/90 + 90/90 |

Plainly: **with many users 128-row windows are zero to slightly negative (-1..-3 %, inside this bench's +-3) and they take 5.9 GiB from the prompt
cache.** Keep `WINDOW_ROWS=32` for USERS16. The feared effect (wide copies of one stream crowding a shared pass) cannot happen in this engine: in a
shared round a copy was 15 tokens whatever the window. The real finding is the opposite one, see 2.

## 2. Copies in shared rounds: `TF_Q38_COPY_SHARED` (MEASURED, B18, references first and last in one window)
Default profile, 128-row windows, predictions sent, total wall tok/s:
| run | 56 predicted-output requests, 4 in flight | agent loop (fix), 4 agents | follow-up sessions, 4 agents | 56 requests, 2 in flight | held chat mix, 4 in flight |
|---|---|---|---|---|---|
| reference 1 | 235.6 | 109.8 | 151.8 | - | - |
| `COPY_SHARED=1` | 278.2 | 116.5 | 158.7 | - | - |
| `COPY_SHARED=2` | 301.2 | 120.6 | 157.8 | 315.6 | 117.8 |
| reference 2 | 235.6 | 110.1 | 153.1 | 208.8 | 117.4 |
`COPY_SHARED=2` against the references: **4 rewrite requests in flight +27.8 %, 2 in flight +51.1 %, 4 agents in a bug-fix loop +9.7 %, follow-up sessions
+3.5 %, chat in flight +0.3 % (zero).** Texts against the one-user reference in every run: 56/56, 90/90, 97/97, 56/56, held 107/107.
(Two agents rewriting at once used to get 209 tok/s in total, less than one alone at 227.)

USERS16 profile (16 x 64K, 32-row windows), predictions sent, one reference (the second was cut by the window):
| run | loop 4 agents | loop 8 agents | 16 predicted-output requests | 12 held chat + 4 predicted-output in flight | distinct 16 users | held 16 in flight |
|---|---|---|---|---|---|---|
| reference | 111.8 | 196.8 | 314.0 | 117.7 + 162.5 | 278.2 / 285.3 | 207.5 |
| `COPY_SHARED=2` | 116.3 | 208.0 | 317.0 | 117.8 + 165.1 | 282.0 / 282.5 | 209.3 |
All texts identical (90/90, 90/90, 56/56, 107/107, 56/56, 107/107). With 16 users the cap equals today's 15 tokens: zero by construction, no loss.
(The 8-agent figure includes prompts left warm by the 4-agent pass in both runs.)

## 3. Chooser between copy drafts and the chain (replay first, then the two rules that survived; one user)
Replay (SIMULATED, `tools/choose11.py` on the round-10 `TF_STEP_LOG=2` log, 295 requests, cost fitted on the run; fit on one half, evaluated on the other):
- A pooled calibrated chooser (survival of a copy by match-length class x what the last copy did, best width, against the stream's own recent rate
  without copies) is **not shippable**: +3..18 % on rewrites, **-12..-21 % on follow-up sessions and regen** (below "no prediction"), in both directions of
  the split. The survival of a short match depends on the workload, which the features do not see; a per-stream correction factor made it worse.
- The chain's own probabilities cannot be used at a copy round without running the chain first (1.7 ms a step); not built.
- What held on both halves were two threshold rules read off the calibration table: first width from the match length (a 16-token match keeps 59-80
  tokens on average, a 4-7 token match 4), and a short match after a late break. Built as `COPY_FIRST` and `COPY_REENTER`.

Engine (MEASURED, B18, default profile, one user, 128-row windows + predictions = the round-10 state as reference, wall tok/s, runs in this order):
| run | numbered | plain rewrite | toolwrite | agent loop (fix) | follow-up | regen | held chat mix |
|---|---|---|---|---|---|---|---|
| reference 1 | 232.8 | 227.5 | 178.1 | 90.9 | 104.7 | 90.9 | 66.64 |
| `COPY_FIRST=1` | 243.9 | 235.7 | 182.6 | 92.1 | 105.7 | 89.6 | - |
| `COPY_FIRST=1` + `COPY_REENTER=4` | 244.1 | 238.4 | 182.2 | 91.6 | 104.6 | 89.0 | 66.53 |
| `COPY_RESET=half` | 241.4 | 235.6 | 178.6 | 91.8 | 105.0 | 91.1 | - |
| `WINDOW_ROWS=64` | 215.1 | 214.6 | 164.8 | 89.5 | 105.1 | 91.2 | - |
| reference 2 | 233.1 | 227.3 | 178.0 | 90.7 | 104.9 | 91.1 | - |
Against the mean of the references: FIRST +4.7 / +3.6 / +2.6 / +1.4 / +0.9 / **-1.5** %; FIRST + REENTER +4.8 / +4.8 / +2.3 / +0.9 / -0.1 / -2.2 %, held -0.2 %
(zero); RESET=half **+3.6 / +3.6 / +0.3 / +1.1 / +0.2 / +0.1 %** (never negative); 64 rows -7.7 / -5.6 / -7.4 / -1.4 / +0.3 / +0.2 %.
Identity in every run: pred 56/56, loop 90/90, follow 97/97, regen 24/24; ident 51/51 and held 107/107 where run.
Decisions: `COPY_RESET=half` -> default with 128-row windows. `COPY_FIRST` -> option (loses 1.5 % on "regenerate with a change"). `COPY_REENTER` -> dropped
(the replay's +2..5 % on numbered did not appear: the proposer's cursor rule already re-enters). 64-row windows keep about 70 % of the 128-row gain on
rewrites: use only where memory is short. Chat: zero for all of them.

## 4. `prediction` on `/v1/responses` and `/v1/messages` (MEASURED, B18, `tools/rt11.py`, 4 `numbered` items, non-streaming, temp 0)
| route | no prediction | with prediction | texts with vs without | malformed predictions answered 200 with the reference text | `"recall": false` |
|---|---|---|---|---|---|
| /v1/chat/completions | 105.2 tok/s | 426.3 | 4/4 | 13/13 | accepted |
| /v1/responses | 104.1 | 426.8 | 4/4 | 13/13 | accepted |
| /v1/messages | 104.2 | 426.5 | 4/4 | 13/13 | accepted |
Malformed shapes tried: a number, a bare string, no `content`, `content` a number, a list mixing numbers / null / objects without text, an empty list, null,
empty text, an unknown `type`, a bool, nested lists, wrong text, odd unicode.

## 5. The engine's recall store with predictions (MEASURED, B18, default profile, 128 rows, predictions sent, record empty at start)
| | loop (fix), 1 agent | follow-up, 1 agent | loop, 4 agents in flight | follow-up, 4 agents |
|---|---|---|---|---|
| prediction only (references first and last) | 90.9 / 90.7 | 104.7 / 104.9 | 109.8 / 110.1 | 151.8 / 153.1 |
| + `--recall`, record cold | 92.6 (+1.9 %) | 110.6 (+5.6 %) | 112.6 (+2.4 %) | 159.6 (+4.7 %) |
| + `--recall`, the same sessions again | 128.7 (+41.7 %) | 156.9 (+49.8 %) | 141.2 (+28.4 %) | 199.7 (+31.0 %) |
Texts 90/90 and 97/97 in all eight cells. Bounds seen: 133,926 token ids = 536 KB on disk (cap 62.9 MB = `--recall-gib` default 1/16 GiB), its own RAM
figure 16.8 MiB; engine RSS 2.2 GB. `"recall": false` on every request: the record stayed empty (0 ids, 0 bytes, 0 proposals), speed = the no-recall
reference, texts 90/90. With predictions already sent, a cold record adds little (+2..6 %); it pays when sessions repeat.
Stays an OPTION (`RECALL=1` -> `--recall --recall-dir <state dir>/recall`): **the record is shared by every request and written to disk. Fine on a
single-user box; on a shared box one user's text can be proposed as a draft in another user's request (never kept unless the model writes it anyway,
but it changes timing and sits on disk).**

## 6. Prompt cache in agent loops (MEASURED on the round-10 log; no engine change)
| trace | follow-up turns | previous prompt is an exact prefix | engine resumed at previous prompt - 7 tokens | prompt growth a turn | of which the previous reply | first-chunk time |
|---|---|---|---|---|---|---|
| loop_fix | 74 | 74/74 | 74/74 | 387 tokens | 102 (26 %) | 0.353 s of 1.392 s (25 %) |
| loop_follow | 81 | 81/81 | 81/81 | 595 tokens | 326 (55 %) | 0.469 s of 3.519 s (13 %) |
Every turn hits the cache. A turn prefills the previous reply (which the engine had just generated; its decode state is not kept) plus the tool result;
first-chunk time is about 0.16 s fixed + 0.55 ms a new prompt token. A client keeps this by: appending only, keeping the system prompt and tool list
byte-identical, sending the assistant's tool call back exactly as received, and putting volatile text in the newest message. Open (builder C's area):
the fixed 0.16 s a turn and keeping the state at the reply's end (26-55 % of a turn's prefilled tokens).

## For the integrator
1. Apply this patch INSTEAD of r10-B's. Recipe: default profile `COPY_ROWS=128` + `WIDE_ROWS=128` + `TF_Q38_COPY_RESET=half` + `TF_Q38_COPY_SHARED=2`;
   USERS16 keeps `COPY_ROWS=32` and may take `TF_Q38_COPY_SHARED=2` (zero at 16 users, +4..6 % with 4-8 agents).
2. `TF_Q38_COPY_FIRST=1` as an option for edit boxes; `RECALL=1` as an option with the sharing note; README text in `API-NOTE.md`.
3. See "Final candidate" below (if present) for the proposed defaults measured together.

## Not done / open
- The 16-user profile with 128-row windows AND `COPY_SHARED` (would need the 5.9 GiB back; not measured).
- Copy-versus-chain choice using the chain's probabilities (needs the chain run before a copy round).
- Tool-call skeleton as forced drafts (round-10 estimate: 9 % of passes on tool turns).
