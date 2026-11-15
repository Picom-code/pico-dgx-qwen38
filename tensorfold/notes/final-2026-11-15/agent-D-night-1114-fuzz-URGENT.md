# URGENT notes from agent D (exactness fuzz) to agent A. Newest last. Plain facts; each says what it changes on the front page.

## U1 (01:35 UTC Nov 15, static, reproduced on the Mac with the repo's recipe.sh sha256 27955316, no engine needed)
recipe.sh picks the LOSSY fp8 KV cache for any KV value that is not the literal `bf16`, without a word:

    +------------------------+----------------------------+--------------------------------------------+
    | option given           | recipe.sh today            | with the fix                               |
    +------------------------+----------------------------+--------------------------------------------+
    | (none)                 | bf16, 3 lanes   (correct)  | same                                       |
    | KV=fp8 / KV_DTYPE=fp8  | fp8, 5 lanes    (correct)  | same + one line "LOSSY KV cache ..."       |
    | KV=BF16                | fp8  (silently lossy)      | stops: "KV is bf16 ... or fp8 ..., not"    |
    | KV=exact, KV=bf-16     | fp8  (silently lossy)      | stops                                      |
    | KV=bf16 KV_DTYPE=fp8   | fp8  (inherited var wins)  | stops: "KV and KV_DTYPE disagree"          |
    | USERS16=yes            | ignored: default profile   | stops: "not a number"                      |
    | CERT=yes               | engine gets CERT=yes       | stops                                      |
    | COPY_ROWS=abc          | bash error, WINDOW_ROWS=abc| stops                                      |
    | CHUNK=banana, LANE_REUSE=-3 | passed to the engine  | stops                                      |
    +------------------------+----------------------------+--------------------------------------------+

What it changes on the front page: the sentence "fp8 only by the explicit option" is NOT true for the recipe.sh in the repo today
(a typo in KV gives fp8). With the fix it is true. `recipe.sh start` with no options = bf16 exact profile: true before and after.
FIX (ready, 2 hunks, recipe.sh only; valid option sets give a byte-identical engine environment, checked for default, USERS16, AGENTS, fp8):
    handoff/D/night-1114-fuzz/fix-recipe-validate/recipe-validate.patch   (applies to recipe.sh sha256 27955316 = main 42291bf)
    handoff/D/night-1114-fuzz/fix-recipe-validate/recipe.sh               (the patched file, sha256 66fe0314)
The same patch (01:50 UTC version) also refuses QSA_SLACK below 32 millionths (2x the worst-case fp32 summation error; below 16 the certified select's bound is not rigorous,
so the option could silently void the exactness of the indexer select) and USERS16=1 with AGENTS=1 (gave 16 lanes with a 131,072-token pool lane without a word).
Checked: 24 valid option sets (default, USERS16, AGENTS, fp8, levers off, R9/R10/R11=0, GATE, DEEP ...) give the byte-identical engine environment
before and after; 15 nonsense sets stop with rc 2 and one line.

## U2 (01:35 UTC Nov 15, from the levers-off reference of 14 Nov 19:30-20:30 UTC, binary I11 = tree 17a4c9d, bf16, 625 requests; BASE behaviour, no lever involved)
Three facts about the engine with every lever OFF that the README's exactness section must not contradict:
  1. `logprobs` is refused (HTTP 400 "logprobs are not supported by this model or backend") and `n` > 1 is refused (400 "n must be 1").
     So "logprobs identical" cannot be claimed or tested: say "not supported".
  2. A streamed reply and the same request non-streamed are NOT the same bytes in two cases, levers off:
     (a) thinking on: the streamed reasoning text ends with one extra newline (37 of 37 thinking requests);
     (b) a tool call cut off by max_tokens: non-streamed returns the raw text "<tool_call>...", streamed returns a parsed tool call (5 of 5 such requests).
     Answer text, finish reason and token counts are equal. Wording: "streaming does not change the tokens", not "identical JSON".
  3. Levers off: 2 requests in flight == solo for 556 of 556 answered requests; a `prediction` field == no field for 120 of 120.

## U3 (06:15 UTC Nov 15, measured on the head, final image binary F11 af27a1e7, bf16) : "sampled output identical per seed" is NOT true on /v1/messages, levers on or off
/v1/messages does not forward `seed` (server/anthropic_translate.zig passes only temperature, top_p, top_k, prediction, recall, stop_sequences), so every sampled
request on that route draws with a fresh seed: two levers-off runs of the same request already differ (7 of the first 7 sampled /v1/messages requests of matrix 3).
This is the BASE engine (upstream translate code + our prediction line), not a lever. /v1/chat/completions and /v1/responses forward the seed and are fine so far.
Front page: say "sampled output identical per seed on /v1/chat/completions and /v1/responses; /v1/messages has no seed field" (or merge the fix).
FIX offered (one line, NOT BUILT by me: no builds on the head tonight): handoff/D/night-1114-fuzz/fix-messages-seed/messages-seed.patch (adds seed and min_p to the
forwarded names; against integ11 17a4c9d). Needs a build + a run of `rt.py pair cases3.jsonl sel=route_msg` before it may be claimed.
Also confirmed by the probe (base, for the README's API notes): repetition_penalty / presence_penalty / frequency_penalty / logit_bias / min_tokens are accepted with
HTTP 200 and IGNORED (same reply as without them); logprobs and n > 1 are refused with 400.
STATUS so far (all on F11, default profile 2 x 262K on the head): identity set on vs knob-off 51/51; matrix 1: 619 requests, 0 different (601 identical, 18 refused the
same way), base deterministic 619/619; live sessions 190 turns: on 190/190, after a cancelled stream 190/190.

## U4 (07:55 UTC, status, nothing new that changes the front page)
Final image binary F11 on the head, bf16, default profile: levers on vs knob-off in one engine 1,788 comparisons, 0 different (matrix 1 619, matrix 3 648, live
sessions 190 + 190 after a cancelled stream, long prompts 6, long outputs 4, predictions started). Second engine start with every lever off by its flags:
identity set 51/51 (= knob-off = levers on), 1,138 requests against knob-off 0 different, session turns cold vs resumed 190/190. The head's own build of the
same tree (I11) gives the same 619 replies as the image's binary. Still to run: streamed pairs, 2-16 in flight, check modes, USERS16 / AGENTS profiles.

## U5 (11:45 UTC, FINAL): the exactness sentence is safe; five footnotes
Final image binary on the head, bf16: levers on vs levers off 7,210 comparisons, 0 different (7,135 identical, 75 refused the same way); flags-off engine vs
knob-off 1,138, 0 different; identity set 51/51 three ways; check modes 0 mismatches; USERS16 (16 in flight) and AGENTS profiles included at head-sized pools.
Footnotes the front page needs (all base-engine facts): (1) sampled "per seed" holds on /v1/chat/completions and /v1/responses, NOT on /v1/messages (no seed);
(2) logprobs / n > 1 refused; penalties, logit_bias, min_tokens accepted but ignored; (3) streaming: same tokens, not the same JSON form (reasoning trailing
newlines, cut tool call); (4) fp8 outside the claim, and merge fix-recipe-validate before writing "fp8 only by the explicit option"; (5) the recipe's own
profile sizes (3 lanes, 16 x 40,960, 5 agents) were not startable on the head: say "tested at 2 lanes / 16 x 16K / 3-4 agent lanes on the head".
Full report: handoff/D/night-1114-fuzz/FUZZ.md. Lease D released 11:35 UTC.
