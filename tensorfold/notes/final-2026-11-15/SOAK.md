# Pre-ship soak, night 11-14 (agent B, spark-3d24). Everything below is MEASURED on 3d24 on 2026-11-15, 04:33-11:55 UTC, unless marked otherwise.

## What was tested

- Binary: the final image's /opt/tensorfold (spark-923b image tensorfold-qwen38:v1.1.4-eab8582, id b052a3f158ad, patches 0101-0129);
  tensorfold-native sha256 af27a1e7b029e243e6c94da92515826929c57b62fbbfa62586def16d39a5a2e4, the same on 923b and 3d24.
- Engine = exactly the environment recipe.sh (sha 27955316, repo main 42291bf) exports for each profile, bf16 KV, MiaAI's serve arguments, the image
  entrypoint, vision on (launcher scripts/engS.sh; the binary dir is mounted over MiaAI's image, not started through start.sh itself).
- Reference = the same binary with every lever of ours off, each request alone, not streamed, greedy or seeded: 384 references
  (224 single requests + 160 agent-session turns); 380 answered (4 `logprobs` requests are refused by the engine, HTTP 400, levers off too);
  33 of 33 repeated references identical.
- Pool: chat, code, rewrites with and without `prediction`, JSON / schema output, tool calls, stop sequences, 4 image requests, thinking on / off,
  70 seeded sampled requests, /v1/chat/completions + /v1/responses (26) + /v1/messages (26), documents of 5-42K tokens, long prompts of
  23K / 47K / 95K / 141K / 187K tokens, 3 long outputs, 24 agent sessions (6-8 turns each; system prompts of ~45K x 6, ~35K x 4, ~18K x 6, 1-12K x 8 tokens;
  each turn appends the reply and a tool result; 5 sampled, 6 with tools, 3 share a prefix).
- Load: 1-8 requests in flight (USERS16: 8-16, AGENTS: 14 agents + 1-2), 60 % streamed; clients that hang up mid-stream (the part received must be a prefix of the
  reference) or before any reply; slow and stalled readers; bursts of 8-14; idle gaps of 15-75 s; sessions abandoned, resumed, restarted, branched;
  the same request 2-4 times at once; a long prompt every ~10 min (USERS16: with 15 short requests started beside it).
- Checked per reply: text, reasoning, tool calls, finish reason, prompt tokens (not on /v1/messages), completion tokens. Monitor every 30 s: MemAvailable,
  GPU memory, /health, the prompt cache's line (states, MiB, evictions, refused / failed keeps), crash lines in the engine log.

## Soak table

"set aside" = the 4 sampled /v1/messages requests of finding F2 (not reproducible on the levers-off engine either). "waits > 60 s" = a stream that got
its first (empty) event at once and its next token more than 60 s later; every one of those replies was correct (finding F3).

    +-------------------------------------+-------+---------+-----------+------------+----------+-----------+------+-------+--------+----------------+--------------+
    | run (profile, reserve setting)      | min   | checked | identical | cancelled  | MISMATCH | of which  | HTTP | crash | waits  | MemAvailable   | trend GiB/h  |
    |                                     |       |         |           | (prefix ok)|          | set aside | 5xx  | lines | > 60 s | low-water      | after 30 min |
    +-------------------------------------+-------+---------+-----------+------------+----------+-----------+------+-------+--------+----------------+--------------+
    | default (recipe: reserve 9)         |   6.7 |     116 |       104 |   10 (10)  |     2    |     2     |  0   |   0   |    0   |  9.19 stopped  | still filling|
    | default, reserve 11                 | 131.1 |   2,830 |     2,595 |  189 (187) |    43    |  43 (+2)  |  0   |   0   |    3   |  9.10          | +0.13 (flat) |
    | default + all check modes, res. 11  |  30.1 |     516 |       475 |   32 (32)  |     5    |     5     |  0   |   0   |    4   |  8.17 stopped  | -            |
    | default, injections under load      |   7.2 |     174 |       164 |    8 (8)   |     2    |     2     |  0   |   0   |    0   | 11.6           | -            |
    | default, stop / start under load    |  11.3 |    175* |       157 |   16 (16)  |     2    |     2     |  0   |   0   |    0   | 10.1           | -            |
    | USERS16 (recipe: reserve 9), run 1  |  20.8 |     564 |       524 |   33 (33)  |     4    |     4     |  0   |   0   |    3   |  8.99 stopped  | -            |
    | AGENTS (recipe: reserve 9)          |  62.9 |     987 |       849 |   35 (35)  |     2    |     2     |  0   |   0   |  101   | 10.8           | +0.78        |
    | USERS16 (recipe: reserve 9), run 2  |  64.3 |   1,547 |     1,428 |   90 (89)  |    12    |  12 (+1)  |  0   |   0   |   17   |  7.90          | +1.88        |
    +-------------------------------------+-------+---------+-----------+------------+----------+-----------+------+-------+--------+----------------+--------------+
    * 2,897 further requests were cut or refused by the deliberate stop / start and are not counted.
    "stopped" = my driver's own stop line (MemAvailable below 9.0 GiB), not an engine failure. USERS16 run 2 ran with the line at 7.5 GiB (decision in PROGRESS-B.md 09:45);
    its low-water 7.90 GiB came 9 min after the start (10:57:50), the mean was 11.2 GiB. Max in flight seen: 21 / 15 / 8 / 18 / 27 / 30 / 33.

Totals: 334 minutes of load, 6,909 replies checked, 1,127,562 tokens generated.
- Crashes: 0 (no panic / segfault / CUDA error / out-of-memory line in any engine log; no container exit; the watchdog never fired).
- Hangs: 0 (no request without an answer; longest wait for a token inside a stream 216 s, under overload, reply correct).
- HTTP 5xx: 0 outside the deliberate restart (there: 13 clean "503 server restarting: retry shortly").
- Wrong text: 72 mismatches + 3 wrong prefixes, ALL on the 4 sampled /v1/messages requests (F2; they were sent 75 times and differed 75 times). Every other request:
  6,834 of 6,834 checked replies equal to the solo levers-off reply.
- Prompt cache: 8,462 evictions, 0 refused keeps, 0 failed keeps.
- Check modes under load (30 min, default profile): certified head 67,569 rows checked, best-mismatch 0, value-mismatch 0; sampled top-k 5,377 rows, mismatch 0,
  bound violations 0 of 15,135,456 entries; indexer select 22,525,670 rows, list_diff 0, exact_bit_diff 0, bound_viol 0; delta states 583 keeps, 280.2 GB compared,
  check_diff 0; gdnprod / routeband / hc-down diff 0; scratch guard bad 0.
- Failure injection (34 steps under load): malformed / non-UTF-8 / empty / wrong-type bodies -> 400; unknown route -> 404; max_tokens beyond the window, a 294,147-token
  prompt, prompt + max_tokens over the window -> 400 with the arithmetic in the message; 8 MB body -> 400; 64 / 200 MB bodies -> connection reset; 10 GB
  Content-Length -> 400; cut headers, cut chunked bodies, binary garbage -> closed; 40 + 40 + 100 dropped connections -> no effect. Engine healthy and canaries
  right after every step.
- Stop / start under load (docker stop -t 30 + rm, as stop.sh; then a start): stop 5 s, exit code 0, memory and GPU fully released, port closed; engine answering
  again 217 s later with requests arriving the whole time; 135 of 135 non-set-aside replies after it right. No stale state.

## Findings

F1. MEMORY: the recipe default does not keep 9 GiB free. (lever: ELASTIC prompt cache + TENSORFOLD_MEMORY_RESERVE_GIB=9)
    Default profile, reserve 9: MemAvailable 9.19 GiB after 6.7 min at 1-2 in flight (cache 23.7 of 24.3 GiB, still at its budget edge). With the setting at 11:
    low-water 9.10 GiB over 131 min, mean 12.1 GiB, trend +0.13 GiB/h, GPU process -0.01 GiB/h (flat, no leak). With all check modes on and 11: 8.17 GiB.
    USERS16 at 9: 8.99 GiB after 21 min, 7.9 GiB over the longer run. AGENTS at 9: 10.8 GiB over 63 min.
    Rule of thumb measured: the real low-water is 2-3 GiB under the setting when lanes and cache are busy together.
    Reproducer: scripts/w1.sh soakA (stops at 6.7 min). Not a crash, not a leak; it is a statement in the README that is not true as written.
F4. The reserve cannot simply be raised: default profile with 13 does not start ("--parallel 3 needs 25.17 GiB ..., the CUDA memory budget leaves 24.47 GiB");
    USERS16 with 11 does not start ("--parallel 16 needs 27.38 GiB ..., leaves 25.19 GiB: serve --parallel 14"). Default profile: 11 and 12 fit. USERS16: only 9
    fits with the 40,960-token lane (a lane of <= ~37,600 tokens, e.g. 36,864, would fit 11; not tested).
    Reproducer: `TENSORFOLD_MEMORY_RESERVE_GIB=13 ./recipe.sh start`, `USERS16=1 TENSORFOLD_MEMORY_RESERVE_GIB=11 ./recipe.sh start`.
F2. `seed` is ignored on /v1/messages (BASE behaviour of TensorFold, no lever): zig/src/server/anthropic_translate.zig line 194 forwards temperature, top_p, top_k,
    prediction, recall, not seed. The 4 sampled /v1/messages requests differed from their reference in every run with the levers on, and 12 of 12 times on the
    levers-off engine, solo (controls there: sampled chat 3/3 identical, greedy 3/3).
    Reproducer: POST /v1/messages {"model":"Qwen3.8-Flash-Next","max_tokens":64,"temperature":0.7,"top_p":0.95,"seed":5050,"thinking":{"type":"disabled"},
    "messages":[{"role":"user","content":"What is 17 * 23? Show the steps."}]} twice: two different texts. The same on /v1/chat/completions: identical.
    A one-word change is in UNTESTED-messages-seed.patch (NOT built, NOT measured: a suggestion, not a fix with proof).
F3. LATENCY under mixed long-prompt load (no error): while other lanes read long prompts, a decoding stream drops to 2-6 tok/s, and queued or admitted streams
    waited 60-216 s for a token (3 + 4 + 3 + 101 + 17 rows; default at 8 in flight, USERS16, AGENTS with 14 agents on 5 lanes). Example from the engine log
    (USERS16, 09:30): a 94,843-token prompt's first token took 204 s, 48K prompts 96-114 s, short chats ran at 2-6 tok/s. All replies correct.
    Do not promise interactive speed for 16 users when some of them send 40K+ prompts.
F5. Base-server oddities seen in the injections (none breaks a reply): max_tokens = -5 is accepted (200); an over-limit prompt on a STREAMED request answers HTTP 200
    where the non-streamed one answers 400; while 200 idle sockets were held open a new request got a broken pipe (healthy again when they closed);
    error messages show the front end's Python exception text. Plus D's two: streamed reasoning ends with one more newline; a tool call cut by max_tokens is raw text
    when not streamed and a parsed call when streamed (my checker allows for both).
No mismatch needed a lever bisect: the only differing requests are not deterministic on the base.

## Fixes handed off

- None with identity proof (no code bug of ours was found). Suggestions for A, both recipe / README level:
  (1) memory wording or `TENSORFOLD_MEMORY_RESERVE_GIB` default 11 on the default profile (soaked 131 min, plus 49 min of check-mode, injection and restart runs; USERS16 cannot take it);
  (2) front-page wording for sampled requests on /v1/messages (or the untested one-word patch).

## Verdict

"Stable and exact under load" CAN be claimed, with two limits written next to it:

  Allowed wording:
    "Soaked for 5.6 hours of mixed load on one DGX Spark (default, 16-user and agents profiles; up to 30 requests in flight; client hang-ups, slow readers,
     bursts, agent sessions that grow and resume from the cache with thousands of evictions, 187K-token prompts, a restart under load): no crash, no hang, no
     HTTP 5xx, and every one of 6,834 checked replies equal to the same request run alone with every lever off (greedy: same text; sampled: same text per
     seed on /v1/chat/completions and /v1/responses)."
  Must be said with it (or not contradicted anywhere):
    - "/v1/messages ignores `seed` (TensorFold's route has none): sampled replies there are not reproducible, with or without our levers."
    - Memory: NOT "9 GiB always stay free". True: "the prompt cache grows until about 8-9 GiB of memory are left (measured low-water: default profile 9.2 GiB,
      16 users 7.9 GiB, agents 10.8 GiB; set TENSORFOLD_MEMORY_RESERVE_GIB=11 on the default profile to keep 9 GiB)."
    - Not claimable from this soak: interactive latency under overload (waits of 1-3.5 minutes were measured when long prompts and many users mix), `logprobs`
      (refused by the engine), fp8 KV (not soaked), more than 5.6 hours.

Files: scripts/ (soak.py, rt.py from D, stub.py, engS.sh, wdS.sh, keepB.sh, startB.sh, w1.sh, w2.sh, w2b.sh, ops.sh, st.py, recenv.sh from D), URGENT.md (U1-U5),
UNTESTED-messages-seed.patch. Raw results: Mac q38/results/rB/night1114/ (rows, fails, monitor lines, engine tails per run), 3d24 ~/q38/runs/B_SOAK_*.
