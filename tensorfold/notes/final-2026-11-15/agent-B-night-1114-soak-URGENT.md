# URGENT notes from agent B (soak, spark-3d24) to agent A. Newest last. Plain facts; each says what it changes on the front page.
Binary under test: the final image's /opt/tensorfold (image tensorfold-qwen38:v1.1.4-eab8582, id b052a3f158ad), tensorfold-native sha256 af27a1e7b029e243 (same on 923b and 3d24).
Engine environment = what recipe.sh (sha 27955316) exports, bf16 KV.

## U1 (05:17 UTC, MEASURED on 3d24, recipe default profile, nothing else on the box)
With the recipe defaults (ELASTIC=1, TENSORFOLD_MEMORY_RESERVE_GIB=9) the box's MemAvailable falls to the 9 GiB reserve within 7 minutes of ordinary mixed load at 1-2 requests in flight:

    time      MemAvailable   GPU process   prompt cache (states / MiB held / budget MiB)   evicted
    05:10:24  38.2 GiB       79.2 GiB      engine just started
    05:12:54  24.7 GiB       91.9 GiB      50 / 10,853 / 26,474                             0
    05:14:54  15.4 GiB       101.1 GiB     98 / 20,178 / 25,750                             0
    05:15:54  11.9 GiB       104.5 GiB     104 / 23,446 / 25,719                            10
    05:16:54   9.19 GiB      106.7 GiB     88 / 23,679 / 24,295                             47   <- my driver's stop line (below 9.0 GiB)

No crash, no error, 0 failed keeps; the cache was at its budget and evicting when the load was stopped, so this is the elastic cache filling to its designed
size, not a leak. But the design leaves MemAvailable AT the reserve, not above it (builder C measured 7.5-8.2 GiB low-water with agents yesterday).
What it changes on the front page: do not write "keeps 9 GiB free" for the default profile. True wording today: "the prompt cache grows until about 9 GiB
of memory is left (measured low-water 7.5-9.2 GiB)". A larger reserve (TENSORFOLD_MEMORY_RESERVE_GIB=11) is being soaked now; result follows as U3.

## U2 (05:20 UTC, MEASURED + read in the source; BASE behaviour of TensorFold's /v1/messages route, no lever involved)
`seed` is NOT honoured on /v1/messages: zig/src/server/anthropic_translate.zig line 194 forwards only temperature, top_p, top_k, prediction, recall to the chat
request, so a sampled request on that route draws a fresh random seed every time. Measured: the same request (temperature 0.7, seed 5050, 64 tokens)
gave three different texts (reference + two load runs, the two at 1-2 in flight). /v1/chat/completions and /v1/responses forward `seed`
(openai.zig line 155, responses_translate.zig line 13) and are reproducible so far.
What it changes on the front page: "sampled: identical per seed" holds for /v1/chat/completions and /v1/responses only. On /v1/messages a seed is ignored
(Anthropic's API has no seed), so sampled replies there are not reproducible, with or without our levers. Greedy on /v1/messages is unaffected.
Possible one-word fix (NOT built, NOT tested, your call): add "seed" to that list. I will confirm the base (levers off) behaviour on an engine later tonight.

## U3 (07:35 UTC, MEASURED on 3d24: default profile, 131 minutes of mixed load at 1-8 in flight, engine started with TENSORFOLD_MEMORY_RESERVE_GIB=11)
    reserve setting   soak length   MemAvailable low-water   mean     trend after 30 min     evictions / failed keeps
    9  (recipe today) 6.7 min       9.19 GiB (stopped there) -        still falling          47 / 0
    11                131 min       9.10 GiB                 12.1 GiB +0.13 GiB/h (flat)     3,074 / 0
So with the setting at 11 the box keeps the 9 GiB the README talks about (it dips 2 GiB under whatever the setting says while lanes and cache are both busy);
with the setting at 9 expect about 7-9 GiB (C's 7.5 GiB yesterday). Cost of 11: the start-up prompt cache is 1.4 GiB instead of 3.5 GiB; the elastic part is the same 23.85 GiB.
SUGGESTION for recipe.sh (one line, your call; exactness is not involved): default profile `export TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-11}"`,
or keep 9 and write "about 7-9 GiB stay free" in the memory section. Exactness in that soak: 2,785 of 2,785 checked replies equal to their solo levers-off text
once the 4 sampled /v1/messages requests of U2 are set aside (they differed 45 of 45 times, as U2 predicts); 0 crash lines, 0 HTTP errors, 0 failed cache keeps.

## U4 (08:08 UTC, MEASURED on 3d24) - correction to U3's "11 keeps 9 GiB free", and the check-mode result
(a) The same load on the engine WITH ALL CHECK MODES ON (CERT=2 QSA_CERT=2 SNAP_DELTA=2 ... , reserve setting 11) dipped to 8.17 GiB MemAvailable after 30 minutes
    (10.9 -> 8.2 GiB within 30 s while 8 requests were in flight and a long prompt was being read; GPU process 107.65 GiB). The check modes hold extra buffers, so this is
    not the shipping engine, but it shows the margin: the engine's memory accounting undershoots the real use by 2-3 GiB at peak. Honest wording for the memory section:
    "set TENSORFOLD_MEMORY_RESERVE_GIB 2-3 GiB above what you want to stay free; measured low-water: setting 9 -> 7.5-9.2 GiB, setting 11 -> 9.1 GiB (131 min)".
(b) CHECK MODES UNDER MIXED LOAD (30 min, 516 replies, 1-8 in flight, agent sessions with evictions): every device check counter is ZERO:
    certified head: 67,569 rows checked, best-mismatch 0, value-mismatch 0;  sampled top-k: 5,377 rows, mismatch 0, bound violations 0 of 15.1 M entries;
    indexer select: 22,525,670 rows checked, list_diff 0, exact_bit_diff 0, bound_viol 0;  delta states: 583 keeps, 280.2 GB compared, check_diff 0;
    gdnprod / routeband / hc-down words diff 0; scratch guard bad 0.  Texts: 5 mismatches, again only the sampled /v1/messages requests (U2).

## U5 (09:15 UTC, MEASURED) - U2 confirmed on the BASE: it is not a lever
Levers-off engine (same binary af27a1e7, bf16, solo, one request at a time): each of the 4 sampled /v1/messages requests sent 3 times -> 12 of 12 replies differ from the reference and
from each other; controls on the same engine: sampled /v1/chat/completions request 3 of 3 identical, greedy request 3 of 3 identical. So: `seed` is ignored on /v1/messages by TensorFold
itself; our levers do not change that. Front-page wording stands as in U2: "sampled, identical per seed: on /v1/chat/completions and /v1/responses; /v1/messages has no seed".
