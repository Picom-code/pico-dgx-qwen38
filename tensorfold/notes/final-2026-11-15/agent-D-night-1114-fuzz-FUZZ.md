# Exactness fuzz, night of 14-15 Nov 2026 (agent D, head Spark)

Question: does a request served with the recipe's speed levers return the same output as the same engine with the levers off?
Binary: the FINAL IMAGE's engine (tensorfold-qwen38:v1.1.4-eab8582, patches 0101-0129; tensorfold-native sha256 af27a1e7b029e243),
copied from spark-923b to the head. KV cache bf16. 04:36 - 11:35 UTC, 9 engines that served (5 more starts were refused for memory on the head), 0 crash lines in every engine log.

## Verdict

    +-----------------------------------------------------------+---------+-----------+-----------+------------------------+
    | comparison                                                | tested  | identical | DIFFERENT | errored (same refusal) |
    +-----------------------------------------------------------+---------+-----------+-----------+------------------------+
    | levers ON vs levers OFF (knob-off, same engine or solo)   |   7,210 |     7,135 |         0 |                     75 |
    | second engine started with the FLAGS off vs knob-off      |   1,138 |     1,118 |         0 |                     20 |
    | identity set (51): on = knob-off = flags-off              |      51 |        51 |         0 |                      0 |
    | levers off twice (base determinism)                       |   1,290 |     1,251 |        10 |                     29 |
    +-----------------------------------------------------------+---------+-----------+-----------+------------------------+

NO LEVER DIFFERENCE WAS FOUND: 0 of 7,210. Nothing to bisect. The 10 base differences are one shape (sampled /v1/messages: the
route drops `seed`), present with the levers off; that shape (52 requests) is left out of the lever table.

"Same output as with the levers off" is SAFE as worded for what was tested, with these footnotes (each is a base-engine fact, not a lever):

  1. Routes. Greedy: token for token on /v1/chat/completions, /v1/responses and /v1/messages. Sampled "identical per seed": on
     /v1/chat/completions and /v1/responses only. /v1/messages does not pass `seed` to the engine, so a sampled request there is never
     repeatable (levers on or off). Fix offered (not built): fix-messages-seed/messages-seed.patch.
  2. A sampled request WITHOUT a seed differs call to call (by design): "per seed" must stay in the sentence.
  3. logprobs and n > 1 are refused (HTTP 400): nothing to claim. repetition_penalty, presence_penalty, frequency_penalty, logit_bias and
     min_tokens are accepted and IGNORED by the engine (same reply with and without): do not list them as supported.
  4. Streaming does not change the tokens (streamed on vs streamed off: 745 pairs, 738 identical, 7 refused the same way, 0 different). The streamed and non-streamed FORMS of one
     reply differ in the base engine: non-streamed reasoning text is right-trimmed (streamed keeps its trailing newlines), and a tool call cut
     off by max_tokens comes back as raw text non-streamed but as a parsed call streamed. Say "same tokens", not "same JSON".
  5. KV=fp8 is outside the claim (lossy; not tested tonight). With today's recipe.sh a mistyped KV value silently gives fp8: merge
     fix-recipe-validate/ before saying "fp8 only by the explicit option".

NOT covered tonight (word the claim so it does not promise these): the recipe's own sizes of the default profile (3 lanes; the head ran 2),
USERS16 (16 x 40,960; the head ran 16 x 16,384) and AGENTS (5 lanes; the head ran 4 and 3): the head keeps a 13 GiB reserve, those sizes do not
start here; 50K-200K-token prompts with many requests in flight; video, JPEG and WebP inputs (PNG only); the RECALL, COPY_FIRST, ARENA and
GATE=structured options (off by default); a soak (agent B). Matrix 4 and the prediction / streamed runs have no second levers-off run of
their own (their shapes are the ones of matrices 1 and 3, which have). Levers without a knob word were compared through the flags-off
engine on 1,138 requests (matrix 1, half of matrix 3, sessions, long prompts and outputs), and through every in-flight / profile run
(reference = solo knob-off of the default engine, which equals flags-off).

## Findings

    +----+----------+----------------------------------------------------------------+---------------------------+------------------------------+
    | id | class    | what                                                           | lever                     | fix                          |
    +----+----------+----------------------------------------------------------------+---------------------------+------------------------------+
    | B1 | base     | sampled /v1/messages is not repeatable: `seed` is not forwarded| none (levers off: 10 of 23| messages-seed.patch (1 line, |
    |    |          | (server/anthropic_translate.zig forwards temperature, top_p,   | differ between two runs)  | NOT built / not proven)      |
    |    |          | top_k, prediction, recall, stop_sequences)                     |                           |                              |
    | B2 | base     | streamed reasoning keeps trailing newlines, non-streamed trims | none (off_s vs off differ,| none (document)              |
    |    |          | them; cut tool call: raw text non-streamed, parsed streamed    | on_s == off_s)            |                              |
    | B3 | base     | penalties, logit_bias, min_tokens accepted and ignored;        | none                      | none (document)              |
    |    |          | logprobs and n > 1 refused                                     |                           |                              |
    | R1 | recipe   | KV=BF16 / KV=exact / any typo selects the LOSSY fp8 cache      | recipe.sh, not the engine | recipe-validate.patch        |
    |    |          | silently; inherited KV_DTYPE=fp8 beats KV=bf16; nonsense values|                           | (tested: 24 valid option sets|
    |    |          | (USERS16=yes, CERT=yes, COPY_ROWS=abc, CHUNK=banana) pass on   |                           | identical env, 15 bad stop)  |
    | R2 | recipe   | QSA_SLACK below 32 would void the certified select's margin;   | recipe.sh option          | same patch                   |
    |    |          | USERS16=1 + AGENTS=1 gives 16 lanes x 131,072 without a word   |                           |                              |
    | H1 | head only| USERS16 16 x 32,768 and AGENTS 5 lanes do not start beside a   | none                      | none (13 GiB reserve is the  |
    |    |          | 13 GiB reserve (needs 23.62 GiB, 18.49 left)                   |                           | head's rule, not the recipe) |
    +----+----------+----------------------------------------------------------------+---------------------------+------------------------------+

Minimal reproducers (any engine start; levers do not matter):

    B1  curl -s 127.0.0.1:8888/v1/messages -H 'content-type: application/json' -H 'anthropic-version: 2023-06-01' -d '{"model":"Qwen3.8-Flash-Next",
        "max_tokens":64,"temperature":1.0,"seed":7,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"Tell me a short story about a lighthouse."}]}'
        twice: two different texts (the shape of cases x0119, x0388 ... of cases3.jsonl; this exact curl line was not run). The same shape on
        /v1/chat/completions with a seed gives the same text twice (3,007 sampled comparisons identical).
    B2  /v1/chat/completions, thinking on, "max_tokens" small enough to end inside the reasoning: "stream":true gives reasoning_content ending
        in "\n\n", "stream":false gives it without (case x0606 of cases3.jsonl; same 420 completion tokens).
    B3  any greedy request with and without "repetition_penalty":1.5 (or presence_penalty, frequency_penalty, logit_bias, min_tokens): same reply.
    R1  KV=BF16 ./recipe.sh start  -> recipe.sh exports KV_DTYPE=fp8 (shown with recenv.sh; the engine was not started that way).

## Device check modes ON (713 requests of the fuzz: 0 mismatches)

    certified head, greedy   14,000 calls, 72,876 rows checked against the whole head: best-mismatch 0, value-mismatch 0; whole-head fallbacks 2
    certified head, top-k     5,234 calls, 18,222 rows checked: topk-mismatch 0; bound violations 0 of 16.3 million entries
    indexer select            4,191,551 rows (3,280,182 through the list path): list_diff 0, exact_bit_diff 0, bound_viol 0, list_miss 0, fallback 0
    fused read-out            1.7 million read-outs, 20.8 billion words against the five-launch chain: 0 differ
    DeltaNet producers        14,652 launches, 62 billion words: diff 0      router bands 3,648 passes: diff 0      scratch guard 16,960 passes: bad 0
    replies in check mode     702 compared with the solo knob-off replies: 691 identical, 11 same refusals, 0 different
    (no delta-state check line and no tall read-out check line appeared in the log: those two check modes are NOT confirmed by a counter)

## Base determinism (levers off twice)

1,290 pairs (matrix 1 and matrix 3): 1,251 identical, 29 refused the same way, 10 different. All 10 are sampled /v1/messages requests
(10 of the 23 such pairs; the seed never reaches the engine; the other 13 were equal by chance). Every other shape: deterministic. Cross-build: the head's own build of the same tree (I11
0efcbbf3) and the image's binary (F11 af27a1e7; 6 kernel files and the binary differ in bytes) give the same 619 replies, flags off.

## Knob audit

    +-----------+--------------------------------------------------------------+-------------------------------------------------------------+
    | profile   | recipe.sh hands the engine (recenv.sh)                       | engine start log on the head                                |
    +-----------+--------------------------------------------------------------+-------------------------------------------------------------+
    | default   | bf16, 3 lanes, CHUNK 4608, CERT 1 sm8 TOPK 20 ROWS 8 GATHER, | matches every printed line: SM8 copy, top-k up to 20; trim  |
    | (no opts) | HEAD2 r9 FAST 3 SAMPLED, MANY 256, WINDOW/WIDE 128 half,     | 3072; head2 on rank 512 fast 3; many 256; hc down 3072; gdn |
    |           | ELASTIC 1, SNAP_DELTA 1, GDN_PROD 2, ROUTE_BAND 2048, TRIM   | prod 2; route band 2048; qsa cert=1 cert_min=1 slack=100e-6 |
    |           | 3072, HC_DN_TALL 3072, QSA 1/1/1 MIN 1, WARM 3072, DEEP 0.9,6| half=1 list=4096 rows=1; warm 3072; elastic; passes of 4608;|
    |           | DRAFTS 12, STOP2 0.5, NGRAM_SPLIT 256, PREFIX_WAIT 60000     | 8.39 GiB a stream (2 lanes here: the head's reserve)        |
    | USERS16   | bf16, 16 lanes, pool 40,960, CHUNK 6144, WINDOW 32 / WIDE    | started only with pool 16,384: "16 lanes share 16,489 MiB", |
    |           | 128, ELASTIC 2, LANE_REUSE 5, CERT_TOPK 20 + ROWS 8          | passes of 6144, top-k up to 20, elastic 2,647 + 13,289 MiB  |
    | AGENTS    | bf16, 5 lanes, pool 131,072, CHUNK 6144, ELASTIC 2,          | started with 4 and 3 lanes: "4 lanes share 19,095 MiB (4,773|
    |           | LANE_REUSE 5, WINDOW 128                                     | a lane at 131,072)"; prompt cache 12 MiB at start + elastic |
    | KV=fp8    | fp8, 5 lanes, CHUNK 6144 (only KV=fp8 or KV_DTYPE=fp8 ...    | not started (outside the claim)                             |
    |           | and, today, any KV word that is not `bf16`: finding R1)      |                                                             |
    +-----------+--------------------------------------------------------------+-------------------------------------------------------------+

- `recipe.sh start` with no options = bf16 exact profile: TRUE (environment and the engine's own line "--kv-cache bf16", 8.39 GiB a stream).
- KV=fp8 reachable only by the explicit option: FALSE today (R1), TRUE with the fix.
- Audit gap: the engine prints at start only the levers listed above. It prints nothing for HCF, HALF_BELOW, MTP_LAST, LATE_SKIP, KEEP_INPASS,
  LANE_REUSE, GDN_MANY, BEST / ATTN_LANES, WINDOW_ROWS / COPY_*, DEEP, STOP2, SNAP_DELTA, NGRAM_SPLIT, CERT_GATHER, CERT_TOPK_ROWS, PREFIX_WAIT:
  for those the audit is the environment only (MiaAI's start.sh forwards every TENSORFOLD_* / TF_FLASHNEXT_* variable, so they do arrive).
  The same forwarding passes stale variables of the calling shell (e.g. a leftover TF_FLASHNEXT_Q38_CERT=2) into the container: worth one README line.
- On a tight budget the AGENTS profile's fixed prompt cache is ~0 (12 MiB) and everything rides on the elastic cache: informational.
- Lowest MemAvailable on the head during the night: 9.35 GiB (default profile, elastic cache grown; reserve 13, watchdog line 6).

## The matrix (levers on vs levers off; a request counts once per run it was in)

A. LEVERS ON against LEVERS OFF, every comparison of the night added up, by request shape (a request counts once per run it was in)
+------------------------+--------+-----------+-----------+---------+
| shape                  | tested | identical | DIFFERENT | errored |
+------------------------+--------+-----------+-----------+---------+
| PROMPT                 |        |           |           |         |
| chat                   |    256 |       256 |         0 |       0 |
| nonenglish             |    123 |       123 |         0 |       0 |
| unicode                |    133 |       133 |         0 |       0 |
| code                   |    264 |       264 |         0 |       0 |
| table                  |     11 |        11 |         0 |       0 |
| base64                 |     30 |        30 |         0 |       0 |
| copy                   |    931 |       927 |         0 |       4 |
| near                   |    186 |       186 |         0 |       0 |
| rewrite                |    111 |       111 |         0 |       0 |
| boundary               |    118 |       118 |         0 |       0 |
| cross                  |      6 |         6 |         0 |       0 |
| long                   |      9 |         9 |         0 |       0 |
| image                  |     87 |        87 |         0 |       0 |
| longtext               |     13 |        13 |         0 |       0 |
| rand                   |   3054 |      3041 |         0 |      13 |
| held                   |   1734 |      1722 |         0 |      12 |
| fresh                  |   1320 |      1319 |         0 |       1 |
+------------------------+--------+-----------+-----------+---------+
| OUTPUT                 |        |           |           |         |
| eos                    |     51 |        51 |         0 |       0 |
| long_out               |      7 |         7 |         0 |       0 |
| mt4096                 |      4 |         4 |         0 |       0 |
| stop                   |    703 |       703 |         0 |       0 |
| ignore_eos             |      3 |         3 |         0 |       0 |
+------------------------+--------+-----------+-----------+---------+
| PARAMETERS             |        |           |           |         |
| greedy                 |   3895 |      3858 |         0 |      37 |
| sampled                |   3041 |      3007 |         0 |      34 |
| sampgrid               |    110 |       110 |         0 |       0 |
| grid                   |    207 |       207 |         0 |       0 |
| T0.2                   |    207 |       207 |         0 |       0 |
| T0.7                   |    387 |       387 |         0 |       0 |
| T1.0                   |    230 |       230 |         0 |       0 |
| T1.5                   |    212 |       212 |         0 |       0 |
| topk_dflt              |    185 |       185 |         0 |       0 |
| topk_0                 |    108 |       108 |         0 |       0 |
| topk_1                 |    101 |       101 |         0 |       0 |
| topk_2                 |     78 |        78 |         0 |       0 |
| topk_5                 |     70 |        70 |         0 |       0 |
| topk_19                |     67 |        67 |         0 |       0 |
| topk_20                |    122 |       122 |         0 |       0 |
| topk_21                |    130 |       130 |         0 |       0 |
| topk_50                |    113 |       113 |         0 |       0 |
| topk_100               |     62 |        62 |         0 |       0 |
| penalty                |    227 |       227 |         0 |       0 |
| param_edge             |     52 |        48 |         0 |       4 |
| logprobs               |     58 |         0 |         0 |      58 |
| think                  |   1039 |      1032 |         0 |       7 |
| budget                 |     20 |        20 |         0 |       0 |
| effort                 |     13 |        13 |         0 |       0 |
| json                   |    237 |       237 |         0 |       0 |
| schema                 |     87 |        87 |         0 |       0 |
| grammar                |     17 |        17 |         0 |       0 |
| tools                  |    297 |       291 |         0 |       6 |
| tc_auto                |     72 |        66 |         0 |       6 |
| tc_required            |    107 |       107 |         0 |       0 |
| tc_named               |     12 |        12 |         0 |       0 |
| tc_none                |     21 |        21 |         0 |       0 |
+------------------------+--------+-----------+-----------+---------+
| PREDICTION             |        |           |           |         |
| pred                   |    368 |       368 |         0 |       0 |
| predwrong              |     35 |        35 |         0 |       0 |
| predpart               |     34 |        34 |         0 |       0 |
| pred2                  |    150 |       150 |         0 |       0 |
| pred_exact             |     16 |        16 |         0 |       0 |
| pred_wrong             |     18 |        18 |         0 |       0 |
| pred_part              |     18 |        18 |         0 |       0 |
| pred_longer            |     15 |        15 |         0 |       0 |
| pred_shorter           |     27 |        27 |         0 |       0 |
| pred_shift             |     14 |        14 |         0 |       0 |
| pred_empty             |     18 |        18 |         0 |       0 |
| pred_rsn               |     24 |        24 |         0 |       0 |
+------------------------+--------+-----------+-----------+---------+
| ROUTE                  |        |           |           |         |
| route_chat             |   6494 |      6432 |         0 |      62 |
| route_resp             |    433 |       433 |         0 |       0 |
| route_msg              |    283 |       270 |         0 |      13 |
+------------------------+--------+-----------+-----------+---------+
| CACHE / SESSION        |        |           |           |         |
| seq                    |     84 |        84 |         0 |       0 |
| sysprefix              |   1025 |      1020 |         0 |       5 |
| sysS                   |    120 |       120 |         0 |       0 |
| sysM                   |    195 |       190 |         0 |       5 |
| sysL                   |    196 |       196 |         0 |       0 |
| multiturn              |    365 |       360 |         0 |       5 |
| edit                   |    146 |       146 |         0 |       0 |
| same                   |     12 |        12 |         0 |       0 |
| doc                    |      8 |         8 |         0 |       0 |
| grow                   |     16 |        16 |         0 |       0 |
| agent                  |     10 |        10 |         0 |       0 |
| sess                   |   1325 |      1325 |         0 |       0 |
| turn0                  |    210 |       210 |         0 |       0 |
| turn1                  |    210 |       210 |         0 |       0 |
| turn2                  |    210 |       210 |         0 |       0 |
| turn3+                 |    695 |       695 |         0 |       0 |
| regen                  |    105 |       105 |         0 |       0 |
+------------------------+--------+-----------+-----------+---------+
| OTHER                  |        |           |           |         |
| boundary_miss          |      8 |         8 |         0 |       0 |
| long0                  |      3 |         3 |         0 |       0 |
| long1                  |      3 |         3 |         0 |       0 |
| long2                  |      2 |         2 |         0 |       0 |
| long3                  |      1 |         1 |         0 |       0 |
+------------------------+--------+-----------+-----------+---------+
| ALL COMPARISONS        |   7210 |      7135 |         0 |      75 |
+------------------------+--------+-----------+-----------+---------+


## Run by run

| run                                                                                                                                       | tested | identical | DIFFERENT | errored |
| BASE determinism, matrix 1 (625 fixed shapes): off2 vs off (levers off twice)                                                             |    619 |       601 |         0 |      18 |
| matrix 1 (625 fixed shapes): on vs off                                                                                                    |    619 |       601 |         0 |      18 |
| BASE determinism, matrix 3 (random cross product): off2 vs off (levers off twice)                                                         |    671 |       650 |        10 |      11 |
| matrix 3 (random cross product): on vs off                                                                                                |    648 |       644 |         0 |       4 |
| live multi-turn sessions : on vs off                                                                                                      |    190 |       190 |         0 |       0 |
| live multi-turn sessions : on_c vs off                                                                                                    |    190 |       190 |         0 |       0 |
| long prompts (40K-185K words): on vs off                                                                                                  |      6 |         6 |         0 |       0 |
| long outputs (max_tokens 4,096): on vs off                                                                                                |      4 |         4 |         0 |       0 |
| prediction field built from the real reply: on vs off                                                                                     |    150 |       150 |         0 |       0 |
| streamed, matrix 3: on_s vs off_s                                                                                                         |    240 |       238 |         0 |       2 |
| streamed, matrix 1: on_s vs off_s                                                                                                         |    130 |       125 |         0 |       5 |
| request after the same request was cancelled mid-stream: on_c vs off                                                                      |     75 |        75 |         0 |       0 |
| re-check of x0606 (streamed reasoning, levers off and on): on vs off                                                                      |      1 |         1 |         0 |       0 |
| re-check of x0606 (streamed reasoning, levers off and on): on_s vs off_s                                                                  |      1 |         1 |         0 |       0 |
| matrix 4 (second random cross product, new seed): on vs off                                                                               |    642 |       642 |         0 |       0 |
| live multi-turn sessions, STREAMED: on_s vs off_s                                                                                         |    185 |       185 |         0 |       0 |
| streamed, matrix 4: on_s vs off_s                                                                                                         |    189 |       189 |         0 |       0 |
| 2 in flight + 1 client that cancels streams (default profile) [2 replies without a usable solo reference skipped]                         |    218 |       215 |         0 |       3 |
| 3 clients on 2 lanes, each request after a cancelled copy of itself, streamed and not                                                     |    359 |       347 |         0 |      12 |
| device check modes ON, matrix 1                                                                                                           |    260 |       251 |         0 |       9 |
| device check modes ON, session turns replayed                                                                                             |    190 |       190 |         0 |       0 |
| device check modes ON, matrix 3 [11 replies without a usable solo reference skipped]                                                      |    249 |       247 |         0 |       2 |
| device check modes ON, long prompts                                                                                                       |      3 |         3 |         0 |       0 |
| USERS16 profile, 16 in flight (burst) + 2 cancelling clients, matrix 1                                                                    |    616 |       598 |         0 |      18 |
| USERS16 profile, 8 in flight, streamed, slow readers, matrix 3 [11 replies without a usable solo reference skipped]                       |    319 |       317 |         0 |       2 |
| USERS16 profile, 4 in flight, each after a cancelled copy [10 replies without a usable solo reference skipped]                            |    230 |       230 |         0 |       0 |
| USERS16 profile, 3 sessions in flight (turns replayed)                                                                                    |    190 |       190 |         0 |       0 |
| USERS16 profile, 16 in flight (burst), matrix 4 [29 replies without a usable solo reference skipped]                                      |    642 |       642 |         0 |       0 |
| AGENTS profile (4 lanes on the head), 4 sessions in flight                                                                                |    190 |       190 |         0 |       0 |
| AGENTS profile (3 lanes on the head), 3 sessions in flight                                                                                |    190 |       190 |         0 |       0 |
| AGENTS profile, 3 in flight, matrix 4, each after a cancelled copy, streamed and not [14 replies without a usable solo reference skipped] |    284 |       284 |         0 |       0 |
| FLAGS-OFF ENGINE, session turns COLD (reverse order, fresh engine) vs resumed knob-off                                                    |    190 |       190 |         0 |       0 |
| FLAGS-OFF ENGINE vs knob-off, matrix 1                                                                                                    |    619 |       601 |         0 |      18 |
| FLAGS-OFF ENGINE vs knob-off, matrix 3                                                                                                    |    319 |       317 |         0 |       2 |
| FLAGS-OFF ENGINE vs knob-off, long prompts                                                                                                |      6 |         6 |         0 |       0 |
| FLAGS-OFF ENGINE vs knob-off, long outputs                                                                                                |      4 |         4 |         0 |       0 |

## Method (what was compared with what)

- Binary under test: the final image's engine (tensorfold-qwen38:v1.1.4-eab8582, patches 0101-0129), copied from spark-923b to the head,
  tensorfold-native sha256 af27a1e7b029e243 (116 files, 0 sha mismatches). KV cache bf16. Head launcher = the environment recipe.sh computes
  (recenv.sh) with 2 x 262K lanes instead of 3 (the head keeps a 13 GiB reserve for DeepSeek's rank 0, the gateway and the vision encoder).
- Reference = the SAME running engine with every lever that has a knob-file word switched off (23 words: cert, cert_topk, cert_gather, hcf,
  half_below, mtp_last, gdn_many, best_lanes, attn_lanes, late_skip, keep_inpass, lane_reuse, head2, head2_sampled, deep_off, stop2,
  ngram_split, qsa_cert, qsa_list, qsa_rows, qsa_cert_min, exp_half, prefix_wait_ms). Every request runs off / on back to back (order rotated),
  and off a second time: a request whose two levers-off replies differ is "base not deterministic" and is not counted for or against a lever.
- Levers WITHOUT a knob word (128-row copy windows + COPY_RESET + COPY_SHARED, PREFILL_MANY, elastic prompt cache, delta prompt states,
  prompt-pass levers GDN_PROD / ROUTE_BAND / SCRATCH_TRIM + CHUNK / HC_DN_TALL / HEAD_KV / WARM, QSA_HALF, HEAD2_FAST, CERT_PLAIN, CERT_CODE,
  DRAFTS=12) stay ON in the knob-off reference. They are covered by a SECOND ENGINE START of the same binary with every lever off by its
  flag (recipe options CERT=0 MANY=0 COPY_ROWS=16 FUSED=0 HALF=0 MTP_LAST=0 GDN_MANY=0 BEST_LANES=0 ATTN_LANES=0 LATE_SKIP=0 KEEP_INPASS=0
  LANE_REUSE=0 HEAD2=0 R9=0 R10=0 R11=0 PREFIX_WAIT=0, TF_FLASHNEXT_Q38_PREDICTION=0): its replies are compared with the knob-off replies.
- The `prediction` field: the reference is the same request WITHOUT the field (levers off).
- Compared per reply: HTTP status, error text, answer text, reasoning text, tool calls (name + arguments as canonical JSON), finish reason,
  prompt tokens, completion tokens (prompt tokens are not compared on /v1/messages: Anthropic accounting leaves cached tokens out).
- Streaming: the concatenated chunks of a streamed reply are compared with the streamed levers-off reply (and with the non-streamed one).
- In flight: each reply of a run with N requests in flight is compared with the SOLO levers-off reply of the same request.
- Sessions: live multi-turn conversations (each real reply goes back as the assistant turn), with "regenerate" and "edit an earlier turn"
  steps, under levers off / on / on after a client-cancelled stream; then every turn's request is replayed COLD (fresh flags-off engine,
  last turn first) and compared with the resumed reply.
- Tools: Mac q38/kitD/redteam (rt.py: gen / gen2 / gen3 / pair / pairsum / sess / run / cmp / bisect / probe; fuzzrep.py: the tables;
  rtq.sh / rtchain.sh / run12.sh / rj.sh: head queue with the window cap and the 12:04 UTC cutoff; stub.py: dry-check engine).

## Files

- Raw results: Mac q38/results/rD12/N/ (every reply of every run, report.txt, D_RT_N_alldiff.jsonl, engine start logs, runner logs); head ~/q38/runs/D_RT_N_*.
- Matrices: tools/matrix/ cases1.jsonl (625, exact token boundaries made against the engine), cases2.jsonl (345 predictions), cases3.jsonl (675),
  cases4.jsonl (671), sess.jsonl + sess4.jsonl (60 session scripts, 375 turns).
- Rebuild the tables: python3 -I tools/fuzzrep.py RUNS_DIR tools/matrix
- Fixes: fix-recipe-validate/ (recipe-validate.patch sha256 5732f061, patched recipe.sh 66fe0314), fix-messages-seed/ (source, one line, NOT built).
- Notes sent during the night: URGENT.md (U1-U5).
