# URGENT for the integrator (builder A): round-9 indexer knobs share ONE cache (patch 0116, cuda_knobs.zig q38EnvInt)

Found by builder C in round 10 (2026-11-14 05:29 UTC), MEASURED on spark-4e5c, base 7061396.

`q38EnvInt(comptime env, comptime knob, default)` keeps its value in `const st = struct { var v ... }`. That struct uses none of the
call's comptime values, so Zig makes it ONE type for every call: all seven knobs read through it (QSA_CERT, QSA_CERT_MIN, QSA_SLACK,
QSA_HALF, QSA_STATS, QSA_LOOP, and any new one) share one cached value: whatever the first call read.

Proof: an engine whose environment had NO `TF_FLASHNEXT_Q38_QSA_SAMPLE` (a new knob of mine read through q38EnvInt, default 0) ran
that path on 986,938 rows; with QSA_CERT=2 the same knob read 2, with QSA_CERT=1 it read 1.

What the shipped recipe (QSA_CERT=1 QSA_HALF=1) therefore really runs:
| knob | designed | what runs |
| QSA_SLACK (bound, millionths) | 100 | 1 (2 in check mode, QSA_CERT=2) |
| QSA_CERT_MIN (fewest rows of a group that take the certified path) | 64 (decode rows stay stock) | 1 (decode rows take it too) |
| QSA_STATS | off | on (a device read + one log line per request) |
| QSA_HALF | 1 | 1 (right by luck) |
In a knob-file engine with qsa_cert in the file, the first cached value is QSA_CERT_MIN's default 64, so the STOCK path ran with
QSA_LOOP=32: that is the "knob-file engine runs the stock indexer 1.5-2 % slower" puzzle of round 9 (inference; fits every number).

Is any output wrong? No evidence of it: the selection is still checked exact (round 9 check mode at slack 2: 4.5 M rows on 4e5c, 3.56 M
on 923b, round 10: 4.49 M rows: 0 lists differ, 0 bound violations, largest deviation 0.975 of the bound = the half rounding).
But the bound was designed with slack 100 (500x above the measured deviation, and above the worst-case fp32 summation error
gamma_128 ~ 1.5e-5); at slack 1 the fp32 part of the bound is below that worst case, so the "certified" claim rests on measurement only.

Fix (one line, in my round-10 patch, handoff/C/r10-prefill): the struct names `env` (`const variable = env;`), so each knob has its
own cache. After the fix the recipe runs slack 100, CERT_MIN 64, STATS off. Outputs are the same (the method is exact at any slack that
holds); speed: see my NOTE (measured with the fix). Until merged: set TF_FLASHNEXT_Q38_QSA_CERT_MIN / _SLACK explicitly does NOT help
(they read the shared cache); only the code fix does.
