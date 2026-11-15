# Self-test: the exactness fuzzer (agent D, night of 2026-11-15)

`fuzz/rt.py` sends the request matrices in `fuzz/matrix/` (2,316 request shapes + 60 session scripts) to a running engine twice, once with the
levers on and once with every lever off, and compares every reply (text, reasoning, tool calls, finish reason, token counts).
`fuzz/fuzzrep.py RUNS_DIR fuzz/matrix` rebuilds the tables. Method, results and what was NOT covered:
[../notes/final-2026-11-15/agent-D-night-1114-fuzz-FUZZ.md](../notes/final-2026-11-15/agent-D-night-1114-fuzz-FUZZ.md).

Result of that night on the final engine build: 7,210 levers-on against levers-off comparisons, 0 different (75 refused the same way);
713 requests with the device check modes on, 0 mismatches. The scripts are the ones that ran on our boxes: paths and container names in the
`*.sh` files are ours and need editing for another setup; `rt.py` and `fuzzrep.py` are stdlib Python.
