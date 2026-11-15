# Changes in pico-dgx-qwen38

## tensorfold-fp4-local branch (2026-11-12)

`tensorfold/`: the official NVIDIA FP4-Local checkpoint on TensorFold + MiaAI-Lab patches, measured against custom engine R and a third engine on one Spark; our exact levers (patches 0101-0104) and why most do not pay; recipe.sh. See tensorfold/README.md.

## 2026-11-14: the TensorFold FP4-Local recipe becomes the main content

- `tensorfold/` (patches 0101-0129 on MiaAI-Lab's TensorFold recipe, official NVIDIA FP4-Local checkpoint, bf16 KV cache by default) merged into `main`;
  the top-level README now describes it. Status: ship candidate; open items in `tensorfold/notes/wrapup-2026-11-14/WRAPUP.md`.
- Round-by-round changes, corrections (incl. the patch 0120 knob-cache fix) and dead ends: `tensorfold/README.md`.


## 2026-11-15: final measured night (four agents, one Spark each)

Numbers: top-level README; every table and raw file name: `tensorfold/notes/final-2026-11-15/FINAL.md`.

- **Final tables on the final image** (patches 0101-0129, bf16 KV): one user levers off / recipe default / levers off in one window on spark-923b
  (320 held prompts 59.1 / 66.9 / 58.4 tok/s; first token 180 / 120 / 183 ms; file rewrites 140 / 227 / 138), identity counts, check-mode counters,
  16 users on the bf16 pool against MiaAI-Lab's shipped shape in one window (288 against 220 / 219 tok/s; last user's first token 0.9-1.0 s against 6.8 s).
- **Fix: `USERS16=1 ./recipe.sh start` did not start on a fresh box.** The bf16 pool default 16 x 40,960 tokens needed 0.2 GiB more than the engine's
  budget on spark-4e5c (it fitted on spark-923b by 0.6 GiB). Now 16 x 32,768 (measured on both boxes).
- **Fix: a typo in `KV=` silently selected the lossy fp8 cache.** recipe.sh treated every value but the literal `bf16` as fp8. Now only `KV=fp8` selects
  it (with a one-line warning) and any other value stops. Other nonsense option values stop too.
- **recipe.sh for strangers** (from the from-scratch install): the Hugging Face cache follows `HF_HOME`; the base image is pinned by digest; its own image,
  container and state names (no collision with MiaAI-Lab's recipe on the same box); `./recipe.sh verify` (checkpoint sha256 list); `bench` prints the
  table's one-user rows; a clear message when `start` runs before `prepare`.
- **Claims reworded** (measured this night): "10 agents x 56K: 142 s -> 5.5 s" was an fp8-KV number; on the bf16 default 10 agents read 64.5 s (13 of 20
  follow-ups from the cache) and 5 agents 1.3 s (25 of 25). custom engine R "faster on chat" -> ties on the 3-prompt chat / code bench (same-window runs on spark-4e5c).
  No prefill lead: a vLLM recipe reads a cold prompt at 3.5-4.0K tok/s against our 2.4-2.6K. "Sampled output identical per seed" holds on
  /v1/chat/completions and /v1/responses; /v1/messages ignores `seed` in the base engine.
- **Not merged:** a one-line engine patch that forwards `seed` on /v1/messages (not built, not tested).
- Docs: private network addresses in older harness scripts and notes replaced by host names.
