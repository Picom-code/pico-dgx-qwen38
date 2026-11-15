# From-scratch install of the README's "run it" block (agent C, spark-4e5c, 2026-11-15 UTC)

Box: spark-4e5c (DGX Spark, GB10, 121 GiB, DGX OS, driver 580.x), which never had a `recipe.sh prepare` image. Test window 1 (04:30-09:10 UTC).
Repo: FRESH `gh repo clone Picom-code/pico-dgx-qwen38 -- --branch main` on the Mac at 04:29 UTC = commit 42291bf, copied over without `.git`.
Nothing of our earlier work was reused: no build tree, no binary, no checkpoint copy, no head2 file. Reused: only the base image
`nvcr.io/nvidia/pytorch:26.07-py3`, after I checked by hand that its RepoDigest is sha256:2140e699...f34a1c, the digest named in recipe.sh's header
(the recipe itself does NOT check it: correction 6). The 102 GB checkpoint was downloaded again by the recipe, as a stranger's would be.
Only addition to the README's commands: `CONTAINER_NAME=q38-run-C` (the test window's harness only sees containers named q38-run*).
Raw logs: spark-4e5c ~/q38/fresh-1114/log/ (chain.log = one line per result), copies in Mac q38/results/night1114-C/.

## VERDICT

Does a stranger get a working exact engine from the README?

- `./recipe.sh prepare`, `./recipe.sh start`, `./recipe.sh bench`, `AGENTS=1 ./recipe.sh start`: **YES**. Built, started on the exact profile
  (bf16 KV, 3 streams at 262,144), and gave the same texts as spark-923b's final image, at the front page's speed or slightly above.
- `USERS16=1 ./recipe.sh start` (third line of "run it"): **NO, it does not start** as shipped at 42291bf. One-number fix, measured (correction 1).
- Two more cases fail for some strangers: `HF_HOME` users (correction 2) and people who already run MiaAI-Lab's recipe on the box (correction 3).

## Steps and timings (measured)

| step | command (README, literal) | result | time |
|---|---|---|---|
| 1 | `cd tensorfold && ./recipe.sh prepare` | rc 0 | 37 min 49 s total |
| 1a | - clone MiaAI recipe at 20e15e4, copy patches 0101-0129, `docker build` (TensorFold eab8582 + 46 patches, Zig 0.17.0, codecs) | image tensorfold-qwen38:v1.1.4-eab8582, 24.5 GB, patches hash 1a42c7208df0 | 10 min 40 s (the Zig download alone took 4.5 min: another download of mine shared the uplink; expect ~7 min on a free link) |
| 1b | - checkpoint download, 102.26 GB, 33 files, no token ("unauthenticated requests" warning only; host has no `hf` CLI: it downloads from inside the container) | complete | 27 min 3 s at 47-63 MB/s |
| 1c | - checkpoint check + two-stage head file | "Checkpoint OK ... 375487 tensors in 21 shards"; head2 sha256 fce7c913060d45f2 "= the measured file" | 6 s |
| 2 | `./recipe.sh start` | rc 0; "Window 262144 tokens - 3 at once - bf16 KV - drafts MTP"; engine: "3 streams at once, 8.39 GiB each at a 262144-token window, of 27.0 GiB left after a 9.0 GiB reserve"; MemAvailable 35 GiB, GPU 80.1 GiB | 156 s (214 s when the checkpoint is not in the page cache) |
| 3 | `./recipe.sh bench` | rc 0, table below | 14 min 54 s |
| 4 | `./recipe.sh stop` (in recipe.sh, not on the front page) | rc 0; no container, port 8888 free, no GPU process, MemAvailable back to 116 GiB | 5 s |
| 5 | `USERS16=1 ./recipe.sh start` | **FAILED**: "tensorfold: --parallel 16 needs 27.38 GiB for its streams at a 262144-token window, and the CUDA memory budget leaves 27.18 GiB"; "ERROR: the server exited (code 1) before it was ready" | 211 s to fail |
| 6 | `AGENTS=1 ./recipe.sh start` | rc 0; 5 streams, pool 5 x 131,072; first token 121 ms; 5 users at once 175.4 tok/s (last first token 0.84 s); chat / code 51.0 / 91.9 (3 runs); prefill 29K 2610 tok/s; stop clean | 151 s |
| 7 | tensorfold/README "every lever off" line, literal | rc 0; first token 180 ms; chat / code 47.5 / 78.0; held 59.14; stop clean | 153 s |

Start line knobs (step 2, `docker inspect` + engine log): KV_DTYPE bf16, PARALLEL 3, CONTEXT 262144, and every knob variable recipe.sh sets for the default
profile: the container's environment was compared with the list evaluated from recipe.sh on the Mac (q38/C1114/expect/default.env): 42 of 42 equal, plus the three
HEAD2 variables (file, FAST=3, SAMPLED=1). Same check for AGENTS (43 of 43), the every-lever-off line (27 of 27) and USERS16 (43, POOL_CTX as overridden). Engine log lines: "q38 cert: SM8 copy of the head
made (625 MiB) ... top-k up to 20", "q38 head2: on, rank 512 ... fast 3", "q38 scratch trim", "q38 gdn prod: 2", "q38 route band: 2048", "q38 qsa knobs: cert=1
cert_min=1 slack=100.0e-6", "q38 warm: a 3072-row prompt pass", "q38 elastic: the prompt cache takes 1914 MiB at start-up and up to 23850 MiB more".

## Reproduced numbers next to the front page (fresh install, one user, default profile, bf16 KV, 3 x 262K)

| bench | fresh install, spark-4e5c, window 1 | front page (spark-923b, 2026-11-14) | verdict |
|---|---|---|---|
| first token, short prompt (median of 7; 3 runs) | 123 / 119 ms (patched recipe run: 112) | 119 ms | same |
| 3-prompt chat / code, 9 runs (median) | 54.9 / 90.9 | 52.1 / 90.8 | same (chat single runs swing 49.8-57.7) |
| sparkDash-method prose / code / structured | 71.9 / 120.3 / 167.2 | 69.7 / 116.5 / 162.6 (table rounds to 69 / 116 / 160) | +3 % |
| held prompts, every third of the 320 (104 scored) | 68.56 | 66.87 on the same 104 prompts (65.98 on all 315) | +2.5 % |
| prefill 7.4K / 29K / 116K tok/s | 2585 / 2478 / 2494 | 2559 / 2499 / 2401 | same |
| levers off (README line), same window: first token / chat / code / held | 180 ms / 47.5 / 78.0 / 59.14 | 183 ms / 46.1 / 75.8 / 57.4 | same; default is +15.9 % on held here (front page +15.0 %) |
| USERS16 with the fix (POOL_CTX=32768): 4 / 8 / 16 users distinct | 162.5 / 220.7 / 278.6, 282.3, 289.1; last first token 0.38 / 0.62 / 0.9-1.0 s | 158 / 221 / 287-292; 0.7-0.9 s | same |

Texts (greedy, compared with `qk.py cmp`, exact string equality):

| pair | identity set | held slice |
|---|---|---|
| fresh install default vs spark-923b final image F11_on | 51/51 identical | 107/107 identical (same token counts) |
| fresh install: every-lever-off line vs default | 51/51 identical | 107/107 identical |
| fresh install levers off vs spark-923b F11_off | 51/51 identical | - |
| patched recipe.sh (below) vs unpatched fresh install | 51/51 identical | - |

Checkpoint integrity: after the download, all 23 large files match the sha256 list of revision b9ce18a6 from the Hub (`recipe.sh verify` of the patch: "23 of 23 files OK").

## Corrections (each as an exact change). Tested ones are marked.

Files in this directory: `recipe.sh.patch` (against tensorfold/recipe.sh at 42291bf; contains corrections 1, 2, 3, 6, 7, 8, 9),
`checkpoint-b9ce18a6.sha256` (new file for tensorfold/, used by `recipe.sh verify`), `README.patch.md` (text changes), `URGENT.md`.

1. **BLOCKER. `USERS16=1 ./recipe.sh start` does not start.** recipe.sh line 94: `POOL16_BF16=40960` -> `POOL16_BF16=32768`.
   Measured tonight on the fresh install: 40960 fails (needs 27.38 GiB, budget 27.18); 36864 starts but leaves 0.4 GiB of prompt cache; 32768 starts with
   3.5 GiB of prompt cache and the same speed (table above). The budget itself moved between 25.9 and 27.2 GiB across starts, so 36864 has no margin.
   Also change the texts that name the lane size: recipe.sh comment "16 x POOL16_BF16 tokens", WRAPUP / README "32-41K lanes" -> "32,768-token lanes".
   TESTED (as `POOL_CTX=32768`).
2. **Fails for `HF_HOME` users.** recipe.sh finds the checkpoint and the head2 file under `${HF_CACHE:-$HOME/.cache/huggingface}`; MiaAI's scripts use
   `${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}`. Measured: with `HF_HOME` set and nothing under ~/.cache/huggingface, `prepare` ends rc 2 right after
   MiaAI's "Done": `ls: cannot access '.../.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/*'`.
   Fix: one line near the top, `export HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"`, and `$HF_CACHE` in the four places. TESTED (rc 0, same head2 sha).
3. **Collides with MiaAI-Lab's own recipe on the same box.** Image tag `tensorfold-qwen38:v1.1.4-eab8582`, container `qwen38-flash-next-tf`, state dir and kernel
   cache are MiaAI's names: our `prepare` overwrites their image tag, our `start` stops and removes their running container, and afterwards each recipe sees
   "patch set changed". Fix: recipe.sh exports its own `IMAGE=pico-dgx-qwen38-tf:v1.1.4-eab8582`, `CONTAINER_NAME=pico-qwen38-tf`, `STATE_DIR`, `KERNEL_CACHE`, `LOG_DIR`.
   TESTED (prepare 2 s from the build cache, start 216 s, 51/51 texts). Note: MiaAI's start.sh footer still prints "Restart ./start.sh restart / Stop ./stop.sh";
   the README must say `./recipe.sh stop`.
4. **`recipe.sh bench` is not "the benches behind the table".** It runs `distinct 4` and `distinct 5` on the default profile, which has 3 streams: measured 124.9 and
   139.2 tok/s with the last user's first token at 10.2 and 10.7 s (they queue). The table's "4 / 8 users 158 / 221" row is the USERS16 profile. It has no
   first-token and no prefill line. Fix in the patch: bench adds `ttft` and `prefill 8000,32000,128000`, and runs `distinct 3` on the default profile,
   `4 8 16` with `USERS16=1` (`BENCH_USERS` overrides). README: say which rows `bench` reproduces and that the many-user rows need `USERS16=1 ./recipe.sh start` first.
   Patch NOT run end to end (pieces run by hand).
5. **tensorfold/README.md "Reproduce" is stale.** (a) The "every lever off" line lacks `R11=0`: 8 round-11 knobs stay on (CHUNK=4608, COPY_SHARED, GDN_PROD, HC_DN_TALL,
   NGRAM_SPLIT, ROUTE_BAND, SCRATCH_TRIM, SNAP_DELTA). Change the line to end `... R9=0 R10=0 R11=0 PREFIX_WAIT=0 ./recipe.sh start   # every lever of ours off,
   bf16 KV, 3 streams` and add `KV=fp8` for "MiaAI-Lab's configuration". (Texts are identical either way, measured; the speed column was measured with R11=0.)
   (b) "`./recipe.sh start` ... (5 users at 262K)" -> "(exact profile: bf16 KV, 3 users at 262K)". (c) delete the line `KV_DTYPE=bf16 PARALLEL=3 ./recipe.sh start`
   (it is the default) and add `KV=fp8 ./recipe.sh start   # MiaAI-Lab's lossy KV cache, 5 users at 262K`. (d) configuration table row "KV cache | fp8 (MiaAI's default)"
   -> "bf16 (exact); `KV=fp8` is the opt-in". (e) the headline block at the top is the round-10 fp8 table (patches 0101-0123); "Files" stops at patch 0118.
6. **Pins that are not enforced.** The base image is pinned by tag; the digest in recipe.sh's header is only a comment. The checkpoint is checked for "every shard is
   there and not empty", no hashes. Fix: `export BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699...` (prepare.sh and mkhead2 both use it) and a new
   `./recipe.sh verify` (sha256 of the 23 large files against checkpoint-b9ce18a6.sha256; 11 s when the files are in the page cache). TESTED (both).
7. `prepare` first tries `docker pull ghcr.io/miaai-lab/...:v1.1.4-eab8582-<hash>`, a tag that cannot exist. Fix: `export PULL="${PULL:-0}"`. TESTED.
8. `./recipe.sh start` or `stop` before `prepare`: a bare `cd: ...: No such file or directory`. Fix: a one-line message. `prepare` never removes stale `01*.patch`
   from MiaAI's tree: `rm -f "$SRC"/patches/01*.patch` before the copy. In the patch; the message path not exercised.
9. The head2 file was written by a root container into the user's HF cache (root-owned file). Fix: `--user "$(id -u):$(id -g)" -e HOME=/tmp` in mkhead2.
   TESTED (same sha256 fce7c913060d45f2, owner = the user).
10. **Front page "run it" lacks what a stranger needs** (text in README.patch.md): prerequisites (Docker with the NVIDIA runtime, user in the `docker` group, git,
    python3, curl, ~160 GB of free disk, internet to github.com, nvcr.io, ziglang.org, storage.googleapis.com, huggingface.co; no HF token), time (image ~10 min,
    download 102 GB = 27 min at 60 MB/s, start 2.5 min), a test request, `./recipe.sh stop`, where the log is (`docker logs -f <container>`).
11. Front-page table: column 1 "MiaAI config (our levers off)" has "KV cache: fp8 (lossy)" while the footnote says the one-user rows of columns 1-2 were measured
    with bf16 KV. Say "bf16 for the one-user rows; fp8 for the many-user rows (c)" in that cell, or split the column.
12. Front page: "Per-round tables ... are in tensorfold/README.md and CHANGES.md": the top-level CHANGES.md is the old vLLM build's list (2026-11-06). Drop "and CHANGES.md"
    or move that file into vllm-another vLLM recipe/.
13. Table numbers to refresh if A agrees (fresh install is at or above every one-user row; nothing needs to go DOWN).

## The PATCHED recipe.sh, every line of the corrected "run it" block (window 2, 09:15-09:58 UTC, measured)

`recipe.sh.patch` applied to a copy of the fresh clone (corrections 1, 2, 3, 4, 6, 7, 8, 9 together), commands typed as in README.patch.md:

| command | result |
|---|---|
| `./recipe.sh prepare` | rc 0 in 2 s (image `pico-dgx-qwen38-tf:v1.1.4-eab8582` built earlier from the layer cache with the digest-pinned base; checkpoint present) |
| `./recipe.sh verify` | "23 of 23 files OK", 13 s |
| `./recipe.sh start` | rc 0, 169 s, bf16 KV, 3 x 262K |
| the README's test `curl` | answers "Hello!" |
| `./recipe.sh bench` (patched: + first token, + prefill, 3 users) | rc 0, 15 min 44 s: first token 113 ms; chat / code 53.7 / 88.4; sparkDash-method 69.9 / 117.3 / 163.2; held 67.55; prefill 2587 / 2547 / 2495; 3 users 138.4 tok/s (last first token 0.48 s) |
| `./recipe.sh stop` | clean |
| `USERS16=1 ./recipe.sh start` | rc 0, 220 s: "16 lanes share 24189 MiB (1511 MiB a lane at a 32768-token window)" |
| `USERS16=1 ./recipe.sh bench` | rc 0, 16 min 28 s: first token 117 ms; chat / code 53.8 / 89.5; held 67.51; prefill 2557 / 2625 / 2554; 4 / 8 / 16 users 156.1 / 222.0 / 284.4 (last first token 0.45 / 0.58 / 0.98 s) |
| `AGENTS=1 ./recipe.sh start` | rc 0; first token ~120 ms; 5 users 179.7 tok/s (last first token 0.74 s); stop clean |

Texts: patched default vs the first fresh-install run: identity set 51/51, held 107/107 identical. USERS16 profile vs default profile: 51/51 and 107/107 identical
(the 16-lane pool gives one user the same text and the same speed as the default profile).

## What was not checked
- `recipe.sh bench` of the PATCH end to end; `RECALL=1`, `KV=fp8`, `GATE=structured`, `ELASTIC=0`, the `prediction` field, images; a 262K-token request; anything longer than a few minutes of load.
- A box with less free memory than this one (116 GiB MemAvailable before start), or a different DGX OS / driver.
