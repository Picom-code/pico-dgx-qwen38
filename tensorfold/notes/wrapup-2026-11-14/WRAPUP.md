# Wrap-up, 2026-11-14 (work stopped at 21:08 UTC; the Spark setup is being taken down)

State: SHIP CANDIDATE, not shipped, repo private. Image = MiaAI base + patches 0101-0129. Default KV cache = bf16 (exact). All builders stopped by the owner's
request in the middle of the final suite, so the tables below are partial; what is missing is listed.

## Final image, one user, spark-923b, documented path, bf16 KV, 3 x 262K, one window (measured)

| bench | levers off (19:10-20:23) | recipe default (20:23-21:07) |
|---|---|---|
| first token, short prompt | 183 ms | 119 ms |
| 3-prompt chat / code, 9 runs (median) | 46.1 / 75.8 | 52.1 / 90.8 |
| ALL 320 held prompts (315 answered) | 57.39 | 65.98 (+15.0 %) |
| sparkDash-method prose / code / structured (T0) | 58.3 / 91.5 / 105.8 | 69.7 / 116.5 / 162.6 |
| sparkDash 1.8.8 itself, c1 prose / code / structured | ~57.9 / ~91.2 / ~104 | ~68.8 / ~115.8 / ~160 |
| sparkDash 1.8.8 catalogs prose / structural / code | (see suite.log) | 48.3 / 76.1 / 96.1 |
| prefill 7.4K / 29K / 116K tok/s | 2261 / 2310 / 2134 | 2559 / 2499 / 2401 |
| 182K-token needle | - | 3/3, first token 77.2 s |
| image / qeval | - | PASS / 32 of 35 (math 11/14) |

Identity, default vs levers off, this image, bf16 KV: identity set 51/51 and all 320 held texts identical (compared on the Mac from the pulled run files).
NOT RUN on the final image (killed at wrap-up): second levers-off run (so the pair is not bracketed), edit / agent sets, seeded sampled pair, 227K prompt,
check modes (certified head counters, indexer select, delta states, scratch guard), the many-user suite, agent traces, `recipe.sh prepare` after the last two
recipe.sh edits (POOL16_BF16=40960; AGENTS default 5 lanes under bf16: committed with this note, never started).
Raw: Mac q38/results/wrapup/r923_F11.tgz.

## bf16 KV, many users (builder C, spark-4e5c, one window, references first and last, measured)

- bf16 KV costs no decode speed at 1-16 users: 16-lane pool, 16 users 289.1 / 286.7 / 292.0 (fp8 277.7-289.2 and 282.5-284.4); held prompts 16 in flight 197.2 vs 197.3 / 195.6.
- Fit: 5 x 262K is refused (41.8 GiB needed, 28.1 budget); 3 x 262K fits (8.36 GiB a stream); the 16-lane pool starts at a 32,768-token lane window
  (1,745 MiB a lane; a full 262K request takes 8.5 GiB and is admitted by bytes); 12 x 64K and 8 x 96K are worse for 12-16 users.
- What fp8 KV changes: 37 of 51 identity-set replies differ in wording; ~7.7 divergence events per 1,000 generated tokens; same quality score within noise.
- Agents under bf16: a kept state costs 30.1 KiB a token (fp8 17.5); with elastic cache + delta states 5 agents x 56K resume 25/25 (follow-up median 1.7 s on 3 lanes);
  ELASTIC low-water 7.5 GiB on the default profile (below the 9 GiB reserve): a reserve of 11 GiB was queued, not measured.
- PROFILES.md was not written; the U16 bf16 agents run and the check modes under bf16 were in progress. Raw: Mac q38/results/wrapup/r4e5c_C14.tgz, log PROGRESS-C.md 'Round 14'.

## Not done before shipping (in order)

1. Bracketed final suite on the final image (off / on / off), incl. edit, agent, sampled, 227K, check-mode counters.
2. Many-user final table on bf16 (16-lane pool, 32-41K lanes) against MiaAI's configuration; agent traces; identity 16 in flight under bf16.
3. Red team: exactness fuzz and soak were started twice and stopped within minutes: NOTHING was tested (tools: Mac q38/kitD/redteam, head ~/q38/kit-D/redteam).
4. From-scratch install on a box that never had our builds.
5. README pass for shipping (first table on bf16; exact / not exact section; credits; memory section), SHIP-CHECKLIST.md (not written).
6. Owner decisions: public or private; ELASTIC default and reserve; whether fp8 'more users' stays as an option at all (bf16 serves 16 users at the same speed,
   with a 32-41K lane window instead of 48-64K).

## Where things are

- Private repo Picom-code/pico-dgx-qwen38, branch tensorfold-fp4-local: patches 0101-0129, recipe.sh, README, notes per round.
- Mac ~/.hermes/cache/scratch/q38: PROGRESS.md (rounds 1-7), PROGRESS-A/B/C/D.md (rounds 8-14), handoff/ (every builder's patch + NOTE), wt/ (source worktrees;
  integration branch integ11 = 17a4c9d), results/, shared/ (window harness), research/.
- On the Sparks (kept, nothing deleted): ~/q38 on each box: models (official checkpoint on all four), build trees, binaries, run logs; head ~/q38/shared
  (lease.sh, qws.sh, reaper.sh). DeepSeek production restored and verified at 21:12 UTC (health 200, 391, session store 12.00 GiB, image).
- Skills: llm-inference-systems-playbook (roadblock table, lever ledger, invention protocol, rulers, paper maps); dsgx10-spark-deepseek-engine reference
  qwen38-fastest-build.md.
