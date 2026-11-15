# Builder D handoff (head spark-3caa, many-user MoE path), 2026-11-13

Base commit for every patch: 52b32e1 (patches 0101-0109). Branch `builder-D` in Mac `q38/wt/D` (commit 0c0eda1, local only).
Builds on the head: `~/q38/tfbin/D1` (XSTAT aid), `D2` (+ EXP_PAIRS), `D3` (+ ROW_TRIM; sha 51c11631eda83102). All flags default to stock.

## Bottom line
No positive lever came out of the MoE lever area. The expert path is at the memory bus for the rows a pass holds, and the rows cannot be
arranged to need fewer experts. Three exact experiments were built and measured: one negative, two zero. Nothing here should change the recipe.
What is worth integrating is the log aid (measuring only).

## 1. xstat-log-aid/  (INTEGRATE if you want the measuring aid; changes no value)
- `xstat.patch`: `TF_FLASHNEXT_Q38_XSTAT=FILE` (optional `TF_FLASHNEXT_Q38_XSTAT_MAX=blocks`). Every MoE block appends: u32 magic 0x58333851,
  model layers, layer, rows, slots, lanes, each lane's row count, picks i32 [rows, slots], weights f32 [rows, slots]. One stream wait a block.
  Files: zig/src/families/flashnext/cuda_batch.zig (1 line), cuda_batch_moe.zig (struct Xstat).
- `xstat.py FILE` (numpy; on the head run it in the pytorch image: `docker run --rm --cpus 2 --memory 4g -v ~/q38/prof-D:/p:ro -v ~/q38/kit-D:/k:ro
  --entrypoint python3 nvcr.io/nvidia/pytorch:26.07-py3 /k/xstat.py /p/x16.bin`), `lanehist.py FILE` (stdlib).
- Speed with the aid on: 16 users 268.8 / 275.2 (reference 270-274): the per-layer wait costs nothing measurable. Texts with 16 in flight 107/107 = kp25.
- Raw output: `xstat_raw_8_16_users.txt`.

### Expert statistics (MEASURED, head, D1, 16 x 32K slots, distinct coding prompts 512 tokens)
```
+----------------------+-------+----------+--------+---------+-----------+-------------+---------------+---------------+
| load                 | rows  | distinct | of 512 | rows an | GiB a pass| extra row   | 16-pair items | a lane alone /|
|                      | a pass| a layer  |        | expert  | (experts) | (experts)   | re-read       | pair Jaccard  |
+----------------------+-------+----------+--------+---------+-----------+-------------+---------------+---------------+
| 8 users (5-8 lanes)  | 49.0  | 166.4    | 32.5 % | 2.94    | 21.8      | +2.40/layer | 1.1 %         | 36.0 / 0.118  |
| 16 users (13-16)     | 78.5  | 228.3    | 44.6 % | 3.44    | 30.1      | +1.22/layer | 1.9 %         | 30.7 / 0.085  |
| no lane table (16 u)*| 25.4  | 61.1     | 11.9 % | 4.16    | 8.5       | -           | 6.9 %         | -             |
+----------------------+-------+----------+--------+---------+-----------+-------------+---------------+---------------+
```
  * passes without a lane table: 144 lone-stream decode passes of 1-7 rows (the tail of a bench) and 64 prompt passes (24 of them 65-255 rows).
- An expert is 2.7648 MB (gate + up + down NVFP4 codes and e4m3 scales). Extra row: 8 users 319 MB = 1.42 ms, 16 users 162 MB = 0.72 ms (at 225 GB/s).
- Round 7's "159 distinct experts a layer" was inferred from an assumed 225 GB/s. The measured union is 228. With GPROF on the head
  (72.6 rows, MoE group 117.4 ms) the MoE group moves >= 254 GB/s, about 93 % of LPDDR5X-8533 x 16 channels (273 GB/s, theoretical).
- By layer (16 users): layer 0 reads 317 distinct, layers 36-47 about 204.
- Rows a lane (16 users, coding + held prompts): 2: 17 %, 3: 12 %, 4: 15 %, 5: 11 %, 6: 9 %, 7: 35 % (the cap: bonus + 6 drafts); mean 5.1.
  With DRAFTS=8 on coding prompts alone: 9 rows: 39 %, mean 7.05, 108 rows a pass, and the same 265-272 tok/s (section 4).
- Routing weights exactly zero: 0 of 25.7 million (smallest 0.0028). Experts are SwiGLU. No exact skip exists.
- Used experts adjacent in id: 0.447, chance is 0.446. A storage order by co-activation has nothing to give (and the reads are at the bus rate anyway).
- SIMULATED: splitting a 16-user pass into two sub-passes grouped by expert overlap reads x1.31 the bytes (random halves x1.36, each lane alone x2.06).
  One pass is the minimum for its rows; composing passes differently can only add reads.
- Router (ARITHMETIC on KPROF 16 users, 3d24 U2_kp16): logits + top-k + plan kernels are 1.4 % of serialised GPU time; the weights are 2.6 MB a layer.
  A certified coarse router is bounded below 1 % even at zero cost. Not built.

## 2. dead-exp-pairs/  (DO NOT INTEGRATE: negative)
`TF_FLASHNEXT_Q38_EXP_PAIRS=N`: decode expert items of up to N pairs instead of 16, so an expert picked by 17+ rows is read once (1.9 % of bytes).
```
| run (D2, head, one window)  | 16 users distinct      | 16 copies |
| stock 16-pair items         | 271.7 / 272.9 / 271.3  | 343.6     |
| EXP_PAIRS=64                | 264.4 / 265.0 / 265.7  | 320.7     |   -2.5 % / -6.7 %
```
Cause: units are dealt to warps by a fixed stride; a long item makes one warp walk the same block 2-4 times while its neighbours finish.
Stock's second item runs on another warp at the same time. Applies on top of xstat.patch.

## 3. dead-row-trim/  (DO NOT INTEGRATE: zero)
`TF_FLASHNEXT_Q38_ROW_TRIM=1`: 64-row lane tiles (qmmf.cu) multiply only the 16-row tiles that hold live rows (78 rows: 80 rows of mma instead of 128).
The flag travels in bit 16 of the kernel's `group` argument; the stage loop is a generic lambda over a compile-time tile count.
```
| run (D3, head, one window) | 16 users distinct      | copies | 8 users | held 16 in flight | GPROF ms: GDN, hc, QSA, head, MoE (pass) |
| ref1                       | 272.5 / 273.5 / 274.3  | 339.7  | 210.3   | 179.8             | 33.27, 7.94+7.02, 8.41, 7.09, 118.9 (183.7) |
| ROW_TRIM=1                 | 272.4 / 271.2 / 265.5  | 339.3  | 211.4   | 178.7             | 33.29, 7.96+6.99, 8.36, 7.16, 119.6 (184.5) |
| ref2                       | 269.3 / 267.2 / 269.0  | 333.8  | 208.4   | 176.9             | 33.23, 7.92+7.01, 8.41, 7.11, 119.0 (183.8) |
```
Identity: held 16 in flight 107/107, waves of 16 128/128 against ref1. Exact, engaged (log line), and every dense group is unchanged to 0.05 ms:
the padded row tiles' mma work is not what the lane kernels wait for. My estimate (dense compute-bound at ~28 TFLOP/s) was wrong.

## 4. Draft depth and gate at 16 users (settings only, D3, window W5; each run: distinct 16 x3, distinct 8, held 16 in flight)
```
| run                           | 16 users distinct      | 8 users | held16 | GPROF rows / ms a pass | texts vs ref |
| ref1 (W4, 25 min earlier)     | 272.5 / 273.5 / 274.3  | 210.3   | 179.8  | (other load mix)       | -            |
| DRAFTS=8                      | 275.5 / 273.9 / 273.4  | 210.3   | 176.0  | 75.4 / 180.9           | 107/107      |
| DRAFTS=10 + WIDE_ROWS=192     | 263.3 / 265.5 / 268.2  | 205.5   | 176.6  | 77.0 / 182.7           | 107/107      |
| DRAFTS=8 + CONFIDENCE=-0.3    | 265.7 / 259.7 / 263.1  | 209.8   | 175.7  | 77.7 / 181.4           | 107/107      |
| ref3 (0.4 / 6, last)          | 266.3 / 270.7 / 273.0  | 210.4   | 177.6  | 70.8 / 179.4           | 107/107      |
```
DRAFTS=8 adds 4.6 rows a pass for 1.5 ms (0.33 ms a row: rows are cheap, as the statistics say) but the tokens gained are inside the
reference spread (274.3 against 273.4 and 270.0; held16 176.0 against 179.8 and 177.6). Deeper or looser is negative. MiaAI's 0.4 / 6 stays.

## What remains (for other builders; ESTIMATES from the statistics)
- Prompt passes under churn: 64 of 880 passes in the 16-user runs (2 x 16 distinct + 107 held prompts with 16 in flight) were prompt passes, so
  batching (PREFILL_MANY) already folds 139 prompts into 64 passes. Riding in the decode pass instead (upstream KEEP_INPASS, queue item U3,
  builder C's area) could save at most ~50 ms each = ~2 % on held / waves loads, nothing on the 16-arrive-together bench.
- Everything else in the expert path is bytes at the bus rate. A faster 16-user pass needs fewer distinct experts a token: better draft acceptance
  (builder B's side), not kernel or scheduling work.

## Head as a test box (measured)
- Total 119.7 GiB. DeepSeek stopped: 7.6 GiB used (gateway 31 MiB, vision encoder 1.26 GiB host + 1.67 GiB GPU, system), MemAvailable 112 GiB.
- Engine: GPU 78.7 GiB. It sizes its streams from MemAvailable - 9 GiB = 23.4 GiB on the head, so 5 x 262K (24.2 GiB) and 16 x 64K (25.1 GiB)
  are refused by the engine itself. Use 3 x 262K for one user and 16 x 32K for many users.
- MemAvailable while running: READY 31-32 GiB; one user 24.9 GiB minimum; 16 users 18-19 GiB; waves of 16: 15.9 GiB. The watchdog (6 GiB) never fired.
- Speed = the workers: one user chat 50.6 / code 82.3 (3 runs), ttft 178 ms; 16 users 270-274; 8 users 208-211; held 16 in flight 177-180.
- Scripts: head `~/q38/kit-D/{run,wd,stop,build}.sh` (Mac `q38/kitD/`). run.sh refuses without a covering lease or below 95 GiB MemAvailable,
  container `--memory 12g`; build.sh is 3 CPUs / 8 GiB and removes itself when the window closes.
