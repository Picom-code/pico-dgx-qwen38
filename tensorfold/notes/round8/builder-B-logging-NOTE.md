# 0110 draft-study logging (builder B) : measuring aids only, off unless set

Patch: `0110-q38-draft-study-logging.patch` against base 52b32e1 (patches 0101-0109). Two files: `zig/src/core/lanes/engine.zig`,
`zig/src/families/flashnext/cuda_mtp_ops.zig`. No behaviour change without the variables (built as B1 / B2 on spark-3d24;
B2 sha 75083f7649aa3bf7; a default-gate held run with logging on read 61.22 / 60.98 tok/s).

| variable | what it adds |
|---|---|
| `TF_STEP_LOG=2` | everything `=1` prints, plus `QTOP j tok:prob x8` (the draft head's 8 best tokens of chain step j), `QCTOK` (the chain's tokens), `QWIN` (the window's draft tokens), `QTOK` (the tokens the round commits) |
| `TF_FLASHNEXT_Q38_MTPDUMP=FILE` (with `TF_STEP_LOG=2`) | the MTP layer's output streams of every chain step, [10240] bf16 a step |
| `TF_FLASHNEXT_Q38_MTPDUMP2=FILE` | the draft head's input row of every scored head call, [2560] bf16 (what `kit/mkhead2.py` fits its rotation on) |

Tools (handoff/B/study): `dparse.py` (log -> rounds), `dsim.py` (policies on always-8 rounds), `dsim2.py` (sibling rows on real
default-gate rounds), `gatetrain.py` (learned gate on the dumped states), `headstudy.py` (draft-head variants replayed on dumped inputs).
Run recipe: `TFENV='-e TF_STEP_LOG=2 -e TF_FLASHNEXT_Q38_MTPDUMP2=/prof/x.mix' ... eng.sh tfb default`, then `qk.py held`.
