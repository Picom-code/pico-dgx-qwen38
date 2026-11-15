#!/bin/bash
# w9a.sh (923b, builder A, round 9): ONE engine (binary I2 = patches 0101-0115, HEAD2 off, tfb default one user) with the draft-head input dump on:
# the 260-prompt fitting corpus, then the 123-prompt fresh set, then the held prompts (stride 3). Then the engine stops and fit9.sh runs.
. $HOME/q38/kit-A/libA.sh; K9=$HOME/q38/kit-A9
rm -f $HOME/q38/prof/r9_all.mix
go R9_dump I2 "$B -e TF_FLASHNEXT_Q38_ATTN_LANES=1 -e TF_FLASHNEXT_Q38_LATE_SKIP=16 -e TF_FLASHNEXT_Q38_KEEP_INPASS=4096 -e TF_FLASHNEXT_Q38_LANE_REUSE=5 -e TF_STEP_LOG=2 -e TF_FLASHNEXT_Q38_MTPDUMP2=/prof/r9_all.mix" \
  "python3 -I $K9/runset.py $K9/corpus/fit.jsonl \$O/fit.jsonl | tail -1; QK_HELD=$K9/corpus/fresh.jsonl $Q held \$O/fresh.jsonl | tail -7; QK_HELD_STRIDE=3 $Q held \$O/held.jsonl | tail -4; $C $RUNS/G_ref/held.jsonl \$O/held.jsonl" default 3300
ls -la $HOME/q38/prof/r9_all.mix; grep -c '^info: prompt cache:' $RUNS/R9_dump/engine.log; grep -c '^QTOP ' $RUNS/R9_dump/engine.log
bash $K9/fit9.sh > $RUNS/R9_fit.log 2>&1; tail -30 $RUNS/R9_fit.log
echo "w9a done $(date -u +%T)"
