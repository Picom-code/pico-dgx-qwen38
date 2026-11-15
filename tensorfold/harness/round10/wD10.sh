#!/bin/bash
# wD10.sh BIN (923b, builder A, round 10): LM-head input dumps (TF_FLASHNEXT_Q38_HDUMP; values never change): engine 1 the 260-prompt fitting corpus
# -> ~/q38/prof/c10_fit.{x,n}; engine 2 fresh set + held/3 + agent + edit sets -> ~/q38/prof/c10_test.{x,n} with call marks in ~/q38/runs/C10_marks.txt.
. $HOME/q38/kit-A/libA.sh; K9=$HOME/q38/kit-A9; BIN=${1:-J1}; P=$HOME/q38/prof
E10="$B -e TF_FLASHNEXT_Q38_ATTN_LANES=1 -e TF_FLASHNEXT_Q38_LATE_SKIP=16 -e TF_FLASHNEXT_Q38_KEEP_INPASS=4096 -e TF_FLASHNEXT_Q38_LANE_REUSE=5 -e TF_FLASHNEXT_Q38_HDUMP_MAX=900000"
docker run --rm -v $P:/p alpine sh -c 'rm -f /p/c10_fit.x /p/c10_fit.n /p/c10_test.x /p/c10_test.n'
mk() { echo "$1:$(( $(stat -c %s $P/c10_test.n) / 4 ))" >> $RUNS/C10_marks.txt; }
: > $RUNS/C10_marks.txt
go C10_fit $BIN "$E10 -e TF_FLASHNEXT_Q38_HDUMP=/prof/c10_fit" "python3 -I $K9/runset.py $K9/corpus/fit.jsonl \$O/fit.jsonl | tail -1" default 1800
go C10_test $BIN "$E10 -e TF_FLASHNEXT_Q38_HDUMP=/prof/c10_test" "QK_HELD=$K9/corpus/fresh.jsonl $Q held \$O/fresh.jsonl | tail -1; mk fresh; QK_HELD_STRIDE=3 $Q held \$O/held.jsonl | tail -1; mk held; QK_HELD=$KA/sets/agent.jsonl $HP 1 \$O/agent.jsonl 900 | tail -1; mk agent; QK_HELD=$KA/sets/edit.jsonl $HP 1 \$O/edit.jsonl 1500 | tail -1; mk edit; $C $RUNS/P_ref1/held.jsonl \$O/held.jsonl" default 1500
ls -la $P/c10_*; cat $RUNS/C10_marks.txt | tr '\n' ' '; echo; echo "wD10 done $(date -u +%T)"
