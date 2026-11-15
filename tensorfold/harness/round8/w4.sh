#!/bin/bash
# w4.sh T (923b, builder A): A2 one user (5 x 262K default shape, whole stack): ref / LATE_SKIP=T / ref; held 107 x 256 one in flight, ident, edit set 4 in flight; texts.
. $HOME/q38/kit-A/libA.sh; BIN=${BIN:-A2}; T=${1:-16}; AL="-e TF_FLASHNEXT_Q38_ATTN_LANES=1 -e TF_FLASHNEXT_Q38_GDN_LANES_MANY=2 -e TF_FLASHNEXT_Q38_BEST_LANES=1"
while pgrep -f 'kit-A/w[123].sh' >/dev/null; do sleep 5; done
ONE="$HP 1 \$O/held1.jsonl 256 3 | tail -1; docker logs q38-run-A > \$O/engine1.log 2>&1; python3 -I $KA/tfsteps.py \$O/engine1.log 2>/dev/null | tail -3 | cut -c1-200"
go A2_1ref $BIN "$B $AL" "$ONE; $Q ident \$O/ident.json | tail -1; $C $RUNS/A_ref1/ident.json \$O/ident.json; QK_HELD=$KA/sets/edit.jsonl $HP 4 \$O/edit4.jsonl 1200 | tail -1" default 760
go A2_1t $BIN "$B $AL -e TF_FLASHNEXT_Q38_LATE_SKIP=$T" "$ONE; $C $RUNS/A2_1ref/held1.jsonl \$O/held1.jsonl; $Q ident \$O/ident.json | tail -1; $C $RUNS/A_ref1/ident.json \$O/ident.json; QK_HELD=$KA/sets/edit.jsonl $HP 4 \$O/edit4.jsonl 1200 | tail -1; $C $RUNS/A2_1ref/edit4.jsonl \$O/edit4.jsonl" default 760
go A2_1ref2 $BIN "$B $AL" "$ONE; $C $RUNS/A2_1ref/held1.jsonl \$O/held1.jsonl" default 600
echo "w4 done $(date -u +%T)"
