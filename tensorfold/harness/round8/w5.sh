#!/bin/bash
# w5.sh (923b, builder A): GPU time by group at 16 users with LATE_SKIP=16 (A2), against A_gp_on (A1, same flags without it).
. $HOME/q38/kit-A/libA.sh; BIN=${BIN:-A2}; AL="-e TF_FLASHNEXT_Q38_ATTN_LANES=1"
while pgrep -f 'kit-A/w[1234].sh' >/dev/null; do sleep 5; done
GP="$Q distinct 16 | tail -1; $Q distinct 16 | tail -1; $HP 16 \$O/held16.jsonl 256 3 | tail -1; docker logs q38-run-A 2>&1 | grep 'q38 gprof 9+rows' | tail -1 | cut -c1-330"
go A2_gp_t16 $BIN "$MUE $AL -e TF_FLASHNEXT_Q38_LATE_SKIP=16 -e TF_FLASHNEXT_Q38_GPROF=1" "$GP; $C $RUNS/A_ref1/held16.jsonl \$O/held16.jsonl" pc_16_262144 420
go A2_gp_ref $BIN "$MUE $AL -e TF_FLASHNEXT_Q38_GPROF=1" "$GP" pc_16_262144 420
echo "w5 done $(date -u +%T)"
