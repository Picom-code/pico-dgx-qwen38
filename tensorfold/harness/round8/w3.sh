#!/bin/bash
# w3.sh (923b, builder A): A2 = A1 + TF_FLASHNEXT_Q38_LATE_SKIP=T. 16 users: ref (ATTN_LANES on, skip off) / T=16 / T=32 / T=8 / ref; texts with 16 in flight.
. $HOME/q38/kit-A/libA.sh; BIN=${BIN:-A2}; AL="-e TF_FLASHNEXT_Q38_ATTN_LANES=1"
while pgrep -f 'kit-A/w[12].sh' >/dev/null; do sleep 5; done
MB="$Q distinct 16 | tail -1; $Q distinct 16 | tail -1; $Q distinct 16 | tail -1; $HP 16 \$O/held16.jsonl 256 3 | tail -1"
CM="$C $RUNS/A_ref1/held16.jsonl \$O/held16.jsonl"
sha256sum ~/q38/tfbin/$BIN/native/bin/tensorfold-native | cut -c1-16
go A2_ref1 $BIN "$MUE $AL" "$MB; $CM" pc_16_262144 440
go A2_t16 $BIN "$MUE $AL -e TF_FLASHNEXT_Q38_LATE_SKIP=16" "$MB; $CM; $W 16 \$O/waves16.jsonl 256 1 6 | tail -1; $C $RUNS/A_ref1/waves16.jsonl \$O/waves16.jsonl" pc_16_262144 600
go A2_t32 $BIN "$MUE $AL -e TF_FLASHNEXT_Q38_LATE_SKIP=32" "$MB; $CM" pc_16_262144 440
go A2_t8 $BIN "$MUE $AL -e TF_FLASHNEXT_Q38_LATE_SKIP=8" "$MB; $CM" pc_16_262144 440
go A2_ref2 $BIN "$MUE $AL" "$MB; $CM" pc_16_262144 440
echo "w3 done $(date -u +%T)"
