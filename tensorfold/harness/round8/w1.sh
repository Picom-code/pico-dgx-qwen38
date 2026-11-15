#!/bin/bash
# w1.sh (923b, builder A, first window): A1 = kp25 + TF_FLASHNEXT_Q38_ATTN_LANES. Reference first and last, flag on between, then GPU time by
# group off / on, per-kernel profile with it on, 12 / 9 users, one-user identity set.
. $HOME/q38/kit-A/libA.sh; BIN=${BIN:-A1}; AL="-e TF_FLASHNEXT_Q38_ATTN_LANES=1"
MB="$Q distinct 16 | tail -1; $Q distinct 16 | tail -1; $Q distinct 16 | tail -1; $HP 16 \$O/held16.jsonl 256 3 | tail -1; $W 16 \$O/waves16.jsonl 256 1 6 | tail -1"
GP="$Q distinct 16 | tail -1; $Q distinct 16 | tail -1; grep 'q38 gprof 9+rows' <(docker logs q38-run-A 2>&1) | tail -1 | cut -c1-330"
sha256sum ~/q38/tfbin/$BIN/native/bin/tensorfold-native | cut -c1-16
go A_ref1 $BIN "$MUE" "$MB; $C \$O/waves16.jsonl $RUNS/U_mu/waves16.jsonl; $Q ident \$O/ident.json | tail -1"
go A_on $BIN "$MUE $AL" "$MB; $C $RUNS/A_ref1/held16.jsonl \$O/held16.jsonl; $C $RUNS/A_ref1/waves16.jsonl \$O/waves16.jsonl; $Q ident \$O/ident.json | tail -1; $C $RUNS/A_ref1/ident.json \$O/ident.json; $Q distinct 12 | tail -1; $Q distinct 9 | tail -1"
go A_ref2 $BIN "$MUE" "$MB; $C $RUNS/A_ref1/held16.jsonl \$O/held16.jsonl; $Q distinct 12 | tail -1; $Q distinct 9 | tail -1"
go A_gp_off $BIN "$MUE -e TF_FLASHNEXT_Q38_GPROF=1" "$GP" pc_16_262144 400
go A_gp_on $BIN "$MUE $AL -e TF_FLASHNEXT_Q38_GPROF=1" "$GP" pc_16_262144 400
go A_kp_on $BIN "$MUE $AL -e TF_KPROF=400000" "$Q distinct 16 | tail -1; $Q distinct 16 | tail -1; docker logs q38-run-A > \$O/engine.log 2>&1; python3 -I $KA/kprof.py \$O/engine.log > \$O/kprof.txt; head -60 \$O/kprof.txt | cut -c1-150" pc_16_262144 400
go A_on2 $BIN "$MUE $AL" "$MB; $C $RUNS/A_ref1/held16.jsonl \$O/held16.jsonl"
echo "w1 done $(date -u +%T)"
