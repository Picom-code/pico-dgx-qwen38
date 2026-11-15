#!/bin/bash
# w6.sh (923b, builder A): I2 = integ + B's two-stage draft head. One user, whole recipe environment: HEAD2 off / on; held 107 x 256, texts.
. $HOME/q38/kit-A/libA.sh; BIN=${BIN:-I2}
ALL="$B -e TF_FLASHNEXT_Q38_GDN_LANES_MANY=2 -e TF_FLASHNEXT_Q38_BEST_LANES=1 -e TF_FLASHNEXT_Q38_ATTN_LANES=1 -e TF_FLASHNEXT_Q38_LATE_SKIP=16 -e TF_FLASHNEXT_Q38_KEEP_INPASS=4096 -e TF_FLASHNEXT_Q38_LANE_REUSE=5"
while pgrep -f 'q38-build-A' >/dev/null || docker ps --format '{{.Names}}' | grep -q q38-build-A; do sleep 5; done
[ -x ~/q38/tfbin/$BIN/native/bin/tensorfold-native ] || { echo "no $BIN"; exit 1; }
sha256sum ~/q38/tfbin/$BIN/native/bin/tensorfold-native ~/q38/prof/head2_r512.bin | cut -c1-16
ONE="$HP 1 \$O/held1.jsonl 256 3 | tail -1; docker logs q38-run-A > \$O/engine1.log 2>&1; python3 -I $KA/tfsteps.py \$O/engine1.log 2>/dev/null | sed -n '2,3p' | cut -c1-200"
go I2_1ref $BIN "$ALL" "$ONE; $C $RUNS/A2_1ref/held1.jsonl \$O/held1.jsonl" default 560
go I2_1h2 $BIN "$ALL -e TF_FLASHNEXT_Q38_HEAD2=/prof/head2_r512.bin" "$ONE; $C $RUNS/A2_1ref/held1.jsonl \$O/held1.jsonl; docker logs q38-run-A 2>&1 | grep -i 'head2' | head -3 | cut -c1-200" default 560
echo "w6 done $(date -u +%T)"
