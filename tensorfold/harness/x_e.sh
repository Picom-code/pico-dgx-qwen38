#!/bin/bash
# x_e.sh WIN DEADLINE_EPOCH BOX(w2|w3) : spare lane inside open window WIN (kp23): lane-batched DeltaNet for more than 8 lanes (TF_FLASHNEXT_Q38_GDN_LANES_MANY).
# No engine starts after the deadline or once the window's restore has begun; waits for the box's own window lane to finish first.
WIN=$1; DL=$2; BX=$3; KP=kp23; D=$HOME/q38/runs/$WIN; exec > $D/x_e_$BX.log 2>&1
w2() { ssh -n -o ConnectTimeout=10 -o ServerAliveInterval=30 pico@spark-3d24 "$@"; }
w3() { ssh -n -o ConnectTimeout=10 -o ServerAliveInterval=30 pico@spark-4e5c "$@"; }
open_() { [ ! -e $D/DONE ] && [ "$(systemctl --user is-active q38-win)" = active ] && ! grep -q "restore begin\|work done" $D/run.log && [ $(date +%s) -lt $DL ]; }
LAST=S2_depth; [ $BX = w3 ] && LAST=S3_ref16
while open_ && ! grep -q "$LAST done" $D/s_*.log; do sleep 10; done
E='QK_MODEL=Qwen3.8-Flash-Next'; NE=TFENTRY=/opt/nvidia/nvidia_entrypoint.sh
TW="TFBIN=\$HOME/q38/tfbin/$KP TFMODEL=\$HOME/q38/models/nvidia-fp4-local $NE"
go() { open_ || { echo "window closing or deadline: stop"; exit 0; }
  $BX "$TW TFENV='$1' bash ~/q38/kit/eng.sh tfb $2" 2>&1 | tail -1 | cut -c1-200
  open_ || { $BX "bash ~/q38/kit/eng.sh stop >/dev/null"; echo "window closing after start: stopped"; exit 0; }
  $BX "export $E O=\$HOME/q38/runs/$3; mkdir -p \$O; $4; docker logs q38-run-tf > \$O/engine.log 2>&1; bash ~/q38/kit/eng.sh stop >/dev/null"; echo "$3 done $(date -u +%T)"; }
C='python3 -I ~/q38/kit/qk.py cmp'; Q='python3 -I ~/q38/kit/qk.py'
B="-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_CERT_PLAIN=1 -e TF_FLASHNEXT_Q38_PREFILL_MANY=256 -e TF_FLASHNEXT_Q38_WINDOW_ROWS=32 -e TF_FLASHNEXT_Q38_HCF=3 -e TF_FLASHNEXT_Q38_HCF_ST=3 -e TF_FLASHNEXT_Q38_HCF_HALF=1 -e TF_FLASHNEXT_Q38_HALF_BELOW=96 -e TF_FLASHNEXT_Q38_MTP_LAST=1"
MUE="$B -e TF_FLASHNEXT_Q38_WIDE_ROWS=128 -e TF_FLASHNEXT_Q38_POOL_CTX=65536"
G="-e TF_FLASHNEXT_Q38_GDN_LANES_MANY=1"
MB="$Q distinct 16 | tail -1; $Q distinct 16 | tail -1; $Q distinct 16 | tail -1; $Q distinct 8 | tail -1; python3 -I ~/q38/kit/heldpar.py 16 \$O/held16.jsonl 256 3 | tail -1"
$BX "sha256sum ~/q38/tfbin/$KP/native/bin/tensorfold-native | cut -c1-16"
if [ $BX = w3 ]; then
  go "$MUE" pc_16_262144 XE3_ref "$MB"
  go "$MUE $G" pc_16_262144 XE3_on "$MB; $C \$HOME/q38/runs/XE3_ref/held16.jsonl \$O/held16.jsonl; docker logs q38-run-tf 2>&1 | grep -ciE 'panic|segfault'"
  go "$MUE" pc_16_262144 XE3_ref2 "$MB; $C \$HOME/q38/runs/XE3_ref/held16.jsonl \$O/held16.jsonl"
else
  GP="$Q distinct 16 | tail -1; $Q distinct 16 | tail -1; python3 -I ~/q38/kit/heldpar.py 16 \$O/held16.jsonl 256 3 | tail -1; docker logs q38-run-tf 2>&1 | grep 'q38 gprof 9+rows' | tail -1 | cut -c1-330"
  go "$MUE $G -e TF_FLASHNEXT_Q38_GPROF=1" pc_16_262144 XE2_gp_on "$GP"
  W="$Q distinct 12 | tail -1; $Q distinct 12 | tail -1; python3 -I ~/q38/kit/waves.py 16 \$O/waves16.jsonl 256 1 6 | tail -1"
  go "$MUE $G" pc_16_262144 XE2_on "$W"
  go "$MUE" pc_16_262144 XE2_off "$W; $C \$HOME/q38/runs/XE2_on/waves16.jsonl \$O/waves16.jsonl"
fi
echo "x e lane done $(date -u +%T)"
