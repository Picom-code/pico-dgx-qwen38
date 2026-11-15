#!/bin/bash
# x_b.sh WIN BOX BUILD : spare lane inside open window WIN on one worker: half tiles (TF_FLASHNEXT_Q38_HALF_BELOW, upstream #1171) paired speed + identity,
# fused read-out check with 16 in flight, free env tests (expert tile rows at 16 users, prompt chunk size).
WIN=$1; BX=$2; KP=${3:-kp19}; D=$HOME/q38/runs/$WIN; exec > $D/x_b_$BX.log 2>&1
w2() { ssh -n -o ConnectTimeout=10 -o ServerAliveInterval=30 pico@spark-3d24 "$@"; }
w3() { ssh -n -o ConnectTimeout=10 -o ServerAliveInterval=30 pico@spark-4e5c "$@"; }
r3() { ssh -n -o ConnectTimeout=10 -o ServerAliveInterval=30 picom@spark-923b "$@"; }
open_() { [ ! -e $D/DONE ] && [ "$(systemctl --user is-active q38-win)" = active ]; }
E='QK_MODEL=Qwen3.8-Flash-Next'; NE=TFENTRY=/opt/nvidia/nvidia_entrypoint.sh
TW="TFBIN=\$HOME/q38/tfbin/$KP TFMODEL=\$HOME/q38/models/nvidia-fp4-local $NE"
go() { open_ || { echo "window closed: stop"; exit 0; }
  $BX "$TW TFENV='$1' bash ~/q38/kit/eng.sh tfb $2" 2>&1 | tail -1 | cut -c1-200
  $BX "export $E O=\$HOME/q38/runs/$3; mkdir -p \$O; $4; docker logs q38-run-tf > \$O/engine.log 2>&1; bash ~/q38/kit/eng.sh stop >/dev/null"; echo "$3 done $(date -u +%T)"; }
C='python3 -I ~/q38/kit/qk.py cmp'; Q='python3 -I ~/q38/kit/qk.py'
CK="docker logs q38-run-tf 2>&1 | grep 'q38 hcf check' | sed -n '1p;\$p'; docker logs q38-run-tf 2>&1 | grep -ciE 'panic|error\\.|segfault'"
H1="python3 -I ~/q38/kit/heldpar.py 1 \$O/held.jsonl 256 6 | tail -1; python3 -I ~/q38/kit/tfsteps.py <(docker logs q38-run-tf 2>&1) | sed -n '2,3p'"
B="-e TF_FLASHNEXT_Q38_CERT=1"
MU="-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_WIDE_ROWS=128 -e TF_FLASHNEXT_Q38_PREFILL_MANY=256"
for i in $(seq 90); do $BX "ls ~/q38/build/out/$KP/native/bin/tensorfold-native >/dev/null 2>&1 && ! docker ps --format {{.Names}} | grep -q q38-build" && break; sleep 10; done
$BX "rm -rf ~/q38/tfbin/$KP; cp -a ~/q38/build/out/$KP ~/q38/tfbin/$KP; sha256sum ~/q38/tfbin/$KP/native/bin/tensorfold-native | cut -c1-16; tail -1 ~/q38/build/$KP.log | cut -c1-80"
go "$B" default XB_ref "$H1"
go "$B -e TF_FLASHNEXT_Q38_HALF_BELOW=96" default XB_half "$H1; $C \$HOME/q38/runs/XB_ref/held.jsonl \$O/held.jsonl"
go "$B -e TF_FLASHNEXT_Q38_HALF_BELOW=96 -e TF_FLASHNEXT_Q38_HCF=3" default XB_half_hcf "$H1; $C \$HOME/q38/runs/XB_ref/held.jsonl \$O/held.jsonl"
go "$MU -e TF_FLASHNEXT_Q38_HCF=3 -e TF_FLASHNEXT_Q38_HCF_MAX=128 -e TF_FLASHNEXT_Q38_HCF_CHECK=1 -e TF_FLASHNEXT_Q38_HALF_BELOW=96" p16 XB_chk16 "python3 -I ~/q38/kit/heldpar.py 16 \$O/held16.jsonl 256 3 | tail -1; $CK"
MB="$Q distinct 16 | tail -1; $Q distinct 16 | tail -1; $Q distinct 8 | tail -1; python3 -I ~/q38/kit/heldpar.py 16 \$O/held16.jsonl 256 3 | tail -1"
go "$MU" p16 XB_mu "$MB"
for r in 64 128; do go "$MU -e TF_FLASHNEXT_PROMPT_EXPERT_TILE_ROWS=$r" p16 XB_et$r "$MB; $C \$HOME/q38/runs/XB_mu/held16.jsonl \$O/held16.jsonl"; done
PF="$Q prefill 8000,32000,128000 | tail -4; $Q ctx 100000 | tail -1; docker logs q38-run-tf 2>&1 | grep 'lhash first' | sort -k5,5nr | head -3 | cut -c1-160; nvidia-smi --query-compute-apps=used_memory --format=csv,noheader"
go "$B -e TF_FLASHNEXT_Q38_LHASH=1" default XB_c3072 "$PF"
for c in 4096 8192; do go "$B -e TF_FLASHNEXT_Q38_LHASH=1 -e TF_FLASHNEXT_CHUNK=$c" default XB_c$c "$PF"; done
echo "x b lane done $(date -u +%T)"
