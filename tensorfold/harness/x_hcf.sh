#!/bin/bash
# x_hcf.sh WIN BOX(w2|w3|r3) BUILD : fused hyper-connection read-out (TF_FLASHNEXT_Q38_HCF) check + paired speed on one worker INSIDE open window WIN
# (spare lane). Every engine start first checks that the window is still open.
WIN=$1; BX=$2; KP=${3:-kp18}; D=$HOME/q38/runs/$WIN; exec > $D/x_hcf_$BX.log 2>&1
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
$BX "[ -d ~/q38/tfbin/$KP ] || cp -a ~/q38/build/out/$KP ~/q38/tfbin/$KP; sha256sum ~/q38/tfbin/$KP/native/bin/tensorfold-native | cut -c1-16"
go "$B" default XH_ref "$Q ident \$O/ident.json | tail -1; $H1"
go "$B -e TF_FLASHNEXT_Q38_HCF=3 -e TF_FLASHNEXT_Q38_HCF_CHECK=1" default XH_chk3 "$Q ident \$O/ident.json | tail -1; python3 -I ~/q38/kit/heldpar.py 1 \$O/held.jsonl 256 6 | tail -1; $CK; $C \$HOME/q38/runs/XH_ref/ident.json \$O/ident.json; $C \$HOME/q38/runs/XH_ref/held.jsonl \$O/held.jsonl"
go "$B -e TF_FLASHNEXT_Q38_HCF=3" default XH_f3 "$Q ident \$O/ident.json | tail -1; $H1; $C \$HOME/q38/runs/XH_ref/ident.json \$O/ident.json; $C \$HOME/q38/runs/XH_ref/held.jsonl \$O/held.jsonl"
go "$B -e TF_FLASHNEXT_Q38_HCF=1" default XH_f1 "$H1; $C \$HOME/q38/runs/XH_ref/held.jsonl \$O/held.jsonl"
go "$B" default XH_ref2 "$H1; $C \$HOME/q38/runs/XH_ref/held.jsonl \$O/held.jsonl"
go "$B -e TF_FLASHNEXT_Q38_HCF=3 -e TF_FLASHNEXT_Q38_HCF_ST=3" default XH_f3s3 "$H1; $C \$HOME/q38/runs/XH_ref/held.jsonl \$O/held.jsonl"
go "$B -e TF_FLASHNEXT_Q38_HCF=3 -e TF_FLASHNEXT_Q38_HCF_ST=6" default XH_f3s6 "$H1; $C \$HOME/q38/runs/XH_ref/held.jsonl \$O/held.jsonl"
go "$B -e TF_FLASHNEXT_Q38_WIDE_ROWS=128 -e TF_FLASHNEXT_Q38_HCF=3 -e TF_FLASHNEXT_Q38_HCF_MAX=128 -e TF_FLASHNEXT_Q38_HCF_CHECK=1" p16 XH_chk16 "python3 -I ~/q38/kit/heldpar.py 16 \$O/held16.jsonl 256 3 | tail -1; $CK"
echo "x hcf lane done $(date -u +%T)"
