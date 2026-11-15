#!/bin/bash
# x_d.sh WIN DEADLINE_EPOCH : 4e5c spare lane inside open window WIN (kp21): short context copies on the edit / agent / chat sets. No engine starts after the deadline
# or once the window's restore has begun.
WIN=$1; DL=$2; BX=w3; KP=kp21; D=$HOME/q38/runs/$WIN; exec > $D/x_d_$BX.log 2>&1
w3() { ssh -n -o ConnectTimeout=10 -o ServerAliveInterval=30 pico@spark-4e5c "$@"; }
open_() { [ ! -e $D/DONE ] && [ "$(systemctl --user is-active q38-win)" = active ] && ! grep -q "restore begin\|work done" $D/run.log && [ $(date +%s) -lt $DL ]; }
E='QK_MODEL=Qwen3.8-Flash-Next'; NE=TFENTRY=/opt/nvidia/nvidia_entrypoint.sh
TW="TFBIN=\$HOME/q38/tfbin/$KP TFMODEL=\$HOME/q38/models/nvidia-fp4-local $NE"
go() { open_ || { echo "window closing or deadline: stop"; exit 0; }
  $BX "$TW TFENV='$1' bash ~/q38/kit/eng.sh tfb $2" 2>&1 | tail -1 | cut -c1-200
  $BX "export $E O=\$HOME/q38/runs/$3; mkdir -p \$O; $4; docker logs q38-run-tf > \$O/engine.log 2>&1; bash ~/q38/kit/eng.sh stop >/dev/null"; echo "$3 done $(date -u +%T)"; }
C='python3 -I ~/q38/kit/qk.py cmp'; ST="python3 -I ~/q38/kit/tfsteps.py"
one() { echo "L0=\$(docker logs q38-run-tf 2>&1 | wc -l); QK_HELD=$1 python3 -I ~/q38/kit/heldpar.py ${5:-1} \$O/$2.jsonl $3 ${4:-1} | tail -1; docker logs q38-run-tf 2>&1 | tail -n +\$((L0+1)) > \$O/eng_$2.log; echo '== $2'; $ST \$O/eng_$2.log | sed -n '2,3p'"; }
S3="python3 -I ~/q38/kit/mkedit.py ~/q38/kit/sets | tail -1; $(one \$HOME/q38/kit/sets/edit.jsonl edit 1500); $(one \$HOME/q38/kit/sets/agent.jsonl agent 900); $(one \$HOME/q38/kit/prompts.jsonl chat 256 8)"
CS="for f in edit agent chat; do $C \$HOME/q38/runs/XD_ref/\$f.jsonl \$O/\$f.jsonl; done; docker logs q38-run-tf 2>&1 | grep -ciE 'panic|segfault|HeldDrafts'"
B="-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_CERT_PLAIN=1 -e TF_FLASHNEXT_Q38_PREFILL_MANY=256 -e TF_FLASHNEXT_Q38_WINDOW_ROWS=32 -e TF_FLASHNEXT_Q38_HCF=3 -e TF_FLASHNEXT_Q38_HCF_ST=3 -e TF_FLASHNEXT_Q38_HALF_BELOW=96 -e TF_FLASHNEXT_Q38_MTP_LAST=1"
$BX "rm -rf ~/q38/tfbin/$KP; cp -a ~/q38/build/out/$KP ~/q38/tfbin/$KP; sha256sum ~/q38/tfbin/$KP/native/bin/tensorfold-native | cut -c1-16"
go "$B" default XD_ref "$S3"
go "$B -e TF_FLASHNEXT_Q38_SHORT_COPY=3,7" default XD_sc3 "$S3; $CS"
echo "x d lane done $(date -u +%T)"
