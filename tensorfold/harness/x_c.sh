#!/bin/bash
# x_c.sh WIN : 3d24 spare lane in window WIN with kp20: MTP last-row experts and the 2-stage fused down kernel, 54 held prompts, vs XH_ref2 (kp18, same box, same window).
WIN=$1; BX=w2; KP=kp20; D=$HOME/q38/runs/$WIN; exec > $D/x_c_$BX.log 2>&1
w2() { ssh -n -o ConnectTimeout=10 -o ServerAliveInterval=30 pico@spark-3d24 "$@"; }
open_() { [ ! -e $D/DONE ] && [ "$(systemctl --user is-active q38-win)" = active ]; }
E='QK_MODEL=Qwen3.8-Flash-Next'; NE=TFENTRY=/opt/nvidia/nvidia_entrypoint.sh
TW="TFBIN=\$HOME/q38/tfbin/$KP TFMODEL=\$HOME/q38/models/nvidia-fp4-local $NE"
go() { open_ || { echo "window closed: stop"; exit 0; }
  $BX "$TW TFENV='$1' bash ~/q38/kit/eng.sh tfb $2" 2>&1 | tail -1 | cut -c1-200
  $BX "export $E O=\$HOME/q38/runs/$3; mkdir -p \$O; $4; docker logs q38-run-tf > \$O/engine.log 2>&1; bash ~/q38/kit/eng.sh stop >/dev/null"; echo "$3 done $(date -u +%T)"; }
C='python3 -I ~/q38/kit/qk.py cmp'
H1="python3 -I ~/q38/kit/heldpar.py 1 \$O/held.jsonl 256 6 | tail -1; python3 -I ~/q38/kit/tfsteps.py <(docker logs q38-run-tf 2>&1) | sed -n '2,3p'; $C \$HOME/q38/runs/XH_ref2/held.jsonl \$O/held.jsonl; docker logs q38-run-tf 2>&1 | grep -ciE 'panic|segfault'"
B="-e TF_FLASHNEXT_Q38_CERT=1"
go "$B -e TF_FLASHNEXT_Q38_MTP_LAST=1" default XC_mtplast "$H1"
go "$B -e TF_FLASHNEXT_Q38_HCF=3 -e TF_FLASHNEXT_Q38_HCF_ST=2" default XC_f3s2 "$H1"
go "$B -e TF_FLASHNEXT_Q38_HCF=3 -e TF_FLASHNEXT_Q38_HCF_ST=2 -e TF_FLASHNEXT_Q38_HCF_HALF=1 -e TF_FLASHNEXT_Q38_HCF_CHECK=1" default XC_chkh "python3 -I ~/q38/kit/heldpar.py 1 \$O/held.jsonl 128 12 | tail -1; docker logs q38-run-tf 2>&1 | grep 'q38 hcf check' | tail -1"
echo "x c lane done $(date -u +%T)"
