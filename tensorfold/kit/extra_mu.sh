#!/bin/bash
# extra_mu.sh WAITLABEL : spare-capacity runs on a worker inside an open window, after the window's own lane on this box wrote ~/q38/runs/WAITLABEL/engine.log.
# Many-user variants on top of wide rows; each: 1/4/8/16 distinct, 16 copies, 16 distinct again. Output ~/q38/runs/X_*.txt
export QK_MODEL=Qwen3.8-Flash-Next; Q="python3 -I $HOME/q38/kit/qk.py"
for i in $(seq 720); do [ -f $HOME/q38/runs/$1/engine.log ] && ! docker ps --format '{{.Names}}' | grep -q '^q38-run' && break; sleep 10; done
run() { # run LABEL VARIANT ENV [TFDRAFTS] [TFCONF]
  TFBIN=$HOME/q38/tfbin/kp11 TFMODEL=$HOME/q38/models/nvidia-fp4-local TFENTRY=/opt/nvidia/nvidia_entrypoint.sh TFENV="$3" TFDRAFTS=${4:-6} TFCONF=${5:--0.4} bash ~/q38/kit/eng.sh tfb $2 2>&1 | tail -1 | cut -c1-200
  { for n in 1 4 8 16; do $Q distinct $n | tail -1; done; $Q copies 16 | tail -1; $Q distinct 16 | tail -1
    python3 -I ~/q38/kit/mustep.py <(docker logs q38-run-tf 2>&1) 2>/dev/null | tail -6; docker logs q38-run-tf 2>&1 | grep 'mtpstat\|q38 cert' | tail -2; } 2>&1
  bash ~/q38/kit/eng.sh stop >/dev/null; echo "$1 done $(date -u +%T)"; }
W="-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_WIDE_ROWS"
{ run X_w160_d8 p16 "$W=160" 8
  run X_w160_d8_c5 p16 "$W=160" 8 -0.5
  run X_w128_d6_c3 p16 "$W=128" 6 -0.3
  run X_w192_d10 p16 "$W=192" 10
} > $HOME/q38/runs/X_extra.txt 2>&1
echo EXTRA_DONE >> $HOME/q38/runs/X_extra.txt
