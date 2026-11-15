# libA.sh (923b, builder A): helpers for run scripts. Source it; then  go LABEL BIN "EXTRA_ENV" "COMMANDS" [VARIANT]
# No engine starts after the deadline in ~/q38/runs/A_deadline (epoch seconds; the Mac side rewrites it on lease renew).
export QK_MODEL=Qwen3.8-Flash-Next
KA=$HOME/q38/kit-A; RUNS=$HOME/q38/runs
C="python3 -I $KA/qk.py cmp"; Q="python3 -I $KA/qk.py"; W="python3 -I $KA/waves.py"; HP="python3 -I $KA/heldpar.py"
B="-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_CERT_PLAIN=1 -e TF_FLASHNEXT_Q38_PREFILL_MANY=256 -e TF_FLASHNEXT_Q38_WINDOW_ROWS=32 -e TF_FLASHNEXT_Q38_HCF=3 -e TF_FLASHNEXT_Q38_HCF_ST=3 -e TF_FLASHNEXT_Q38_HCF_HALF=1 -e TF_FLASHNEXT_Q38_HALF_BELOW=96 -e TF_FLASHNEXT_Q38_MTP_LAST=1"
MUE="$B -e TF_FLASHNEXT_Q38_WIDE_ROWS=128 -e TF_FLASHNEXT_Q38_POOL_CTX=65536 -e TF_FLASHNEXT_Q38_GDN_LANES_MANY=2 -e TF_FLASHNEXT_Q38_BEST_LANES=1"
left() { echo $(( $(cat $RUNS/A_deadline 2>/dev/null || echo 0) - $(date +%s) )); }
go() { # LABEL BIN ENV CMDS [VARIANT] [NEED_SECONDS]
  local L=$1 BIN=$2 ENV=$3 CMDS=$4 V=${5:-pc_16_262144} NEED=${6:-540}
  [ $(left) -ge $NEED ] || { echo "$L SKIPPED: $(left) s left, need $NEED"; return 1; }
  export O=$RUNS/$L; mkdir -p $O
  TFBIN=$HOME/q38/tfbin/$BIN TFMODEL=$HOME/q38/models/nvidia-fp4-local TFENTRY=/opt/nvidia/nvidia_entrypoint.sh TFENV="$ENV" bash $KA/eng.sh tfb $V 2>&1 | tail -1 | cut -c1-200
  if curl -sf -m 5 http://127.0.0.1:18301/v1/models >/dev/null; then eval "$CMDS"; else echo "$L ENGINE NOT UP"; fi
  docker logs q38-run-A > $O/engine.log 2>&1; echo "crash lines: $(grep -ciE 'panic|segfault|error\.' $O/engine.log)"
  bash $KA/eng.sh stop >/dev/null; echo "$L done $(date -u +%T)"; }
