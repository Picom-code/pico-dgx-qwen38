#!/bin/bash
# run12.sh BIN PROFILE LEV MIN ["recipe options"] ["-e NAME=v ..."] : builder D round 12 (red team). The engine on the HEAD with EXACTLY the
# environment recipe.sh computes (recenv.sh), bf16 KV by default, started with the head's safety rules:
#   window open + not closing + younger than 262 min; lease D live; MemAvailable >= 95 GiB; reserve 13 GiB; container --memory 12g;
#   watchdog (wd.sh: < 6 GiB MemAvailable, lease gone, window closing, deadline). Container q38-run-D on 127.0.0.1:18301.
# PROFILE: def (recipe default; PAR=n overrides --parallel) | u16 (USERS16=1; POOL=n overrides the pool window)
# LEV: on (recipe defaults) | off (every lever of ours off = the reference).  rc 3 refused, rc 4 refused by the window cap.
BIN=$1; PROF=${2:-def}; LEV=${3:-on}; MIN=${4:-20}; ROPT=$5; X=$6
Q=$HOME/q38; S=$Q/shared; K=$Q/kit-D; RT=$K/redteam; PORT=18301
OFFOPT="CERT=0 MANY=0 COPY_ROWS=16 FUSED=0 HALF=0 MTP_LAST=0 GDN_MANY=0 BEST_LANES=0 ATTN_LANES=0 LATE_SKIP=0 KEEP_INPASS=0 LANE_REUSE=0 HEAD2=0 R9=0 R10=0 R11=0 PREFIX_WAIT=0"
ma() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
[ -f $S/OPEN ] && [ ! -f $S/CLOSING ] || { echo "REFUSED: no open window"; exit 3; }
O=$(cat $S/OPEN); now=$(date +%s)
[ $now -gt $(( O + 262*60 )) ] && { echo "REFUSED BY CAP: window is $(( (now-O)/60 )) min old"; exit 4; }
left=$(( (O + 270*60 - now) / 60 )); [ $MIN -gt $left ] && MIN=$left
# night 11-14: every engine of D is gone by 12:04 UTC (lease released by 12:10)
CUTE=${CUTE:-$(date -u -d "2026-11-15 12:04:00" +%s)}; leftc=$(( (CUTE - now) / 60 )); [ $leftc -lt 6 ] && { echo "REFUSED BY THE NIGHT CUTOFF (12:04 UTC)"; exit 5; }; [ $MIN -gt $leftc ] && MIN=$leftc
L=$(cat $S/leases/D 2>/dev/null || echo 0); [ $L -gt $(( now + 300 )) ] || { echo "REFUSED: lease D is not live"; exit 3; }
DL=$(( now + MIN*60 )); echo $DL > $K/eng_dl
docker rm -f q38-run-D >/dev/null 2>&1; [ -f $K/wd.pid ] && kill $(cat $K/wd.pid) 2>/dev/null
for i in $(seq 30); do [ $(ma) -ge 95 ] && break; sleep 2; done
[ $(ma) -ge 95 ] || { echo "REFUSED: MemAvailable $(ma) GiB < 95"; exit 3; }
RO=""; XE=""
case $PROF in def) RO="${PAR:+PARALLEL=$PAR}" ;; u16) RO="USERS16=1 ${POOL:+POOL_CTX=$POOL}" ;; *) echo "profile?"; exit 2 ;; esac
if [ $LEV = off ]; then RO="$RO $OFFOPT"; XE="-e TF_FLASHNEXT_Q38_PREDICTION=0"; fi
RO="$RO TENSORFOLD_MEMORY_RESERVE_GIB=${RESERVE:-13} $ROPT"
HEAD2FILE=$Q/prof-D/head2_r9.bin bash $RT/recenv.sh "$RO" > $K/eng_env.txt || { echo "recenv failed"; exit 2; }
KV=$(sed -n 's/^KV_DTYPE=//p' $K/eng_env.txt); PARN=$(sed -n 's/^PARALLEL=//p' $K/eng_env.txt)
ENVS="-e HF_HUB_OFFLINE=1 -e TENSORFOLD_CUDA_KERNELS=native -e TF_FLASHNEXT_Q38_KNOBFILE=/prof/knobs"
while IFS= read -r l; do case "$l" in KV_DTYPE=*|PARALLEL=*) ;; TF_FLASHNEXT_Q38_HEAD2=*) ENVS="$ENVS -e TF_FLASHNEXT_Q38_HEAD2=/prof/head2_r9.bin" ;; *) ENVS="$ENVS -e $l" ;; esac; done < $K/eng_env.txt
ENVS="$ENVS $XE $X"
echo "run12: $BIN $PROF lev=$LEV kv=$KV parallel=$PARN window age $(( (now-O)/60 )) min, engine minutes $MIN; pre-start MemAvailable $(ma) GiB"
: > $Q/prof-D/knobs; chmod 666 $Q/prof-D/knobs 2>/dev/null
IMG=ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold:zig-db28187-4144696f21e3
VIS="--vision --vision-max-images 50 --vision-max-videos 4 --vision-image-tokens 16384"; [ -n "$NOVISION" ] && VIS=""
mkdir -p $Q/tf-D/kcache-rt12; T0=$(date +%s)
docker run -d --name q38-run-D --memory ${CMEM:-12g} --memory-swap ${CMEM:-12g} --gpus all --ipc=host --network host --ulimit memlock=-1 --ulimit stack=67108864 $ENVS \
  -v $Q/models/nvidia-fp4-local:/model:ro -v $Q/tf-D/kcache-rt12:/cache -v $Q/tfbin/$BIN:/opt/tensorfold:ro -v $Q/prof-D:/prof --entrypoint /opt/nvidia/nvidia_entrypoint.sh $IMG \
  /opt/tensorfold/native/bin/tensorfold-native serve /model --name Qwen3.8-Flash-Next --host 127.0.0.1 --port $PORT --context 262144 --parallel $PARN --max-tokens 32768 \
  --temperature 1.0 --top-p 0.95 --top-k 20 --thinking --kv-cache $KV $VIS $SERVEX >/dev/null || exit 1
nohup setsid bash $K/wd.sh $DL > $K/wd.log 2>&1 < /dev/null &
for i in $(seq 450); do
  curl -sf -m 3 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { echo "READY in $(( $(date +%s)-T0 )) s; MemAvailable $(ma) GiB (min while loading $(awk 'NF==2{print $2}' $K/wd.log | sort -n | head -1) MiB); GPU MiB $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | paste -sd+)"; exit 0; }
  [ "$(docker inspect -f '{{.State.Running}}' q38-run-D 2>/dev/null)" = true ] || { echo "FAILED: container gone"; tail -2 $K/wd.log; docker logs --tail 12 q38-run-D 2>&1 | cut -c1-400; exit 1; }
  sleep 2
done; echo "FAILED: not ready after 900 s"; docker rm -f q38-run-D; exit 1
