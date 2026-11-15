#!/bin/bash
# engS.sh PROFILE LEV ["recipe options"] ["-e NAME=v ..."] : agent B night 11-14 soak engine on spark-3d24, container q38-run-B on 127.0.0.1:18301.
# The engine gets EXACTLY the environment `recipe.sh start` hands to MiaAI's start.sh (recenv.sh: recipe.sh's exports + config.sh's defaults), the same
# serve arguments and the image's own entrypoint; the binary is the final image's /opt/tensorfold (copied from spark-923b) mounted over the MiaAI image.
#   PROFILE def (bf16 KV, 3 x 262,144, ELASTIC) | u16 (USERS16=1) | agents (AGENTS=1)
#   LEV     on (recipe defaults) | off (every lever of ours off = the reference engine; `prediction` ignored)
#           chk (recipe defaults + every device check mode) | knob (recipe defaults + the knob file /prof/knobs, for bisecting)
# engS.sh stop [DIR] : stops as MiaAI's stop.sh does (docker stop -t 30, then rm); with DIR the engine log is saved there first.
# REFUSES (rc 3) unless the keeper's stamp ~/q38/kit-B/soak/win is fresh, the window is younger than 262 min, lease B is live and it is before 12:00 UTC.
Q=$HOME/q38; K=$Q/kit-B/soak; PORT=18301; N=q38-run-B; BIN=${BIN:-F29}; CUT=${CUT:-1794744000}
ma() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
if [ "$1" = stop ]; then
  [ -n "$2" ] && { mkdir -p $2; docker logs $N > $2/engine_$(date -u +%H%M%S).log 2>&1; }
  [ -f $K/wd.pid ] && kill $(cat $K/wd.pid) 2>/dev/null; rm -f $K/wd.pid
  docker stop -t 30 $N >/dev/null 2>&1; docker rm -f $N >/dev/null 2>&1; exit 0
fi
PROF=${1:-def}; LEV=${2:-on}; ROPT=$3; X=$4
read O L W < $K/win 2>/dev/null; now=$(date +%s)
[ -n "$W" ] && [ $(( now - W )) -lt 200 ] || { echo "REFUSED: window stamp missing or stale (keeper not running?)"; exit 3; }
[ "$O" -gt 0 ] && [ $now -lt $(( O + 262*60 )) ] || { echo "REFUSED BY CAP: window is $(( (now-O)/60 )) min old"; exit 3; }
[ "$L" -gt $(( now + 300 )) ] || { echo "REFUSED: lease B is not live"; exit 3; }
[ $now -lt $CUT ] || { echo "REFUSED: past the cutoff"; exit 3; }
docker ps --format '{{.Names}}' | grep -q '^q38-run' && { echo "REFUSED: another q38-run container is up"; exit 3; }
[ -f $K/wd.pid ] && kill $(cat $K/wd.pid) 2>/dev/null
for i in $(seq 45); do [ $(ma) -ge 95 ] && break; sleep 2; done
[ $(ma) -ge 95 ] || { echo "REFUSED: MemAvailable $(ma) GiB < 95 (DeepSeek still up?)"; exit 3; }
OFFOPT="CERT=0 MANY=0 COPY_ROWS=16 FUSED=0 HALF=0 MTP_LAST=0 GDN_MANY=0 BEST_LANES=0 ATTN_LANES=0 LATE_SKIP=0 KEEP_INPASS=0 LANE_REUSE=0 HEAD2=0 R9=0 R10=0 R11=0 PREFIX_WAIT=0"
CHK="CERT=2 QSA_CERT=2 TF_FLASHNEXT_Q38_QSA_STATS=1 GDN_PROD=12 TF_FLASHNEXT_Q38_ROUTE_BAND_CHECK=1 TF_FLASHNEXT_Q38_SCRATCH_CHECK=1 TF_FLASHNEXT_Q38_HC_DN_TALL_CHECK=1 SNAP_DELTA=2"
case $PROF in def) RO="" ;; u16) RO="USERS16=1" ;; agents) RO="AGENTS=1" ;; *) echo "profile?"; exit 2 ;; esac
XE=""
case $LEV in on) ;; off) RO="$RO $OFFOPT"; XE="-e TF_FLASHNEXT_Q38_PREDICTION=0" ;; chk) RO="$RO $CHK"; XE="-e TF_FLASHNEXT_Q38_PLOG=1" ;; knob) XE="-e TF_FLASHNEXT_Q38_KNOBFILE=/prof/knobs" ;; *) echo "lev?"; exit 2 ;; esac
RO="$RO $ROPT"
HEAD2FILE=$Q/prof-S/head2_r9.bin bash $K/recenv.sh "$RO" > $K/eng_env.txt || { echo "recenv failed"; exit 2; }
KV=$(sed -n 's/^KV_DTYPE=//p' $K/eng_env.txt); PARN=$(sed -n 's/^PARALLEL=//p' $K/eng_env.txt)
ENVS="-e HF_HUB_OFFLINE=1 -e TENSORFOLD_CUDA_KERNELS=native -e TENSORFOLD_MAX_IMAGES=50 -e TENSORFOLD_IMAGE_TOKENS=16384 -e TENSORFOLD_VISION_WORKSPACE_MIB=2048"
while IFS= read -r l; do case "$l" in KV_DTYPE=*|PARALLEL=*) ;; TF_FLASHNEXT_Q38_HEAD2=*) ENVS="$ENVS -e TF_FLASHNEXT_Q38_HEAD2=/prof/head2_r9.bin" ;; *) ENVS="$ENVS -e $l" ;; esac; done < $K/eng_env.txt
ENVS="$ENVS $XE $X"
SHA=$(sha256sum $Q/tfbin/$BIN/native/bin/tensorfold-native | cut -c1-16)
echo "engS: $BIN ($SHA) profile $PROF lev=$LEV kv=$KV parallel=$PARN [$RO] [$XE $X]; window age $(( (now-O)/60 )) min; pre-start MemAvailable $(ma) GiB; $(date -u +%T)"
: > $Q/prof-S/knobs; chmod 666 $Q/prof-S/knobs 2>/dev/null
IMG=ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold:zig-db28187-4144696f21e3
mkdir -p $Q/tf/kcache-soak; T0=$(date +%s)
docker run -d --name $N --gpus all --ipc=host --network host --ulimit memlock=-1 --ulimit stack=67108864 $ENVS \
  -v $Q/models/nvidia-fp4-local:/model:ro -v $Q/tf/kcache-soak:/cache -v $Q/tfbin/$BIN:/opt/tensorfold:ro -v $Q/prof-S:/prof --entrypoint /opt/nvidia/nvidia_entrypoint.sh $IMG \
  /opt/tensorfold/native/bin/tensorfold-native serve /model --name Qwen3.8-Flash-Next --host 127.0.0.1 --port $PORT --context 262144 --parallel $PARN --max-tokens 32768 \
  --temperature 1.0 --top-p 0.95 --top-k 20 --thinking --kv-cache $KV --vision --vision-max-images 50 --vision-max-videos 4 --vision-image-tokens 16384 $SERVEX >/dev/null || exit 1
nohup setsid bash $K/wdS.sh > $K/wd.log 2>&1 < /dev/null & echo $! > $K/wd.pid
SL='streams at once|prompt cache:? |q38 pool|q38 elastic|q38 arena|refus|does not fit|not enough|budget|sequence pool|q38 qsa knobs|q38 head2|q38 cert|q38 snap|kv'
for i in $(seq 450); do
  if curl -sf -m 3 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; then
    echo "READY in $(( $(date +%s)-T0 )) s; MemAvailable $(ma) GiB; GPU MiB $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | paste -sd+)"
    docker logs $N 2>&1 | grep -v 'prompt cache: [0-9]* tokens' | grep -iE "$SL" | cut -c1-300 | head -16; exit 0; fi
  [ "$(docker inspect -f '{{.State.Running}}' $N 2>/dev/null)" = true ] || { echo "FAILED: container gone after $(( $(date +%s)-T0 )) s"; docker logs --tail 15 $N 2>&1 | cut -c1-400; docker rm -f $N >/dev/null 2>&1; exit 1; }
  sleep 2
done; echo "FAILED: not ready after 900 s"; docker rm -f $N >/dev/null 2>&1; exit 1
