#!/bin/bash
# rtq.sh BIN TAG JOBFILE : red-team job runner on the head (builder D, round 12). Resumable; spans windows by itself.
#   ENGINE name|PROFILE|LEV|recipe options|-e words|shell env for run12 (PAR=2 ...)   (re)defines the engine; out dir $O = runs/D_RT_TAG_name
#   [NEED=min ]command          runs with $O $K $RT $KK; done once (marker $O/.done_N). rc 3 = "deadline: call me again on a fresh engine".
# Lease D: acquired before every engine start (the start happens only after "WINDOW OPEN"), renewed every 8 min while the runner lives,
# released at the end and whenever the runner waits for a new window. Engine never started after window age 262 min (run12.sh).
BIN=$1; T=$2; J=$3; Q=$HOME/q38; S=$Q/shared; export K=$Q/kit-D KK=$Q/kit; export RT=$K/redteam; R=$Q/runs; export QK_MODEL=Qwen3.8-Flash-Next
CUTE=$(date -u -d "2026-11-15 12:04:00" +%s); export CUTE
touch $K/RT_HOLD
( while kill -0 $$ 2>/dev/null; do sleep 480; kill -0 $$ 2>/dev/null && [ -f $K/RT_HOLD ] && [ -f $S/OPEN ] && [ ! -f $S/CLOSING ] && bash $S/lease.sh renew D 120 >/dev/null 2>&1; done ) & BG=$!
UP=0; ENG_DL=0; ES=0; O=""; SKIP=none
stop_engine() { [ $UP = 1 ] || return 0; ES=$(ls -d $O/eng* 2>/dev/null | wc -l); ES=$((ES+1)); mkdir -p $O/eng$ES
  KEEP=1 bash $K/stop.sh $O/eng$ES | tail -1; cp $K/eng_env.txt $O/eng$ES/ 2>/dev/null
  echo "engine log: crash lines $(grep -ciE 'panic|segfault|error\(|abort|out of memory' $O/eng$ES/engine.log), min MemAvailable $(awk 'NF==2{print $2}' $O/eng$ES/wd.log | sort -n | head -1) MiB"; UP=0; }
trap 'stop_engine; docker rm -f q38-run-D >/dev/null 2>&1; kill $BG 2>/dev/null; if [ -f $K/RT_KEEP ] && [ $(date +%s) -lt $CUTE ]; then echo "lease D kept for the next job file"; else rm -f $K/RT_HOLD; python3 $KK/evict.py $Q/models $Q/tfbin >/dev/null 2>&1; bash $S/lease.sh release D | tail -1; fi; echo "rtq end $(date -u +%T) MemAvailable $(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo) GiB"' EXIT
start_engine() {
  for try in $(seq 40); do
    [ -f $K/STOPRT ] && return 1
    [ $(date +%s) -gt $(( CUTE - 420 )) ] && { echo "night cutoff: no engine start"; return 1; }
    A=$(bash $S/lease.sh acquire D 120 2>&1); echo "$A" | grep -q "WINDOW OPEN" || { echo "acquire: no window ($(echo "$A" | tail -1 | cut -c1-100)); wait"; sleep 60; continue; }
    touch $K/RT_HOLD
    env $CUR_SH bash $RT/run12.sh $BIN $CUR_PROF $CUR_LEV 268 "$CUR_RO" "$CUR_ENV"; rc=$?
    if [ $rc = 0 ]; then UP=1; ENG_DL=$(( $(cat $K/eng_dl) - 120 )); docker logs q38-run-D 2>&1 | head -400 > $O/start_$(date -u +%H%M%S).log; return 0; fi
    echo "engine start failed rc $rc (try $try) $(date -u +%T)"; rm -f $K/RT_HOLD; bash $S/lease.sh release D >/dev/null 2>&1
    [ $rc = 1 ] && [ $try -ge 2 ] && return 1
    [ $rc = 5 ] && return 1
    sleep 240
  done; return 1; }
alive() { curl -sf -m 8 http://127.0.0.1:18301/v1/models >/dev/null 2>&1; }
echo "rtq start $(date -u +%T) $BIN $T $J"
n=0
while IFS= read -r -u 3 line; do
  n=$((n+1))
  case "$line" in ''|\#*) continue ;; ENGINE\ *) stop_engine; IFS='|' read -r NM CUR_PROF CUR_LEV CUR_RO CUR_ENV CUR_SH <<< "${line#ENGINE }"; O=$R/D_RT_${T}_$NM; mkdir -p $O
      echo "== ENGINE $NM $CUR_PROF lev=$CUR_LEV [$CUR_RO] [$CUR_ENV] [$CUR_SH] $(date -u +%T)"; continue ;; esac
  need=4; case "$line" in NEED=*) need=${line%% *}; need=${need#NEED=}; line=${line#* } ;; esac
  [ -f $O/.done_$n ] && continue
  [ "$SKIP" = "$NM" ] && continue
  [ -f $K/STOPRT ] && { echo "STOPRT"; break; }
  [ $(date +%s) -gt $(( CUTE - 120 )) ] && { echo "NIGHT CUTOFF: stop"; break; }
  for attempt in 1 2 3 4 5 6; do
    now=$(date +%s)
    if [ $UP = 0 ] || [ $(( ENG_DL - now )) -lt $(( need*60 )) ] || ! alive; then stop_engine; start_engine || { echo "NO ENGINE for [$NM]: skipping this engine's commands"; SKIP=$NM; break; }; fi
    echo "-- [$n] $line  ($(date -u +%T))"
    O=$O DL=$ENG_DL bash -c "$line"; rc=$?
    if [ $rc = 3 ]; then UP=1; stop_engine; continue; fi
    if alive; then touch $O/.done_$n; break; fi
    echo "ENGINE NOT ANSWERING after the command (rc $rc) $(date -u +%T): saving its log, restart, repeat"; docker logs --tail 8 q38-run-D 2>&1 | cut -c1-400; tail -2 $K/wd.log; UP=1; stop_engine
  done
done 3< $J
