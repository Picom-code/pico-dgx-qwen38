#!/bin/bash
# w2.sh "STEPS" : agent B night 11-14, WINDOW 2 chain on spark-3d24 (log ~/q38/runs/B_SOAK_w2.log). STEPS default: u16 ag
#   u16  USERS16 profile (bf16, 16 x 40,960 pool): 8-16 in flight, a full-window request arriving while 15 short ones run, lanes reused across sizes
#   ag   AGENTS profile: 14 agent runners with growing histories (24 sessions, churn, branches) + light other traffic + long prompts
#   ref  (only if window 1's reference pass was cut) the rest of the references
Q=$HOME/q38; K=$Q/kit-B/soak; R=$Q/runs; S="python3 -I $K/soak.py"; POOL=$K/pool.jsonl; REF=$R/B_SOAK_ref/ref.jsonl; RESF=$K/reserve_ok; CUT=1794744300
export RT_URL=http://127.0.0.1:18301 RT_MODEL=Qwen3.8-Flash-Next SOAK_CONTAINER=q38-run-B RT_KNOBS=$Q/prof-S/knobs
winend() { read O_ L_ W_ < $K/win; echo $(( O_ + 266*60 )); }
left() { a=$(( ( $(winend) - $(date +%s) ) / 60 )); b=$(( ( CUT - $(date +%s) ) / 60 )); echo $(( a < b ? a : b )); }
eng() { bash $K/engS.sh "$@" > $O/start_$(date -u +%H%M%S).txt 2>&1; rc=$?; tail -n 18 $(ls -t $O/start_*.txt | head -1) | cut -c1-300; return $rc; }
fin() { # save the log, stop as stop.sh does, count crash lines
  bash $K/engS.sh stop $O; L=$(ls -t $O/engine_*.log | head -1)
  echo "engine log $L: lines $(wc -l < $L); crash lines $(grep -ciE 'panic|segfault|SIGSEGV|core dumped|CUDA error|out of memory|unreachable|assertion' $L); check_diff nonzero $(grep -c 'check_diff=[1-9]' $L); watchdog stops: $(grep -c 'WATCHDOG STOP' $K/wd.log)"; }
counters() { docker logs q38-run-B > $O/engine_chk.log 2>&1
  for k in 'q38 qsacert' 'q38 cert:' 'q38 cert topk' 'q38 cert gather' 'q38 gdnprod' 'q38 routeband' 'q38 scratch check' 'q38 hcdn' 'q38 snapdelta keep' 'q38 qsa knobs'; do grep "$k" $O/engine_chk.log | tail -1 | cut -c1-420; done; }
soakrun() { # NAME PROFILE LEV KEEP_MIN MAX_MIN load-options... : one soak; when the load stops on LOW WATER (MemAvailable < 9 GiB) the next engine gets a larger memory reserve
  local name=$1 prof=$2 lev=$3 keep=$4 maxm=$5 try RES M nxt; shift 5
  for try in 1 2 3; do
    RES=$(cat $RESF 2>/dev/null); export O=$R/B_SOAK_${name}$([ $try -gt 1 ] && echo _r$try); mkdir -p $O
    M=$(( $(left) - keep - 5 )); [ $M -gt $maxm ] && M=$maxm
    [ $M -lt 12 ] && { echo "$name: only $M min: skipped"; return 0; }
    echo "--- $name try $try: profile $prof lev $lev reserve ${RES:-9 (recipe default)} GiB, load $M min, $(date -u +%T)"
    eng $prof $lev "${RES:+TENSORFOLD_MEMORY_RESERVE_GIB=$RES}" || { echo "ENGINE DID NOT START"; return 1; }
    $S load $POOL $REF $O $M winend=$(winend) "$@" > $O/load.log 2>&1
    grep -v '^  !!' $O/load.log | tail -2 | cut -c1-700; echo "mismatch / error lines: $(grep -c '^  !!' $O/load.log)"
    [ $lev = chk ] && counters
    fin; $S report $O | cut -c1-700
    if grep -q '"why": "LOW WATER' $O/end.json 2>/dev/null; then nxt=$([ -z "$RES" ] && echo 11 || echo $(( RES + 2 ))); echo "LOW WATER with reserve ${RES:-9}: the next engine gets TENSORFOLD_MEMORY_RESERVE_GIB=$nxt"; echo $nxt > $RESF; continue; fi
    return 0
  done; }
STEPS=${1:-"base u16 ag"}; U16MIN=${U16MIN:-52}; AGMIN=${AGMIN:-52}
for st in $STEPS; do
  echo "=== $st $(date -u +%T) minutes left $(left)"
  case $st in
  ref) export O=$R/B_SOAK_ref; mkdir -p $O
    eng def off || { echo "REF ENGINE DID NOT START"; exit 1; }
    $S ref $POOL $REF dl=$(( $(date +%s) + ${REFMIN:-25}*60 )) 2>&1 | tail -30 | cut -c1-240; fin ;;
  base) export O=$R/B_SOAK_base; mkdir -p $O     # F2: is the BASE (levers off) deterministic for sampled /v1/messages requests? and for a sampled chat request (control)
    eng def off || { echo "BASE ENGINE DID NOT START"; continue; }
    for id in q0200 q0209 q0218 q0227 q0001 q0004; do $S solo $POOL $REF $id 3 2>&1 | cut -c1-300; done; fin ;;
  u16) : > $RESF; soakrun U u16 on ${KEEPFOR:-0} $U16MIN seed=41 levels=8,12,16,10,14,16,9,16 per=200 longgap=400 full15=15 sess=25 burst=300 idle=600 ;;
  u16b) : > $RESF; soakrun U2 u16 on 0 $U16MIN seed=47 low=${LOW:-7.5} levels=8,12,16,10,14,16,9,16 per=200 longgap=400 full15=15 sess=25 burst=300 idle=600 ;;   # stop line 7.5 GiB (see PROGRESS-B 09:45)
  ag) : > $RESF; soakrun G agents on ${KEEPFOR2:-0} $AGMIN seed=43 agents=${NAG:-14} levels=1,2,1,2 per=200 sess=0 longgap=600 burst=500 idle=0 ;;
  esac
done
echo "W2 CHAIN DONE $(date -u +%T)"
