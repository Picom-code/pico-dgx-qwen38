#!/bin/bash
# w1.sh "STEPS" : agent B night 11-14, WINDOW 1 chain on spark-3d24 (detached; log ~/q38/runs/B_SOAK_w1.log). STEPS default: ref soakA chk inj
#   ref   levers-off engine: reference pass (every request solo), at most REFMIN minutes
#   soakA recipe default profile, mixed load 1-8 in flight, until (window end - what chk and inj need)
#   chk   recipe default + every device check mode, mixed load CHKMIN minutes; check counters printed
#   inj   recipe default: HTTP failure injection UNDER load, then stop / start under load (as recipe.sh stop / start: docker stop -t 30; rm; run)
# Every engine start is refused by engS.sh outside the window; the load driver stops itself at window start + 266 min, at 12:05 UTC and below 9 GiB MemAvailable.
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
STEPS=${1:-"ref soakA chk inj"}; REFMIN=${REFMIN:-48}; CHKMIN=${CHKMIN:-46}; INJMIN=${INJMIN:-27}
for st in $STEPS; do
  echo "=== $st $(date -u +%T) window minutes left $(left)"
  case $st in
  ref) export O=$R/B_SOAK_ref; mkdir -p $O
    eng def off || { echo "REF ENGINE DID NOT START"; exit 1; }
    $S ref $POOL $REF dl=$(( $(date +%s) + REFMIN*60 )) 2>&1 | tail -45 | cut -c1-240
    fin ;;
  soakA) : > $RESF; soakrun A def on $(( CHKMIN + INJMIN + 9 )) 150 seed=11 ;;
  chk) soakrun C def chk $(( INJMIN + 4 )) $CHKMIN seed=23 longgap=420 ;;
  inj) export O=$R/B_SOAK_I; mkdir -p $O/load1 $O/load2; RES=$(cat $RESF 2>/dev/null)
    M=$(( $(left) - 3 )); [ $M -lt 14 ] && { echo "inj: only $M min: skipped"; continue; }
    eng def on "${RES:+TENSORFOLD_MEMORY_RESERVE_GIB=$RES}" || { echo "ENGINE DID NOT START"; continue; }
    # 1. HTTP failure injection while a mixed load runs (levels 2-4)
    $S load $POOL $REF $O/load1 7 winend=$(winend) seed=31 levels=3,4,2 per=120 longgap=0 idle=0 maxp=20000 > $O/load1.log 2>&1 & LP=$!
    sleep 45; $S inject $POOL $REF $O 2>&1 | cut -c1-260 | tail -45
    wait $LP; grep -v '^  !!' $O/load1.log | tail -1 | cut -c1-600
    echo "after injection: container running $(docker inspect -f '{{.State.Running}} restarts={{.RestartCount}}' q38-run-B), MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GiB"
    # 2. stop / start under load (what recipe.sh stop / start do: docker stop -t 30, rm, run), the load keeps coming; then every reply must be right again
    M2=$(( $(left) - 2 )); [ $M2 -gt 11 ] && M2=11
    if [ $M2 -ge 9 ]; then
      touch $O/load2/OUTAGE_OK
      $S load $POOL $REF $O/load2 $M2 winend=$(winend) seed=37 levels=4,6,3 per=120 longgap=0 idle=0 maxp=20000 drain=60 > $O/load2.log 2>&1 & LP=$!
      sleep 100; T0=$(date +%s); docker logs q38-run-B > $O/engine_before_restart.log 2>&1
      docker stop -t 30 q38-run-B >/dev/null; RC=$(docker inspect -f '{{.State.ExitCode}}' q38-run-B); docker rm -f q38-run-B >/dev/null; T1=$(date +%s); sleep 3
      echo "stop under load: took $(( T1 - T0 )) s, exit code $RC; after stop: MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GiB, GPU MiB in use [$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | paste -sd+)], port listening: $(ss -ltn 'sport = :18301' | grep -c LISTEN); crash lines before the stop: $(grep -ciE 'panic|segfault|SIGSEGV|core dumped|CUDA error|out of memory' $O/engine_before_restart.log)"
      eng def on "${RES:+TENSORFOLD_MEMORY_RESERVE_GIB=$RES}"; T2=$(date +%s); echo "$T0 $T2" > $O/load2/OUTAGE; echo "restart under load: engine answers again after $(( T2 - T0 )) s"
      wait $LP; grep -v '^  !!' $O/load2.log | tail -1 | cut -c1-600
    else echo "restart test skipped: $M2 min left"; fi
    fin; $S report $O/load1 $O/load2 | cut -c1-700 ;;
  esac
done
echo "W1 CHAIN DONE $(date -u +%T)"
