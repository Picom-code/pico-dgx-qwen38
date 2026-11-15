#!/bin/bash
# startN.sh "STEPS" LOGNAME [MAXAGE_MIN=15] [NOTBEFORE_EPOCH] (HEAD, agent A, night 11-14): wait until NOTBEFORE (default 05:00 UTC Nov 15), acquire lease A;
# if the open window is older than MAXAGE minutes, release and wait for it to close, then acquire a fresh one; start wN.sh on spark-923b + keeper.
S=$HOME/q38/shared; B=picom@spark-923b; L=$HOME/q38/startN.log; STEPS=$1; LN=$2; MAXAGE=${3:-15}; NB=${4:-$(date -u -d '2026-11-15 05:00:05' +%s)}; CUT=$(date -u -d '2026-11-15 11:40:00' +%s)
while [ $(date +%s) -lt $NB ]; do sleep 5; done
pkill -f 'q38/keep[N].sh' 2>/dev/null; sleep 1
for t in $(seq 400); do
  [ $(date +%s) -ge $CUT ] && { echo "$(date -u +%T) past 11:40 UTC: no new chain" >> $L; exit 2; }
  bash $S/lease.sh acquire A 120 >> $L 2>&1 || { sleep 20; continue; }
  o=$(cat $S/OPEN 2>/dev/null || echo 0); age=$(( ($(date +%s) - o) / 60 ))
  [ $o != 0 ] && [ $age -le $MAXAGE ] && break
  bash $S/lease.sh release A >> $L 2>&1; echo "$(date -u +%T) window is $age min old: released, waiting" >> $L; sleep 45
done
[ -f $S/leases/A ] || { echo "$(date -u +%T) no lease: stop" >> $L; exit 2; }
o=$(cat $S/OPEN); echo "$(date -u +%T) lease ok; window opened $(date -u -d @$o +%T), usable until $(date -u -d @$(( o + 270*60 )) +%T)" >> $L
ssh -n $B "echo $(( $(date +%s) + 118*60 )) > ~/q38/runs/A_deadline; echo $(( o + 270*60 )) > ~/q38/runs/A_winend; date +%s > ~/q38/runs/A_stamp; rm -f ~/q38/runs/A_chain_done; (nohup bash -c 'bash ~/q38/kit-N/${SC:-wN.sh} \"$STEPS\" > ~/q38/runs/$LN.log 2>&1' > /dev/null 2>&1 &); echo started" >> $L 2>&1
nohup bash $HOME/q38/keepN.sh > /dev/null 2>&1 &
echo "$(date -u +%T) chain ${SC:-wN.sh} [$STEPS] -> $LN.log + keeper started" >> $L
