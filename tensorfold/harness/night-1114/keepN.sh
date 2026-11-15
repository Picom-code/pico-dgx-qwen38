#!/bin/bash
# keepN.sh (HEAD, agent A, night 11-14): keeps lease A alive while a chain runs on spark-923b; stamps the box every minute:
# ~/q38/runs/A_winend (window start + 270 min), A_deadline (lease end - 2 min), A_stamp (now). Releases when the chain is done, the window
# ends, or at the night's hard cutoff 12:06 UTC (then it also removes the engine container on the box).
S=$HOME/q38/shared; B=picom@spark-923b; L=$HOME/q38/keepN.log; CUT=$(date -u -d '2026-11-15 12:06:00' +%s); R='~/q38/runs'
while :; do
  now=$(date +%s); o=$(cat $S/OPEN 2>/dev/null || echo 0); we=$(( o + 270*60 ))
  if [ $now -ge $CUT ]; then ssh -n -o ConnectTimeout=10 $B "echo 0 > $R/A_deadline; echo 0 > $R/A_winend; docker rm -f q38-run-A >/dev/null 2>&1; touch $R/A_chain_done"; bash $S/lease.sh release A >> $L 2>&1; echo "$(date -u +%T) CUTOFF: engine removed, released, exit" >> $L; exit 0; fi
  if [ $o = 0 ] || [ $(( we - now )) -lt 120 ]; then ssh -n -o ConnectTimeout=10 $B "echo 0 > $R/A_deadline; echo 0 > $R/A_winend"; bash $S/lease.sh release A >> $L 2>&1; echo "$(date -u +%T) no window / window end: stamps 0, released, exit" >> $L; exit 0; fi
  if ssh -n -o ConnectTimeout=10 $B "test -f $R/A_chain_done"; then bash $S/lease.sh release A >> $L 2>&1; ssh -n $B "echo 0 > $R/A_deadline"; echo "$(date -u +%T) chain done: released" >> $L; exit 0; fi
  e=$(cat $S/leases/A 2>/dev/null || echo 0); left=$(( e - now ))
  if [ $left -lt 6300 ]; then
    if bash $S/lease.sh renew A 120 >> $L 2>&1; then e=$(cat $S/leases/A); echo "$(date -u +%T) renewed" >> $L
    else ssh -n $B "echo 0 > $R/A_deadline; echo 0 > $R/A_winend"; echo "$(date -u +%T) renew failed: stamps 0, exit" >> $L; exit 3; fi
  fi
  ssh -n -o ConnectTimeout=10 $B "echo $(( e - 120 )) > $R/A_deadline; echo $we > $R/A_winend; date +%s > $R/A_stamp"
  sleep 60
done
