#!/bin/bash
# keepA11.sh (HEAD, builder A, round 11): keeps lease A alive while a chain runs on spark-923b and releases it when the chain is done.
# Every 2 min: writes 923b ~/q38/runs/A_winend (window start + 270 min) and A_deadline (lease end - 2 min); renews the lease when < 105 min are left
# and the window still has room; releases when 923b has ~/q38/runs/A_chain_done or the window end is under 2 min away. No window / renew fails: both stamps 0.
S=$HOME/q38/shared; B=picom@spark-923b; L=$HOME/q38/keepA.log
while :; do
  now=$(date +%s); o=$(cat $S/OPEN 2>/dev/null || echo 0); we=$(( o + 270*60 ))
  if [ $o = 0 ] || [ $(( we - now )) -lt 120 ]; then ssh -n -o ConnectTimeout=10 $B 'echo 0 > ~/q38/runs/A_deadline; echo 0 > ~/q38/runs/A_winend'; bash $S/lease.sh release A >> $L 2>&1; echo "$(date -u +%T) no window / window end: stamps 0, released, exit" >> $L; exit 0; fi
  if ssh -n -o ConnectTimeout=10 $B 'test -f ~/q38/runs/A_chain_done'; then bash $S/lease.sh release A >> $L 2>&1; ssh -n $B 'echo 0 > ~/q38/runs/A_deadline'; echo "$(date -u +%T) chain done: released" >> $L; exit 0; fi
  e=$(cat $S/leases/A 2>/dev/null || echo 0); left=$(( e - now ))
  if [ $left -lt 6300 ]; then
    if bash $S/lease.sh renew A 120 >> $L 2>&1; then e=$(cat $S/leases/A); echo "$(date -u +%T) renewed" >> $L
    else ssh -n $B 'echo 0 > ~/q38/runs/A_deadline; echo 0 > ~/q38/runs/A_winend'; echo "$(date -u +%T) renew failed: stamps 0, exit" >> $L; exit 3; fi
  fi
  ssh -n -o ConnectTimeout=10 $B "echo $(( e - 120 )) > ~/q38/runs/A_deadline; echo $we > ~/q38/runs/A_winend"
  sleep 120
done
