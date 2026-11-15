#!/bin/bash
# keepA.sh (HEAD, builder A): keeps lease A alive while a chain runs on spark-923b and releases it when the chain is done.
# Every 5 min: if 923b has ~/q38/runs/A_chain_done -> release A, exit. If lease A has < 105 min left -> renew 120 and write the deadline file on 923b.
# If the renew fails (window closing / no window): deadline 0 on 923b (no further engine starts), exit. Log ~/q38/keepA.log.
S=$HOME/q38/shared; B=picom@spark-923b; L=$HOME/q38/keepA.log
while :; do
  if ssh -n -o ConnectTimeout=10 $B 'test -f ~/q38/runs/A_chain_done'; then bash $S/lease.sh release A >> $L 2>&1; echo "$(date -u +%T) chain done: released" >> $L; exit 0; fi
  e=$(cat $S/leases/A 2>/dev/null || echo 0); left=$(( e - $(date +%s) ))
  if [ $left -lt 6300 ]; then
    if bash $S/lease.sh renew A 120 >> $L 2>&1; then ssh -n $B "echo $(( $(date +%s) + 118*60 )) > ~/q38/runs/A_deadline"; echo "$(date -u +%T) renewed" >> $L
    else ssh -n $B 'echo 0 > ~/q38/runs/A_deadline'; echo "$(date -u +%T) renew failed: deadline 0, exit" >> $L; exit 3; fi
  fi
  sleep 300
done
