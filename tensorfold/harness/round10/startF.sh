#!/bin/bash
# startF.sh "STEPS" (HEAD, builder A): acquire lease A (waits for a closing window, opens a new one), then start the final chain on spark-923b and the keeper.
S=$HOME/q38/shared; B=picom@spark-923b; L=$HOME/q38/startF.log; STEPS=${1:-"off on g0 fly off2 mu mia chk"}
for i in 1 2 3 4 5 6; do bash $S/lease.sh acquire A 120 >> $L 2>&1 && break; sleep 20; done
[ -f $S/leases/A ] || { echo "$(date -u +%T) no lease: stop" >> $L; exit 2; }
echo "$(date -u +%T) lease ok; window opened $(date -u -d @$(cat $S/OPEN) +%T)" >> $L
ssh -n $B "echo $(( $(date +%s) + 118*60 )) > ~/q38/runs/A_deadline; rm -f ~/q38/runs/A_chain_done; (nohup bash -c 'P=F10 bash ~/q38/kit-A10/wF10.sh \"$STEPS\" > ~/q38/runs/wF10.log 2>&1' > /dev/null 2>&1 &); echo started" >> $L 2>&1
nohup bash $HOME/q38/keepA.sh > /dev/null 2>&1 &
echo "$(date -u +%T) chain + keeper started" >> $L
