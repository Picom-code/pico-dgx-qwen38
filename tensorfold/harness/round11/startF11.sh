#!/bin/bash
# startF11.sh SCRIPT "STEPS" LOGNAME [MAXAGE_MIN=12] (HEAD, builder A): acquire lease A; if the open window is older than MAXAGE_MIN minutes, release and wait
# for it to close (other builders' leases keep it open until its cap), then acquire a fresh one; start the chain on spark-923b and the keeper.
S=$HOME/q38/shared; B=picom@spark-923b; L=$HOME/q38/startF.log; SC=$1; STEPS=$2; LN=$3; MAXAGE=${4:-12}
for t in $(seq 400); do
  bash $S/lease.sh acquire A 120 >> $L 2>&1 || { sleep 20; continue; }
  o=$(cat $S/OPEN 2>/dev/null || echo 0); age=$(( ($(date +%s) - o) / 60 ))
  [ $o != 0 ] && [ $age -le $MAXAGE ] && break
  bash $S/lease.sh release A >> $L 2>&1; echo "$(date -u +%T) window is $age min old: released, waiting" >> $L; sleep 45
done
[ -f $S/leases/A ] || { echo "$(date -u +%T) no lease: stop" >> $L; exit 2; }
o=$(cat $S/OPEN); echo "$(date -u +%T) lease ok; window opened $(date -u -d @$o +%T), usable until $(date -u -d @$(( o + 270*60 )) +%T)" >> $L
ssh -n $B "echo $(( $(date +%s) + 118*60 )) > ~/q38/runs/A_deadline; echo $(( o + 270*60 )) > ~/q38/runs/A_winend; rm -f ~/q38/runs/A_chain_done; (nohup bash -c 'P=F11 bash ~/q38/kit-A11/$SC \"$STEPS\" > ~/q38/runs/$LN.log 2>&1' > /dev/null 2>&1 &); echo started" >> $L 2>&1
nohup bash $HOME/q38/keepA11.sh > /dev/null 2>&1 &
echo "$(date -u +%T) chain $SC [$STEPS] + keeper started" >> $L
