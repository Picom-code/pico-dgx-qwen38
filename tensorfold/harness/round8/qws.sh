#!/bin/bash
# qw.sh LABEL GUARD_MIN SCRIPT : a Qwen test window on the three WORKERS (never the head). Modelled on ~/s5/sw.sh.
# Stops DeepSeek prod, sources SCRIPT (cwd ~/q38/runs/LABEL = $D; r3 = spark-923b, w2 = spark-3d24, w3 = spark-4e5c),
# then ALWAYS, on every worker: stops Qwen containers, deletes temporary checkpoint copies, evicts Qwen files from page cache;
# then restarts DeepSeek and verifies it. Trap + guard unit q38-guard, no LLM needed.
#   systemd-run --user --unit=q38-win /bin/bash ~/q38/qw.sh LABEL GUARD_MIN ~/q38/w_x.sh
L=$1; GMIN=$2; CMDS=$3
export D=$HOME/q38/runs/$L; mkdir -p $D; exec > $D/run.log 2>&1
B3=picom@spark-923b; B2=pico@spark-3d24; B4=pico@spark-4e5c
SSH="ssh -n -o ConnectTimeout=10 -o ServerAliveInterval=30"
r3() { $SSH $B3 "$@"; }; w2() { $SSH $B2 "$@"; }; w3() { $SSH $B4 "$@"; }
T0=$(date +%s)
stopall() { bash $HOME/q38/kit/qstop.sh 2>&1 | tail -1 & for h in $B3 $B2 $B4; do timeout 400 ssh -n -o ConnectTimeout=10 $h 'bash ~/q38/kit/qstop.sh' 2>&1 | tail -1 & done; wait; }
restore() {
  trap - EXIT INT TERM
  echo "restore begin $(date -u +%FT%TZ)"; kill $TP 2>/dev/null
  touch $HOME/q38/shared/CLOSING; rm -f $HOME/q38/shared/OPEN $HOME/q38/shared/leases/*
  stopall
  $HOME/dsgx10-wide/prod_up.sh
  systemctl --user stop q38-guard.timer q38-guard 2>/dev/null
  bash $HOME/q38/verify_prod.sh
  echo "prod restored; outage $(( $(date +%s)-T0 ))s  start $(date -u -d @$T0 +%FT%TZ) end $(date -u +%FT%TZ)"
  echo "$L $(date -u -d @$T0 +%FT%TZ) $(date -u +%FT%TZ) $(( $(date +%s)-T0 ))s" >> $HOME/q38/windows.log
  echo DONE > $D/DONE
  rm -f $HOME/q38/shared/CLOSING
}
trap restore EXIT INT TERM
systemctl --user stop q38-guard.timer q38-guard 2>/dev/null; systemctl --user reset-failed q38-guard 2>/dev/null
systemd-run --user --unit=q38-guard --on-active=$((GMIN*60)) /bin/bash -c "pkill -TERM -f '^/bin/bash /home/pico/q38/shared/qws.sh'; sleep 420; bash /home/pico/q38/kit/qstop.sh; for h in $B3 $B2 $B4; do timeout 400 ssh -n -o ConnectTimeout=10 \$h 'bash ~/q38/kit/qstop.sh'; done; curl -s -o /dev/null -w %{http_code} 127.0.0.1:18900/health | grep -q 200 || /home/pico/dsgx10-wide/prod_up.sh; bash /home/pico/q38/verify_prod.sh >> /home/pico/q38/guard.log 2>&1"
( sleep $((GMIN*60-200)); echo "WATCHDOG: cap hit $(date -u +%T)"; kill -TERM $$ ) & WD=$!
echo "start $(date -u +%FT%TZ) label $L guard ${GMIN}min"
r3 'docker ps --format "{{.Names}}" | grep "^q38-dl" | xargs -r docker pause' >/dev/null 2>&1   # downloads do not run during measurements
bash $HOME/dsgx10/serve.sh stop; sleep 3
for i in $(seq 60); do a=$(r3 "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo"); [ "${a:-0}" -ge 105 ] && break; sleep 2; done
echo "spark-923b MemAvailable ${a} GiB after prod stop ($(( $(date +%s)-T0 ))s)"
# GPU temperature, power and clock of all four boxes every 30 s (rule 5)
( while :; do l="$(date -u +%T)"; l="$l head $(nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits | tr -d ' ')"; for h in $B2 $B4 $B3; do l="$l ${h#*@spark-} $(timeout 8 ssh -n -o ConnectTimeout=5 $h 'nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits; cat /sys/class/thermal/thermal_zone*/temp | sort -n | tail -1' 2>/dev/null | tr -d ' ' | paste -sd/)"; done; echo "$l"; sleep 30; done > $D/temps.log ) & TP=$!
# xcopy SRCDIR_ON_923b NAME : temporary copy 923b -> 3d24:~/q38/tmpmodels/NAME over the fabric (deleted by qstop before prod restarts)
xcopy() { w2 "python3 -I ~/q38/kit/xfer.py recv spark-3d24 18398 ~/q38/tmpmodels/$2" & sleep 2; r3 "python3 -I ~/q38/kit/xfer.py send spark-3d24 18398 $1"; wait $!; w2 "df -h ~ | tail -1"; }
cd $D && source "$CMDS"
kill $WD 2>/dev/null
