#!/bin/bash
# rtchain.sh BIN QUEUEFILE : run the job files named in QUEUEFILE one after another ("TAG JOBFILE" a line; the file is read again after
# every job, so lines can be appended while it runs). Each job runs through rtq.sh (lease D, window cap, night cutoff 12:04 UTC).
# Lease D is kept between job files (marker kit-D/RT_KEEP) and released at the end. Stop: touch ~/q38/kit-D/STOPRT.
Q=$HOME/q38; K=$Q/kit-D; RT=$K/redteam; BIN=$1; QF=$2; CUTE=$(date -u -d "2026-11-15 12:04:00" +%s)
while pgrep -f "[r]tq.sh" >/dev/null; do sleep 5; done
touch $K/RT_KEEP
trap 'rm -f $K/RT_KEEP $K/RT_HOLD; docker rm -f q38-run-D >/dev/null 2>&1; python3 $Q/kit/evict.py $Q/models $Q/tfbin >/dev/null 2>&1; bash $Q/shared/lease.sh release D | tail -1; echo "chain end $(date -u +%T) MemAvailable $(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo) GiB"' EXIT
n=0
while :; do
  n=$((n+1)); j=$(sed -n "${n}p" $QF); [ -z "$j" ] && { sleep 20; j=$(sed -n "${n}p" $QF); [ -z "$j" ] && break; }
  [ -f $K/STOPRT ] && break; [ $(date +%s) -gt $(( CUTE - 420 )) ] && { echo "night cutoff"; break; }
  set -- $j; echo "== chain: $j $(date -u +%T)"; bash $RT/rtq.sh ${3:-$BIN} $1 $RT/$2 > $RT/log_$1_$2.log 2>&1
done
