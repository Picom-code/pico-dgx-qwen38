#!/bin/bash
# wdS.sh : watchdog beside the soak engine (spark-3d24). Removes q38-run-B when: MemAvailable < 5 GiB (hard floor; the soak driver stops the load at 9),
# the keeper's window stamp is older than 6 min, the window is older than 270 min, lease B ended, or it is 12:08 UTC. Logs MemAvailable every 10 s.
Q=$HOME/q38; K=$Q/kit-B/soak; N=q38-run-B; CUT=${CUT:-1794744480}
while sleep 10; do
  [ "$(docker inspect -f '{{.State.Running}}' $N 2>/dev/null)" = true ] || { echo "$(date -u +%T) container not running: watchdog ends"; exit 0; }
  A=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); now=$(date +%s); read O L W < $K/win 2>/dev/null
  echo "$(date -u +%T) $A"
  why=""
  [ $A -lt 5120 ] && why="MemAvailable $A MiB < 5 GiB"
  [ -z "$W" ] || [ $(( now - W )) -gt 360 ] && why="window stamp stale"
  [ -n "$O" ] && [ $now -gt $(( O + 270*60 )) ] && why="window cap (270 min)"
  [ -n "$L" ] && [ $now -gt "$L" ] && why="lease ended"
  [ $now -gt $CUT ] && why="cutoff 12:08 UTC"
  if [ -n "$why" ]; then echo "$(date -u +%T) WATCHDOG STOP: $why"; mkdir -p $Q/runs/B_SOAK_wd; docker logs $N > $Q/runs/B_SOAK_wd/engine_$(date -u +%H%M%S).log 2>&1; docker rm -f $N >/dev/null 2>&1; exit 0; fi
done
