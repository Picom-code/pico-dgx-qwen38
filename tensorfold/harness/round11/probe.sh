#!/bin/bash
# probe.sh "NAME|RECIPE ARGS" ... (923b): start line / budget line of recipe.sh start variants (no measurement). Needs the lease stamps.
. $HOME/q38/kit-A/libA.sh; . $HOME/q38/kit-A9/lib9.sh; . $HOME/q38/kit-A11/lib11.sh
for v in "$@"; do n=${v%%|*}; a=${v#*|}; [ $(left) -ge 300 ] || { echo "$n SKIPPED: $(left) s left"; continue; }
  bash $KA/rsA.sh $a > $RUNS/probe_$n.txt 2>&1; docker logs q38-run-A > $RUNS/probe_$n.eng 2>&1
  echo "== $n [$a] $(date -u +%T): $(head -c 160 $RUNS/probe_$n.txt | head -1)"; grep -hE 'streams at once|needs .* GiB|prompt cache|q38 arena|q38 elastic' $RUNS/probe_$n.eng | cut -c1-330 | tail -4
  awk '/MemAvailable/{print "   MemAvailable GiB " int($2/1048576)}' /proc/meminfo; bash $KA/eng.sh stop >/dev/null; sleep 3
done; echo "probe done $(date -u +%T)"; touch $RUNS/A_probe_done
