#!/bin/bash
# go_n.sh : night 11-14 launcher (agent D). Sleeps until 05:00:15 UTC (no lease / engine before 05:00), then runs the queue q1.txt on the
# final image's binary F11 (af27a1e7, copied from spark-923b); if F11 cannot start an engine on the head, the same queue on I11
# (0efcbbf3: the head's own build of the same tree 17a4c9d). Everything else (lease D, window cap, 12:04 UTC cutoff) is in rtchain / rtq / run12.
Q=$HOME/q38; RT=$Q/kit-D/redteam; T0=$(date -u -d "2026-11-15 05:00:15" +%s); now=$(date +%s)
[ $now -lt $T0 ] && sleep $(( T0 - now ))
echo "go $(date -u +%T)"
bash $RT/rtchain.sh F11 $RT/q1.txt
if [ ! -s $Q/runs/D_RT_N_on/pair1.jsonl ] && [ ! -f $Q/kit-D/STOPRT ]; then echo "F11 produced nothing: same queue on I11 $(date -u +%T)"; bash $RT/rtchain.sh I11 $RT/q1.txt; fi
echo "go end $(date -u +%T)"
