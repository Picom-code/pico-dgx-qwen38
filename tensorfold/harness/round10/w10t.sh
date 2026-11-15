#!/bin/bash
# w10t.sh (923b): after w10e: does the custom engine R launcher still start here (before the final window)? 2 runs of the 3-prompt bench, then stop.
. $HOME/q38/kit-A/libA.sh
while pgrep -f 'kit-A10/w10e.sh' >/dev/null; do sleep 3; done; rm -f $RUNS/A_chain_done
if [ $(left) -ge 1500 ]; then bash $KA/eng.sh fly default 2>&1 | tail -2 | cut -c1-200; curl -sf -m 5 http://127.0.0.1:18301/v1/models | cut -c1-200; echo; $Q c1 2 | tail -2; $Q distinct 8 | tail -1; docker logs q38-run-fly 2>&1 | tail -3 | cut -c1-200; bash $KA/eng.sh stop; fi
echo "w10t done $(date -u +%T)"; touch $RUNS/A_chain_done
