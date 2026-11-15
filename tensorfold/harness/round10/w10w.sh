#!/bin/bash
# w10w.sh (923b): when the running wF10.sh has finished its "mu" step, stop that chain cleanly (script, launcher children, container) and start
# wF10b.sh "on off3 chk" (the default run that did not start at 15:12, a third levers-off run after it, the check modes). Log ~/q38/runs/wF10b.log.
R=$HOME/q38/runs
while ! grep -q 'F10_mu done' $R/wF10.log; do sleep 5; done
for pat in 'kit-A10/wF10.s[h]' 'kit-A/rsA.s[h]' 'recipe.s[h] start' 'start.s[h] restart'; do for p in $(pgrep -f "$pat"); do kill $p 2>/dev/null; done; done
sleep 20; docker ps -a --format '{{.Names}}' | grep '^q38-run' | xargs -r docker rm -f >/dev/null 2>&1; sleep 5
echo "old chain stopped $(date -u +%T)" > $R/wF10b.log
P=F10 bash $HOME/q38/kit-A10/wF10b.sh "on off3 chk" >> $R/wF10b.log 2>&1
