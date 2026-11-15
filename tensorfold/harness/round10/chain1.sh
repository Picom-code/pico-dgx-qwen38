#!/bin/bash
# chain1.sh (923b): after the image build: knob-fix pair, head dumps, cascade study, leftovers. Log ~/q38/runs/A_c1.log
while pgrep -f 'kit-A10/prep.sh' >/dev/null; do sleep 3; done
tail -1 ~/q38/recipe_work/prepare10b.log | cut -c1-200
bash ~/q38/kit-A10/w10k.sh
bash ~/q38/kit-A10/wD10.sh J2
bash ~/q38/kit-A10/casc_run.sh 3000 > ~/q38/runs/C10_casc.log 2>&1; tail -45 ~/q38/runs/C10_casc.log
bash ~/q38/kit-A10/w10a.sh "one users"
echo "chain1 done $(date -u +%T)"
