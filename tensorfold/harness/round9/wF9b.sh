#!/bin/bash
# wF9.sh "STEPS" (923b, builder A, round 9): FINAL suite on the DOCUMENTED path (recipe.sh prepare, recipe.sh start on that image, vision on).
# STEPS: prep chk off on cmp mu mia pw mu2. Every step checks ~/q38/runs/A_deadline first. Labels ${P}_*, P default F9.
. $HOME/q38/kit-A/libA.sh; . $HOME/q38/kit-A9/lib9.sh; STEPS=${1:-"off on cmp"}; P=${P:-F9}
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
CL="docker logs q38-run-A 2>&1 | grep -E 'q38 cert|q38 hcf|q38 head2|q38 elastic|q38 qsacert|streams at once' | cut -c1-230 | awk '!s[substr(\$0,1,18)]++ || /calls/' | tail -12"
SQ="SQ_QK=$KA/qk.py python3 -I $K9/sq.py"
SAMP="$SQ run \$O/sq07off.jsonl s07off 24 256 | tail -1; $SQ run \$O/sq07on.jsonl s07on 24 256 | tail -1; $SQ sd \$O/sqsd.jsonl 6 | tail -1"
while pgrep -f 'kit-A9/wF9.sh' >/dev/null; do sleep 5; done
if has prep; then ( cd ~/q38/repo_tf && WORK=$HOME/q38/recipe_work HF_CACHE=$HOME/q38/hf bash ./recipe.sh prepare > ~/q38/recipe_work/prepare9f.log 2>&1; echo "prepare rc $?"; ls ~/q38/recipe_work/miaai-recipe/patches | tail -1; tail -2 ~/q38/recipe_work/prepare9f.log | cut -c1-200; docker images --format "{{.Repository}}:{{.Tag}} {{.CreatedSince}}" | grep tensorfold-qwen38; sha256sum ~/q38/hf/q38/head2_*.bin | cut -c1-90 ); fi
# check modes: the certified head (greedy + sampled top-k) and the certified indexer select compare themselves with the stock kernels on the device
has chk && gor "CERT=2 QSA_CERT=2 TF_FLASHNEXT_Q38_QSA_STATS=1" ${P}_chk "$Q prefill 32000,128000 | tail -2; $Q sd 1 | tail -5; $SQ run \$O/sq07off.jsonl s07off 12 256 | tail -1; $Q ident \$O/ident.json | tail -1; $C $RUNS/P_ref1/ident.json \$O/ident.json; docker logs q38-run-A 2>&1 | grep -E 'q38 qsacert|q38 cert' | tail -6 | cut -c1-400" 900
has off && gor "$OFF" ${P}_off "MAXU=5 HELD_STRIDE=1 SKIP=multi bash ~/q38/kit/suite.sh ${P}_off full | tail -2; $SETS; $SAMP" 5400
has on && gor "" ${P}_on "MAXU=5 HELD_STRIDE=1 SKIP=multi bash ~/q38/kit/suite.sh ${P}_on full | tail -2; $SETS; $SAMP; $CL" 5400
if has cmp; then for f in ident_on.json held.jsonl edit.jsonl agent.jsonl; do $C $RUNS/${P}_off/$f $RUNS/${P}_on/$f; done; for f in sq07off sq07on sqsd; do eval "$SQ cmp $RUNS/${P}_off/$f.jsonl $RUNS/${P}_on/$f.jsonl" | tail -3; done; fi
H16="$HP 16 \$O/held16.jsonl 256 3 | tail -1; $C $RUNS/A_ref1/held16.jsonl \$O/held16.jsonl"
has mu && gor "USERS16=1" ${P}_mu "$MU; $WV; python3 -I $KA/poolt.py 150000x1+15 | cut -c1-420; $Q ctx 250000 | tail -1; $H16; $CL" 1800
has mia && gor "$OFF PARALLEL=16 CONTEXT=65536 TF_FLASHNEXT_PRODUCT_STREAMS=16" ${P}_mia16 "$MU; $WV; $C \$O/waves16.jsonl $RUNS/${P}_mu/waves16.jsonl; $C \$O/waves4.jsonl $RUNS/${P}_mu/waves4.jsonl" 1500
has pw && gor "USERS16=1 PREFIX_WAIT=60000" ${P}_pw "$MU; $WV; $H16; $C $RUNS/${P}_mu/waves16.jsonl \$O/waves16.jsonl" 1500
has mu2 && gor "USERS16=1" ${P}_mu2 "$MU" 900
echo "wF9 [$STEPS] done $(date -u +%T)"
