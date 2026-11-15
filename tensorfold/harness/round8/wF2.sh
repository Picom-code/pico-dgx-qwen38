#!/bin/bash
# wF.sh [STEPS] (923b, builder A): FINAL suite through the DOCUMENTED path (recipe.sh prepare, then recipe.sh start on that image, vision on).
# STEPS (default all): prep off on cmp mu mia. Every step checks the deadline in ~/q38/runs/A_deadline before it starts an engine.
. $HOME/q38/kit-A/libA.sh; STEPS=${1:-"off onl cmp"}; P=${P:-F}
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
gor() { # "ARGS" LABEL CMDS NEED_SECONDS
  [ $(left) -ge ${4:-900} ] || { echo "$2 SKIPPED: $(left) s left, need ${4:-900}"; return 1; }
  export O=$RUNS/$2; mkdir -p $O
  bash $KA/rsA.sh $1 2>&1 | tail -1 | cut -c1-300
  if curl -sf -m 5 http://127.0.0.1:18301/v1/models >/dev/null; then eval "$3"; else echo "$2 ENGINE NOT UP"; fi
  docker logs q38-run-A > $O/engine.log 2>&1; echo "crash lines: $(grep -ciE 'panic|segfault' $O/engine.log)"
  bash $KA/eng.sh stop >/dev/null; echo "$2 done $(date -u +%T)"; }
ST="python3 -I $KA/tfsteps.py"
one() { echo "L0=\$(docker logs q38-run-A 2>&1 | wc -l); QK_HELD=$1 $HP ${5:-1} \$O/$2.jsonl $3 ${4:-1} | tail -1; docker logs q38-run-A 2>&1 | tail -n +\$((L0+1)) > \$O/eng_$2.log; echo '== $2'; $ST \$O/eng_$2.log | sed -n '2,3p'"; }
SETS="$(one $KA/sets/edit.jsonl edit 1500); $(one $KA/sets/agent.jsonl agent 900)"
MU="for n in 1 4 8 12 16; do $Q distinct \$n | tail -1; done; $Q distinct 16 | tail -1; $Q distinct 16 | tail -1; $Q copies 16 | tail -1"
WV="$W 16 \$O/waves16.jsonl 256 1 8 | tail -1; $W 4 \$O/waves4.jsonl 128 5 | tail -1"
CL="docker logs q38-run-A 2>&1 | grep -E 'q38 cert|q38 hcf' | sed -n '1p;\$p' | cut -c1-230"
OFF="CERT=0 MANY=0 COPY_ROWS=16 FUSED=0 HALF=0 MTP_LAST=0 GDN_MANY=0 BEST_LANES=0 ATTN_LANES=0 LATE_SKIP=0 KEEP_INPASS=0 LANE_REUSE=0"
SU="K=$KA"   # suite.sh reads its kit from ~/q38/kit (same files)
while pgrep -f 'kit-A/wF.sh' >/dev/null; do sleep 5; done
if has prep; then ( cd ~/q38/repo_tf && WORK=$HOME/q38/recipe_work HF_CACHE=$HOME/q38/hf bash ./recipe.sh prepare > ~/q38/recipe_work/prepareA.log 2>&1; echo "prepare rc $?"; ls ~/q38/recipe_work/miaai-recipe/patches | tail -1; tail -1 ~/q38/recipe_work/prepareA.log | cut -c1-200; docker images --format "{{.Repository}}:{{.Tag}} {{.CreatedSince}}" | grep tensorfold-qwen38 ); fi
has off && gor "$OFF" ${P}_off "MAXU=5 HELD_STRIDE=3 SKIP='prefill sparkdash qeval img needle ctx' bash ~/q38/kit/suite.sh ${P}_off full | tail -2; $SETS" 2400
while pgrep -f 'kit-A/wF.sh' >/dev/null; do sleep 5; done
has onl && [ -d $RUNS/${P}_off ] && gor "" ${P}_on "MAXU=5 HELD_STRIDE=3 SKIP='sparkdash' bash ~/q38/kit/suite.sh ${P}_on full | tail -2; $SETS; $CL" 2700
has on && gor "" ${P}_on "MAXU=5 HELD_STRIDE=1 bash ~/q38/kit/suite.sh ${P}_on full | tail -2; $SETS; $CL" 5400
has cmp && for f in ident_on.json held.jsonl edit.jsonl agent.jsonl; do $C $RUNS/${P}_off/$f $RUNS/${P}_on/$f; done
has mu && gor "USERS16=1" ${P}_mu "$MU; $WV; python3 -I $KA/poolt.py 150000x1+15 | cut -c1-420; $Q ctx 250000 | tail -1; $HP 16 \$O/held16.jsonl 256 3 | tail -1; $C $RUNS/A_ref1/held16.jsonl \$O/held16.jsonl" 1500
has mia && gor "$OFF PARALLEL=16 CONTEXT=65536 TF_FLASHNEXT_PRODUCT_STREAMS=16" ${P}_mia16 "$MU; $WV; $C \$O/waves16.jsonl $RUNS/${P}_mu/waves16.jsonl; $C \$O/waves4.jsonl $RUNS/${P}_mu/waves4.jsonl" 1200
echo "wF done $(date -u +%T)"
