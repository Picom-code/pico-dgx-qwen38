#!/bin/bash
# wF10.sh "STEPS" (923b, builder A, round 10): FINAL suite on the DOCUMENTED path (recipe.sh prepare, recipe.sh start on that image, vision on).
# STEPS in the order given below: prep chk off on fly u8 off2 mu mia mu2. Labels ${P}_*, P default F10. Every step checks ~/q38/runs/A_deadline.
#   off / off2 = every lever of ours off (first AND last), on = recipe defaults, fly = custom engine R custom engine R v0.3.13 on turboderp's EXL3 4.05 bpw
#   checkpoint (RE-QUANTISED weights: not the official checkpoint; same prompts, same window), u8 = ours with USERS16 at 1 / 4 / 8 users.
. $HOME/q38/kit-A/libA.sh; . $HOME/q38/kit-A9/lib9.sh; STEPS=${1:-"off on fly u8 off2"}; P=${P:-F10}
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
OFF="$OFF R10=0"
CL="docker logs q38-run-A 2>&1 | grep -E 'q38 cert|q38 hcf|q38 head2|q38 elastic|q38 qsa knobs|q38 warm|streams at once' | cut -c1-230 | awk '!s[substr(\$0,1,18)]++ || /calls/' | tail -12"
SQ="SQ_QK=$KA/qk.py python3 -I $K9/sq.py"
SAMP="$SQ run \$O/sq07off.jsonl s07off 24 256 | tail -1; $SQ run \$O/sq07on.jsonl s07on 24 256 | tail -1; $SQ sd \$O/sqsd.jsonl 6 | tail -1"
U148="for n in 1 4 8; do $Q distinct \$n | tail -1; done"
if has prep; then ( cd ~/q38/repo_tf && WORK=$HOME/q38/recipe_work HF_CACHE=$HOME/q38/hf bash ./recipe.sh prepare > ~/q38/recipe_work/prepare10f.log 2>&1; echo "prepare rc $?"; ls ~/q38/recipe_work/miaai-recipe/patches | tail -1; tail -2 ~/q38/recipe_work/prepare10f.log | cut -c1-200; docker images --format "{{.Repository}}:{{.Tag}} {{.CreatedSince}}" | grep tensorfold-qwen38; sha256sum ~/q38/hf/q38/head2_*.bin | cut -c1-90 ); fi
for s in $STEPS; do case $s in
chk) gor "CERT=2 QSA_CERT=2 TF_FLASHNEXT_Q38_QSA_STATS=1" ${P}_chk "$Q prefill 32000,128000 | tail -2; $Q ctx 250000 | tail -1; $Q sd 1 | tail -5; $SQ run \$O/sq07off.jsonl s07off 12 256 | tail -1; $Q ident \$O/ident.json | tail -1; $C $RUNS/P_ref1/ident.json \$O/ident.json; docker logs q38-run-A 2>&1 | grep -E 'q38 qsacert|q38 cert|q38 qsa knobs' | tail -6 | cut -c1-420" 900 ;;
off) gor "$OFF" ${P}_off "MAXU=5 HELD_STRIDE=1 SKIP=multi bash ~/q38/kit/suite.sh ${P}_off full | tail -2; $SETS; $SAMP; $Q distinct 4 | tail -1" 5400 ;;
on) gor "" ${P}_on "MAXU=5 HELD_STRIDE=1 SKIP=multi bash ~/q38/kit/suite.sh ${P}_on full | tail -2; $SETS; $SAMP; $Q distinct 4 | tail -1; $CL" 5400
    for f in ident_on.json held.jsonl edit.jsonl agent.jsonl; do $C $RUNS/${P}_off/$f $RUNS/${P}_on/$f; done; for f in sq07off sq07on sqsd; do $SQ cmp $RUNS/${P}_off/$f.jsonl $RUNS/${P}_on/$f.jsonl | tail -3; done ;;
off2) gor "$OFF" ${P}_off2 "MAXU=5 HELD_STRIDE=1 SKIP='multi prefill qeval img needle ctx' bash ~/q38/kit/suite.sh ${P}_off2 full | tail -2; $SETS; $SQ run \$O/sq07off.jsonl s07off 24 256 | tail -1" 3600
    for f in ident_on.json held.jsonl edit.jsonl agent.jsonl; do $C $RUNS/${P}_off/$f $RUNS/${P}_off2/$f; done ;;
fly) if [ $(left) -ge 3600 ]; then export O=$RUNS/${P}_fly; mkdir -p $O; bash $KA/eng.sh fly default 2>&1 | tail -1 | cut -c1-200
      if curl -sf -m 5 http://127.0.0.1:18301/v1/models >/dev/null; then MAXU=8 HELD_STRIDE=1 SKIP='ident multi prefill qeval img needle ctx' bash ~/q38/kit/suite.sh ${P}_fly full | tail -2; eval "$U148"; else echo "fly ENGINE NOT UP"; fi
      docker logs q38-run-fly > $O/engine.log 2>&1; bash $KA/eng.sh stop >/dev/null; echo "${P}_fly done $(date -u +%T)"; else echo "fly SKIPPED: $(left) s left"; fi ;;
g0) gor "CERT_GATHER=0" ${P}_g0 "$SAMP" 900; for f in sq07off sq07on sqsd; do $SQ cmp $RUNS/${P}_g0/$f.jsonl $RUNS/${P}_on/$f.jsonl | tail -1; done ;;
u8) gor "USERS16=1" ${P}_u8 "$U148" 700 ;;
mu) gor "USERS16=1" ${P}_mu "$MU; $WV; python3 -I $KA/poolt.py 150000x1+15 | cut -c1-420; $HP 16 \$O/held16.jsonl 256 3 | tail -1; $C $RUNS/A_ref1/held16.jsonl \$O/held16.jsonl; QK_EXTRA='{\"temperature\":0.7,\"seed\":4242}' $HP 16 \$O/helds16.jsonl 256 3 | tail -1; $CL" 1800 ;;
mia) gor "$OFF PARALLEL=16 CONTEXT=65536 TF_FLASHNEXT_PRODUCT_STREAMS=16" ${P}_mia16 "$MU; $WV; $C \$O/waves16.jsonl $RUNS/${P}_mu/waves16.jsonl; $C \$O/waves4.jsonl $RUNS/${P}_mu/waves4.jsonl; $HP 16 \$O/held16.jsonl 256 3 | tail -1; $C $RUNS/A_ref1/held16.jsonl \$O/held16.jsonl" 1500 ;;
mu2) gor "USERS16=1" ${P}_mu2 "$MU; $HP 16 \$O/held16.jsonl 256 3 | tail -1" 900 ;;
esac; done
echo "wF10 [$STEPS] done $(date -u +%T)"; touch $RUNS/A_chain_done
