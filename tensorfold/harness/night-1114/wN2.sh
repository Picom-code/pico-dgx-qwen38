#!/bin/bash
# wN.sh "STEPS" (spark-923b, agent A, night 2026-11-14/15): the FINAL suites on the DOCUMENTED path (recipe.sh start on the image that
# recipe.sh prepare built = patches 0101-0129), unattended. Labels N_*. Every engine start goes through gor(), which refuses to start
# unless left() (libN.sh: lease, window start + 270 min, 12:02 UTC cutoff, fresh keeper stamp) leaves the step's time.
#  window 1: prep u16s ags off on off2 chk          window 2: miafp mu miafp2 agA offs agmia [chk]
#   prep   = recipe.sh prepare (idempotent check: image id before / after)
#   u16s / ags = USERS16=1 / AGENTS=1 started ONCE each with the recipe's defaults + a request; if a start fails, the next smaller size
#            is tried and the working arguments are written to $RUNS/N_u16_args / N_ag_args (used by mu / agA)
#   off / off2 = every lever of ours off (bf16 KV, 3 x 262K), first and last; on = `recipe.sh start` with NO options
#   chk    = every device check mode at once (sample)
#   miafp / miafp2 = MiaAI-Lab's shipped shape with our levers off: fp8 KV, 16 fixed slots x 65,536 (first and last)
#   mu     = USERS16=1 (bf16 KV pool); agA = AGENTS=1 traces; agmia = MiaAI default (fp8, 5 x 262K, levers off) traces
#   offs   = levers off, solo, seeded sampled 256-token held/3 (the reference for "16 sampled in flight")
. $HOME/q38/kit-A/libA.sh; . $HOME/q38/kit-A9/lib9.sh; . $HOME/q38/kit-A11/lib11.sh; . $HOME/q38/kit-N/libN.sh
STEPS=${1:-"prep u16s ags off on off2 chk"}; P=N
PF="$Q prefill 32000,128000 | tail -2"
waitclear() { for i in $(seq 72); do pgrep -f 'start[.]sh restart' >/dev/null || return 0; sleep 5; done; echo "another start.sh still running after 6 min"; }
H16="$HP 16 \$O/held16.jsonl 256 3 | tail -1"; S16="QK_EXTRA='{\"temperature\":0.7,\"seed\":4242}' $HP 16 \$O/helds16.jsonl 256 3 | tail -1"
T5="$AT run \$O/at5.json 5 6 56000 0 300 2000 | tail -12"; T10="$AT run \$O/at10.json 10 3 56000 0 300 2000 | tail -9"
PC="docker logs q38-run-A 2>&1 | grep -E 'streams at once|q38 elastic' | cut -c1-300 | tail -2; docker logs q38-run-A 2>&1 | grep 'prompt cache:' | tail -1 | cut -c1-200"
CHKENV="CERT=2 QSA_CERT=2 TF_FLASHNEXT_Q38_QSA_STATS=1 GDN_PROD=12 TF_FLASHNEXT_Q38_ROUTE_BAND_CHECK=1 TF_FLASHNEXT_Q38_SCRATCH_CHECK=1 TF_FLASHNEXT_Q38_HC_DN_TALL_CHECK=1 SNAP_DELTA=2 TF_FLASHNEXT_Q38_PLOG=1"
FULL="MAXU=3 HELD_STRIDE=1"
for s in $STEPS; do waitclear; echo "### step $s $(date -u +%T) left $(left) s"; case $s in
prep) I0=$(docker images -q pico-dgx-qwen38-tf:v1.1.4-eab8582); T0=$(date +%s)
  ( cd $HOME/q38/repo_tf && env WORK=$HOME/q38/recipe_work HF_CACHE=$HOME/q38/hf bash ./recipe.sh prepare > $RUNS/N_prepare.log 2>&1 ); rc=$?
  I1=$(docker images -q pico-dgx-qwen38-tf:v1.1.4-eab8582); echo "recipe.sh prepare rc $rc in $(( $(date +%s) - T0 )) s; image before $I0 after $I1 $([ "$I0" = "$I1" ] && echo UNCHANGED || echo CHANGED); recipe.sh sha $(sha256sum $HOME/q38/repo_tf/recipe.sh | cut -c1-16); patches $(ls $HOME/q38/repo_tf/patches/01*.patch | wc -l) sha $(cat $HOME/q38/repo_tf/patches/01*.patch | sha256sum | cut -c1-16)"
  tail -4 $RUNS/N_prepare.log | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-200 ;;
u16s) rm -f $RUNS/N_u16_args; for a in "" "POOL_CTX=36864" "POOL_CTX=32768"; do rm -f $RUNS/${P}_u16s/ok
    gor "USERS16=1 $a" ${P}_u16s "touch \$O/ok; $ENVD; $Q ttft | tail -1; $NU 16 \$O/users.jsonl 2 | tail -1; $PC; $MEM" 420
    [ -f $RUNS/${P}_u16s/ok ] && { echo "USERS16=1 $a" > $RUNS/N_u16_args; echo "USERS16 STARTS with [USERS16=1 $a]"; break; }
    echo "USERS16 [$a] DID NOT START:"; sed 's/\x1b\[[0-9;]*m//g' $HOME/q38/logs/recipe_start.log | grep -iE 'needs|budget|error|refus|fail' | tail -4 | cut -c1-300; done ;;
ags) rm -f $RUNS/N_ag_args; for a in "" "PARALLEL=4" "PARALLEL=3"; do rm -f $RUNS/${P}_ags/ok
    gor "AGENTS=1 $a" ${P}_ags "touch \$O/ok; $ENVD; $Q ttft | tail -1; $NU 5 \$O/users.jsonl 1 | tail -1; $AT run \$O/at2.json 2 2 20000 0 300 2000 | tail -5; $PC; $MEM" 420
    [ -f $RUNS/${P}_ags/ok ] && { echo "AGENTS=1 $a" > $RUNS/N_ag_args; echo "AGENTS STARTS with [AGENTS=1 $a]"; break; }
    echo "AGENTS [$a] DID NOT START:"; sed 's/\x1b\[[0-9;]*m//g' $HOME/q38/logs/recipe_start.log | grep -iE 'needs|budget|error|refus|fail' | tail -4 | cut -c1-300; done ;;
off) gor "$OFF" ${P}_off "$ENVD; $FULL SKIP=multi bash ~/q38/kit/suite.sh ${P}_off full | tail -2; $SETS; $SAMP; $PRED; $Q distinct 3 | tail -1; $MEM" 4800
  [ -f $RUNS/${P}_off/suite.log ] || { echo "off DID NOT RUN: chain stops $(date -u +%T)"; touch $RUNS/A_chain_done; exit 3; }
  echo "-- off vs the wrap-up run of the same image (F11_off)"; cmpall F11_off ${P}_off ;;
on) gor "" ${P}_on "$ENVD; $FULL SKIP=multi bash ~/q38/kit/suite.sh ${P}_on full | tail -2; $SETS; $SAMP; $PRED; $RT; $Q distinct 3 | tail -1; $CL; $MEM" 4800
  echo "-- on vs off (same window)"; cmpall ${P}_off ${P}_on ;;
off2) if [ $(left) -ge 4300 ]; then SK="multi qeval img needle"; X="$SETS; $SAMP; $PRED"; NEEDT=4300; else SK="multi qeval img needle ctx sparkdash prefill"; X="$SETS; $SAMP"; NEEDT=2700; echo "off2 SHORT form (no sparkDash-own, prefill, 227K, prediction sets)"; fi
  gor "$OFF" ${P}_off2 "$ENVD; $FULL SKIP='$SK' bash ~/q38/kit/suite.sh ${P}_off2 full | tail -2; $X; $MEM" $NEEDT
  echo "-- off2 vs off"; cmpall ${P}_off ${P}_off2; echo "-- on vs off2"; cmpall ${P}_off2 ${P}_on ;;
chk) gor "$CHKENV" ${P}_chk "$ENVD; $PF; $Q ctx 250000 | tail -1; $Q sd 1 | tail -5; $SQ run \$O/sq07off.jsonl s07off 12 256 | tail -1; $Q ident \$O/ident_on.json | tail -1; $AT run \$O/at2.json 2 4 20000 0 300 2000 | tail -8; AR_PRED=1 timeout 300 $AR set $S11/pred.jsonl \$O/pred1.jsonl held | tail -2; docker logs q38-run-A > \$O/engine_chk.log 2>&1; grep -E 'q38 [a-z0-9 ]*(cert|check|chk|qsa|gdnprod|routeband|scratch|hcdn|snapdelta|delta)' \$O/engine_chk.log | cut -c1-520 | awk '{k=substr(\$0,1,22); l[k]=\$0} END{for(k in l) print l[k]}' | sort | head -40; echo \"mismatch-like lines: \$(grep -ciE 'MISMATCH|VIOLAT|DIFFER' \$O/engine_chk.log)\"" 1200
  for r in on off; do [ -f $RUNS/${P}_$r/ident_on.json ] && { echo -n "chk ident vs ${P}_$r: "; $C $RUNS/${P}_$r/ident_on.json $RUNS/${P}_chk/ident_on.json | tail -1; }; done
  [ -f $RUNS/${P}_on/sq07off.jsonl ] && { echo -n "chk seeded sampled (12) vs on: "; eval "$SQ cmp $RUNS/${P}_on/sq07off.jsonl $RUNS/${P}_chk/sq07off.jsonl" | tail -1; }
  [ -f $RUNS/${P}_on/pred1.jsonl ] && { echo -n "chk rewrite(held) vs on: "; $AR cmp $RUNS/${P}_on/pred1.jsonl $RUNS/${P}_chk/pred1.jsonl | tail -1; } ;;
miafp|miafp2) gor "$OFF KV=fp8 PARALLEL=16 CONTEXT=65536 TF_FLASHNEXT_PRODUCT_STREAMS=16" ${P}_$s "$ENVD; $USERS; $([ $s = miafp ] && echo "$H16; $WV;") $PC; $MEM" $([ $s = miafp ] && echo 900 || echo 480) ;;
mu) A=$(cat $RUNS/N_u16_args 2>/dev/null || echo "USERS16=1")
  gor "$A" ${P}_mu "$ENVD; $USERS; $H16; $HP 16 \$O/held16f.jsonl 384 1 | tail -1; $S16; $WV; $Q ident \$O/ident_on.json | tail -1; python3 -I $KA/poolt.py 150000x1+15 | cut -c1-420; $CL; $PC; $MEM" 1500
  echo -n "ALL 320 held, 16 in flight (USERS16) vs SOLO levers-off (N_off, window 1): "; $C $RUNS/${P}_off/held.jsonl $RUNS/${P}_mu/held16f.jsonl | tail -1
  echo -n "ALL 320 held, 16 in flight (USERS16) vs solo recipe default (N_on): "; $C $RUNS/${P}_on/held.jsonl $RUNS/${P}_mu/held16f.jsonl | tail -1
  echo -n "identity set on USERS16 (one at a time) vs solo levers-off (N_off): "; $C $RUNS/${P}_off/ident_on.json $RUNS/${P}_mu/ident_on.json | tail -1 ;;
agA) A=$(cat $RUNS/N_ag_args 2>/dev/null || echo "AGENTS=1"); gor "$A" ${P}_agA "$ENVD; $T10; $T5; $PC; $MEM" 1200 ;;
agmia) gor "$OFF KV=fp8" ${P}_agmia "$ENVD; $T10; $PC; $MEM" 1500 ;;
agoff) A=$(cat $RUNS/N_ag_args 2>/dev/null || echo "AGENTS=1"); gor "$A $OFF" ${P}_agoff "$ENVD; $T10; $PC; $MEM" 1500
  echo -n "at10 agA vs the same profile with every lever off: "; $AT cmp $RUNS/${P}_agoff/at10.json $RUNS/${P}_agA/at10.json | tail -1 ;;
bench) gor "" ${P}_bench "$ENVD; ( cd $HOME/q38/repo_tf && HOST=127.0.0.1 PORT=18301 timeout 1500 bash ./recipe.sh bench ) > \$O/bench.log 2>&1; echo \"recipe.sh bench rc \$?\"; grep -E 'ttft|1 user|sd-method|held ALL|prefill|distinct|ident saved' \$O/bench.log | cut -c1-150; $MEM" 1300 ;;
offs) gor "$OFF" ${P}_offs "$ENVD; QK_EXTRA='{\"temperature\":0.7,\"seed\":4242}' $HP 1 \$O/helds1.jsonl 256 3 | tail -1; $MEM" 1080
  echo -n "seeded sampled held/3: 16 in flight (USERS16, N_mu) vs SOLO levers-off: "; $C $RUNS/${P}_offs/helds1.jsonl $RUNS/${P}_mu/helds16.jsonl | tail -1 ;;
esac; done
echo "wN [$STEPS] done $(date -u +%T)"; docker rm -f q38-run-A >/dev/null 2>&1; touch $RUNS/A_chain_done
