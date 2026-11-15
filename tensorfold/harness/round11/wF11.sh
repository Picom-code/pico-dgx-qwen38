#!/bin/bash
# wF11.sh "STEPS" (923b, builder A, round 11): FINAL ONE-USER suite on the DOCUMENTED path (recipe.sh start on the image recipe.sh prepare built,
# patches 0101-0129), bf16 KV, default profile (3 x 262,144). STEPS in order: smoke off on off2 r10 r11p chk. Labels ${P}_*.
#   off / off2 = every lever of ours off, bf16 KV (first AND last); on = recipe defaults; r10 / r11p = round-10 defaults then recipe defaults again (prefill pair);
#   chk = every device check mode at once. Every engine start checks min(lease deadline, window start + 270 min).
. $HOME/q38/kit-A/libA.sh; . $HOME/q38/kit-A9/lib9.sh; . $HOME/q38/kit-A11/lib11.sh; STEPS=${1:-"smoke off on off2 r10 r11p chk"}; P=${P:-F11}
PF="$Q prefill 8000,32000,128000 | tail -3; $Q ctx 250000 | tail -1"
waitclear() { for i in $(seq 72); do pgrep -f 'start[.]sh restart' >/dev/null || return 0; sleep 5; done; echo "another start.sh still running after 6 min"; }
need() { [ -f $RUNS/${P}_$1/suite.log ] || { echo "$1 DID NOT RUN: chain stops $(date -u +%T)"; touch $RUNS/A_chain_done; exit 3; }; }
for s in $STEPS; do waitclear; case $s in
smoke) gor "" ${P}_smoke "$Q ttft | tail -1; $Q ident \$O/ident_on.json | tail -1; $C $RUNS/F10_kbf/ident.json \$O/ident_on.json | tail -1; $Q prefill 8000 | tail -1; $RT; $CL; $MEM" 600
  grep -q . $RUNS/${P}_smoke/ident_on.json 2>/dev/null || { echo "SMOKE FAILED: chain stops"; touch $RUNS/A_chain_done; exit 3; } ;;
off) gor "$OFF" ${P}_off "MAXU=3 HELD_STRIDE=1 SKIP=multi bash ~/q38/kit/suite.sh ${P}_off full | tail -2; $SETS; $SAMP; $PRED; $Q distinct 3 | tail -1; $MEM" 5400; need off
  echo "-- off vs previous image's levers-off (F10_kbfoff, bf16)"; for f in ident held edit agent; do x=$f.jsonl; [ $f = ident ] && x=ident.json; y=$x; [ $f = ident ] && y=ident_on.json; echo -n "$f: "; $C $RUNS/F10_kbfoff/$x $RUNS/${P}_off/$y | tail -1; done ;;
on) gor "" ${P}_on "MAXU=3 HELD_STRIDE=1 SKIP=multi bash ~/q38/kit/suite.sh ${P}_on full | tail -2; $SETS; $SAMP; $PRED; $RT; $Q distinct 3 | tail -1; $CL; $MEM" 5400; need on
  echo "-- on vs off (same window)"; cmpall ${P}_off ${P}_on
  echo "-- on vs previous image's default (F10_kbf, bf16)"; for f in ident held edit agent; do x=$f.jsonl; [ $f = ident ] && x=ident.json; y=$x; [ $f = ident ] && y=ident_on.json; echo -n "$f: "; $C $RUNS/F10_kbf/$x $RUNS/${P}_on/$y | tail -1; done
  echo -n "sq07off vs F10_kbf: "; eval "$SQ cmp $RUNS/F10_kbf/sq07off.jsonl $RUNS/${P}_on/sq07off.jsonl" | tail -1 ;;
off2) gor "$OFF" ${P}_off2 "MAXU=3 HELD_STRIDE=1 SKIP='multi prefill qeval img needle ctx' bash ~/q38/kit/suite.sh ${P}_off2 full | tail -2; $SETS; $SAMP; $MEM" 3300
  echo "-- off2 vs off"; cmpall ${P}_off ${P}_off2 ;;
r10) gor "R11=0" ${P}_r10 "$Q ttft | tail -1; $PF" 900 ;;
r11p) gor "" ${P}_r11p "$Q ttft | tail -1; $PF" 900 ;;
chk) gor "CERT=2 QSA_CERT=2 TF_FLASHNEXT_Q38_QSA_STATS=1 GDN_PROD=12 TF_FLASHNEXT_Q38_ROUTE_BAND_CHECK=1 TF_FLASHNEXT_Q38_SCRATCH_CHECK=1 TF_FLASHNEXT_Q38_HC_DN_TALL_CHECK=1 SNAP_DELTA=2 TF_FLASHNEXT_Q38_PLOG=1" ${P}_chk "$Q prefill 32000,128000 | tail -2; $Q ctx 250000 | tail -1; $Q sd 1 | tail -5; $SQ run \$O/sq07off.jsonl s07off 12 256 | tail -1; $Q ident \$O/ident_on.json | tail -1; $C $RUNS/${P}_on/ident_on.json \$O/ident_on.json | tail -1; $AT run \$O/at2.json 2 4 20000 0 300 2000 | tail -8; docker logs q38-run-A 2>&1 > \$O/engine_chk.log; for k in 'q38 qsacert' 'q38 cert:' 'q38 cert topk' 'q38 cert gather' 'q38 gdnprod' 'q38 routeband' 'q38 scratch check' 'q38 hcdn' 'q38 snapdelta keep' 'q38 qsa knobs'; do grep \"\$k\" \$O/engine_chk.log | tail -2 | cut -c1-520; done" 900 ;;
esac; done
echo "wF11 [$STEPS] done $(date -u +%T)"; touch $RUNS/A_chain_done
