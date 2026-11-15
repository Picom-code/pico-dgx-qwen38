# lib9.sh (923b, builder A, round 9): documented-path runner. Source after libA.sh.
K9=$HOME/q38/kit-A9; ST="python3 -I $KA/tfsteps.py"
gor() { # "RECIPE ARGS" LABEL CMDS NEED_SECONDS : recipe.sh start on the prepared image (container q38-run-A, 127.0.0.1:18301), run CMDS, stop
  [ $(left) -ge ${4:-900} ] || { echo "$2 SKIPPED: $(left) s left, need ${4:-900}"; return 1; }
  export O=$RUNS/$2; mkdir -p $O
  bash $KA/rsA.sh $1 2>&1 | tail -1 | cut -c1-300
  if curl -sf -m 5 http://127.0.0.1:18301/v1/models >/dev/null; then eval "$3"; else echo "$2 ENGINE NOT UP"; fi
  docker logs q38-run-A > $O/engine.log 2>&1; echo "crash lines: $(grep -ciE 'panic|segfault' $O/engine.log)"; grep 'q38 head2' $O/engine.log | head -1 | cut -c1-120
  bash $KA/eng.sh stop >/dev/null; echo "$2 done $(date -u +%T)"; }
one() { echo "L0=\$(docker logs q38-run-A 2>&1 | wc -l); QK_HELD=$1 $HP ${5:-1} \$O/$2.jsonl $3 ${4:-1} | tail -1; docker logs q38-run-A 2>&1 | tail -n +\$((L0+1)) > \$O/eng_$2.log; echo '== $2'; $ST \$O/eng_$2.log | sed -n '2,3p'"; }
SETS="$(one $KA/sets/edit.jsonl edit 1500); $(one $KA/sets/agent.jsonl agent 900)"
OFF="CERT=0 MANY=0 COPY_ROWS=16 FUSED=0 HALF=0 MTP_LAST=0 GDN_MANY=0 BEST_LANES=0 ATTN_LANES=0 LATE_SKIP=0 KEEP_INPASS=0 LANE_REUSE=0 HEAD2=0 R9=0"
MU="for n in 1 4 8 12 16; do $Q distinct \$n | tail -1; done; $Q distinct 16 | tail -1; $Q distinct 16 | tail -1; $Q copies 16 | tail -1"
WV="$W 16 \$O/waves16.jsonl 256 1 8 | tail -1; $W 4 \$O/waves4.jsonl 128 5 | tail -1"
