#!/bin/bash
# wG.sh (923b, builder A, second window): documented path with patches 0101-0115: recipe.sh head2 (file from the checkpoint + shipped rotation), recipe.sh prepare,
# then one user recipe default (HEAD2 off) -> HEAD2=1 -> default again: first token, 3-prompt x9, held 104, ident; texts.
. $HOME/q38/kit-A/libA.sh
gor() { [ $(left) -ge ${4:-900} ] || { echo "$2 SKIPPED: $(left) s left"; return 1; }
  export O=$RUNS/$2; mkdir -p $O
  bash $KA/rsA.sh $1 2>&1 | tail -1 | cut -c1-300
  if curl -sf -m 5 http://127.0.0.1:18301/v1/models >/dev/null; then eval "$3"; else echo "$2 ENGINE NOT UP"; fi
  docker logs q38-run-A > $O/engine.log 2>&1; echo "crash lines: $(grep -ciE 'panic|segfault' $O/engine.log)"; grep 'q38 head2' $O/engine.log | head -1 | cut -c1-120
  bash $KA/eng.sh stop >/dev/null; echo "$2 done $(date -u +%T)"; }
( cd ~/q38/repo_tf && rm -f ~/q38/hf/q38/head2_r512.bin; WORK=$HOME/q38/recipe_work HF_CACHE=$HOME/q38/hf bash ./recipe.sh head2 2>&1 | tail -3 | cut -c1-200
  WORK=$HOME/q38/recipe_work HF_CACHE=$HOME/q38/hf bash ./recipe.sh prepare > ~/q38/recipe_work/prepareB.log 2>&1; echo "prepare rc $?"; ls ~/q38/recipe_work/miaai-recipe/patches | tail -1; tail -1 ~/q38/recipe_work/prepareB.log | cut -c1-120 )
S="$Q ttft | tail -1; $Q c1 9 | tail -2; QK_HELD_STRIDE=3 $Q held \$O/held.jsonl | tail -4; $Q ident \$O/ident.json | tail -1"
gor "" G_ref "$S; $C $RUNS/F_on/ident_on.json \$O/ident.json; $C $RUNS/F_on/held.jsonl \$O/held.jsonl" 1500
gor "HEAD2=1" G_h2 "$S; $C $RUNS/G_ref/ident.json \$O/ident.json; $C $RUNS/G_ref/held.jsonl \$O/held.jsonl; $Q img | tail -1; $Q distinct 4 | tail -1" 1500
gor "" G_ref2 "$Q ttft | tail -1; $Q c1 9 | tail -2; QK_HELD_STRIDE=3 $Q held \$O/held.jsonl | tail -4; $C $RUNS/G_ref/held.jsonl \$O/held.jsonl" 1200
echo "wG done $(date -u +%T)"
