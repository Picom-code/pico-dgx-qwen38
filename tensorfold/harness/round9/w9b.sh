#!/bin/bash
# w9b.sh (923b, builder A, round 9): the two-stage draft head with the round-9 rotation, PAIRED on the documented path (image 0101-0115):
# reference (HEAD2=0) -> HEAD2=1 new rotation r9 -> reference -> HEAD2=1 old rotation r512 (if time). Each: first token, 3-prompt x9, held/3
# (now OUTSIDE the fit), the fresh set (never fitted on), identity set, edit set, agent set; texts against the first reference.
. $HOME/q38/kit-A/libA.sh; . $HOME/q38/kit-A9/lib9.sh
( cd ~/q38/repo_tf && WORK=$HOME/q38/recipe_work HF_CACHE=$HOME/q38/hf bash ./recipe.sh head2 2>&1 | tail -2 | cut -c1-220; sha256sum ~/q38/hf/q38/head2_*.bin ~/q38/head2r9/head2_r9.bin | cut -c1-90 )
S="$Q ttft | tail -1; $Q c1 9 | tail -2; QK_HELD_STRIDE=3 $Q held \$O/held.jsonl | tail -4; QK_HELD=$K9/corpus/fresh.jsonl $Q held \$O/fresh.jsonl | tail -7; $Q ident \$O/ident.json | tail -1; $SETS"
CM="for f in ident.json held.jsonl fresh.jsonl edit.jsonl agent.jsonl; do $C $RUNS/P_ref1/\$f \$O/\$f; done"
gor "HEAD2=0" P_ref1 "$S; $C $RUNS/G_ref/ident.json \$O/ident.json; $C $RUNS/G_ref/held.jsonl \$O/held.jsonl" 2700
gor "HEAD2=1" P_new "$S; $CM" 2700
gor "HEAD2=0" P_ref2 "$S; $CM" 2700
gor "HEAD2=1 HEAD2_ROT=r512" P_old "$S; $CM" 2700
echo "w9b done $(date -u +%T)"
