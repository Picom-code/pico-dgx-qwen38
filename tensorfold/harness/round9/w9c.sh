#!/bin/bash
# w9c.sh (923b, builder A, round 9): image with patches 0101-0118 (recipe.sh prepare), then the round-9 levers of B, C, D TOGETHER, paired on the
# documented path: Y_ref (R9=0: round-8 defaults + HEAD2 r9) -> Y_all (R9=1) -> Y_ref2 (R9=0). held/3 + the fresh set; texts against P_ref1.
. $HOME/q38/kit-A/libA.sh; . $HOME/q38/kit-A9/lib9.sh
while pgrep -f 'kit-A9/w9b.sh' >/dev/null; do sleep 3; done
( cd ~/q38/repo_tf && WORK=$HOME/q38/recipe_work HF_CACHE=$HOME/q38/hf bash ./recipe.sh prepare > ~/q38/recipe_work/prepare9.log 2>&1; echo "prepare rc $? $(date -u +%T)"; ls ~/q38/recipe_work/miaai-recipe/patches | tail -1; tail -2 ~/q38/recipe_work/prepare9.log | cut -c1-200; docker images --format "{{.Repository}}:{{.Tag}} {{.CreatedSince}}" | grep tensorfold-qwen38 )
S="QK_HELD_STRIDE=3 $Q held \$O/held.jsonl | tail -4; QK_HELD=$K9/corpus/fresh.jsonl $Q held \$O/fresh.jsonl | tail -7"
CM="for f in held.jsonl fresh.jsonl; do $C $RUNS/P_ref1/\$f \$O/\$f; done"
EL="grep -E 'q38 head2|q38 elastic|q38 cert|q38 qsa|streams at once|q38 deep|q38 stop' \$O/../Y_tmp.log | cut -c1-200 | head -12"
gor "R9=0" Y_ref "$S; $CM" 1500
gor "R9=1" Y_all "$Q ttft | tail -1; $S; $CM; docker logs q38-run-A > $RUNS/Y_tmp.log 2>&1; $EL" 1500
gor "R9=0" Y_ref2 "$S; $CM" 1500
echo "w9c done $(date -u +%T)"
