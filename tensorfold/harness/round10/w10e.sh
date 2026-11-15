#!/bin/bash
# w10e.sh (923b, builder A, round 10): image 0101-0122 (recipe.sh prepare), then on the documented path, paired, reference first and last:
# E_ref1 (COPY_ROWS=32 = before B's patch setting) -> E_win (recipe default: 128-row copy windows) -> E_half0 (default + HALF=0: builder D's audit
# says HALF is slightly negative for a sampled user) -> E_win2 (default) -> E_ref2 (COPY_ROWS=32).
. $HOME/q38/kit-A/libA.sh; . $HOME/q38/kit-A9/lib9.sh
while pgrep -f 'kit-A10/w10b.sh' >/dev/null; do sleep 3; done
( cd ~/q38/repo_tf && WORK=$HOME/q38/recipe_work HF_CACHE=$HOME/q38/hf bash ./recipe.sh prepare > ~/q38/recipe_work/prepare10e.log 2>&1; echo "prepare rc $? $(date -u +%T) $(ls ~/q38/recipe_work/miaai-recipe/patches | tail -1)" )
SQ="SQ_QK=$KA/qk.py python3 -I $K9/sq.py"
S="$Q ttft | tail -1; $SQ run \$O/sq07off.jsonl s07off 40 256 | tail -1; QK_HELD_STRIDE=3 $Q held \$O/held.jsonl | tail -1; $SETS; $Q ident \$O/ident.json | tail -1"
CM="$SQ cmp $RUNS/E_ref1/sq07off.jsonl \$O/sq07off.jsonl | tail -1; for f in held.jsonl edit.jsonl agent.jsonl ident.json; do $C $RUNS/E_ref1/\$f \$O/\$f; done"
EL="docker logs q38-run-A 2>&1 | grep -E 'streams at once' | cut -c1-200 | tail -1"
gor "COPY_ROWS=32" E_ref1 "$S; $C $RUNS/L_all/held.jsonl \$O/held.jsonl; $C $RUNS/L_all/ident.json \$O/ident.json; $EL" 1300
gor "" E_win "$S; $CM; $EL" 1300
gor "HALF=0" E_half0 "$S; $CM; $SQ cmp $RUNS/E_win/sq07off.jsonl \$O/sq07off.jsonl | tail -1" 1300
gor "" E_win2 "$S; $CM; $SQ cmp $RUNS/E_win/sq07off.jsonl \$O/sq07off.jsonl | tail -1" 1300
gor "COPY_ROWS=32" E_ref2 "$S; $CM" 1300
gor "USERS16=1 COPY_ROWS=128" E_u128 "$EL; $Q distinct 16 | tail -1" 400
echo "w10e done $(date -u +%T)"; touch $RUNS/A_chain_done
