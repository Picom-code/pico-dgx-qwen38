#!/bin/bash
# w10h.sh (923b, builder A, round 10): builder D's audit says HALF (small projections on 32-column tiles) is slightly negative for one sampled user.
# Own pair on the documented path: H_ref1 (recipe default) -> H_half0 (HALF=0) -> H_ref2. Seeded sampled chat 40 + greedy held/3 + first token.
. $HOME/q38/kit-A/libA.sh; . $HOME/q38/kit-A9/lib9.sh
SQ="SQ_QK=$KA/qk.py python3 -I $K9/sq.py"
S="$Q ttft | tail -1; $SQ run \$O/sq07off.jsonl s07off 40 256 | tail -1; QK_HELD_STRIDE=3 $Q held \$O/held.jsonl | tail -1"
CM="$SQ cmp $RUNS/H_ref1/sq07off.jsonl \$O/sq07off.jsonl | tail -1; $C $RUNS/H_ref1/held.jsonl \$O/held.jsonl; python3 -I $HOME/q38/kit-A10/hcmp.py $RUNS/H_ref1/held.jsonl \$O/held.jsonl"
gor "" H_ref1 "$S; $C $RUNS/L_all/held.jsonl \$O/held.jsonl" 1000
gor "HALF=0" H_half0 "$S; $CM" 1000
gor "" H_ref2 "$S; $CM" 1000
echo "w10h done $(date -u +%T)"; touch $RUNS/A_chain_done
