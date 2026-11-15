#!/bin/bash
# sum.sh [grep-pattern] : every table of the night from the files that exist (head). Prints only titles, TOTAL lines and DIFF lines unless FULL=1.
RT=$HOME/q38/kit-D/redteam; M=$RT/matrix; R=$HOME/q38/runs; ON=$R/D_RT_N_on; f() { if [ -n "$FULL" ]; then cat; else grep -E "^==|TOTAL|DIFF|identical|self check"; fi; }
ps() { [ -s $1 ] && { echo "#### pairsum $(basename $(dirname $1))/$(basename $1) ($(wc -l < $1) replies)"; python3 -I $RT/rt.py pairsum $1 $2 ${1%.jsonl}.diff | f; }; }
cm() { [ -s $2 ] && [ -s $1 ] && { echo "#### cmp $(basename $(dirname $2))/$(basename $2) vs $(basename $1) label ${4:-any} ($(wc -l < $2) replies)"; python3 -I $RT/rt.py cmp $1 $2 ${2%.jsonl}.diff $3 $4 | f; }; }
ps $ON/pair1.jsonl $M/cases1.jsonl; ps $ON/sess.jsonl $ON/sess.jsonl.cases; ps $ON/pair3.jsonl $M/cases3.jsonl; ps $ON/pairL.jsonl $M/cases1.jsonl; ps $ON/pairO.jsonl $M/cases3.jsonl
E=$R/D_RT_N_off; cm $ON/sess.jsonl $E/cold.jsonl $ON/sess.jsonl.cases off; cm $ON/pair1.jsonl $E/env1.jsonl $M/cases1.jsonl off; cm $ON/pair3.jsonl $E/env3.jsonl $M/cases3.jsonl off; cm $ON/pairL.jsonl $E/envL.jsonl $M/cases1.jsonl off; cm $ON/pairO.jsonl $E/envO.jsonl $M/cases3.jsonl off
cm $R/D_RT_A_off/ref.jsonl $E/env1.jsonl $M/cases1.jsonl
B=$R/D_RT_N_on2; ps $B/pair2.jsonl $M/cases2.jsonl; ps $B/pairS3.jsonl $M/cases3.jsonl; ps $B/pairS1.jsonl $M/cases1.jsonl; ps $B/pairC.jsonl $M/cases3.jsonl; cm $ON/pair3.jsonl $B/c2.jsonl $M/cases3.jsonl off; cm $ON/pair1.jsonl $B/c3.jsonl $M/cases1.jsonl off
C=$R/D_RT_N_chk; cm $ON/pair1.jsonl $C/k1.jsonl $M/cases1.jsonl off; cm $ON/sess.jsonl $C/ks.jsonl $ON/sess.jsonl.cases off; cm $ON/pair3.jsonl $C/k3.jsonl $M/cases3.jsonl off; cm $ON/pairL.jsonl $C/kL.jsonl $M/cases1.jsonl off
U=$R/D_RT_N_u16; cm $ON/pair1.jsonl $U/u16.jsonl $M/cases1.jsonl off; cm $ON/pair3.jsonl $U/u8s.jsonl $M/cases3.jsonl off; cm $ON/pair3.jsonl $U/u4.jsonl $M/cases3.jsonl off; cm $ON/sess.jsonl $U/u3.jsonl $ON/sess.jsonl.cases off
cm $ON/pair3.jsonl $R/D_RT_N_u16d/u16d.jsonl $M/cases3.jsonl off; cm $ON/sess.jsonl $R/D_RT_N_ag/ag.jsonl $ON/sess.jsonl.cases off
