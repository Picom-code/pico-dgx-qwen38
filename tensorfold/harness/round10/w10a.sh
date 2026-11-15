#!/bin/bash
# w10a.sh "STEPS" (923b, builder A, round 10): round-9 leftovers on the DOCUMENTED path (image 0101-0119), paired, reference first and last.
# one user: L_ref1 (R10=0) -> L_hs (HEAD2_SAMPLED=1) -> L_hsd (HEAD2_SAMPLED=1 DEEP_MAX=15) -> L_ref2
# 16 users: U_ref1 (USERS16=1 CERT_TOPK=0) -> U_on (USERS16=1 CERT_TOPK=20, row limit 8) -> U_ref2
. $HOME/q38/kit-A/libA.sh; . $HOME/q38/kit-A9/lib9.sh; STEPS=${1:-"one users"}
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
SQ="SQ_QK=$KA/qk.py python3 -I $K9/sq.py"
SAMP="$SQ run \$O/sq07off.jsonl s07off 40 256 | tail -1; $SQ run \$O/sq07on.jsonl s07on 24 256 | tail -1; $SQ run \$O/sqcode.jsonl s07off 24 256 code | tail -1"
GR="QK_HELD_STRIDE=3 $Q held \$O/held.jsonl | tail -4; $Q ident \$O/ident.json | tail -1"
CMS="for f in sq07off sq07on sqcode; do $SQ cmp $RUNS/L_ref1/\$f.jsonl \$O/\$f.jsonl | tail -2; done"
CMG="$C $RUNS/L_ref1/held.jsonl \$O/held.jsonl; $C $RUNS/L_ref1/ident.json \$O/ident.json"
EL="docker logs q38-run-A 2>&1 | grep -E 'q38 cert|q38 head2|streams at once' | cut -c1-200 | awk '!s[substr(\$0,1,18)]++ || /calls/' | tail -6"
if has one; then
gor "R10=0" L_ref1 "$SAMP; $GR; $C $RUNS/P_ref1/held.jsonl \$O/held.jsonl; $C $RUNS/P_ref1/ident.json \$O/ident.json" 1300
gor "R10=0 HEAD2_SAMPLED=1" L_hs "$SAMP; $CMS; $EL" 800
gor "R10=0 HEAD2_SAMPLED=1 DEEP_MAX=15" L_hsd "$SAMP; $GR; $CMS; $CMG; $EL" 1300
gor "R10=0" L_ref2 "$SAMP; $GR; $CMS; $CMG" 1300
fi
HS="for i in a b; do QK_EXTRA='{\"temperature\":0.7,\"seed\":4242}' $HP 16 \$O/helds16\$i.jsonl 256 3 | tail -1; done; QK_EXTRA='{\"temperature\":0.7,\"seed\":4242}' $HP 4 \$O/helds4.jsonl 256 3 | tail -1; QK_EXTRA='{\"temperature\":0.7,\"seed\":4242}' $HP 2 \$O/helds2.jsonl 256 6 | tail -1; $Q distinct 16 | tail -1"
CMU="for f in helds16a helds16b helds4 helds2; do $C $RUNS/U_ref1/\$f.jsonl \$O/\$f.jsonl; done"
if has users; then
gor "R10=0 USERS16=1 CERT_TOPK=0" U_ref1 "$HS; $C \$O/helds16a.jsonl \$O/helds16b.jsonl" 900
gor "R10=0 USERS16=1 CERT_TOPK=20" U_on "$HS; $CMU; $EL" 900
gor "R10=0 USERS16=1 CERT_TOPK=0" U_ref2 "$HS; $CMU" 900
fi
echo "w10a [$STEPS] done $(date -u +%T)"
