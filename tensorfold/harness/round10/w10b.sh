#!/bin/bash
# w10b.sh "STEPS" (923b, builder A, round 10): DOCUMENTED path, image 0101-0121. Paired, reference first and last.
# chk  : device check modes on the fixed image with the round-10 select forms (long prompts) + identity
# one  : L_ref1 (R10=0) -> L_all (R10=1) -> L_d12 (R10=1 DEEP_MAX=12, greedy only) -> L_s100 (R10=1 QSA_SLACK=100, prefill only) -> L_ref2 (R10=0)
# users: U_ref1 (R10=0 USERS16=1: CERT_TOPK=0) -> U_on (R10=1 USERS16=1: CERT_TOPK=20, row limit 8) -> U_ref2
. $HOME/q38/kit-A/libA.sh; . $HOME/q38/kit-A9/lib9.sh; STEPS=${1:-"chk one users"}
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
SQ="SQ_QK=$KA/qk.py python3 -I $K9/sq.py"
SAMP="$SQ run \$O/sq07off.jsonl s07off 40 256 | tail -1; $SQ run \$O/sq07on.jsonl s07on 24 256 | tail -1; $SQ run \$O/sqcode.jsonl s07off 24 256 code | tail -1"
GR="QK_HELD_STRIDE=3 $Q held \$O/held.jsonl | tail -4; $Q ident \$O/ident.json | tail -1"
PF="$Q ttft | tail -1; $Q prefill 8000,32000,128000 | tail -3"
CMS="for f in sq07off sq07on sqcode; do $SQ cmp $RUNS/L_ref1/\$f.jsonl \$O/\$f.jsonl | tail -2; done"
CMG="$C $RUNS/L_ref1/held.jsonl \$O/held.jsonl; $C $RUNS/L_ref1/ident.json \$O/ident.json"
EL="docker logs q38-run-A 2>&1 | grep -E 'q38 cert|q38 head2|streams at once|q38 qsa knobs|q38 warm|q38 arena' | cut -c1-200 | awk '!s[substr(\$0,1,18)]++ || /calls/' | tail -8"
has chk && gor "R10=1 CERT=2 QSA_CERT=2 TF_FLASHNEXT_Q38_QSA_STATS=1" L_chk "$Q prefill 32000,128000 | tail -2; $Q ctx 250000 | tail -1; $SQ run \$O/sq07off.jsonl s07off 12 256 | tail -1; $Q ident \$O/ident.json | tail -1; $C $RUNS/P_ref1/ident.json \$O/ident.json; docker logs q38-run-A 2>&1 | grep -E 'q38 qsacert|q38 cert|q38 qsa knobs' | tail -7 | cut -c1-420" 900
if has one; then
gor "R10=0" L_ref1 "$PF; $SAMP; $GR; $C $RUNS/P_ref1/held.jsonl \$O/held.jsonl; $C $RUNS/P_ref1/ident.json \$O/ident.json; $EL" 1500
gor "R10=1" L_all "$PF; $SAMP; $GR; $CMS; $CMG; $EL" 1500
gor "R10=1 DEEP_MAX=12" L_d12 "$GR; $CMG" 800
gor "R10=1 QSA_SLACK=100" L_s100 "$PF; $EL" 400
gor "R10=0" L_ref2 "$PF; $SAMP; $GR; $CMS; $CMG" 1500
fi
HS="for i in a b; do QK_EXTRA='{\"temperature\":0.7,\"seed\":4242}' $HP 16 \$O/helds16\$i.jsonl 256 3 | tail -1; done; QK_EXTRA='{\"temperature\":0.7,\"seed\":4242}' $HP 4 \$O/helds4.jsonl 256 3 | tail -1; QK_EXTRA='{\"temperature\":0.7,\"seed\":4242}' $HP 2 \$O/helds2.jsonl 256 6 | tail -1; $Q distinct 16 | tail -1; $Q distinct 16 | tail -1"
CMU="for f in helds16a helds16b helds4 helds2; do $C $RUNS/U_ref1/\$f.jsonl \$O/\$f.jsonl; done"
if has users; then
gor "R10=0 USERS16=1" U_ref1 "$HS; $C \$O/helds16a.jsonl \$O/helds16b.jsonl" 900
gor "R10=1 USERS16=1" U_on "$HS; $CMU; $EL" 900
gor "R10=0 USERS16=1" U_ref2 "$HS; $CMU" 900
fi
echo "w10b [$STEPS] done $(date -u +%T)"
