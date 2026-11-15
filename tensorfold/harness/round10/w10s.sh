#!/bin/bash
# w10s.sh (923b): smoke of the final image 0101-0123: USERS16 start line with the 16 x 48K pool, 16 users, identity set against round-9 texts.
. $HOME/q38/kit-A/libA.sh; . $HOME/q38/kit-A9/lib9.sh
gor "USERS16=1" S_u16 "docker logs q38-run-A 2>&1 | grep -E 'streams at once|q38 cert' | cut -c1-260 | head -3; $Q distinct 16 | tail -1; $Q ident \$O/ident.json | tail -1; $C $RUNS/P_ref1/ident.json \$O/ident.json; QK_EXTRA='{\"temperature\":0.7,\"seed\":4242}' $HP 16 \$O/helds16a.jsonl 256 3 | tail -1; $C $RUNS/U_ref1/helds16a.jsonl \$O/helds16a.jsonl" 600
echo "w10s done $(date -u +%T)"
