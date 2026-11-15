# lib11.sh (923b, builder A, round 11): source after libA.sh + lib9.sh. Window-safe runner pieces.
K11=$HOME/q38/kit-A11; S11=$K11/sets; AR="python3 -I $K11/ar.py"; AT="python3 -I $K11/at.py"
# seconds left = min(lease deadline, window end = OPEN + 270 min); both files are written by the head's keeper (keepA11.sh)
left() { local d w n; d=$(cat $RUNS/A_deadline 2>/dev/null || echo 0); w=$(cat $RUNS/A_winend 2>/dev/null || echo 0); n=$(date +%s); [ $w -lt $d ] && d=$w; echo $(( d - n )); }
OFF="$OFF R10=0 R11=0"
CL="docker logs q38-run-A 2>&1 | grep -E 'q38 cert|q38 hcf|q38 head2|q38 elastic|q38 qsa knobs|q38 warm|q38 scratch|q38 snap|streams at once|prompt cache' | cut -c1-260 | awk '!s[substr(\$0,1,18)]++ || /calls/' | tail -14"
SQ="SQ_QK=$KA/qk.py python3 -I $K9/sq.py"
SAMP="$SQ run \$O/sq07off.jsonl s07off 24 256 | tail -1; $SQ run \$O/sq07on.jsonl s07on 24 256 | tail -1; $SQ sd \$O/sqsd.jsonl 6 | tail -1"
# predictions: the 56 single-turn rewrite requests without / with the `prediction` field, the agent loop (held half) without / with, the three routes
PRED="AR_PRED=0 timeout 900 $AR set $S11/pred.jsonl \$O/pred0.jsonl all | tail -6; AR_PRED=1 timeout 900 $AR set $S11/pred.jsonl \$O/pred1.jsonl all | tail -6; echo -n 'pred1 vs pred0: '; $AR cmp \$O/pred0.jsonl \$O/pred1.jsonl; AR_PRED=0 timeout 900 $AR loop $S11/tasks.json \$O/loop0.jsonl held | tail -3; AR_PRED=1 timeout 900 $AR loop $S11/tasks.json \$O/loop1.jsonl held | tail -3; echo -n 'loop1 vs loop0: '; $AR cmp \$O/loop0.jsonl \$O/loop1.jsonl"
RT="timeout 600 python3 -I $K11/rt11.py $S11/pred.jsonl \$O/rt.json 4 | tail -9"
MEM="awk '/MemAvailable/{print \"MemAvailable GiB \" int(\$2/1048576)}' /proc/meminfo"
cmpall() { # A B : every identity file of run A against run B
  for f in ident_on.json held.jsonl edit.jsonl agent.jsonl; do [ -f $RUNS/$1/$f ] && [ -f $RUNS/$2/$f ] && { echo -n "$f $1 vs $2: "; $C $RUNS/$1/$f $RUNS/$2/$f | tail -1; }; done
  for f in sq07off sq07on sqsd; do [ -f $RUNS/$1/$f.jsonl ] && [ -f $RUNS/$2/$f.jsonl ] && { echo -n "$f: "; eval "$SQ cmp $RUNS/$1/$f.jsonl $RUNS/$2/$f.jsonl" | tail -1; }; done
  for f in pred0 pred1 loop0 loop1; do [ -f $RUNS/$1/$f.jsonl ] && [ -f $RUNS/$2/$f.jsonl ] && { echo -n "$f: "; $AR cmp $RUNS/$1/$f.jsonl $RUNS/$2/$f.jsonl | tail -1; }; done; }
