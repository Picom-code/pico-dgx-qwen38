# libN.sh (923b, agent A, night 11-14): source after libA.sh + lib9.sh + lib11.sh.
# left() = seconds until the EARLIEST of: lease deadline, window start + 270 min (both stamped every minute by the head's keeper keepN.sh),
# and the night's hard cutoff (12:02 UTC). A stamp older than 6 minutes counts as "no window" (the keeper died): no engine starts.
KN=$HOME/q38/kit-N; NU="python3 -I $KN/nu.py"; CUTOFF=$(date -u -d '2026-11-15 12:02:00' +%s)
left() { local d w n s; d=$(cat $RUNS/A_deadline 2>/dev/null || echo 0); w=$(cat $RUNS/A_winend 2>/dev/null || echo 0); s=$(cat $RUNS/A_stamp 2>/dev/null || echo 0); n=$(date +%s)
  [ $(( n - s )) -gt 360 ] && { echo 0; return; }; [ $w -lt $d ] && d=$w; [ $CUTOFF -lt $d ] && d=$CUTOFF; echo $(( d - n )); }
# start facts of the running engine: the knobs recipe.sh exported (no key-like names), the engine's own start lines
ENVD="docker inspect q38-run-A --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^(TF_|TENSORFOLD_|KV_DTYPE|PARALLEL|CONTEXT|DRAFTS)' | grep -viE 'key|token|secret|pass' | sort > \$O/env.txt; echo \"env knobs: \$(wc -l < \$O/env.txt) -> \$(grep -E '^(KV_DTYPE|PARALLEL|CONTEXT|TF_FLASHNEXT_Q38_POOL_CTX|TF_FLASHNEXT_CHUNK|TF_FLASHNEXT_Q38_WINDOW_ROWS|TF_FLASHNEXT_Q38_SNAP_DELTA|TF_FLASHNEXT_Q38_CACHE_ELASTIC|TF_FLASHNEXT_Q38_LANE_REUSE|TF_FLASHNEXT_Q38_CERT|TF_FLASHNEXT_Q38_HEAD2)=' \$O/env.txt | tr '\n' ' ')\"; docker logs q38-run-A 2>&1 | grep -iE 'streams at once|kv.?(cache|dtype)|bf16 k|fp8 k|prompt cache|q38 elastic|q38 qsa knobs|needs .* GiB' | cut -c1-300 | awk '!s[\$0]++' | head -8; echo \"image \$(docker inspect -f '{{.Image}}' q38-run-A | cut -c8-19)\""
USERS="for n in 1 4 8 12 16; do $NU \$n \$O/users.jsonl 3 | tail -1; done"
