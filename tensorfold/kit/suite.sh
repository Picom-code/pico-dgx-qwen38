#!/bin/bash
# suite.sh LABEL [quick] : the bake-off measurement suite against the Qwen server on 127.0.0.1:18301 (runs on spark-923b).
# Same prompts, settings and order for every engine. Output: ~/q38/runs/LABEL/*.{log,json}
L=$1; MODE=${2:-full}; O=$HOME/q38/runs/$L; mkdir -p $O; cd $O
K=$HOME/q38/kit; Q="python3 -I $K/qk.py"
export QK_URL=${QK_URL:-http://127.0.0.1:18301/v1/chat/completions}
MAXU=${MAXU:-16}
( while :; do echo "$(date +%s) $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | paste -sd+ | bc 2>/dev/null) $(nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.current.graphics --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"; sleep 5; done > mem.log ) & MP=$!
trap "kill $MP 2>/dev/null" EXIT
s() { echo "== $1 $(date -u +%T)"; shift; timeout ${TMO:-1500} "$@" 2>&1 | tail -${TAILN:-12}; }
{
echo "suite $L mode $MODE start $(date -u +%FT%TZ) model $QK_MODEL"
# greedy probe: an engine that refuses temperature 0 is measured sampled (T 0.7, top_p 0.8, seed 1234) and says so
if ! python3 -I - <<PYEOF
import json, os, sys, urllib.request
b = {"model": os.environ.get("QK_MODEL", "x"), "messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 4, "temperature": 0, "top_p": 1}
try: urllib.request.urlopen(urllib.request.Request("$QK_URL", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=120).read()
except Exception as e:
    print("greedy probe failed:", e, getattr(e, "read", lambda: b"")()[:300]); sys.exit(1)
PYEOF
then echo "GREEDY REFUSED by this engine: every 'temp 0' number below is SAMPLED at temperature 0.7, top_p 0.8, seed 1234"; export QK_EXTRA='{"temperature": 0.7, "top_p": 0.8, "seed": 1234}'; fi
echo "idle after load: MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GiB, GPU MiB $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | paste -sd+)"
sk() { case " $SKIP " in *" $1 "*) return 0;; esac; return 1; }
if [ "$MODE" = quick ]; then
s ttft $Q ttft; s c1 $Q c1 2; s sd $Q sd 1; s img $Q img; s distinct4 $Q distinct 4; s prefill $Q prefill 8000
[ -n "$NODRAFT_REQ" ] && { echo "== nodraft probe"; QK_EXTRA='{"draft": false}' $Q c1 1 2>&1 | tail -2; }
else
s ttft $Q ttft
s c1 $Q c1 9
s sd $Q sd 5
sk ident || s ident $Q ident $O/ident_on.json
if [ -n "$NODRAFT_REQ" ] && ! sk ident; then TAILN=1 s ident_nodraft $Q ident $O/ident_off.json nodraft; s identcmp $Q cmp $O/ident_on.json $O/ident_off.json; fi
if ! sk multi; then
for n in 4 8 16; do [ $n -le $MAXU ] && { s distinct$n $Q distinct $n; s copies$n $Q copies $n; }; done
[ $MAXU -lt 16 ] && { s distinct_max$MAXU $Q distinct $MAXU; s copies_max$MAXU $Q copies $MAXU; s distinct16_queued $Q distinct 16; }
s distinct16_again $Q distinct $( [ $MAXU -lt 16 ] && echo $MAXU || echo 16 )
s sdn1 $Q sdn 1; s sdn4 $Q sdn 4; s sdn5 $Q sdn 5
fi
sk prefill || s prefill $Q prefill 8000,32000,128000
sk held || QK_HELD_STRIDE=${HELD_STRIDE:-3} TMO=3000 s held $Q held $O/held.jsonl
if ! sk sparkdash; then mkdir -p $K/sdw/out/$L; echo none > $K/sdw/key
echo "== sparkDash 1.8.8 real $(date -u +%T)"; docker run --rm --net=host -v $K/sdw:/w -w /w -e SD_ROOT=/w/sd -e BENCH_HISTORY_PATH=/w/out/$L/bh.json -e BENCH_ACTIVE_PATH=/w/out/$L/ba.json node:22-bookworm-slim timeout 900 node /w/sdrun.mjs decode 18301 /w/key /w/out/$L/sd.jsonl > $O/sdrun.log 2>&1; grep "mean\|c1 prose\|c1 code\|c1 struct\|failed" $O/sdrun.log | cut -c1-120; fi
sk qeval || { echo "== qeval c4 $(date -u +%T)"; ( cd $O && timeout 1500 python3 -I $K/qeval_q.py run ${L} --url $QK_URL --concurrency 4 2>&1 | tail -4 ); }
sk img || s img $Q img
sk needle || TMO=900 s needle200k $Q needle 200000
sk ctx || TMO=1200 s ctx $Q ctx ${CTXMAX:-250000}
fi
echo "min MemAvailable during suite: $(awk 'NR>1{print $2}' mem.log | sort -n | head -1) MiB; max GPU MiB $(awk '{print $3}' mem.log | sort -n | tail -1)"
echo "gpu temp C min/max during suite: $(awk '{split($4,a,","); print a[1]}' mem.log | sort -n | sed -n '1p;$p' | paste -sd/) on $(hostname)"
echo "suite end $(date -u +%FT%TZ)"
} > $O/suite.log 2>&1
echo SUITE_DONE >> $O/suite.log
