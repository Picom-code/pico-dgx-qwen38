# Window M (round 5): build kp13 = kp11 + lever 1 (TF_FLASHNEXT_Q38_PREFILL_MANY: waiting prompts in one prompt pass) + lever 2
# (TF_FLASHNEXT_Q38_POOL_CTX: streams admitted by bytes against a pool) + TF_FLASHNEXT_Q38_CERT_PLAIN + the LHASH aid.
#   4e5c: lever 1. 16 x 64K, 128-row passes, certified head: many off vs on (benches, waves of 16 held prompts, texts + score hashes compared).
#   3d24: lever 2. fixed 16 x 64K vs pool (16 lanes, 64K pool windows, 262K largest request): benches + mixed-size scenarios + memory.
#   923b: lever 3 measurement (the copy drafts TensorFold already has: edit / agent / chat sets), then --no-drafts with the certified head on plain rounds.
E='QK_MODEL=Qwen3.8-Flash-Next'; NE=TFENTRY=/opt/nvidia/nvidia_entrypoint.sh
TW="TFBIN=\$HOME/q38/tfbin/kp13 TFMODEL=\$HOME/q38/models/nvidia-fp4-local $NE"
go() { $1 "$2 bash ~/q38/kit/eng.sh $3 $4" 2>&1 | tail -1 | cut -c1-200
  $1 "export $E O=\$HOME/q38/runs/$5; mkdir -p \$O; $6; docker logs q38-run-tf > \$O/engine.log 2>&1; bash ~/q38/kit/eng.sh stop >/dev/null"; echo "$5 done $(date -u +%T)"; }
C='python3 -I ~/q38/kit/qk.py cmp'; Q='python3 -I ~/q38/kit/qk.py'
BASE="-e TF_FLASHNEXT_Q38_WIDE_ROWS=128 -e TF_FLASHNEXT_Q38_CERT=1"
MU="for n in 1 4 8 16; do $Q distinct \$n | tail -1; done; $Q copies 16 | tail -1; $Q distinct 16 | tail -1; $Q copies 8 | tail -1; $Q distinct 8 | tail -1; $Q distinct 4 | tail -1"
WV="python3 -I ~/q38/kit/waves.py 16 \$O/waves16.jsonl 256 1 12 | tail -1; python3 -I ~/q38/kit/waves.py 4 \$O/waves4.jsonl 128 5 | tail -1"
GL="docker logs q38-run-tf 2>&1 | grep -E 'q38 many|q38 pool|q38: prompts' | sed -n '1,3p;\$p' | cut -c1-200; docker logs q38-run-tf 2>&1 | grep -c 'q38 many'"
( go w3 "$TW TFENV='$BASE -e TF_FLASHNEXT_Q38_LHASH=1'" tfb p16 M3_off "$MU; $WV"
  go w3 "$TW TFENV='$BASE -e TF_FLASHNEXT_Q38_LHASH=1 -e TF_FLASHNEXT_Q38_PREFILL_MANY=256'" tfb p16 M3_many "$MU; $WV; $GL; $C \$HOME/q38/runs/M3_off/waves16.jsonl \$O/waves16.jsonl; $C \$HOME/q38/runs/M3_off/waves4.jsonl \$O/waves4.jsonl; docker logs q38-run-tf > \$O/engine.log 2>&1; python3 -I ~/q38/kit/lhcmp.py \$HOME/q38/runs/M3_off/engine.log \$O/engine.log"
  go w3 "$TW TFENV='$BASE -e TF_FLASHNEXT_Q38_LHASH=1 -e TF_FLASHNEXT_Q38_PREFILL_MANY=256'" tfb p16 M3_many2 "$Q distinct 16 | tail -1; $Q distinct 16 | tail -1; $Q copies 16 | tail -1; python3 -I ~/q38/kit/waves.py 16 \$O/waves16b.jsonl 384 1 20 | tail -1; $GL"
  go w3 "$TW TFENV='$BASE -e TF_FLASHNEXT_Q38_LHASH=1'" tfb p16 M3_off2 "$Q distinct 16 | tail -1; $Q distinct 16 | tail -1; $Q copies 16 | tail -1; python3 -I ~/q38/kit/waves.py 16 \$O/waves16b.jsonl 384 1 20 | tail -1; $C \$HOME/q38/runs/M3_many2/waves16b.jsonl \$O/waves16b.jsonl; docker logs q38-run-tf > \$O/engine.log 2>&1; python3 -I ~/q38/kit/lhcmp.py \$HOME/q38/runs/M3_many2/engine.log \$O/engine.log"
) > $D/m_4e5c.log 2>&1 & P3=$!
PT="python3 -I ~/q38/kit/poolt.py"
( go w2 "$TW TFENV='$BASE -e TF_FLASHNEXT_Q38_POOL_CTX=65536'" tfb pc_16_262144 M2_pool "$GL; $MU; $PT 150000x1+15 100000x3+13 200000x1+8 60000x8+8 200000x5+0; $Q ctx 250000 | tail -1; $GL; docker logs q38-run-tf 2>&1 | grep -ciE 'out of memory|LaneMemory|does not fit'"
  go w2 "$TW TFENV='$BASE'" tfb p16 M2_fixed "$MU"
  go w2 "$TW TFENV='$BASE -e TF_FLASHNEXT_Q38_POOL_CTX=65536 -e TF_FLASHNEXT_Q38_PREFILL_MANY=256'" tfb pc_16_262144 M2_poolmany "$MU; $PT 150000x1+15 | tail -1; $GL"
) > $D/m_3d24.log 2>&1 & P2=$!
ST="python3 -I ~/q38/kit/tfsteps.py"
SET="python3 -I ~/q38/kit/mkedit.py ~/q38/kit/sets | tail -1"
one() { echo "L0=\$(docker logs q38-run-tf 2>&1 | wc -l); QK_HELD=$1 python3 -I ~/q38/kit/heldpar.py 1 \$O/$2.jsonl $3 ${4:-1} | tail -1; docker logs q38-run-tf 2>&1 | tail -n +\$((L0+1)) > \$O/eng_$2.log; echo '== $2'; $ST \$O/eng_$2.log | sed -n '1,4p'"; }
( go r3 "$TW TFENV='-e TF_FLASHNEXT_Q38_CERT=1'" tfb default M_copy "$SET; $(one \$HOME/q38/kit/sets/edit.jsonl edit 1500); $(one \$HOME/q38/kit/sets/agent.jsonl agent 900); $(one \$HOME/q38/kit/prompts.jsonl chat 384 8)"
  go r3 "$TW TFENV='-e TF_FLASHNEXT_Q38_CERT=1'" tfb nodraft M_nd "$(one \$HOME/q38/kit/prompts.jsonl held 256 12)"
  go r3 "$TW TFENV='-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_CERT_PLAIN=1'" tfb nodraft M_ndc "$(one \$HOME/q38/kit/prompts.jsonl held 256 12); $C \$HOME/q38/runs/M_nd/held.jsonl \$O/held.jsonl; docker logs q38-run-tf 2>&1 | grep 'q38 cert' | tail -1"
  go r3 "$TW TFENV='-e TF_FLASHNEXT_Q38_CERT=2 -e TF_FLASHNEXT_Q38_CERT_PLAIN=1'" tfb nodraft M_ndc2 "QK_HELD=\$HOME/q38/kit/prompts.jsonl python3 -I ~/q38/kit/heldpar.py 5 \$O/held5.jsonl 256 4 | tail -1; docker logs q38-run-tf 2>&1 | grep 'q38 cert' | tail -2"
) > $D/m_923b.log 2>&1 & P1=$!
wait $P1 $P2 $P3
echo "window M work done $(date -u +%T)"
