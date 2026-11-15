# Window N (round 5): build kp14 = kp13 + the fill-queue batch (core/lane_host_fill.many: requests that arrive together meet in the fill
# list, so that is where their prompts are batched) + TF_FLASHNEXT_Q38_WINDOW_ROWS (32-row verify windows for copied continuations) +
# TF_FLASHNEXT_Q38_GPROF (GPU time by group from CUDA events).
#   4e5c: lever 1 again: 16 x 64K, 128-row passes, cert: many off vs on (benches, waves of 16 and 4, texts + score hashes).
#   3d24: lever 3: window rows 16 vs 32 on the edit / agent / chat sets (texts compared), then the 16-user bench with 32-row windows.
#   923b: lever 5: GPU time by group, drafts on and off; then the candidate stack (pool + many + 32-row windows) many-user benches.
E='QK_MODEL=Qwen3.8-Flash-Next'; NE=TFENTRY=/opt/nvidia/nvidia_entrypoint.sh
TW="TFBIN=\$HOME/q38/tfbin/kp14 TFMODEL=\$HOME/q38/models/nvidia-fp4-local $NE"
go() { $1 "$2 bash ~/q38/kit/eng.sh $3 $4" 2>&1 | tail -1 | cut -c1-200
  $1 "export $E O=\$HOME/q38/runs/$5; mkdir -p \$O; $6; docker logs q38-run-tf > \$O/engine.log 2>&1; bash ~/q38/kit/eng.sh stop >/dev/null"; echo "$5 done $(date -u +%T)"; }
C='python3 -I ~/q38/kit/qk.py cmp'; Q='python3 -I ~/q38/kit/qk.py'
BASE="-e TF_FLASHNEXT_Q38_WIDE_ROWS=128 -e TF_FLASHNEXT_Q38_CERT=1"
MU="for n in 1 4 8 16; do $Q distinct \$n | tail -1; done; $Q copies 16 | tail -1; $Q distinct 16 | tail -1; $Q copies 8 | tail -1; $Q distinct 8 | tail -1; $Q distinct 4 | tail -1"
MS="$Q distinct 16 | tail -1; $Q copies 16 | tail -1; $Q distinct 16 | tail -1; $Q distinct 8 | tail -1; $Q distinct 4 | tail -1"
WV="python3 -I ~/q38/kit/waves.py 16 \$O/waves16.jsonl 256 1 12 | tail -1; python3 -I ~/q38/kit/waves.py 4 \$O/waves4.jsonl 128 5 | tail -1"
GL="docker logs q38-run-tf 2>&1 | grep -E 'q38 many|q38 pool|q38: prompts' | sed -n '1,4p;\$p' | cut -c1-200; docker logs q38-run-tf 2>&1 | grep -c 'q38 many: [0-9]'"
LH="docker logs q38-run-tf > \$O/engine.log 2>&1; python3 -I ~/q38/kit/lhcmp.py"
( go w3 "$TW TFENV='$BASE -e TF_FLASHNEXT_Q38_LHASH=1 -e TF_FLASHNEXT_Q38_PREFILL_MANY=256'" tfb p16 N3_many "$MS; $WV; $GL"
  go w3 "$TW TFENV='$BASE -e TF_FLASHNEXT_Q38_LHASH=1'" tfb p16 N3_off "$MS; $WV; $C \$O/waves16.jsonl \$HOME/q38/runs/N3_many/waves16.jsonl; $C \$O/waves4.jsonl \$HOME/q38/runs/N3_many/waves4.jsonl; $LH \$O/engine.log \$HOME/q38/runs/N3_many/engine.log"
  go w3 "$TW TFENV='$BASE -e TF_FLASHNEXT_Q38_LHASH=1 -e TF_FLASHNEXT_Q38_PREFILL_MANY=256'" tfb p16 N3_many2 "$MS; python3 -I ~/q38/kit/waves.py 16 \$O/waves16b.jsonl 384 1 20 | tail -1; python3 -I ~/q38/kit/waves.py 8 \$O/waves8.jsonl 128 2 | tail -1; $GL"
  go w3 "$TW TFENV='$BASE -e TF_FLASHNEXT_Q38_LHASH=1'" tfb p16 N3_off2 "python3 -I ~/q38/kit/waves.py 16 \$O/waves16b.jsonl 384 1 20 | tail -1; python3 -I ~/q38/kit/waves.py 8 \$O/waves8.jsonl 128 2 | tail -1; $C \$O/waves16b.jsonl \$HOME/q38/runs/N3_many2/waves16b.jsonl; $C \$O/waves8.jsonl \$HOME/q38/runs/N3_many2/waves8.jsonl; $LH \$O/engine.log \$HOME/q38/runs/N3_many2/engine.log"
) > $D/n_4e5c.log 2>&1 & P3=$!
ST="python3 -I ~/q38/kit/tfsteps.py"
one() { echo "L0=\$(docker logs q38-run-tf 2>&1 | wc -l); QK_HELD=$1 python3 -I ~/q38/kit/heldpar.py 1 \$O/$2.jsonl $3 ${4:-1} | tail -1; docker logs q38-run-tf 2>&1 | tail -n +\$((L0+1)) > \$O/eng_$2.log; echo '== $2'; $ST \$O/eng_$2.log | sed -n '2,3p'"; }
SETS="python3 -I ~/q38/kit/mkedit.py ~/q38/kit/sets | tail -1; $(one \$HOME/q38/kit/sets/edit.jsonl edit 1500); $(one \$HOME/q38/kit/sets/agent.jsonl agent 900); $(one \$HOME/q38/kit/prompts.jsonl chat 384 8)"
CMP3="for s in edit agent chat; do $C \$HOME/q38/runs/N2_r16/\$s.jsonl \$O/\$s.jsonl; done"
( go w2 "$TW TFENV='-e TF_FLASHNEXT_Q38_CERT=1'" tfb default N2_r16 "$SETS"
  go w2 "$TW TFENV='-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_WINDOW_ROWS=32'" tfb default N2_r32 "$SETS; $CMP3"
  go w2 "$TW TFENV='-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_WINDOW_ROWS=24'" tfb default N2_r24 "$SETS; $CMP3"
  go w2 "$TW TFENV='$BASE -e TF_FLASHNEXT_Q38_WINDOW_ROWS=32'" tfb p16 N2_r32mu "$MU"
) > $D/n_3d24.log 2>&1 & P2=$!
GP="docker logs q38-run-tf 2>&1 | grep 'q38 gprof' | tail -3 | cut -c1-330"
( go r3 "$TW TFENV='-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_GPROF=1'" tfb default N_gp "$(one \$HOME/q38/kit/prompts.jsonl held 256 8); $GP"
  go r3 "$TW TFENV='-e TF_FLASHNEXT_Q38_GPROF=1'" tfb nodraft N_gp_nd "$(one \$HOME/q38/kit/prompts.jsonl held 128 16); $GP"
  go r3 "$TW TFENV='$BASE -e TF_FLASHNEXT_Q38_POOL_CTX=65536 -e TF_FLASHNEXT_Q38_PREFILL_MANY=256 -e TF_FLASHNEXT_Q38_WINDOW_ROWS=32'" tfb pc_16_262144 N_stack "$MU; $GL; python3 -I ~/q38/kit/poolt.py 150000x1+15 200000x5+0 | cut -c1-420; $Q ctx 250000 | tail -1"
  go r3 "$TW TFENV='-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_GPROF=1'" tfb p16 N_gp16 "$Q distinct 16 | tail -1; $Q distinct 16 | tail -1; $GP"
) > $D/n_923b.log 2>&1 & P1=$!
wait $P1 $P2 $P3
echo "window N work done $(date -u +%T)"
