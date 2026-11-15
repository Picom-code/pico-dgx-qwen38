# Window R (round 7): kp21, no new engine build. Question: is the routed-expert (MoE) path below the box's byte rate because of the kernel FORM?
#   4e5c: expert microbench (decode form vs MiaAI's shared-memory tile vs prefetch depth vs native FP decode, by rows a pass), then one user with the tile from 2 / 6 rows.
#   923b: one user, the tile at every row count (TF_FLASHNEXT_PROMPT_EXPERT_TILE_ROWS=1) in three shapes, gate/up only; reference first and last; texts compared.
#   3d24: 16 users, the same shapes; then the no-drafts GPU time by group with and without the tile.
E='QK_MODEL=Qwen3.8-Flash-Next'; NE=TFENTRY=/opt/nvidia/nvidia_entrypoint.sh
KP=${KP:-kp21}
TW="TFBIN=\$HOME/q38/tfbin/$KP TFMODEL=\$HOME/q38/models/nvidia-fp4-local $NE"
go() { $1 "$TW TFENV='$2' bash ~/q38/kit/eng.sh tfb $3" 2>&1 | tail -1 | cut -c1-200
  $1 "export $E O=\$HOME/q38/runs/$4; mkdir -p \$O; $5; docker logs q38-run-tf > \$O/engine.log 2>&1; bash ~/q38/kit/eng.sh stop >/dev/null"; echo "$4 done $(date -u +%T)"; }
C='python3 -I ~/q38/kit/qk.py cmp'; Q='python3 -I ~/q38/kit/qk.py'
ST="python3 -I ~/q38/kit/tfsteps.py"
B="-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_CERT_PLAIN=1 -e TF_FLASHNEXT_Q38_PREFILL_MANY=256 -e TF_FLASHNEXT_Q38_WINDOW_ROWS=32 -e TF_FLASHNEXT_Q38_HCF=3 -e TF_FLASHNEXT_Q38_HCF_ST=3 -e TF_FLASHNEXT_Q38_HCF_HALF=1 -e TF_FLASHNEXT_Q38_HALF_BELOW=96 -e TF_FLASHNEXT_Q38_MTP_LAST=1"
H1="python3 -I ~/q38/kit/heldpar.py 1 \$O/held.jsonl 256 3 | tail -1; $ST <(docker logs q38-run-tf 2>&1) | sed -n '2,3p'"
T='-e TF_FLASHNEXT_PROMPT_EXPERT_TILE_ROWS'; TC='-e TF_FLASHNEXT_PROMPT_EXPERT_TILE_CFG'
( go r3 "$B" default R_ref "$H1"
  go r3 "$B $T=1" default R_t1 "$H1; $C \$HOME/q38/runs/R_ref/held.jsonl \$O/held.jsonl"
  go r3 "$B $T=1 $TC=32,3,2" default R_t1p32 "$H1; $C \$HOME/q38/runs/R_ref/held.jsonl \$O/held.jsonl"
  go r3 "$B $T=1 $TC=16,3,2" default R_t1p16 "$H1; $C \$HOME/q38/runs/R_ref/held.jsonl \$O/held.jsonl"
  go r3 "$B $T=1 -e TF_FLASHNEXT_PROMPT_EXPERT_TILE=1" default R_t1gu "$H1; $C \$HOME/q38/runs/R_ref/held.jsonl \$O/held.jsonl"
  go r3 "$B" default R_ref2 "$H1; $C \$HOME/q38/runs/R_ref/held.jsonl \$O/held.jsonl"
) > $D/r_923b.log 2>&1 & P1=$!
MUE="$B -e TF_FLASHNEXT_Q38_WIDE_ROWS=128 -e TF_FLASHNEXT_Q38_POOL_CTX=65536"
MB="$Q distinct 16 | tail -1; $Q distinct 16 | tail -1; $Q distinct 16 | tail -1; $Q distinct 8 | tail -1; python3 -I ~/q38/kit/heldpar.py 16 \$O/held16.jsonl 256 3 | tail -1"
CM="$C \$HOME/q38/runs/R2_ref/held16.jsonl \$O/held16.jsonl"
GP="$Q c1 1 | tail -2; docker logs q38-run-tf 2>&1 | grep 'q38 gprof' | tail -3 | cut -c1-330"
( go w2 "$MUE" pc_16_262144 R2_ref "$MB"
  go w2 "$MUE $T=1" pc_16_262144 R2_t1 "$MB; $CM"
  go w2 "$MUE $T=1 $TC=32,3,2" pc_16_262144 R2_t1p32 "$MB; $CM"
  go w2 "$MUE $T=1 $TC=16,3,2" pc_16_262144 R2_t1p16 "$MB; $CM"
  go w2 "$MUE" pc_16_262144 R2_ref2 "$MB; $CM"
  go w2 "-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_GPROF=1" nodraft R2_gp "$GP"
  go w2 "-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_GPROF=1 $T=1" nodraft R2_gpt "$GP"
  go w2 "-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_GPROF=1 $T=1 $TC=16,3,2" nodraft R2_gpt16 "$GP"
) > $D/r_3d24.log 2>&1 & P2=$!
( w3 'mkdir -p ~/q38/runs; bash ~/q38/mb/mbrun.sh mbx 3 > ~/q38/runs/R3_mbx.txt 2>&1; bash ~/q38/mb/mbrun.sh mbxn 3 > ~/q38/runs/R3_mbxn.txt 2>&1; tail -4 ~/q38/runs/R3_mbx.txt | cut -c1-200'
  echo "microbench done $(date -u +%T)"
  go w3 "$B" default R3_ref "$H1"
  go w3 "$B $T=2" default R3_t2 "$H1; $C \$HOME/q38/runs/R3_ref/held.jsonl \$O/held.jsonl"
  go w3 "$B $T=6" default R3_t6 "$H1; $C \$HOME/q38/runs/R3_ref/held.jsonl \$O/held.jsonl"
  go w3 "$B $T=2 $TC=16,3,2" default R3_t2p16 "$H1; $C \$HOME/q38/runs/R3_ref/held.jsonl \$O/held.jsonl"
  go w3 "$B" default R3_ref2 "$H1; $C \$HOME/q38/runs/R3_ref/held.jsonl \$O/held.jsonl"
) > $D/r_4e5c.log 2>&1 & P3=$!
wait $P1 $P2 $P3
echo "window R work done $(date -u +%T)"
