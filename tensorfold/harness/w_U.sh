# Window U (round 7, last): build kp25 = patches 0101-0109 (one launch for every lane's best draft). Final many-user column on the documented path.
#   923b: recipe.sh prepare (0101-0109) -> USERS16 with BEST_LANES=0 -> USERS16 default (GDN table of 16 + BEST_LANES=1, ROWS128 off) with waves, mixed sizes, 227K prompt.
#   4e5c: 16 users ref / BEST_LANES=1 / ref, texts with 16 in flight.   3d24: per-kernel GPU time with it on; 12 and 9 users on / off; one user on / off (texts).
E='QK_MODEL=Qwen3.8-Flash-Next'; NE=TFENTRY=/opt/nvidia/nvidia_entrypoint.sh
KP=${KP:-kp25}
TW="TFBIN=\$HOME/q38/tfbin/$KP TFMODEL=\$HOME/q38/models/nvidia-fp4-local $NE"
go() { $1 "$TW TFENV='$2' bash ~/q38/kit/eng.sh tfb $3" 2>&1 | tail -1 | cut -c1-200
  $1 "export $E O=\$HOME/q38/runs/$4; mkdir -p \$O; $5; docker logs q38-run-tf > \$O/engine.log 2>&1; bash ~/q38/kit/eng.sh stop >/dev/null"; echo "$4 done $(date -u +%T)"; }
gor() { r3 "bash ~/q38/kit/rs.sh $1" 2>&1 | tail -1 | cut -c1-300
  r3 "export $E O=\$HOME/q38/runs/$2; mkdir -p \$O; $3; docker logs q38-run-tf > \$O/engine.log 2>&1; bash ~/q38/kit/eng.sh stop >/dev/null"; echo "$2 done $(date -u +%T)"; }
C='python3 -I ~/q38/kit/qk.py cmp'; Q='python3 -I ~/q38/kit/qk.py'; W='python3 -I ~/q38/kit/waves.py'
MU="for n in 1 4 8 16; do $Q distinct \$n | tail -1; done; $Q copies 16 | tail -1; $Q distinct 16 | tail -1; $Q distinct 16 | tail -1; $Q distinct 12 | tail -1"
( r3 'cd ~/q38/repo_tf && WORK=$HOME/q38/recipe_work HF_CACHE=$HOME/q38/hf bash ./recipe.sh prepare > ~/q38/recipe_work/prepare6.log 2>&1; echo prepare rc $?; ls ~/q38/recipe_work/miaai-recipe/patches | tail -1; tail -1 ~/q38/recipe_work/prepare6.log | cut -c1-200'
  gor "USERS16=1 BEST_LANES=0" U_mu0 "$MU; $W 16 \$O/waves16.jsonl 256 1 8 | tail -1"
  gor "USERS16=1" U_mu "$MU; $W 16 \$O/waves16.jsonl 256 1 8 | tail -1; $W 4 \$O/waves4.jsonl 128 5 | tail -1; $C \$O/waves16.jsonl \$HOME/q38/runs/U_mu0/waves16.jsonl; $C \$O/waves16.jsonl \$HOME/q38/runs/T_mia16/waves16.jsonl; python3 -I ~/q38/kit/poolt.py 150000x1+15 | cut -c1-420; $Q ctx 250000 | tail -1"
) > $D/u_923b.log 2>&1 & P1=$!
B="-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_CERT_PLAIN=1 -e TF_FLASHNEXT_Q38_PREFILL_MANY=256 -e TF_FLASHNEXT_Q38_WINDOW_ROWS=32 -e TF_FLASHNEXT_Q38_HCF=3 -e TF_FLASHNEXT_Q38_HCF_ST=3 -e TF_FLASHNEXT_Q38_HCF_HALF=1 -e TF_FLASHNEXT_Q38_HALF_BELOW=96 -e TF_FLASHNEXT_Q38_MTP_LAST=1"
MUE="$B -e TF_FLASHNEXT_Q38_WIDE_ROWS=128 -e TF_FLASHNEXT_Q38_POOL_CTX=65536 -e TF_FLASHNEXT_Q38_GDN_LANES_MANY=2"
BL="-e TF_FLASHNEXT_Q38_BEST_LANES=1"
MB="$Q distinct 16 | tail -1; $Q distinct 16 | tail -1; $Q distinct 16 | tail -1; $Q distinct 8 | tail -1; python3 -I ~/q38/kit/heldpar.py 16 \$O/held16.jsonl 256 3 | tail -1"
CM="$C \$HOME/q38/runs/U3_ref/held16.jsonl \$O/held16.jsonl; docker logs q38-run-tf 2>&1 | grep -ciE 'panic|segfault'"
( go w3 "$MUE" pc_16_262144 U3_ref "$MB"
  go w3 "$MUE $BL" pc_16_262144 U3_on "$MB; $CM"
  go w3 "$MUE" pc_16_262144 U3_ref2 "$MB; $CM"
) > $D/u_4e5c.log 2>&1 & P2=$!
W2="$Q distinct 12 | tail -1; $Q distinct 9 | tail -1; $Q distinct 4 | tail -1; python3 -I ~/q38/kit/heldpar.py 1 \$O/held1.jsonl 256 6 | tail -1; QK_HELD=\$HOME/q38/kit/sets/edit.jsonl python3 -I ~/q38/kit/heldpar.py 4 \$O/edit4.jsonl 1200 | tail -1"
( go w2 "$MUE $BL -e TF_KPROF=400000" pc_16_262144 U2_kp16 "$Q distinct 16 | tail -1; $Q distinct 16 | tail -1; docker logs q38-run-tf > \$O/engine.log 2>&1; python3 -I ~/q38/kit/kprof.py \$O/engine.log > \$O/kprof.txt; grep -E 'row_best|TOTAL|gdn_rows_late_lanes' \$O/kprof.txt | cut -c1-150"
  go w2 "$MUE $BL" pc_16_262144 U2_on "$W2"
  go w2 "$MUE" pc_16_262144 U2_off "$W2; for f in held1 edit4; do $C \$HOME/q38/runs/U2_on/\$f.jsonl \$O/\$f.jsonl; done"
) > $D/u_3d24.log 2>&1 & P3=$!
wait $P1 $P2 $P3
echo "window U work done $(date -u +%T)"
