# Window T (round 7, FINAL): build kp24 = patches 0101-0108 (GDN lanes for 9-16 streams, 128-row lane tiles).
#   923b (DOCUMENTED path: recipe.sh prepare with 0101-0108, then recipe.sh start, vision on): one user default (identity against window Q's texts) ->
#        USERS16 default -> MiaAI configuration 16 x 64K -> USERS16 with the two new levers off (same-window pair).
#   4e5c: 16 users, each lever alone and together, reference first and last, texts with 16 in flight.
#   3d24: everything on under the checkers (certified head against the whole head, fused read-out against the five-launch chain) with 16 in flight; GPU time by group;
#        12 / 9 users and the edit set 4 in flight, on against off, texts compared.
E='QK_MODEL=Qwen3.8-Flash-Next'; NE=TFENTRY=/opt/nvidia/nvidia_entrypoint.sh
KP=${KP:-kp24}
TW="TFBIN=\$HOME/q38/tfbin/$KP TFMODEL=\$HOME/q38/models/nvidia-fp4-local $NE"
go() { $1 "$TW TFENV='$2' bash ~/q38/kit/eng.sh tfb $3" 2>&1 | tail -1 | cut -c1-200
  $1 "export $E O=\$HOME/q38/runs/$4; mkdir -p \$O; $5; docker logs q38-run-tf > \$O/engine.log 2>&1; bash ~/q38/kit/eng.sh stop >/dev/null"; echo "$4 done $(date -u +%T)"; }
gor() { r3 "bash ~/q38/kit/rs.sh $1" 2>&1 | tail -1 | cut -c1-300
  r3 "export $E O=\$HOME/q38/runs/$2; mkdir -p \$O; $3; docker logs q38-run-tf > \$O/engine.log 2>&1; bash ~/q38/kit/eng.sh stop >/dev/null"; echo "$2 done $(date -u +%T)"; }
C='python3 -I ~/q38/kit/qk.py cmp'; Q='python3 -I ~/q38/kit/qk.py'; W='python3 -I ~/q38/kit/waves.py'
ST="python3 -I ~/q38/kit/tfsteps.py"
MU="for n in 1 4 8 16; do $Q distinct \$n | tail -1; done; $Q copies 16 | tail -1; $Q distinct 16 | tail -1; $Q distinct 12 | tail -1; $Q copies 8 | tail -1"
one() { echo "L0=\$(docker logs q38-run-tf 2>&1 | wc -l); QK_HELD=$1 python3 -I ~/q38/kit/heldpar.py ${5:-1} \$O/$2.jsonl $3 ${4:-1} | tail -1; docker logs q38-run-tf 2>&1 | tail -n +\$((L0+1)) > \$O/eng_$2.log; echo '== $2'; $ST \$O/eng_$2.log | sed -n '2,3p'"; }
SETS="python3 -I ~/q38/kit/mkedit.py ~/q38/kit/sets | tail -1; $(one \$HOME/q38/kit/sets/edit.jsonl edit 1500); $(one \$HOME/q38/kit/sets/agent.jsonl agent 900)"
WV="$W 16 \$O/waves16.jsonl 256 1 8 | tail -1; $W 4 \$O/waves4.jsonl 128 5 | tail -1"
( r3 'cd ~/q38/repo_tf && WORK=$HOME/q38/recipe_work HF_CACHE=$HOME/q38/hf bash ./recipe.sh prepare > ~/q38/recipe_work/prepare5.log 2>&1; echo prepare rc $?; ls ~/q38/recipe_work/miaai-recipe/patches | tail -1; tail -1 ~/q38/recipe_work/prepare5.log | cut -c1-200; docker images --format "{{.Repository}}:{{.Tag}} {{.CreatedSince}}" | grep tensorfold-qwen38'
  gor "" T_on "$Q c1 | tail -2; $Q ttft | tail -1; $(one \$HOME/q38/kit/prompts.jsonl held 256 3); $SETS; for f in held edit agent; do $C \$HOME/q38/runs/Q_on/\$f.jsonl \$O/\$f.jsonl; done"
  gor "USERS16=1" T_mu "$MU; $WV; python3 -I ~/q38/kit/poolt.py 150000x1+15 | cut -c1-420; $Q ctx 250000 | tail -1"
  gor "CERT=0 MANY=0 COPY_ROWS=16 FUSED=0 HALF=0 MTP_LAST=0 GDN_MANY=0 ROWS128=0 PARALLEL=16 CONTEXT=65536 TF_FLASHNEXT_PRODUCT_STREAMS=16" T_mia16 "$MU; $WV; $C \$O/waves16.jsonl \$HOME/q38/runs/T_mu/waves16.jsonl; $C \$O/waves4.jsonl \$HOME/q38/runs/T_mu/waves4.jsonl"
  gor "USERS16=1 GDN_MANY=0 ROWS128=0" T_mu0 "$MU; $WV; $C \$O/waves16.jsonl \$HOME/q38/runs/T_mu/waves16.jsonl"
) > $D/t_923b.log 2>&1 & P1=$!
B="-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_CERT_PLAIN=1 -e TF_FLASHNEXT_Q38_PREFILL_MANY=256 -e TF_FLASHNEXT_Q38_WINDOW_ROWS=32 -e TF_FLASHNEXT_Q38_HCF=3 -e TF_FLASHNEXT_Q38_HCF_ST=3 -e TF_FLASHNEXT_Q38_HCF_HALF=1 -e TF_FLASHNEXT_Q38_HALF_BELOW=96 -e TF_FLASHNEXT_Q38_MTP_LAST=1"
MUE="$B -e TF_FLASHNEXT_Q38_WIDE_ROWS=128 -e TF_FLASHNEXT_Q38_POOL_CTX=65536"
G1="-e TF_FLASHNEXT_Q38_GDN_LANES_MANY=1"; G2="-e TF_FLASHNEXT_Q38_GDN_LANES_MANY=2"; R="-e TF_FLASHNEXT_Q38_ROWS128=2048"
MB="$Q distinct 16 | tail -1; $Q distinct 16 | tail -1; $Q distinct 16 | tail -1; $Q distinct 8 | tail -1; python3 -I ~/q38/kit/heldpar.py 16 \$O/held16.jsonl 256 3 | tail -1"
CM="$C \$HOME/q38/runs/T3_ref/held16.jsonl \$O/held16.jsonl; docker logs q38-run-tf 2>&1 | grep -ciE 'panic|segfault'"
( go w3 "$MUE" pc_16_262144 T3_ref "$MB"
  go w3 "$MUE $G2 $R" pc_16_262144 T3_all "$MB; $CM"
  go w3 "$MUE $R" pc_16_262144 T3_r128 "$MB; $CM"
  go w3 "$MUE $G2" pc_16_262144 T3_g2 "$MB; $CM"
  go w3 "$MUE $G1" pc_16_262144 T3_g1 "$MB; $CM"
  go w3 "$MUE" pc_16_262144 T3_ref2 "$MB; $CM"
  go w3 "$MUE $G2 -e TF_FLASHNEXT_Q38_ROWS128=1024" pc_16_262144 T3_r1024 "$MB; $CM"
) > $D/t_4e5c.log 2>&1 & P2=$!
W2="$Q distinct 12 | tail -1; $Q distinct 9 | tail -1; $W 16 \$O/waves16.jsonl 256 1 6 | tail -1; QK_HELD=\$HOME/q38/kit/sets/edit.jsonl python3 -I ~/q38/kit/heldpar.py 4 \$O/edit4.jsonl 1200 | tail -1; python3 -I ~/q38/kit/heldpar.py 1 \$O/held1.jsonl 256 6 | tail -1"
( go w2 "$MUE $G2 $R -e TF_FLASHNEXT_Q38_HCF_MAX=128 -e TF_FLASHNEXT_Q38_CERT=2 -e TF_FLASHNEXT_Q38_HCF_CHECK=1" pc_16_262144 T2_chk "$W 16 \$O/waves16.jsonl 256 1 10 | tail -1; $Q distinct 16 | tail -1; QK_HELD=\$HOME/q38/kit/sets/edit.jsonl python3 -I ~/q38/kit/heldpar.py 4 \$O/edit4.jsonl 1200 | tail -1; docker logs q38-run-tf 2>&1 | grep -E 'q38 cert: calls|q38 hcf check' | tail -2 | cut -c1-230"
  go w2 "$MUE $G2 $R" pc_16_262144 T2_on "$W2"
  go w2 "$MUE" pc_16_262144 T2_off "$W2; for f in waves16 edit4 held1; do $C \$HOME/q38/runs/T2_on/\$f.jsonl \$O/\$f.jsonl; done"
  go w2 "$MUE $G2 $R -e TF_FLASHNEXT_Q38_GPROF=1" pc_16_262144 T2_gp "$Q distinct 16 | tail -1; $Q distinct 16 | tail -1; python3 -I ~/q38/kit/heldpar.py 16 \$O/held16.jsonl 256 3 | tail -1; docker logs q38-run-tf 2>&1 | grep 'q38 gprof 9+rows' | tail -1 | cut -c1-330"
  go w2 "$MUE $G2 $R -e TF_KPROF=400000" pc_16_262144 T2_kp16 "$Q distinct 16 | tail -1; $Q distinct 16 | tail -1; docker logs q38-run-tf > \$O/engine.log 2>&1; python3 -I ~/q38/kit/kprof.py \$O/engine.log > \$O/kprof.txt; head -34 \$O/kprof.txt | cut -c1-150; tail -1 \$O/kprof.txt"
) > $D/t_3d24.log 2>&1 & P3=$!
wait $P1 $P2 $P3
echo "window T work done $(date -u +%T)"
