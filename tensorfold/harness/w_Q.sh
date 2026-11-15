# Window Q (round 6, FINAL): build kp21 = patches 0101-0107.
#   923b (final numbers, DOCUMENTED path: recipe.sh prepare, then recipe.sh start on that image, vision on): levers off -> recipe default (full suite) -> identity ->
#        USERS16 profile -> MiaAI configuration with 16 x 64K slots (same-window "off" column for many users).
#   3d24: queue E short context copies (TF_FLASHNEXT_Q38_SHORT_COPY) on edit / agent / chat sets + held, identity each; then the no-drafts configuration with the stack.
#   4e5c: 16 users with copy rows 16 vs 32 (same box, same window), GPU time by group of a prompt pass (Q9), whole final stack CHECKED with 16 in flight (cert + fused read-out).
E='QK_MODEL=Qwen3.8-Flash-Next'; NE=TFENTRY=/opt/nvidia/nvidia_entrypoint.sh
KP=${KP:-kp21}
TW="TFBIN=\$HOME/q38/tfbin/$KP TFMODEL=\$HOME/q38/models/nvidia-fp4-local $NE"
go() { $1 "$TW TFENV='$2' bash ~/q38/kit/eng.sh tfb $3" 2>&1 | tail -1 | cut -c1-200
  $1 "export $E O=\$HOME/q38/runs/$4; mkdir -p \$O; $5; docker logs q38-run-tf > \$O/engine.log 2>&1; bash ~/q38/kit/eng.sh stop >/dev/null"; echo "$4 done $(date -u +%T)"; }
gor() { r3 "bash ~/q38/kit/rs.sh $1" 2>&1 | tail -1 | cut -c1-300
  r3 "export $E O=\$HOME/q38/runs/$2; mkdir -p \$O; $3; docker logs q38-run-tf > \$O/engine.log 2>&1; bash ~/q38/kit/eng.sh stop >/dev/null"; echo "$2 done $(date -u +%T)"; }
C='python3 -I ~/q38/kit/qk.py cmp'; Q='python3 -I ~/q38/kit/qk.py'; W='python3 -I ~/q38/kit/waves.py'
MU="for n in 1 4 8 16; do $Q distinct \$n | tail -1; done; $Q copies 16 | tail -1; $Q distinct 16 | tail -1; $Q copies 8 | tail -1; $Q copies 4 | tail -1"
GL="docker logs q38-run-tf 2>&1 | grep -E 'q38 many|q38 pool|q38: prompts' | sed -n '1,2p;\$p' | cut -c1-200; docker logs q38-run-tf 2>&1 | grep -c 'q38 many: [0-9]'"
CL="docker logs q38-run-tf 2>&1 | grep -E 'q38 cert|q38 hcf' | sed -n '1p;\$p' | cut -c1-230"
ST="python3 -I ~/q38/kit/tfsteps.py"
one() { echo "L0=\$(docker logs q38-run-tf 2>&1 | wc -l); QK_HELD=$1 python3 -I ~/q38/kit/heldpar.py ${5:-1} \$O/$2.jsonl $3 ${4:-1} | tail -1; docker logs q38-run-tf 2>&1 | tail -n +\$((L0+1)) > \$O/eng_$2.log; echo '== $2'; $ST \$O/eng_$2.log | sed -n '2,3p'"; }
SETS="python3 -I ~/q38/kit/mkedit.py ~/q38/kit/sets | tail -1; $(one \$HOME/q38/kit/sets/edit.jsonl edit 1500); $(one \$HOME/q38/kit/sets/agent.jsonl agent 900)"
OFF="CERT=0 MANY=0 COPY_ROWS=16 FUSED=0 HALF=0 MTP_LAST=0"
( r3 'cd ~/q38/repo_tf && WORK=$HOME/q38/recipe_work HF_CACHE=$HOME/q38/hf bash ./recipe.sh prepare > ~/q38/recipe_work/prepare4.log 2>&1; echo prepare rc $?; ls ~/q38/recipe_work/miaai-recipe/patches | tail -1; tail -1 ~/q38/recipe_work/prepare4.log | cut -c1-200; docker images --format "{{.Repository}}:{{.Tag}} {{.CreatedSince}}" | grep tensorfold-qwen38'
  gor "$OFF" Q_off "$E MAXU=5 HELD_STRIDE=3 SKIP='prefill sparkdash qeval img needle ctx' bash ~/q38/kit/suite.sh Q_off full | tail -3; $SETS"
  gor "" Q_on "$E MAXU=5 HELD_STRIDE=1 bash ~/q38/kit/suite.sh Q_on full | tail -3; $SETS; $CL; $GL"
  r3 "$C ~/q38/runs/Q_off/ident_on.json ~/q38/runs/Q_on/ident_on.json; $C ~/q38/runs/Q_off/held.jsonl ~/q38/runs/Q_on/held.jsonl; $C ~/q38/runs/Q_off/edit.jsonl ~/q38/runs/Q_on/edit.jsonl; $C ~/q38/runs/Q_off/agent.jsonl ~/q38/runs/Q_on/agent.jsonl"
  gor "USERS16=1" Q_mu "$MU; $W 16 \$O/waves16.jsonl 256 1 8 | tail -1; $W 4 \$O/waves4.jsonl 128 5 | tail -1; python3 -I ~/q38/kit/poolt.py 150000x1+15 | cut -c1-420; $Q ctx 250000 | tail -1; $CL; $GL"
  gor "USERS16=1 COPY_ROWS=16" Q_mu16 "$MU"
  gor "$OFF PARALLEL=16 CONTEXT=65536 TF_FLASHNEXT_PRODUCT_STREAMS=16" Q_mia16 "$MU; $W 16 \$O/waves16.jsonl 256 1 8 | tail -1; $W 4 \$O/waves4.jsonl 128 5 | tail -1; $C \$O/waves16.jsonl \$HOME/q38/runs/Q_mu/waves16.jsonl; $C \$O/waves4.jsonl \$HOME/q38/runs/Q_mu/waves4.jsonl"
) > $D/q_923b.log 2>&1 & P1=$!
B="-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_CERT_PLAIN=1 -e TF_FLASHNEXT_Q38_PREFILL_MANY=256 -e TF_FLASHNEXT_Q38_WINDOW_ROWS=32 -e TF_FLASHNEXT_Q38_HCF=3 -e TF_FLASHNEXT_Q38_HCF_ST=3 -e TF_FLASHNEXT_Q38_HCF_HALF=1 -e TF_FLASHNEXT_Q38_HALF_BELOW=96 -e TF_FLASHNEXT_Q38_MTP_LAST=1"
S3="$SETS; $(one \$HOME/q38/kit/prompts.jsonl chat 256 4)"
CS="for f in edit agent chat; do $C \$HOME/q38/runs/Q2_ref/\$f.jsonl \$O/\$f.jsonl; done"
( go w2 "$B" default Q2_ref "$S3"
  go w2 "$B -e TF_FLASHNEXT_Q38_SHORT_COPY=3,7" default Q2_sc3 "$S3; $CS"
  go w2 "$B -e TF_FLASHNEXT_Q38_SHORT_COPY=5,7" default Q2_sc5 "$S3; $CS"
  go w2 "$B -e TF_FLASHNEXT_Q38_SHORT_COPY=3,15" default Q2_sc3w "$S3; $CS"
  MS="python3 -I ~/q38/kit/heldpar.py 1 \$O/held.jsonl 256 8 | tail -1; docker logs q38-run-tf 2>&1 | grep mtpstat | tail -1 | cut -c1-200"
  go w2 "$B -e TF_FLASHNEXT_Q38_MTP_LAST=0 -e TF_HOSTPROF=15200" default Q2_hp0 "$MS"
  go w2 "$B -e TF_FLASHNEXT_Q38_MTP_LAST=1 -e TF_HOSTPROF=15200" default Q2_hp1 "$MS"
  go w2 "-e TF_FLASHNEXT_Q38_CERT=1" nodraft Q2_nd "$(one \$HOME/q38/kit/prompts.jsonl held 192 8)"
  go w2 "$B" nodraft Q2_ndc "$(one \$HOME/q38/kit/prompts.jsonl held 192 8); $C \$HOME/q38/runs/Q2_nd/held.jsonl \$O/held.jsonl"
) > $D/q_3d24.log 2>&1 & P2=$!
MUE="-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_CERT_PLAIN=1 -e TF_FLASHNEXT_Q38_WIDE_ROWS=128 -e TF_FLASHNEXT_Q38_PREFILL_MANY=256 -e TF_FLASHNEXT_Q38_POOL_CTX=65536 -e TF_FLASHNEXT_Q38_HCF=3 -e TF_FLASHNEXT_Q38_HCF_ST=3 -e TF_FLASHNEXT_Q38_HCF_HALF=1 -e TF_FLASHNEXT_Q38_HALF_BELOW=96 -e TF_FLASHNEXT_Q38_MTP_LAST=1"
MB="$Q distinct 16 | tail -1; $Q distinct 16 | tail -1; $Q distinct 16 | tail -1; $Q distinct 8 | tail -1; python3 -I ~/q38/kit/heldpar.py 16 \$O/held16.jsonl 256 3 | tail -1"
( go w3 "$MUE -e TF_FLASHNEXT_Q38_WINDOW_ROWS=16" pc_16_262144 Q3_r16 "$MB"
  go w3 "$MUE -e TF_FLASHNEXT_Q38_WINDOW_ROWS=32" pc_16_262144 Q3_r32 "$MB; $C \$HOME/q38/runs/Q3_r16/held16.jsonl \$O/held16.jsonl"
  go w3 "$MUE -e TF_FLASHNEXT_Q38_WINDOW_ROWS=16" pc_16_262144 Q3_r16b "$MB; $C \$HOME/q38/runs/Q3_r16/held16.jsonl \$O/held16.jsonl"
  go w3 "$MUE -e TF_FLASHNEXT_Q38_WINDOW_ROWS=32 -e TF_FLASHNEXT_Q38_HCF_MAX=128 -e TF_FLASHNEXT_Q38_CERT=2 -e TF_FLASHNEXT_Q38_HCF_CHECK=1" pc_16_262144 Q3_chk "$W 16 \$O/waves16.jsonl 256 1 10 | tail -1; QK_HELD=\$HOME/q38/kit/sets/edit.jsonl python3 -I ~/q38/kit/heldpar.py 4 \$O/edit4.jsonl 1200 | tail -1; docker logs q38-run-tf 2>&1 | grep -E 'q38 cert: calls|q38 hcf check' | tail -2 | cut -c1-230"
  go w3 "-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_GPROF=1" nodraft Q3_gp "$Q prefill 32000 | tail -1; $Q prefill 32000 | tail -1; $Q c1 1 | tail -2; docker logs q38-run-tf 2>&1 | grep 'q38 gprof' | tail -3 | cut -c1-330"
  go w3 "$MUE -e TF_FLASHNEXT_Q38_WINDOW_ROWS=32" pc_16_262144 Q3_mix "python3 -I ~/q38/kit/poolt.py 150000x1+15 100000x3+13 200000x5+0 | cut -c1-420"
  H2="python3 -I ~/q38/kit/heldpar.py 1 \$O/held.jsonl 256 2 | tail -1; python3 -I ~/q38/kit/tfsteps.py <(docker logs q38-run-tf 2>&1) | sed -n '2,3p'"
  go w3 "-e TF_FLASHNEXT_Q38_CERT=0" default Q3_off "$H2"
  go w3 "$B" default Q3_on "$H2; $C \$HOME/q38/runs/Q3_off/held.jsonl \$O/held.jsonl"
) > $D/q_4e5c.log 2>&1 & P3=$!
wait $P1 $P2 $P3
echo "window Q work done $(date -u +%T)"
