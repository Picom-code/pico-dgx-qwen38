# Window O (round 5, final): build kp17 = patches 0101-0106.
#   923b (final numbers, DOCUMENTED path: recipe.sh start on the image recipe.sh prepare built, vision on): levers off -> stack (full suite) ->
#        USERS16 profile -> MiaAI configuration with 16 x 64K slots.
#   4e5c: lever 1 identity (prompts batched vs not: waves of 16 / 8 / 4 at 16 x 64K and waves of 5 at the default 5 x 262K; texts + first-score hashes),
#        then lever 3 with the copy-width rule (edit / agent / chat, 16 vs 32 rows), then a CERT=2 checked run of the whole stack with 16 in flight.
#   3d24: the whole many-user stack vs everything off with 16 in flight (texts), stack benches on a second box, lever 3 under 4 in flight, mixed sizes.
E='QK_MODEL=Qwen3.8-Flash-Next'; NE=TFENTRY=/opt/nvidia/nvidia_entrypoint.sh
TW="TFBIN=\$HOME/q38/tfbin/kp17 TFMODEL=\$HOME/q38/models/nvidia-fp4-local $NE"
go() { $1 "$2 bash ~/q38/kit/eng.sh $3 $4" 2>&1 | tail -1 | cut -c1-200
  $1 "export $E O=\$HOME/q38/runs/$5; mkdir -p \$O; $6; docker logs q38-run-tf > \$O/engine.log 2>&1; bash ~/q38/kit/eng.sh stop >/dev/null"; echo "$5 done $(date -u +%T)"; }
gor() { r3 "bash ~/q38/kit/rs.sh $1" 2>&1 | tail -1 | cut -c1-300
  r3 "export $E O=\$HOME/q38/runs/$2; mkdir -p \$O; $3; docker logs q38-run-tf > \$O/engine.log 2>&1; bash ~/q38/kit/eng.sh stop >/dev/null"; echo "$2 done $(date -u +%T)"; }
C='python3 -I ~/q38/kit/qk.py cmp'; Q='python3 -I ~/q38/kit/qk.py'; W='python3 -I ~/q38/kit/waves.py'
MU="for n in 1 4 8 16; do $Q distinct \$n | tail -1; done; $Q copies 16 | tail -1; $Q distinct 16 | tail -1; $Q copies 8 | tail -1; $Q copies 4 | tail -1; python3 -I ~/q38/kit/mustep.py <(docker logs q38-run-tf 2>&1) 2>/dev/null | tail -7"
GL="docker logs q38-run-tf 2>&1 | grep -E 'q38 many|q38 pool|q38: prompts' | sed -n '1,3p;\$p' | cut -c1-200; docker logs q38-run-tf 2>&1 | grep -c 'q38 many: [0-9]'"
CL="docker logs q38-run-tf 2>&1 | grep 'q38 cert' | sed -n '1p;\$p'"
LH="docker logs q38-run-tf > \$O/engine.log 2>&1; python3 -I ~/q38/kit/lhcmp.py"
ST="python3 -I ~/q38/kit/tfsteps.py"
one() { echo "L0=\$(docker logs q38-run-tf 2>&1 | wc -l); QK_HELD=$1 python3 -I ~/q38/kit/heldpar.py ${5:-1} \$O/$2.jsonl $3 ${4:-1} | tail -1; docker logs q38-run-tf 2>&1 | tail -n +\$((L0+1)) > \$O/eng_$2.log; echo '== $2'; $ST \$O/eng_$2.log | sed -n '2,3p'"; }
SETS="python3 -I ~/q38/kit/mkedit.py ~/q38/kit/sets | tail -1; $(one \$HOME/q38/kit/sets/edit.jsonl edit 1500); $(one \$HOME/q38/kit/sets/agent.jsonl agent 900)"
OFF="CERT=0 MANY=0 COPY_ROWS=16"
( r3 'cd ~/q38/repo_tf && WORK=$HOME/q38/recipe_work HF_CACHE=$HOME/q38/hf bash ./recipe.sh prepare > ~/q38/recipe_work/prepare3.log 2>&1; echo prepare rc $?; ls ~/q38/recipe_work/miaai-recipe/patches | tail -1; tail -2 ~/q38/recipe_work/prepare3.log | cut -c1-200; docker images --format "{{.Repository}}:{{.Tag}} {{.CreatedSince}}" | grep tensorfold-qwen38'
  gor "$OFF" O_off "$E MAXU=5 HELD_STRIDE=3 SKIP='prefill sparkdash qeval img needle ctx' bash ~/q38/kit/suite.sh O_off full | tail -3; $SETS"
  gor "" O_on "$E MAXU=5 HELD_STRIDE=1 bash ~/q38/kit/suite.sh O_on full | tail -3; $SETS; $CL; $GL"
  r3 "$C ~/q38/runs/O_off/ident_on.json ~/q38/runs/O_on/ident_on.json; $C ~/q38/runs/O_off/held.jsonl ~/q38/runs/O_on/held.jsonl; $C ~/q38/runs/O_off/edit.jsonl ~/q38/runs/O_on/edit.jsonl; $C ~/q38/runs/O_off/agent.jsonl ~/q38/runs/O_on/agent.jsonl"
  gor "USERS16=1" O_mu "$MU; $W 16 \$O/waves16.jsonl 256 1 8 | tail -1; $W 4 \$O/waves4.jsonl 128 5 | tail -1; python3 -I ~/q38/kit/poolt.py 150000x1+15 | cut -c1-420; $Q ctx 250000 | tail -1; $CL; $GL"
  gor "$OFF PARALLEL=16 CONTEXT=65536 TF_FLASHNEXT_PRODUCT_STREAMS=16" O_mia16 "$MU; $W 16 \$O/waves16.jsonl 256 1 8 | tail -1; $W 4 \$O/waves4.jsonl 128 5 | tail -1; $C \$O/waves16.jsonl \$HOME/q38/runs/O_mu/waves16.jsonl; $C \$O/waves4.jsonl \$HOME/q38/runs/O_mu/waves4.jsonl"
) > $D/o_923b.log 2>&1 & P1=$!
BASE="-e TF_FLASHNEXT_Q38_WIDE_ROWS=128 -e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_LHASH=1"
WV="$W 16 \$O/waves16.jsonl 256 1 12 | tail -1; $W 8 \$O/waves8.jsonl 128 2 | tail -1; $W 4 \$O/waves4.jsonl 128 5 | tail -1"
CW="for f in waves16 waves8 waves4; do $C \$HOME/q38/runs/O3_off/\$f.jsonl \$O/\$f.jsonl; done"
STACK="-e TF_FLASHNEXT_Q38_WIDE_ROWS=128 -e TF_FLASHNEXT_Q38_POOL_CTX=65536 -e TF_FLASHNEXT_Q38_PREFILL_MANY=256 -e TF_FLASHNEXT_Q38_WINDOW_ROWS=32 -e TF_FLASHNEXT_Q38_CERT_PLAIN=1"
( go w3 "$TW TFENV='$BASE'" tfb p16 O3_off "$Q distinct 16 | tail -1; $Q copies 16 | tail -1; $Q distinct 16 | tail -1; $WV"
  go w3 "$TW TFENV='$BASE -e TF_FLASHNEXT_Q38_PREFILL_MANY=256'" tfb p16 O3_many "$Q distinct 16 | tail -1; $Q copies 16 | tail -1; $Q distinct 16 | tail -1; $Q distinct 8 | tail -1; $Q distinct 4 | tail -1; $WV; $GL; $CW; $LH \$HOME/q38/runs/O3_off/engine.log \$O/engine.log"
  go w3 "$TW TFENV='-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_LHASH=1'" tfb default O3_d_off "$W 5 \$O/waves5.jsonl 192 2 | tail -1; $SETS; $(one \$HOME/q38/kit/prompts.jsonl chat 384 8)"
  go w3 "$TW TFENV='-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_LHASH=1 -e TF_FLASHNEXT_Q38_PREFILL_MANY=256 -e TF_FLASHNEXT_Q38_WINDOW_ROWS=32'" tfb default O3_d_on "$W 5 \$O/waves5.jsonl 192 2 | tail -1; $SETS; $(one \$HOME/q38/kit/prompts.jsonl chat 384 8); $GL; for f in waves5 edit agent chat; do $C \$HOME/q38/runs/O3_d_off/\$f.jsonl \$O/\$f.jsonl; done; $LH \$HOME/q38/runs/O3_d_off/engine.log \$O/engine.log"
  go w3 "$TW TFENV='$STACK -e TF_FLASHNEXT_Q38_CERT=2'" tfb pc_16_262144 O3_chk "$W 16 \$O/waves16.jsonl 256 1 20 | tail -1; QK_HELD=\$HOME/q38/kit/sets/edit.jsonl python3 -I ~/q38/kit/heldpar.py 4 \$O/edit4.jsonl 1200 | tail -1; docker logs q38-run-tf 2>&1 | grep 'q38 cert' | tail -2; $GL"
) > $D/o_4e5c.log 2>&1 & P3=$!
HP="python3 -I ~/q38/kit/heldpar.py 16 \$O/held16.jsonl 384 3 1 | tail -1"
( go w2 "$TW TFENV='-e TF_FLASHNEXT_Q38_CERT=0'" tfb p16 O2_alloff "$MU; $HP; $(one \$HOME/q38/kit/sets/edit.jsonl edit4 1200 1 4)"
  go w2 "$TW TFENV='$STACK -e TF_FLASHNEXT_Q38_CERT=1'" tfb pc_16_262144 O2_stack "$MU; $HP; $(one \$HOME/q38/kit/sets/edit.jsonl edit4 1200 1 4); $GL; $CL; $C \$HOME/q38/runs/O2_alloff/held16.jsonl \$O/held16.jsonl; $C \$HOME/q38/runs/O2_alloff/edit4.jsonl \$O/edit4.jsonl; python3 -I ~/q38/kit/poolt.py 150000x1+15 100000x3+13 | cut -c1-420"
  go w2 "$TW TFENV='-e TF_FLASHNEXT_Q38_CERT=1'" tfb nodraft O2_nd "$(one \$HOME/q38/kit/prompts.jsonl held 256 6)"
  go w2 "$TW TFENV='-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_CERT_PLAIN=1'" tfb nodraft O2_ndc "$(one \$HOME/q38/kit/prompts.jsonl held 256 6); $C \$HOME/q38/runs/O2_nd/held.jsonl \$O/held.jsonl"
) > $D/o_3d24.log 2>&1 & P2=$!
wait $P1 $P2 $P3
echo "window O work done $(date -u +%T)"
