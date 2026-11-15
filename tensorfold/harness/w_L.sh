# Window L (round 4): build kp11 (= kp9 + TF_FLASHNEXT_Q38_WIDE_ROWS + certified head for wide passes).
#   923b (final numbers): levers off (c1, sparkDash method, ident, held-320) -> stack on (certified head; FULL suite with --vision) -> texts compared ->
#         many users with the stack (16 slots, wide rows from head ~/q38/wide.env, written once the sweep on 3d24 reports) -> MiaAI default many users.
#   3d24: lever 7 sweep: 16 slots at 64K, pass rows 128 / 64 / 192 / 96 (1/4/8/16 users + 107 held prompts 16 in flight, texts compared), then a checked run.
#   4e5c: wide 128 without / with the certified head (texts compared), then TF_FLASHNEXT_Q38_CERT=2 (every call checked against the whole head) 16 in flight.
E='QK_MODEL=Qwen3.8-Flash-Next'; NE=TFENTRY=/opt/nvidia/nvidia_entrypoint.sh
TW="TFBIN=\$HOME/q38/tfbin/kp11 TFMODEL=\$HOME/q38/models/nvidia-fp4-local $NE"
T923="TFIMG=tensorfold-qwen38:v1.1.4-eab8582 TFVISION=1 $TW"
go() { $1 "$2 bash ~/q38/kit/eng.sh $3 $4" 2>&1 | tail -1 | cut -c1-200
  $1 "export $E O=\$HOME/q38/runs/$5; mkdir -p \$O; $6; docker logs q38-run-tf > \$O/engine.log 2>&1; bash ~/q38/kit/eng.sh stop >/dev/null"; echo "$5 done $(date -u +%T)"; }
C='python3 -I ~/q38/kit/qk.py cmp'; Q='python3 -I ~/q38/kit/qk.py'
CL="docker logs q38-run-tf 2>&1 | grep 'q38 cert' | sed -n '1p;\$p'"
MU="for n in 1 4 8 16; do $Q distinct \$n | tail -1; done; $Q copies 16 | tail -1; $Q distinct 16 | tail -1; $Q copies 8 | tail -1; $Q copies 4 | tail -1; python3 -I ~/q38/kit/mustep.py <(docker logs q38-run-tf 2>&1) 2>/dev/null | tail -7"
HP="python3 -I ~/q38/kit/heldpar.py 16 \$O/held16.jsonl 384 3 1 | tail -1"
# mu HOSTFN LABEL VARIANT ENV BASELABEL
mu() { local x="$MU; $HP; $CL"; [ -n "$5" ] && x="$x; $C \$HOME/q38/runs/$5/held16.jsonl \$O/held16.jsonl"; go $1 "$TW TFENV='$4'" tfb $3 $2 "$x"; }
W() { echo "-e TF_FLASHNEXT_Q38_WIDE_ROWS=$1"; }
# 923b runs the DOCUMENTED path: recipe.sh start on the image recipe.sh prepare built (kit/rs.sh), vision on as MiaAI ships it
gor() { r3 "bash ~/q38/kit/rs.sh $1" 2>&1 | tail -1 | cut -c1-300
  r3 "export $E O=\$HOME/q38/runs/$2; mkdir -p \$O; $3; docker logs q38-run-tf > \$O/engine.log 2>&1; bash ~/q38/kit/eng.sh stop >/dev/null"; echo "$2 done $(date -u +%T)"; }
( r3 'tail -2 ~/q38/recipe_work/prepare.log | cut -c1-200; docker images --format "{{.Repository}}:{{.Tag}} {{.CreatedSince}}" | grep tensorfold-qwen38'
  gor "CERT=0" L_off "$E MAXU=5 HELD_STRIDE=1 SKIP='multi prefill sparkdash qeval img needle ctx' bash ~/q38/kit/suite.sh L_off full | tail -3"
  gor "" L_on "$E MAXU=5 HELD_STRIDE=1 bash ~/q38/kit/suite.sh L_on full | tail -3; $CL"
  r3 "$C ~/q38/runs/L_off/ident_on.json ~/q38/runs/L_on/ident_on.json; $C ~/q38/runs/L_off/held.jsonl ~/q38/runs/L_on/held.jsonl"
  WR=$(cat $HOME/q38/wide.env 2>/dev/null || echo 128)
  gor "USERS16=1 WIDE_ROWS=$WR" L_mu_on "echo wide rows $WR; $MU; $CL"
  gor "CERT=0" L_mu_mia "for n in 1 4 5 8 16; do $Q distinct \$n | tail -1; done; $Q copies 4 | tail -1; $Q copies 5 | tail -1; $Q copies 8 | tail -1; $Q copies 16 | tail -1"
  gor "CERT=0 USERS16=1 WIDE_ROWS=64" L_mu_p16 "$MU"
) > $D/l_923b.log 2>&1 & P1=$!
( mu w2 L2_w128 p16 "$(W 128)" ""
  mu w2 L2_w64 p16 "" L2_w128
  mu w2 L2_w192 p16 "$(W 192)" L2_w64
  mu w2 L2_w96 p16 "$(W 96)" L2_w64
  go w2 "$TW TFENV='-e TF_FLASHNEXT_Q38_CERT=2'" tfb default L2_cert2 "python3 -I ~/q38/kit/heldpar.py 5 \$O/held5.jsonl 384 1 2 | tail -1; docker logs q38-run-tf 2>&1 | grep 'q38 cert' | tail -2"
) > $D/l_3d24.log 2>&1 & P2=$!
( mu w3 L3_w128 p16 "$(W 128)" ""
  mu w3 L3_w128c p16 "$(W 128) -e TF_FLASHNEXT_Q38_CERT=1" L3_w128
  go w3 "$TW TFENV='$(W 128) -e TF_FLASHNEXT_Q38_CERT=2'" tfb p16 L3_cert2 "python3 -I ~/q38/kit/heldpar.py 16 \$O/held16.jsonl 384 1 4 | tail -1; docker logs q38-run-tf 2>&1 | grep 'q38 cert' | tail -2"
  mu w3 L3_w128_80k pc_16_81920 "$(W 128)" ""
) > $D/l_4e5c.log 2>&1 & P3=$!
wait $P1 $P2 $P3
echo "window L work done $(date -u +%T)"
