# Window G (round 3): exact kernel knobs of build kp2 (kp1 + TF_Q38_EXP_D / TF_Q38_EXP_D1 / TF_Q38_QMMF_ST / TF_Q38_QMMF_ST_CL, all off unless set).
#   4e5c: qmmf microbench (no model), then no-draft runs: base, expert prefetch depths, TF_NVFP4_DECODE_WARPS=1.
#   3d24: no-draft runs: base, qmmf pipeline stages 8 / 6 / 12 / 3, cluster-only 12.
#   923b: launcher A-B-A (tf / tfb / tf, drafts on, held 104), then gate 0.5 max 8 and 0.4 max 8 (held 104 + ident, compared with tfb base).
E='QK_MODEL=Qwen3.8-Flash-Next'
TW='TFBIN=$HOME/q38/tfbin/kp2 TFMODEL=$HOME/q38/models/nvidia-fp4-local'
T923="TFIMG=tensorfold-qwen38:v1.1.4-eab8582 $TW"
# go RUNNER PREFIX ENGINE VARIANT LABEL CMDS
go() { $1 "$2 bash ~/q38/kit/eng.sh $3 $4" 2>&1 | tail -1 | cut -c1-200
  $1 "export $E O=\$HOME/q38/runs/$5; mkdir -p \$O; $6; docker logs q38-run-tf > \$O/engine.log 2>&1; bash ~/q38/kit/eng.sh stop >/dev/null"; echo "$5 done $(date -u +%T)"; }
H='python3 -I ~/q38/kit/qk.py held'; C='python3 -I ~/q38/kit/qk.py cmp'
nd() { # nd RUNNER LABEL ENVS BASELABEL : no-draft, 6 held prompts, compare texts with the box's base
  go $1 "$TW TFENV='$3'" tfb nodraft $2 "$H \$O/held6.jsonl 6 | tail -1; [ -n '$4' ] && $C \$HOME/q38/runs/$4/held6.jsonl \$O/held6.jsonl"; }
dr() { # dr LABEL ENGINE PREFIX CMPBASE : drafts on, held 104 (+ ident when CMPBASE given)
  local x="QK_HELD_STRIDE=3 $H \$O/held.jsonl | tail -4"
  [ -n "$4" ] && x="$x; python3 -I ~/q38/kit/qk.py ident \$O/ident_on.json | tail -1; $C \$HOME/q38/runs/$4/held.jsonl \$O/held.jsonl; $C \$HOME/q38/runs/$4/ident_on.json \$O/ident_on.json"
  [ "$4" = self ] && x="QK_HELD_STRIDE=3 $H \$O/held.jsonl | tail -4; python3 -I ~/q38/kit/qk.py ident \$O/ident_on.json | tail -1"
  go r3 "$3" $2 default $1 "$x"; }
( dr G_tf_a tf "" ""
  dr G_tfb "tfb" "$T923" self
  dr G_tf_b tf "" ""
  dr G_g50_8 tfb "$T923 TFCONF=-0.5 TFDRAFTS=8" G_tfb
  dr G_g40_8 tfb "$T923 TFCONF=-0.4 TFDRAFTS=8" G_tfb
) > $D/g_923b.log 2>&1 & P1=$!
( nd w2 G2_base "" ""
  nd w2 G2_st8 "-e TF_Q38_QMMF_ST=8" G2_base
  nd w2 G2_st6 "-e TF_Q38_QMMF_ST=6" G2_base
  nd w2 G2_st12 "-e TF_Q38_QMMF_ST=12" G2_base
  nd w2 G2_cl12 "-e TF_Q38_QMMF_ST_CL=12" G2_base
  nd w2 G2_st3 "-e TF_Q38_QMMF_ST=3" G2_base
) > $D/g_3d24.log 2>&1 & P2=$!
( w3 'bash ~/q38/mb/mbrun.sh mb 5 > ~/q38/runs/G3_mb.txt 2>&1; tail -3 ~/q38/runs/G3_mb.txt'
  nd w3 G3_base "" ""
  nd w3 G3_d4 "-e TF_Q38_EXP_D=4 -e TF_Q38_EXP_D1=8" G3_base
  nd w3 G3_d6 "-e TF_Q38_EXP_D=6 -e TF_Q38_EXP_D1=12" G3_base
  nd w3 G3_d3 "-e TF_Q38_EXP_D=3 -e TF_Q38_EXP_D1=4" G3_base
  nd w3 G3_w1 "-e TF_NVFP4_DECODE_WARPS=1" G3_base
  nd w3 G3_d8 "-e TF_Q38_EXP_D=8 -e TF_Q38_EXP_D1=8" G3_base
) > $D/g_4e5c.log 2>&1 & P3=$!
wait $P1 $P2 $P3
echo "window G work done $(date -u +%T)"
