# Window H (round 3): (kp4 = kp3 + TF_Q38_PDL: programmatic dependent launch on the decode step's kernels)
# Window H (round 3): build kp3 = kp2 + TF_Q38_NGRAM_BUFFERED (n-gram rows through the page cache) + PDL-ready qmmf prologue (inactive).
#   All lanes start the engine through the image's own entrypoint (TFENTRY), the difference found between launchers tf and tfb.
#   923b: drafts on, held 104 + ident: entry base; + buffered n-gram + warm; + mmap n-gram + warm; + gate 0.5 / max 8 on top of buffered.
#   3d24: no drafts, 6 held: entry base; + buffered + warm; + mmap + warm; buffered without warm.
#   4e5c: drafts on with TF_FLASHNEXT_GATHER_LOG=1: entry base, + buffered + warm (join wait in drafted rounds); then 16 slots at 64K with buffered + warm.
E='QK_MODEL=Qwen3.8-Flash-Next'; NE=TFENTRY=/opt/nvidia/nvidia_entrypoint.sh
TW="TFBIN=\$HOME/q38/tfbin/kp3 TFMODEL=\$HOME/q38/models/nvidia-fp4-local $NE"
TW4="TFBIN=\$HOME/q38/tfbin/kp4 TFMODEL=\$HOME/q38/models/nvidia-fp4-local $NE"   # kp4 = kp3 + TF_Q38_PDL
T923="TFIMG=tensorfold-qwen38:v1.1.4-eab8582 $TW"
go() { $1 "$2 bash ~/q38/kit/eng.sh $3 $4" 2>&1 | tail -1 | cut -c1-200
  $1 "export $E O=\$HOME/q38/runs/$5; mkdir -p \$O; $6; docker logs q38-run-tf > \$O/engine.log 2>&1; bash ~/q38/kit/eng.sh stop >/dev/null"; echo "$5 done $(date -u +%T)"; }
H='python3 -I ~/q38/kit/qk.py held'; C='python3 -I ~/q38/kit/qk.py cmp'; W='bash ~/q38/kit/warm.sh'
BUF='-e TF_Q38_NGRAM_BUFFERED=1'; MM='-e TF_FLASHNEXT_GATHER_READ=mmap'
nd() { # nd RUNNER LABEL ENVS BASELABEL PRE
  go $1 "$TW TFENV='$3'" tfb nodraft $2 "$5 $H \$O/held6.jsonl 6 | tail -1; [ -n '$4' ] && $C \$HOME/q38/runs/$4/held6.jsonl \$O/held6.jsonl"; }
dr() { # dr RUNNER LABEL PREFIX BASELABEL PRE [EXTRA]
  local x="$5 QK_HELD_STRIDE=3 $H \$O/held.jsonl | tail -4; python3 -I ~/q38/kit/qk.py ident \$O/ident_on.json | tail -1; python3 -I ~/q38/kit/tfsteps.py <(docker logs q38-run-tf 2>&1) | sed -n 2,3p"
  [ -n "$4" ] && x="$x; $C \$HOME/q38/runs/$4/held.jsonl \$O/held.jsonl; $C \$HOME/q38/runs/$4/ident_on.json \$O/ident_on.json"
  go $1 "$3" tfb default $2 "$x $6"; }
( dr r3 H_ent "$T923" G_tfb ""
  dr r3 H_buf "$T923 TFENV='$BUF'" H_ent "$W;"
  dr r3 H_mmap "$T923 TFENV='$MM'" H_ent "$W;"
  dr r3 H_buf_g50 "$T923 TFENV='$BUF' TFCONF=-0.5 TFDRAFTS=8" H_ent "$W;"
) > $D/h_923b.log 2>&1 & P1=$!
( nd w2 H2_ent "" G2_base ""
  nd w2 H2_buf "$BUF" H2_ent "$W;"
  nd w2 H2_mmap "$MM" H2_ent "$W;"
  nd w2 H2_buf_cold "$BUF" H2_ent ""
  TW="$TW4"
  nd w2 H2_k4 "" H2_ent ""
  nd w2 H2_k4_pdl "-e TF_Q38_PDL=1" H2_ent ""
  nd w2 H2_k4_pdl_buf "-e TF_Q38_PDL=1 $BUF" H2_ent "$W;"
) > $D/h_3d24.log 2>&1 & P2=$!
( GL='-e TF_FLASHNEXT_GATHER_LOG=1'
  dr w3 H3_ent "$TW TFENV='$GL'" "" ""
  dr w3 H3_buf "$TW TFENV='$GL $BUF'" H3_ent "$W;"
  dr w3 H3_k4_pdl "$TW4 TFENV='-e TF_Q38_PDL=1'" H3_ent ""
  go w3 "$TW TFENV='$BUF'" tfb p16 H3_p16 "$W; for n in 1 4 8 16; do python3 -I ~/q38/kit/qk.py distinct \$n | tail -1; done; python3 -I ~/q38/kit/qk.py copies 16 | tail -1; python3 -I ~/q38/kit/qk.py distinct 16 | tail -1"
) > $D/h_4e5c.log 2>&1 & P3=$!
wait $P1 $P2 $P3
echo "window H work done $(date -u +%T)"
