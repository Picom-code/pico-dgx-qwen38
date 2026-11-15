# Window I (round 3, final): build kp5 (= kp4 + buffered n-gram reader for real + TF_HOSTPROF; knobs named TF_FLASHNEXT_Q38_*), engine started
# through the image entrypoint everywhere.
#   923b (final numbers): base = no q38 flags (MiaAI's default config on our binary): c1, sd, ident, held 104; then BEST: the full suite, held 104; compared.
#   3d24: n-gram lever for real: no drafts base / buffered+warm / buffered+warm+PDL / mmap+warm+PDL; drafts on: base / buffered+warm (held 104 + ident).
#   4e5c: host launch cost (TF_HOSTPROF) without and with drafts; 16 slots at 64K with BEST: users 1/4/8/16.
E='QK_MODEL=Qwen3.8-Flash-Next'; NE=TFENTRY=/opt/nvidia/nvidia_entrypoint.sh
TW="TFBIN=\$HOME/q38/tfbin/kp5 TFMODEL=\$HOME/q38/models/nvidia-fp4-local $NE"
T923="TFIMG=tensorfold-qwen38:v1.1.4-eab8582 $TW"
BEST_ENV=''; BEST_PRE=''; BEST_GATE='TFCONF=-0.5 TFDRAFTS=8'
go() { $1 "$2 bash ~/q38/kit/eng.sh $3 $4" 2>&1 | tail -1 | cut -c1-200
  $1 "export $E O=\$HOME/q38/runs/$5; mkdir -p \$O; $6; docker logs q38-run-tf > \$O/engine.log 2>&1; bash ~/q38/kit/eng.sh stop >/dev/null"; echo "$5 done $(date -u +%T)"; }
H='python3 -I ~/q38/kit/qk.py held'; C='python3 -I ~/q38/kit/qk.py cmp'; W='bash ~/q38/kit/warm.sh'
BUF='-e TF_FLASHNEXT_Q38_NGRAM_BUFFERED=1'; MM='-e TF_FLASHNEXT_GATHER_READ=mmap'; PDL='-e TF_FLASHNEXT_Q38_PDL=1'
nd() { go $1 "$TW TFENV='$3'" tfb nodraft $2 "$5 $H \$O/held6.jsonl 6 | tail -1; [ -n '$4' ] && $C \$HOME/q38/runs/$4/held6.jsonl \$O/held6.jsonl"; }
dr() { local x="$5 QK_HELD_STRIDE=3 $H \$O/held.jsonl | tail -4; python3 -I ~/q38/kit/qk.py ident \$O/ident_on.json | tail -1; python3 -I ~/q38/kit/tfsteps.py <(docker logs q38-run-tf 2>&1) | sed -n 2,3p"
  [ -n "$4" ] && x="$x; $C \$HOME/q38/runs/$4/held.jsonl \$O/held.jsonl; $C \$HOME/q38/runs/$4/ident_on.json \$O/ident_on.json"
  go $1 "$3" tfb default $2 "$x"; }
( go r3 "$T923" tfb default I_base "$E MAXU=5 HELD_STRIDE=3 SKIP='multi prefill sparkdash qeval img needle ctx' bash ~/q38/kit/suite.sh I_base full | tail -3"
  go r3 "$T923 TFENV='$BEST_ENV' $BEST_GATE" tfb default I_best "$BEST_PRE $E MAXU=5 HELD_STRIDE=3 bash ~/q38/kit/suite.sh I_best full | tail -3"
  r3 "$C ~/q38/runs/I_base/ident_on.json ~/q38/runs/I_best/ident_on.json; $C ~/q38/runs/I_base/held.jsonl ~/q38/runs/I_best/held.jsonl; $C ~/q38/runs/H_ent/held.jsonl ~/q38/runs/I_best/held.jsonl"
) > $D/i_923b.log 2>&1 & P1=$!
( nd w2 I2_nd_base "" H2_ent ""
  nd w2 I2_nd_buf "$BUF" I2_nd_base "$W;"
  nd w2 I2_nd_buf_pdl "$BUF $PDL" I2_nd_base "$W;"
  nd w2 I2_nd_mmap_pdl "$MM $PDL" I2_nd_base "$W;"
  dr w2 I2_dr_base "$TW" "" ""
  dr w2 I2_dr_buf "$TW TFENV='$BUF'" I2_dr_base "$W;"
  dr w2 I2_dr_pdl "$TW TFENV='$PDL'" I2_dr_base ""
) > $D/i_3d24.log 2>&1 & P2=$!
( go w3 "$TW TFENV='-e TF_HOSTPROF=13700'" tfb nodraft I3_hp_nd "$H \$O/held3.jsonl 3 | tail -1; docker logs q38-run-tf 2>&1 | grep hostprof | tail -40 | awk '{d+=\$5; w+=\$7; n+=\$3} END {print \"hostprof no drafts: launches\", n, \"in-driver ms\", d, \"wall ms\", w, \"us per launch\", 1000*d/n}'"
  go w3 "$TW TFENV='-e TF_HOSTPROF=13700'" tfb default I3_hp_dr "QK_HELD_STRIDE=12 $H \$O/held.jsonl | tail -1; docker logs q38-run-tf 2>&1 | grep hostprof | tail -200 | awk '{d+=\$5; w+=\$7; n+=\$3} END {print \"hostprof drafts: launches\", n, \"in-driver ms\", d, \"wall ms\", w, \"us per launch\", 1000*d/n}'"
  go w3 "$TW TFENV='$BEST_ENV' $BEST_GATE" tfb p16 I3_p16_best "$BEST_PRE for n in 1 4 8 16; do python3 -I ~/q38/kit/qk.py distinct \$n | tail -1; done; python3 -I ~/q38/kit/qk.py copies 16 | tail -1; python3 -I ~/q38/kit/qk.py distinct 16 | tail -1"
  go w3 "$TW" tfb p16 I3_p16_base "for n in 1 4 8 16; do python3 -I ~/q38/kit/qk.py distinct \$n | tail -1; done; python3 -I ~/q38/kit/qk.py copies 16 | tail -1; python3 -I ~/q38/kit/qk.py distinct 16 | tail -1"
) > $D/i_4e5c.log 2>&1 & P3=$!
wait $P1 $P2 $P3
echo "window I work done $(date -u +%T)"
