# Window K (round 4): build kp9 (= kp6 + certified LM head + launch-delay knob + sync call sites).
#   923b: lever 4: base vs TF_FLASHNEXT_Q38_CERT=1 (held 107 + ident, texts compared), then CERT=2 (checked against the whole head, 5 in flight).
#   4e5c: lever 1: launch-delay sensitivity (0 / 2000 / 4000 ns before every launch), hostprof with sync call sites; then 16 x 96K slots.
#   3d24: lever 7: 16 slots at 64K unpriced vs TF_FLASHNEXT_ROUND_MS priced (cross-session draft allocation), 12 x 128K.
E='QK_MODEL=Qwen3.8-Flash-Next'; NE=TFENTRY=/opt/nvidia/nvidia_entrypoint.sh
TW="TFBIN=\$HOME/q38/tfbin/kp9 TFMODEL=\$HOME/q38/models/nvidia-fp4-local $NE"
T923="TFIMG=tensorfold-qwen38:v1.1.4-eab8582 $TW"
go() { $1 "$2 bash ~/q38/kit/eng.sh $3 $4" 2>&1 | tail -1 | cut -c1-200
  $1 "export $E O=\$HOME/q38/runs/$5; mkdir -p \$O; $6; docker logs q38-run-tf > \$O/engine.log 2>&1; bash ~/q38/kit/eng.sh stop >/dev/null"; echo "$5 done $(date -u +%T)"; }
H='python3 -I ~/q38/kit/qk.py held'; C='python3 -I ~/q38/kit/qk.py cmp'; Q='python3 -I ~/q38/kit/qk.py'
HPA="docker logs q38-run-tf 2>&1 | grep 'hostprof launches' | awk '{n+=\$3; d+=\$5; w+=\$7; s+=\$9; sm+=\$11} END {print \"hostprof: launches\", n, \"in-driver ms\", d, \"wall ms\", w, \"us/launch\", 1000*d/n, \"syncs\", s, \"sync ms\", sm}'"
CL="docker logs q38-run-tf 2>&1 | grep 'q38 cert' | sed -n '1p;\$p'"
# dr HOSTFN LABEL ENVPREFIX BASELABEL STRIDE IDENT(1|0)
dr() { local x="QK_HELD_STRIDE=$5 $H \$O/held.jsonl | tail -4; python3 -I ~/q38/kit/tfsteps.py <(docker logs q38-run-tf 2>&1) | sed -n 2,3p; $CL"
  [ "$6" = 1 ] && x="$x; $Q ident \$O/ident_on.json | tail -1"
  [ -n "$4" ] && x="$x; $C \$HOME/q38/runs/$4/held.jsonl \$O/held.jsonl"
  [ -n "$4" ] && [ "$6" = 1 ] && x="$x; $C \$HOME/q38/runs/$4/ident_on.json \$O/ident_on.json"
  go $1 "$3" tfb default $2 "$x"; }
MU="for n in 1 4 8 16; do $Q distinct \$n | tail -1; done; $Q copies 16 | tail -1; $Q distinct 16 | tail -1; python3 -I ~/q38/kit/mustep.py <(docker logs q38-run-tf 2>&1) 2>/dev/null | tail -6"
( dr r3 K_base "$T923" "" 3 1
  dr r3 K_cert "$T923 TFENV='-e TF_FLASHNEXT_Q38_CERT=1'" K_base 3 1
  go r3 "$T923 TFENV='-e TF_FLASHNEXT_Q38_CERT=2'" tfb default K_cert2 "python3 -I ~/q38/kit/heldpar.py 5 \$O/held5.jsonl 384 1 1 | tail -1; docker logs q38-run-tf 2>&1 | grep 'q38 cert' | tail -3"
) > $D/k_923b.log 2>&1 & P1=$!
( dr w3 K3_d0 "$TW" "" 6 0
  dr w3 K3_d2000 "$TW TFENV='-e TF_FLASHNEXT_Q38_LAUNCH_DELAY_NS=2000'" K3_d0 6 0
  dr w3 K3_d4000 "$TW TFENV='-e TF_FLASHNEXT_Q38_LAUNCH_DELAY_NS=4000'" K3_d0 6 0
  go w3 "$TW TFENV='-e TF_HOSTPROF=15200 -e TF_HOSTPROF_K=1'" tfb default K3_hp "QK_HELD_STRIDE=12 $H \$O/held.jsonl | tail -1; $HPA"
  go w3 "$TW" tfb pc_16_98304 K3_p16_96k "$MU"
) > $D/k_4e5c.log 2>&1 & P3=$!
( go w2 "$TW" tfb p16 K2_p16_base "$MU"
  go w2 "$TW TFENV='-e TF_FLASHNEXT_ROUND_MS=46+2.45'" tfb p16 K2_p16_r46 "$MU"
  go w2 "$TW TFENV='-e TF_FLASHNEXT_ROUND_MS=40+3.2'" tfb p16 K2_p16_r40 "$MU"
  go w2 "$TW" tfb pc_12_131072 K2_p12_128k "for n in 1 4 8 12; do $Q distinct \$n | tail -1; done; $Q copies 12 | tail -1"
) > $D/k_3d24.log 2>&1 & P2=$!
wait $P1 $P2 $P3
echo "window K work done $(date -u +%T)"
