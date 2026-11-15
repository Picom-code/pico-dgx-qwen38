# Window S (round 7): kp21, no new engine build. Anatomy of a 16-user round (what is outside the verify pass), native FP decode microbench, gate re-fit after the stack.
#   4e5c: microbench with the native E2M1 / E4M3 decodes; 16 users under the host profiler (sync call sites, MTP step timer); 16 users GPU time by group.
#   3d24: per-kernel GPU time (TF_KPROF, serialised launches): one user with drafts, then 16 users.
#   923b: one user, draft gate re-fit now that a verified draft costs 3.7 ms instead of 4.25: 0.4/8, 0.35/8, 0.3/6, 0.5/8 against 0.4/6; reference first and last.
E='QK_MODEL=Qwen3.8-Flash-Next'; NE=TFENTRY=/opt/nvidia/nvidia_entrypoint.sh
KP=${KP:-kp21}
TW="TFBIN=\$HOME/q38/tfbin/$KP TFMODEL=\$HOME/q38/models/nvidia-fp4-local $NE"
go() { $1 "$TW TFENV='$2' bash ~/q38/kit/eng.sh tfb $3" 2>&1 | tail -1 | cut -c1-200
  $1 "export $E O=\$HOME/q38/runs/$4; mkdir -p \$O; $5; docker logs q38-run-tf > \$O/engine.log 2>&1; bash ~/q38/kit/eng.sh stop >/dev/null"; echo "$4 done $(date -u +%T)"; }
C='python3 -I ~/q38/kit/qk.py cmp'; Q='python3 -I ~/q38/kit/qk.py'
ST="python3 -I ~/q38/kit/tfsteps.py"
B="-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_CERT_PLAIN=1 -e TF_FLASHNEXT_Q38_PREFILL_MANY=256 -e TF_FLASHNEXT_Q38_WINDOW_ROWS=32 -e TF_FLASHNEXT_Q38_HCF=3 -e TF_FLASHNEXT_Q38_HCF_ST=3 -e TF_FLASHNEXT_Q38_HCF_HALF=1 -e TF_FLASHNEXT_Q38_HALF_BELOW=96 -e TF_FLASHNEXT_Q38_MTP_LAST=1"
MUE="$B -e TF_FLASHNEXT_Q38_WIDE_ROWS=128 -e TF_FLASHNEXT_Q38_POOL_CTX=65536"
H1="python3 -I ~/q38/kit/heldpar.py 1 \$O/held.jsonl 256 3 | tail -1; $ST <(docker logs q38-run-tf 2>&1) | sed -n '2,3p'"
CR="$C \$HOME/q38/runs/S_ref/held.jsonl \$O/held.jsonl"
( go r3 "$B" default S_ref "$H1"
  go r3 "$B -e TF_FLASHNEXT_DRAFTS=8" default S_d8 "$H1; $CR"
  go r3 "$B -e TF_FLASHNEXT_DRAFTS=8 -e TF_FLASHNEXT_CONFIDENCE=-0.35" default S_c35d8 "$H1; $CR"
  go r3 "$B -e TF_FLASHNEXT_CONFIDENCE=-0.3" default S_c30 "$H1; $CR"
  go r3 "$B -e TF_FLASHNEXT_DRAFTS=8 -e TF_FLASHNEXT_CONFIDENCE=-0.5" default S_c50d8 "$H1; $CR"
  go r3 "$B" default S_ref2 "$H1; $CR"
) > $D/s_923b.log 2>&1 & P1=$!
KP1="python3 -I ~/q38/kit/heldpar.py 1 \$O/held.jsonl 256 12 | tail -1; docker logs q38-run-tf > \$O/engine.log 2>&1; python3 -I ~/q38/kit/kprof.py \$O/engine.log > \$O/kprof.txt; head -48 \$O/kprof.txt | cut -c1-150; tail -1 \$O/kprof.txt"
KP16="$Q distinct 16 | tail -1; python3 -I ~/q38/kit/heldpar.py 16 \$O/held16.jsonl 256 6 | tail -1; docker logs q38-run-tf > \$O/engine.log 2>&1; python3 -I ~/q38/kit/kprof.py \$O/engine.log > \$O/kprof.txt; head -60 \$O/kprof.txt | cut -c1-150; tail -1 \$O/kprof.txt"
( go w2 "$B -e TF_KPROF=150000" default S2_kp1 "$KP1"
  go w2 "$MUE -e TF_KPROF=400000" pc_16_262144 S2_kp16 "$KP16"
  go w2 "$MUE -e TF_FLASHNEXT_DEPTH_LOG=1" pc_16_262144 S2_depth "$Q distinct 16 | tail -1; $Q distinct 16 | tail -1; python3 -I ~/q38/kit/heldpar.py 16 \$O/held16.jsonl 256 3 | tail -1; docker logs q38-run-tf 2>&1 | grep '^DEPTH' | tail -5; python3 -I ~/q38/kit/mustep.py <(docker logs q38-run-tf 2>&1) 2>&1 | tail -6 | cut -c1-200"
) > $D/s_3d24.log 2>&1 & P2=$!
HPA="docker logs q38-run-tf 2>&1 | grep 'hostprof launches' | awk '{n+=\$3; d+=\$5; w+=\$7; s+=\$9; sm+=\$11} END {print \"hostprof: launches\", n, \"in-driver ms\", d, \"wall ms\", w, \"us/launch\", 1000*d/n, \"syncs\", s, \"sync ms\", sm}'"
( w3 'bash ~/q38/mb/mbrun.sh mbxn 3 > ~/q38/runs/S3_mbxn.txt 2>&1; head -2 ~/q38/runs/S3_mbxn.txt | cut -c1-200; grep -E "^==|decode stock|tile rt4 s4|SUM" ~/q38/runs/S3_mbxn.txt | cut -c1-150; nm -C ~/q38/tfbin/kp21/native/bin/tensorfold-native > ~/q38/runs/kp21.nm 2>/dev/null; wc -l < ~/q38/runs/kp21.nm'
  echo "microbench done $(date -u +%T)"
  go w3 "$MUE -e TF_HOSTPROF=400000 -e TF_HOSTPROF_K=1" pc_16_262144 S3_hp16 "$Q distinct 16 | tail -1; $Q distinct 16 | tail -1; python3 -I ~/q38/kit/heldpar.py 16 \$O/held16.jsonl 256 3 | tail -1; $HPA; docker logs q38-run-tf > \$O/engine.log 2>&1; python3 -I ~/q38/kit/syncsites.py \$O/engine.log ~/q38/runs/kp21.nm 2>&1 | head -14 | cut -c1-220; grep mtpstat \$O/engine.log | tail -2 | cut -c1-260; python3 -I ~/q38/kit/mustep.py \$O/engine.log 2>&1 | tail -6 | cut -c1-200"
  go w3 "$MUE -e TF_FLASHNEXT_Q38_GPROF=1" pc_16_262144 S3_gp16 "$Q distinct 16 | tail -1; $Q distinct 16 | tail -1; python3 -I ~/q38/kit/heldpar.py 16 \$O/held16.jsonl 256 3 | tail -1; docker logs q38-run-tf 2>&1 | grep 'q38 gprof' | tail -3 | cut -c1-330; python3 -I ~/q38/kit/mustep.py <(docker logs q38-run-tf 2>&1) 2>&1 | tail -6 | cut -c1-200"
  go w3 "$MUE" pc_16_262144 S3_ref16 "$Q distinct 16 | tail -1; $Q distinct 16 | tail -1; python3 -I ~/q38/kit/heldpar.py 16 \$O/held16.jsonl 256 3 | tail -1; python3 -I ~/q38/kit/mustep.py <(docker logs q38-run-tf 2>&1) 2>&1 | tail -6 | cut -c1-200"
) > $D/s_4e5c.log 2>&1 & P3=$!
wait $P1 $P2 $P3
echo "window S work done $(date -u +%T)"
