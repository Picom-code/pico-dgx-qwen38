# Window J (round 4): build kp6 (= kp5 + per-kernel host launch timer, sync timer, TF_FLASHNEXT_Q38_SCHED, head-row dump).
#   4e5c: lever 1 (why launches cost 4.8 us with drafts): per-kernel hostprof drafted / no drafts, thread view; paired held-52 runs:
#         base, SCHED=spin, SCHED=blocking, SHARED_SIDE=0, HOST_DRAW_THREADS=0.
#   3d24: lever 2 (newer CUDA userland): image compat 610.43 vs NVIDIA cuda-compat 13.4 (615.71) vs 13.5 (620.70), paired held-52, A-B-A.
#   923b: lever 4 step 1: head-row dump (16 users, then 1 user), then the offline pruning study on the free GPU.
E='QK_MODEL=Qwen3.8-Flash-Next'; NE=TFENTRY=/opt/nvidia/nvidia_entrypoint.sh
TW="TFBIN=\$HOME/q38/tfbin/kp6 TFMODEL=\$HOME/q38/models/nvidia-fp4-local $NE"
T923="TFIMG=tensorfold-qwen38:v1.1.4-eab8582 $TW"
go() { $1 "$2 bash ~/q38/kit/eng.sh $3 $4" 2>&1 | tail -1 | cut -c1-200
  $1 "export $E O=\$HOME/q38/runs/$5; mkdir -p \$O; $6; docker logs q38-run-tf > \$O/engine.log 2>&1; bash ~/q38/kit/eng.sh stop >/dev/null"; echo "$5 done $(date -u +%T)"; }
H='python3 -I ~/q38/kit/qk.py held'; C='python3 -I ~/q38/kit/qk.py cmp'
LIB='docker exec q38-run-tf sh -c "cat /proc/[0-9]*/maps 2>/dev/null | grep -o \"[^ ]*libcuda.so[^ ]*\" | sort -u | head -3"'
HPA="docker logs q38-run-tf 2>&1 | grep 'hostprof launches' | awk '{n+=\$3; d+=\$5; w+=\$7; s+=\$9; sm+=\$11; ts+=\$13} END {print \"hostprof: launches\", n, \"in-driver ms\", d, \"wall ms\", w, \"us/launch\", 1000*d/n, \"syncs\", s, \"sync ms\", sm, \"thread switches\", ts}'"
# dr HOSTFN LABEL ENVPREFIX BASELABEL : drafts on, held every 6th prompt (54), step fit; compare texts with BASELABEL
dr() { local x="QK_HELD_STRIDE=6 $H \$O/held.jsonl | tail -4; python3 -I ~/q38/kit/tfsteps.py <(docker logs q38-run-tf 2>&1) | sed -n 2,3p; $LIB; docker logs q38-run-tf 2>&1 | grep -m2 'q38:'"
  [ -n "$4" ] && x="$x; $C \$HOME/q38/runs/$4/held.jsonl \$O/held.jsonl"
  go $1 "$3" tfb default $2 "$x"; }
( go w3 "$TW TFENV='-e TF_HOSTPROF=15200 -e TF_HOSTPROF_K=1'" tfb default J3_hp_dr "(sleep 45; top -H -b -n 1 -p \$(pgrep -f tensorfold-native | head -1) | head -32 > \$O/top.txt; nproc >> \$O/top.txt) & QK_HELD_STRIDE=12 $H \$O/held.jsonl | tail -1; $HPA"
  go w3 "$TW TFENV='-e TF_HOSTPROF=13700 -e TF_HOSTPROF_K=1'" tfb nodraft J3_hp_nd "$H \$O/held3.jsonl 3 | tail -1; $HPA"
  dr w3 J3_base "$TW" ""
  dr w3 J3_spin "$TW TFENV='-e TF_FLASHNEXT_Q38_SCHED=spin'" J3_base
  dr w3 J3_side0 "$TW TFENV='-e TF_FLASHNEXT_SHARED_SIDE=0'" J3_base
  dr w3 J3_hd0 "$TW TFENV='-e TF_FLASHNEXT_HOST_DRAW_THREADS=0'" J3_base
  dr w3 J3_block "$TW TFENV='-e TF_FLASHNEXT_Q38_SCHED=blocking'" J3_base
) > $D/j_4e5c.log 2>&1 & P3=$!
CP='/home/pico/q38/compat'
( dr w2 J2_c610 "$TW" ""
  dr w2 J2_c615 "$TW TFENV='-v $CP/x13-4/usr/local/cuda-13.4/compat:/usr/local/cuda/compat/lib.real:ro'" J2_c610
  dr w2 J2_c620 "$TW TFENV='-v $CP/x13-5/usr/local/cuda-13.5/compat:/usr/local/cuda/compat/lib.real:ro'" J2_c610
  dr w2 J2_c610b "$TW" J2_c610
  go w2 "$TW TFENV='-e TF_HOSTPROF=13700 -v $CP/x13-5/usr/local/cuda-13.5/compat:/usr/local/cuda/compat/lib.real:ro'" tfb nodraft J2_hp_nd620 "$H \$O/held3.jsonl 3 | tail -1; $HPA; $LIB"
  go w2 "$TW TFENV='-e TF_HOSTPROF=13700'" tfb nodraft J2_hp_nd610 "$H \$O/held3.jsonl 3 | tail -1; $HPA; $LIB"
) > $D/j_3d24.log 2>&1 & P2=$!
( r3 'rm -f ~/q38/prof/hd16.* ~/q38/prof/hd1.*'
  go r3 "$T923 TFENV='-e TF_FLASHNEXT_Q38_HDUMP=/prof/hd16 -e TF_FLASHNEXT_Q38_HDUMP_MAX=320000'" tfb p16 J_dump16 "python3 -I ~/q38/kit/heldpar.py 16 \$O/held16.jsonl 384 1 2 | tail -1; ls -la ~/q38/prof/hd16.*"
  go r3 "$T923 TFENV='-e TF_FLASHNEXT_Q38_HDUMP=/prof/hd1 -e TF_FLASHNEXT_Q38_HDUMP_MAX=60000'" tfb default J_dump1 "QK_HELD_STRIDE=8 $H \$O/held.jsonl | tail -1; ls -la ~/q38/prof/hd1.*"
  r3 'for p in hd1 hd16; do timeout 1500 docker run --rm --name q38-run-study --gpus all --ipc=host -v $HOME/q38/models/nvidia-fp4-local:/model:ro -v $(realpath $HOME/q38/models/nvidia-fp4-local):$(realpath $HOME/q38/models/nvidia-fp4-local):ro -v $HOME/q38/prof:/prof -v $HOME/q38/kit:/kit:ro --entrypoint python3 nvcr.io/nvidia/pytorch:26.07-py3 /kit/hstudy.py /model /prof/$p 100000 2>&1 | grep -v Warning | tee ~/q38/runs/J_study_$p.txt | tail -40; done; python3 ~/q38/kit/evict.py ~/q38/prof'
) > $D/j_923b.log 2>&1 & P1=$!
wait $P1 $P2 $P3
echo "window J work done $(date -u +%T)"
