# Window P (round 6): build kp20 = kp17 + fused decode read-out (HCF) + half tiles (#1171 port) + MTP head last-row experts (#1925 port).
#   923b: paired one-user speed, fused read-out variants (stages 3 / 2, half tiles of the fused down kernel), 107 held prompts x 256 tokens, same box, ref first and last.
#   3d24: paired one-user speed: lane half tiles (96 / 192), MTP last-row experts, everything together (+ ident set), ref first and last.
#   4e5c: free env tests: prompt chunk 3072 / 4096 / 8192 (prefill tok/s, first-score hash, answer), 16 users: expert tile rows 128, fused read-out for wide passes, half tiles.
E='QK_MODEL=Qwen3.8-Flash-Next'; NE=TFENTRY=/opt/nvidia/nvidia_entrypoint.sh
KP=${KP:-kp20}
TW="TFBIN=\$HOME/q38/tfbin/$KP TFMODEL=\$HOME/q38/models/nvidia-fp4-local $NE"
go() { $1 "$TW TFENV='$2' bash ~/q38/kit/eng.sh tfb $3" 2>&1 | tail -1 | cut -c1-200
  $1 "export $E O=\$HOME/q38/runs/$4; mkdir -p \$O; $5; docker logs q38-run-tf > \$O/engine.log 2>&1; bash ~/q38/kit/eng.sh stop >/dev/null"; echo "$4 done $(date -u +%T)"; }
C='python3 -I ~/q38/kit/qk.py cmp'; Q='python3 -I ~/q38/kit/qk.py'
H1="python3 -I ~/q38/kit/heldpar.py 1 \$O/held.jsonl 256 3 | tail -1; python3 -I ~/q38/kit/tfsteps.py <(docker logs q38-run-tf 2>&1) | sed -n '2,3p'"
B="-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_CERT_PLAIN=1 -e TF_FLASHNEXT_Q38_PREFILL_MANY=256 -e TF_FLASHNEXT_Q38_WINDOW_ROWS=32"
F="-e TF_FLASHNEXT_Q38_HCF=3"
( CM="$C \$HOME/q38/runs/P_ref/held.jsonl \$O/held.jsonl"
  go r3 "$B" default P_ref "$H1"
  go r3 "$B $F -e TF_FLASHNEXT_Q38_HCF_ST=3" default P_f3s3 "$H1; $CM"
  go r3 "$B $F -e TF_FLASHNEXT_Q38_HCF_ST=2" default P_f3s2 "$H1; $CM"
  go r3 "$B $F -e TF_FLASHNEXT_Q38_HCF_ST=2 -e TF_FLASHNEXT_Q38_HCF_HALF=1" default P_f3s2h "$H1; $CM"
  go r3 "$B $F -e TF_FLASHNEXT_Q38_HCF_ST=3 -e TF_FLASHNEXT_Q38_HCF_HALF=1" default P_f3s3h "$H1; $CM"
  go r3 "$B" default P_ref2 "$H1; $CM"
) > $D/p_923b.log 2>&1 & P1=$!
( CM="$C \$HOME/q38/runs/P2_ref/held.jsonl \$O/held.jsonl"
  go w2 "$B" default P2_ref "$Q ident \$O/ident.json | tail -1; $H1"
  go w2 "$B -e TF_FLASHNEXT_Q38_HALF_BELOW=96" default P2_half96 "$H1; $CM"
  go w2 "$B -e TF_FLASHNEXT_Q38_MTP_LAST=1" default P2_mtplast "$Q ident \$O/ident.json | tail -1; $H1; $CM; $C \$HOME/q38/runs/P2_ref/ident.json \$O/ident.json"
  go w2 "$B -e TF_FLASHNEXT_Q38_HALF_BELOW=192" default P2_half192 "$H1; $CM"
  go w2 "$B $F -e TF_FLASHNEXT_Q38_HCF_ST=3 -e TF_FLASHNEXT_Q38_HALF_BELOW=96 -e TF_FLASHNEXT_Q38_MTP_LAST=1" default P2_all "$Q ident \$O/ident.json | tail -1; $H1; $CM; $C \$HOME/q38/runs/P2_ref/ident.json \$O/ident.json"
  go w2 "$B" default P2_ref2 "$H1; $CM"
) > $D/p_3d24.log 2>&1 & P2=$!
MU="-e TF_FLASHNEXT_Q38_CERT=1 -e TF_FLASHNEXT_Q38_WIDE_ROWS=128 -e TF_FLASHNEXT_Q38_PREFILL_MANY=256 -e TF_FLASHNEXT_Q38_WINDOW_ROWS=32"
MB="$Q distinct 16 | tail -1; $Q distinct 16 | tail -1; $Q distinct 8 | tail -1; python3 -I ~/q38/kit/heldpar.py 16 \$O/held16.jsonl 256 3 | tail -1; python3 -I ~/q38/kit/mustep.py <(docker logs q38-run-tf 2>&1) 2>/dev/null | tail -4"
PF="$Q prefill 8000,32000,128000 | tail -3; $Q ctx 100000 | tail -1; docker logs q38-run-tf 2>&1 | grep 'lhash first' | sort -t= -k3,3n | tail -4 | cut -c1-160; nvidia-smi --query-compute-apps=used_memory --format=csv,noheader; awk '/MemAvailable/{print \"MemAvailable GiB\", int(\$2/1048576)}' /proc/meminfo"
( CM="$C \$HOME/q38/runs/P3_mu/held16.jsonl \$O/held16.jsonl"
  go w3 "$B -e TF_FLASHNEXT_Q38_LHASH=1" default P3_c3072 "$PF"
  go w3 "$B -e TF_FLASHNEXT_Q38_LHASH=1 -e TF_FLASHNEXT_CHUNK=4096" default P3_c4096 "$PF"
  go w3 "$B -e TF_FLASHNEXT_Q38_LHASH=1 -e TF_FLASHNEXT_CHUNK=8192" default P3_c8192 "$PF"
  go w3 "$MU" p16 P3_mu "$MB"
  go w3 "$MU -e TF_FLASHNEXT_PROMPT_EXPERT_TILE_ROWS=128" p16 P3_et128 "$MB; $CM"
  go w3 "$MU $F -e TF_FLASHNEXT_Q38_HCF_ST=3 -e TF_FLASHNEXT_Q38_HALF_BELOW=96 -e TF_FLASHNEXT_Q38_MTP_LAST=1" p16 P3_f16 "$MB; $CM"
  go w3 "$MU $F -e TF_FLASHNEXT_Q38_HCF_ST=3 -e TF_FLASHNEXT_Q38_HCF_MAX=128 -e TF_FLASHNEXT_Q38_HALF_BELOW=96 -e TF_FLASHNEXT_Q38_MTP_LAST=1" p16 P3_f128 "$MB; $CM"
) > $D/p_4e5c.log 2>&1 & P3=$!
wait $P1 $P2 $P3
echo "window P work done $(date -u +%T)"
