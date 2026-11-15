#!/bin/bash
# w10k.sh (923b, builder A, round 10): what the knob-cache fix (patch 0120) changes in speed: binary J1 (0101-0119) -> J2 (+0120) -> J1, same environment
# (round-9 recipe levers through the binary launcher), held prompts stride 6 + prefill 32K / 128K; texts compared.
. $HOME/q38/kit-A/libA.sh
E10="$B -e TF_FLASHNEXT_Q38_ATTN_LANES=1 -e TF_FLASHNEXT_Q38_LATE_SKIP=16 -e TF_FLASHNEXT_Q38_KEEP_INPASS=4096 -e TF_FLASHNEXT_Q38_LANE_REUSE=5 -e TF_FLASHNEXT_Q38_QSA_CERT=1 -e TF_FLASHNEXT_Q38_QSA_HALF=1 -e TF_FLASHNEXT_Q38_CERT_CODE=sm8 -e TF_FLASHNEXT_Q38_CERT_TOPK=20"
S="QK_HELD_STRIDE=6 $Q held \$O/held.jsonl | tail -1; $Q prefill 32000,128000 | tail -2; $Q ttft | tail -1"
go K_j1a J1 "$E10" "$S" default 700
go K_j2 J2 "$E10" "$S; $C $RUNS/K_j1a/held.jsonl \$O/held.jsonl; docker logs q38-run-A 2>&1 | grep -c 'q38 qsacert'" default 700
go K_j1b J1 "$E10" "$S; $C $RUNS/K_j1a/held.jsonl \$O/held.jsonl; docker logs q38-run-A 2>&1 | grep -c 'q38 qsacert'" default 700
echo "w10k done $(date -u +%T)"
