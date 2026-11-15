#!/bin/bash
# pullN.sh : the night's result files of 923b -> Mac q38/results/n14 (everything of ~/q38/runs/N_* except engine logs over 3 MB, + chain logs)
Q=~/.hermes/cache/scratch/q38; K=~/.ssh/codex_spark_ed25519; mkdir -p $Q/results/n14
ssh -n -i $K pico@spark-head "ssh -n picom@spark-923b 'cd ~/q38/runs && find N_* wN*.log N_prepare.log N_u16_args N_ag_args -type f -size -3000k 2>/dev/null | grep -v /eng_ > /tmp/n14.lst; tar czf /tmp/n14.tgz -T /tmp/n14.lst 2>/dev/null; ls -la /tmp/n14.tgz | cut -c20-60' && scp -q picom@spark-923b:/tmp/n14.tgz /tmp/n14.tgz" && scp -q -i $K pico@spark-head:/tmp/n14.tgz $Q/results/n14.tgz && cd $Q/results/n14 && tar xzf ../n14.tgz 2>/dev/null; ls $Q/results/n14 | tr '\n' ' '
