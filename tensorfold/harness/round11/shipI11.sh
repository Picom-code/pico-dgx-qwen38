#!/bin/bash
# shipA.sh NAME [CPUS] [MEM] : send EVERY tracked + new file of Mac q38/wt/I11 to 923b:~/q38/build-A/tf, verify sha256 of all, build NAME there
# (log ~/q38/build-A/NAME.log, output ~/q38/build-A/out/NAME, copied to ~/q38/tfbin/NAME on success).
N=$1; CPUS=${2:-3}; MEM=${3:-8g}; K=~/.ssh/codex_spark_ed25519; BOX=picom@spark-923b
cd ~/.hermes/cache/scratch/q38/wt/I11 || exit 1
( git ls-files; git ls-files --others --exclude-standard ) | sort -u > /tmp/q38A_files.txt
tar czf /tmp/q38A_src.tgz -T /tmp/q38A_files.txt
tr '\n' '\0' < /tmp/q38A_files.txt | xargs -0 shasum -a 256 > /tmp/q38A_sums.txt
scp -q -i $K /tmp/q38A_src.tgz /tmp/q38A_sums.txt pico@spark-head:/tmp/ || exit 1
ssh -n -i $K pico@spark-head "scp -q /tmp/q38A_src.tgz /tmp/q38A_sums.txt $BOX:/tmp/ && ssh -n $BOX 'mkdir -p ~/q38/build-A/tf && cd ~/q38/build-A/tf && tar xzf /tmp/q38A_src.tgz && echo mismatches \$(sha256sum -c /tmp/q38A_sums.txt 2>/dev/null | grep -vc \": OK\$\") of \$(wc -l < /tmp/q38A_sums.txt) && sha256sum -c /tmp/q38A_sums.txt >/dev/null 2>&1 && echo TREE_VERIFIED && (nohup bash ~/q38/kit-A/tfbuildA.sh $N $CPUS $MEM > ~/q38/build-A/$N.log 2>&1 &) ; echo build $N started'"
