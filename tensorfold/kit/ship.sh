#!/bin/bash
# ship.sh NAME BOX(e.g. pico@spark-3d24) : send every file changed since the MiaAI base (6a8e032) to BOX:~/q38/build/tf, verify sha256 of each
# HEAD=user@host of the cluster head must be set (ssh key via your ssh config).
# on the box against the Mac, then start the build NAME there (log ~/q38/build/NAME.log). Round-3 lesson: never build an unverified tree.
N=$1; BOX=$2; cd ~/.hermes/cache/scratch/q38/work/tf || exit 1
( git diff --name-only 6a8e032; git ls-files --others --exclude-standard ) | sort -u > /tmp/q38_files.txt
tar czf /tmp/q38_src.tgz -T /tmp/q38_files.txt
xargs shasum -a 256 < /tmp/q38_files.txt > /tmp/q38_sums.txt
scp -q /tmp/q38_src.tgz /tmp/q38_sums.txt $HEAD:/tmp/ || exit 1
ssh -n $HEAD "scp -q /tmp/q38_src.tgz /tmp/q38_sums.txt $BOX:/tmp/ && ssh -n $BOX 'cd ~/q38/build/tf && tar xzf /tmp/q38_src.tgz && sha256sum -c /tmp/q38_sums.txt 2>&1 | grep -vc OK; sha256sum -c /tmp/q38_sums.txt >/dev/null 2>&1 && echo TREE_VERIFIED \$(wc -l < /tmp/q38_sums.txt) files && (nohup bash ~/q38/kit/tfbuild.sh $N 12 16g > ~/q38/build/$N.log 2>&1 &) ; echo build $N started'"
