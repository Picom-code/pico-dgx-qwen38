#!/bin/bash
# treecheck.sh BOX : sha256 of EVERY tracked file of Mac work/tf against BOX:~/q38/build/tf, plus files on the box that the Mac tree lacks
# under zig/src, zig/kernels, zig/build. ship.sh sends only files that differ from the MiaAI base NOW, so a file changed by an earlier build
# and reverted since stays stale on the box (round 7: kp22's native-decode files); run this before trusting a build.
BOX=$1; K=~/.ssh/codex_spark_ed25519; cd ~/.hermes/cache/scratch/q38/work/tf || exit 1
git ls-files -z | xargs -0 shasum -a 256 > /tmp/q38_all.txt; git ls-files | grep -E '^zig/(src|kernels|build)/' | sort > /tmp/q38_names.txt
scp -q -i $K /tmp/q38_all.txt /tmp/q38_names.txt pico@spark-head:/tmp/ && ssh -n -i $K pico@spark-head "scp -q /tmp/q38_all.txt /tmp/q38_names.txt $BOX:/tmp/ && ssh -n $BOX 'cd ~/q38/build/tf && sha256sum -c /tmp/q38_all.txt 2>/dev/null | grep -v \": OK\$\" | head -20; echo \"mismatches: \$(sha256sum -c /tmp/q38_all.txt 2>/dev/null | grep -vc \": OK\$\") of \$(wc -l < /tmp/q38_all.txt)\"; find zig/src zig/kernels zig/build -type f | sort | comm -23 - /tmp/q38_names.txt | head -10 | sed \"s/^/extra on box: /\"'"
