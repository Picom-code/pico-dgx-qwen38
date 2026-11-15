#!/bin/bash
# mkpatches.sh : regenerate patches 0110+ from the integration branch (Mac q38/wt/I): one patch a commit after 52b32e1 (kp25 = 0101-0109), in order,
# then prove 6a8e032 (MiaAI base) + patches/01* = the branch's HEAD tree (git write-tree).
set -e
Q=~/.hermes/cache/scratch/q38; P=$Q/repo/tensorfold/patches; T=${TREE:-$Q/wt/I}; cd $T
NAMES=(0110-q38-attention-lanes-in-one-launch 0111-q38-drafted-windows-skip-deltanet-end-state-writes 0112-q38-first-token-keep-inpass-lane-reuse-prefix-wait 0113-q38-expert-statistics-log-aid 0114-q38-draft-study-logging 0115-q38-two-stage-draft-head 0116-q38-integration-followup-3)
rm -f $P/011[0-9]-*.patch; prev=52b32e1; i=0
for c in $(git log --reverse --format=%h 52b32e1..HEAD); do n=${NAMES[$i]}; git diff $prev $c > $P/$n.patch; echo "$n.patch $(wc -l < $P/$n.patch) lines ($c)"; prev=$c; i=$((i+1)); done
C=$Q/A/chk; rm -rf $C; git worktree prune; git worktree add -q --detach $C 6a8e032
cd $C && for f in $P/01*.patch; do git apply --whitespace=nowarn "$f"; done && git add -A && T1=$(git write-tree)
cd $T && T2=$(git rev-parse 'HEAD^{tree}'); git worktree remove --force $C
echo "patched tree $T1"; echo "build tree   $T2"; [ "$T1" = "$T2" ] && echo TREE_EQUAL || echo TREE_DIFFERS
