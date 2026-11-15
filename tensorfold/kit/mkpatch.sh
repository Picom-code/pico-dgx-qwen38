#!/bin/bash
# mkpatch.sh [FROM=1c06bc2] [NAME] : regenerate the newest q38 patch (FROM -> HEAD of work/tf) and prove that patches 01xx on the MiaAI base
# (6a8e032) reproduce the build tree (git write-tree).
set -e
Q=~/.hermes/cache/scratch/q38; P=$Q/repo/tensorfold/patches; FROM=${1:-1c06bc2}; N=${2:-0108-q38-gdn-lanes-for-9-16-streams-and-128-row-lane-tiles.patch}
cd $Q/work/tf
git diff $FROM HEAD > $P/$N
rm -rf /tmp/q38_chk && git worktree prune && git worktree add -q --detach /tmp/q38_chk 6a8e032
cd /tmp/q38_chk && for f in $P/01*.patch; do git apply --whitespace=nowarn "$f"; done && git add -A && T1=$(git write-tree)
cd $Q/work/tf && T2=$(git rev-parse 'HEAD^{tree}'); git worktree remove --force /tmp/q38_chk
echo "patched tree $T1"; echo "build tree   $T2"; [ "$T1" = "$T2" ] && echo TREE_EQUAL || echo TREE_DIFFERS
shasum -a 256 $P/$N | cut -c1-16; wc -l $P/$N
