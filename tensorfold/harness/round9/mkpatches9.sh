#!/bin/bash
# mkpatches9.sh : regenerate patches 0116+ from the round-9 integration branch (Mac q38/wt/I9): one patch a commit after 0b5cf9b (= 0101-0115),
# then prove MiaAI base 6a8e032 + patches/01* = the branch's HEAD tree (git write-tree).
set -e
Q=~/.hermes/cache/scratch/q38; P=$Q/repo/tensorfold/patches; T=$Q/wt/I9; cd $T
NAMES=(0116-q38-certified-indexer-select-and-elastic-prompt-cache 0117-q38-certified-head-for-sampled-passes-sm8-coarse-copy 0118-q38-draft-side-deep-chains-early-stop-fast-two-stage-head 0119-q38-integration-followup 0120-q38-integration-followup-2)
rm -f $P/011[6-9]-*.patch $P/012[0-9]-*.patch; prev=0b5cf9b; i=0
for c in $(git log --reverse --format=%h 0b5cf9b..HEAD); do n=${NAMES[$i]}; git diff $prev $c > $P/$n.patch; echo "$n.patch $(wc -l < $P/$n.patch) lines ($c)"; prev=$c; i=$((i+1)); done
C=$Q/A9/chk; rm -rf $C; git worktree prune; git worktree add -q --detach $C 6a8e032
cd $C && for f in $P/01*.patch; do git apply --whitespace=nowarn "$f"; done && git add -A && T1=$(git write-tree)
cd $T && T2=$(git rev-parse 'HEAD^{tree}'); git worktree remove --force $C
echo "patched tree $T1"; echo "build tree   $T2"; [ "$T1" = "$T2" ] && echo TREE_EQUAL || echo TREE_DIFFERS
