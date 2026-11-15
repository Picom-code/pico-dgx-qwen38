#!/bin/bash
# mkpatches10.sh : regenerate patches 0119+ from the round-10 integration branch (Mac q38/wt/I10): one patch a commit after 7061396 (= 0101-0118),
# names from A10/names10.txt (one a line, in commit order), then prove MiaAI base 6a8e032 + patches/01* = the branch's HEAD tree (git write-tree).
set -e
Q=~/.hermes/cache/scratch/q38; P=$Q/repo/tensorfold/patches; T=$Q/wt/I10; cd $T
rm -f $P/0119-*.patch $P/012[0-9]-*.patch $P/013[0-9]-*.patch; prev=7061396; i=0
for c in $(git log --reverse --format=%h 7061396..HEAD); do i=$((i+1)); n=$(sed -n "${i}p" $Q/A10/names10.txt); [ -n "$n" ] || { echo "no name for commit $i ($c)"; exit 2; }
  git diff $prev $c > $P/$n.patch; echo "$n.patch $(wc -l < $P/$n.patch) lines ($c)"; prev=$c; done
C=$Q/A10/chk; rm -rf $C; git worktree prune; git worktree add -q --detach $C 6a8e032
cd $C && for f in $P/01*.patch; do git apply --whitespace=nowarn "$f"; done && git add -A && T1=$(git write-tree)
cd $T && T2=$(git rev-parse 'HEAD^{tree}'); git worktree remove --force $C
echo "patched tree $T1"; echo "build tree   $T2"; [ "$T1" = "$T2" ] && echo TREE_EQUAL || echo TREE_DIFFERS
