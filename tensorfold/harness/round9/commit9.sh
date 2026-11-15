#!/bin/bash
# commit9.sh "MESSAGE" : copy round-9 notes / harness into the repo, scan (key values + gitleaks), commit as Pico, push the PRIVATE branch, confirm private.
set -e
Q=~/.hermes/cache/scratch/q38; R=$Q/repo; T=$R/tensorfold; MSG="$1"
mkdir -p $T/notes/round9 $T/harness/round9
cp $Q/PROGRESS-A.md $T/notes/round9/PROGRESS-A.md
for b in B C D; do [ -f $Q/PROGRESS-$b.md ] && cp $Q/PROGRESS-$b.md $T/notes/round9/PROGRESS-$b.md; done
cp $Q/handoff/BASE.txt $T/notes/round9/BASE.txt
[ -f $Q/handoff/C/r9-qsa-cert-elastic/NOTE.md ] && cp $Q/handoff/C/r9-qsa-cert-elastic/NOTE.md $T/notes/round9/builder-C-qsa-cert-elastic-NOTE.md
[ -f $Q/handoff/D/r9-cert-topk-sm8/NOTE.md ] && cp $Q/handoff/D/r9-cert-topk-sm8/NOTE.md $T/notes/round9/builder-D-cert-topk-sm8-NOTE.md
[ -f $Q/handoff/B/r9-draft-side/NOTE.md ] && cp $Q/handoff/B/r9-draft-side/NOTE.md $T/notes/round9/builder-B-draft-side-NOTE.md
[ -d $Q/handoff/B/r9-tree-verify-design ] && cp $Q/handoff/B/r9-tree-verify-design/*.md $T/notes/round9/ 2>/dev/null || true
cp $Q/handoff/D/r9-note-for-B.md $T/notes/round9/builder-D-note-for-B.md 2>/dev/null || true
cp $Q/handoff/D/r9-cert-topk-sm8/sq.py $T/kit/sq.py
cp $Q/A9/*.sh $Q/A9/runset.py $Q/A9/evalrot.py $Q/A9/overlap.py $T/harness/round9/
for f in $Q/results/final_r9_*.txt; do [ -f "$f" ] && cp "$f" $T/notes/; done
cd $R; git add -A tensorfold
# key values must not be in the tree (values are never printed)
GK=$(cat ~/.config/opencode/deepseek.key 2>/dev/null | tr -d '\n'); EK=$(ssh -n -o ConnectTimeout=15 -i ~/.ssh/codex_spark_ed25519 pico@spark-head 'cat ~/.config/qwen38/api_key 2>/dev/null' | tr -d '\n')
for k in "$GK" "$EK"; do [ ${#k} -ge 8 ] && { if git grep -qF --cached -e "$k"; then echo "KEY VALUE FOUND IN TREE: abort"; exit 3; fi; }; done; echo "key values absent (gateway len ${#GK}, engine len ${#EK})"
git diff --cached --name-only | wc -l
gitleaks protect --staged --no-banner 2>&1 | tail -2
git -c user.name=Pico -c user.email=50418071+Picom-code@users.noreply.github.com commit -q -m "$MSG"
git push -q origin tensorfold-fp4-local 2>&1 | tail -1
echo "pushed $(git rev-parse --short HEAD); author $(git log -1 --format='%an <%ae>')"
echo "repo private: $(gh api repos/Picom-code/pico-dgx-qwen38 --jq .private); anonymous http $(curl -s -o /dev/null -w '%{http_code}' https://github.com/Picom-code/pico-dgx-qwen38)"
