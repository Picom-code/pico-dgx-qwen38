#!/bin/bash
# commit9.sh "MESSAGE" : copy round-9 notes / harness into the repo, scan (key values + gitleaks), commit as Pico, push the PRIVATE branch, confirm private.
set -e
Q=~/.hermes/cache/scratch/q38; R=$Q/repo; T=$R/tensorfold; MSG="$1"
mkdir -p $T/notes/round10 $T/harness/round10
cp $Q/PROGRESS-A.md $T/notes/round10/PROGRESS-A.md
for b in B C D; do [ -f $Q/PROGRESS-$b.md ] && cp $Q/PROGRESS-$b.md $T/notes/round10/PROGRESS-$b.md; done
cp $Q/handoff/BASE.txt $T/notes/round10/BASE.txt
for n in C/r10-prefill/NOTE.md C/URGENT-r9-knob-cache.md B/r10-workload-drafts/NOTE.md D/r10-audit/AUDIT.md D/r10-levers/NOTE.md; do [ -f $Q/handoff/$n ] && cp $Q/handoff/$n $T/notes/round10/builder-$(echo $n | tr / -); done
[ -f $Q/results/casc10_raw.txt ] && cp $Q/results/casc10_raw.txt $T/notes/round10/
[ -f $Q/handoff/C/r9-qsa-cert-elastic/NOTE.md ] && cp $Q/handoff/C/r9-qsa-cert-elastic/NOTE.md $T/notes/round10/builder-C-qsa-cert-elastic-NOTE.md
[ -f $Q/handoff/D/r9-cert-topk-sm8/NOTE.md ] && cp $Q/handoff/D/r9-cert-topk-sm8/NOTE.md $T/notes/round10/builder-D-cert-topk-sm8-NOTE.md
[ -f $Q/handoff/B/r9-draft-side/NOTE.md ] && cp $Q/handoff/B/r9-draft-side/NOTE.md $T/notes/round10/builder-B-draft-side-NOTE.md
[ -d $Q/handoff/B/r9-tree-verify-design ] && cp $Q/handoff/B/r9-tree-verify-design/*.md $T/notes/round10/ 2>/dev/null || true
cp $Q/handoff/D/r9-note-for-B.md $T/notes/round10/builder-D-note-for-B.md 2>/dev/null || true
cp $Q/handoff/D/r9-cert-topk-sm8/sq.py $T/kit/sq.py
cp $Q/A10/*.sh $Q/A10/*.py $Q/A10/names10.txt $T/harness/round10/
for f in $Q/results/final_r10_*.txt; do [ -f "$f" ] && cp "$f" $T/notes/; done
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
