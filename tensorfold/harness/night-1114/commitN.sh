#!/bin/bash
# commitN.sh "MESSAGE" : night 11-14 final commit. Copies the night's notes / harness / raw result files into the repo, scans (key values,
# private addresses, gitleaks), commits as Pico and pushes branch main. The repo's visibility is NOT touched.
set -e
Q=~/.hermes/cache/scratch/q38; R=$Q/repo; T=$R/tensorfold; N=$T/notes/final-2026-11-15; MSG="$1"
mkdir -p $N/raw $T/harness/night-1114
cp $Q/PROGRESS-A.md $N/PROGRESS-A.md; for b in B C D; do [ -f $Q/PROGRESS-$b.md ] && cp $Q/PROGRESS-$b.md $N/PROGRESS-$b.md; done
cp $Q/A12/*.sh $Q/A12/*.py $Q/A12/intended_*.txt $T/harness/night-1114/
# the other agents' notes of the night (as handed off)
H=$Q/handoff; [ -f $H/C/night-1114-install/INSTALL.md ] && cp $H/C/night-1114-install/INSTALL.md $N/INSTALL.md
night-1114-install/the same-window comparison note $N/the same-window comparison note
for f in $H/C/night-1114-install/URGENT.md $H/C/night-1114-install/PRE-RUN-FINDINGS.md $H/D/night-1114-fuzz/*.md $H/B/night-1114-soak/*.md; do [ -f "$f" ] && cp "$f" $N/agent-$(echo ${f#$H/} | tr / -); done
grep -lq '@TOTN@' $N/agent-B-night-1114-soak-SOAK.md 2>/dev/null && rm -f $N/agent-B-night-1114-soak-SOAK.md   # B's unfinished draft (placeholders) stays out
# raw result files: logs, counters, per-run json (texts of the big sets stay out: file names are listed in FINAL.md)
( cd $Q/results/n14 && find . -type f \( -name '*.log' -o -name 'vals.json' -o -name 'users.jsonl' -o -name 'env.txt' -o -name 'qeval-*.json' -o -name 'rt.json' -o -name 'N_*_args' -o -name 'tables.txt' \) -size -1500k | grep -v '/engine' | while read f; do mkdir -p $N/raw/$(dirname $f); cp $f $N/raw/$f; done )
# no private addresses in what the repo holds (hostnames are fine)
grep -rIlE '10\.220\.0\.[0-9]+|100\.97\.13\.15|100\.83\.230\.9' $N $T/harness/night-1114 $T/selftest 2>/dev/null | while read f; do sed -i '' -E 's/10\.220\.0\.1([^0-9]|$)/spark-head\1/g; s/10\.220\.0\.2([^0-9]|$)/spark-3d24\1/g; s/10\.220\.0\.3([^0-9]|$)/spark-4e5c\1/g; s/10\.220\.0\.4([^0-9]|$)/spark-923b\1/g; s/100\.97\.13\.15/spark-head/g; s/100\.83\.230\.9/spark-3d24/g' "$f"; done
cd $R; git add -A .
if git grep -IqE --cached '(^|[^0-9.])(10\.220\.[0-9]+\.[0-9]+|100\.97\.[0-9]+\.[0-9]+|100\.83\.[0-9]+\.[0-9]+)'; then echo "PRIVATE ADDRESS IN TREE: abort"; git grep -IlE --cached '(^|[^0-9.])(10\.220\.[0-9]+\.[0-9]+|100\.97\.[0-9]+\.[0-9]+|100\.83\.[0-9]+\.[0-9]+)' | head; exit 5; fi
# key values must not be in the tree (values are never printed)
GK=$(cat ~/.config/opencode/deepseek.key 2>/dev/null | tr -d '\n'); EK=$(ssh -n -o ConnectTimeout=15 -i ~/.ssh/codex_spark_ed25519 pico@spark-head 'cat ~/.config/qwen38/api_key 2>/dev/null' | tr -d '\n')
for k in "$GK" "$EK"; do [ ${#k} -ge 8 ] && { if git grep -qF --cached -e "$k"; then echo "KEY VALUE FOUND IN TREE: abort"; exit 3; fi; }; done; echo "key values absent (gateway len ${#GK}, engine len ${#EK})"
[ ${#GK} -ge 8 ] && [ ${#EK} -ge 8 ] || echo "WARNING: a key could not be read, so that grep did not run"
echo "files changed: $(git diff --cached --name-only | wc -l)"
# gitleaks on the changed files as plain files (no git) and on the staged diff
rm -rf /tmp/q38_glN; mkdir -p /tmp/q38_glN; git diff --cached --name-only --diff-filter=AM | while read f; do mkdir -p "/tmp/q38_glN/$(dirname "$f")"; cp "$f" "/tmp/q38_glN/$f"; done
gitleaks detect --no-git --source /tmp/q38_glN --no-banner --redact > /tmp/q38_glN.txt 2>&1 || { grep -E 'File:|RuleID:|Line:' /tmp/q38_glN.txt | head -30; echo 'GITLEAKS (no-git) FOUND SOMETHING: abort (nothing committed)'; exit 4; }; tail -1 /tmp/q38_glN.txt
gitleaks protect --staged --no-banner --redact > /tmp/q38_gl.txt 2>&1 || { grep -E 'File:|RuleID:|Line:' /tmp/q38_gl.txt | head -30; echo 'GITLEAKS (staged) FOUND SOMETHING: abort (nothing committed)'; exit 4; }; tail -1 /tmp/q38_gl.txt
[ "$DRY" = 1 ] && { echo "DRY: not committed"; exit 0; }
git -c user.name=Pico -c user.email=50418071+Picom-code@users.noreply.github.com commit -q -m "$MSG"
git push -q origin main 2>&1 | tail -1
echo "pushed $(git rev-parse HEAD); author $(git log -1 --format='%an <%ae>'); remote main $(git ls-remote origin main | cut -c1-12)"
echo "repo private: $(gh api repos/Picom-code/pico-dgx-qwen38 --jq .private); anonymous http $(curl -s -o /dev/null -w '%{http_code}' https://github.com/Picom-code/pico-dgx-qwen38)"
