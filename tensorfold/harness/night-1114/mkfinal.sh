#!/bin/bash
# mkfinal.sh : tensorfold/notes/final-2026-11-15/FINAL.md = the hand-written head (A12/FINAL_head.md) + every table as tableN.py prints it
# + the names of the raw result files. Run after pullN.sh.
Q=~/.hermes/cache/scratch/q38; N=$Q/repo/tensorfold/notes/final-2026-11-15; mkdir -p $N
python3 -I $Q/A12/tableN.py $Q/results/n14 > $Q/results/n14/tables.txt 2>&1
{
  cat $Q/A12/FINAL_head.md
  echo; echo '## Tables (printed by harness/night-1114/tableN.py from the raw files; nothing typed in)'; echo; echo '```'
  cut -c1-400 $Q/results/n14/tables.txt
  echo '```'; echo
  echo '## Raw result files'; echo
  echo 'On spark-923b under `~/q38/runs/`, copied to the Mac `q38/results/n14/`. Logs, counters and per-run JSON are in `raw/` beside this note;'
  echo 'the reply texts of the big sets (held.jsonl, ident_on.json, edit / agent / pred / loop / sq*.jsonl, at*.json) are not in the repo, their names are listed here.'
  echo; echo '```'
  ( cd $Q/results/n14 && find . -type f | sort | while read f; do printf '%-46s %9s bytes\n' "$f" "$(stat -f %z "$f")"; done )
  echo '```'
} > $N/FINAL.md
wc -l $N/FINAL.md
