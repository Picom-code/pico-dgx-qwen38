#!/bin/bash
# w.sh 'cmd' : run a command on 923b (through the head)
exec ssh -n -o ConnectTimeout=15 -o BatchMode=yes -i ~/.ssh/codex_spark_ed25519 pico@spark-head "ssh -n -o ConnectTimeout=10 picom@spark-923b $(printf '%q' "$1")"
