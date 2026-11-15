#!/bin/bash
# A10/lease.sh acquire|renew MIN | release | status : builder A's lease on the shared window + the deadline file on 923b (lease end - 2 min).
Q=~/.hermes/cache/scratch/q38; case $1 in
acquire|renew) $Q/h.sh "bash ~/q38/shared/lease.sh $1 A $2 2>&1 | tail -2; ssh -n picom@spark-923b 'echo \$(( \$(date +%s) + $2*60 - 120 )) > ~/q38/runs/A_deadline; date -u +%T'" ;;
release) $Q/h.sh "bash ~/q38/shared/lease.sh release A 2>&1 | tail -1; ssh -n picom@spark-923b 'echo 0 > ~/q38/runs/A_deadline; docker ps --format {{.Names}} | grep q38-run'; bash ~/q38/shared/lease.sh status 2>&1 | tail -6" ;;
*) $Q/h.sh "bash ~/q38/shared/lease.sh status 2>&1 | tail -8" ;; esac
