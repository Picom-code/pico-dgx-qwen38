#!/bin/bash
# rs.sh [NAME=VALUE ...] : start Qwen through the DOCUMENTED path on this worker: ~/q38/repo_tf/recipe.sh start, i.e. MiaAI-Lab's start.sh on the
# image that `recipe.sh prepare` built (patches 0001-0017 + 0101-0108). Container q38-run-A, API 127.0.0.1:18301. Arguments are recipe settings
# (CERT=0, USERS16=1, WIDE_ROWS=128, GATE=structured) or engine variables.
Q=$HOME/q38; T0=$(date +%s); mkdir -p $Q/logs $Q/tf/kcache-recipe
docker rm -f q38-run-A >/dev/null 2>&1
cd $Q/repo_tf && env WORK=$Q/recipe_work HF_CACHE=$Q/hf KERNEL_CACHE=$Q/tf/kcache-recipe STATE_DIR=$Q/tf/state LOG_DIR=$Q/tf/logs HOST=127.0.0.1 PORT=18301 \
  CONTAINER_NAME=q38-run-A PREPARE=0 WAIT_TIMEOUT=900 "$@" bash ./recipe.sh start > $Q/logs/recipe_start.log 2>&1; rc=$?
if [ $rc = 0 ] && curl -sf -m 5 http://127.0.0.1:18301/v1/models >/dev/null; then
  echo "READY recipe [$*] in $(( $(date +%s)-T0 )) s; MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GiB; GPU MiB $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | paste -sd+); image $(docker inspect -f '{{.Config.Image}}' q38-run-A) entrypoint $(docker inspect -f '{{.Path}}' q38-run-A)"
else echo "FAILED recipe [$*] rc $rc"; tail -15 $Q/logs/recipe_start.log | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-240; fi
