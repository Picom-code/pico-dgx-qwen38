#!/bin/bash
# qstop.sh (any worker): stop every Qwen test container, delete temporary checkpoint copies (~/q38/tmpmodels, used on boxes whose
# disk DeepSeek's session-store disk tier needs), evict the Qwen files + container image layers from page cache. No sudo.
docker ps -a --format '{{.Names}}' | grep '^q38-run' | xargs -r docker rm -f >/dev/null 2>&1
pkill -f 'q38/kit/xfer.py' 2>/dev/null
for i in $(seq 30); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && break; sleep 1; done
rm -rf $HOME/q38/tmpmodels $HOME/q38/tmpkeep
EV=$HOME/q38/kit/evict.py
[ -f $EV ] && python3 $EV ~/q38/models ~/q38/hf ~/q38/custom engine R ~/q38/tf ~/q38/tf127 ~/q38/tfbin ~/q38/flare ~/q38/img ~/q38/nvref ~/models/qwen38-flash-next-nvfp4
docker rm -f q38-evict >/dev/null 2>&1; [ -f $EV ] && timeout 90 docker run --rm --name q38-evict --memory 1g -v /var/lib/docker:/d:ro -v /var/lib/containerd:/c:ro -v $HOME/q38/kit:/k:ro python:3.12-slim python /k/evict.py /d /c
docker rm -f q38-evict >/dev/null 2>&1
docker ps -a --format '{{.Names}}' | grep '^q38-dl' | xargs -r docker unpause >/dev/null 2>&1
echo "$(hostname) $(free -g | sed -n 2p) disk_free $(df -h ~ | tail -1 | awk '{print $4}') gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
