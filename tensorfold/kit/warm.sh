#!/bin/bash
# warm.sh : read the 7 checkpoint shards that hold the n-gram table (30 GiB) into the page cache (after the engine has loaded).
T0=$(date +%s); cat $(cat ~/q38/kit/ple_files.txt) > /dev/null; echo "warm: n-gram shards read in $(( $(date +%s)-T0 )) s; MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GiB; Cached $(awk '/^Cached/{print int($2/1048576)}' /proc/meminfo) GiB"
