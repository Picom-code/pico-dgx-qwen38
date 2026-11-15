#!/bin/bash
# wdog.sh HH:MM:SS : at that UTC time today remove the study container (so no GPU job of ours is alive when the window's guard restarts DeepSeek)
T=$(date -u -d "$(date -u +%F) $1" +%s); sleep $(( T - $(date +%s) )); docker rm -f q38-study-A >/dev/null 2>&1; echo "watchdog fired $(date -u +%T)" >> ~/q38/runs/A_c1.log
