#!/bin/bash
# rj.sh NAME rt.py-arguments... : run one rt.py command, log to $O/NAME.log, print its last lines, return ITS exit code (3 = deadline: resumable)
N=$1; shift; python3 -I $RT/rt.py "$@" >> $O/$N.log 2>&1; rc=$?; grep -c "LIVE DIFF" $O/$N.log | sed "s/^/[$N] live differences so far: /"; tail -2 $O/$N.log | cut -c1-400; exit $rc
