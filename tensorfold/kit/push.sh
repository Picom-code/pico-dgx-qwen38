#!/bin/bash
# push.sh BOX DESTDIR FILE... : copy Mac files to a worker directory through the head.
BOX=$1; DEST=$2; shift 2; K=~/.ssh/codex_spark_ed25519
ssh -n -i $K pico@spark-head 'mkdir -p /tmp/q38push && rm -f /tmp/q38push/*'
scp -q -i $K "$@" pico@spark-head:/tmp/q38push/ && ssh -n -i $K pico@spark-head "scp -q /tmp/q38push/* $BOX:$DEST/ && ssh -n $BOX 'cd $DEST && sha256sum $(for f in "$@"; do printf '%s ' $(basename $f); done) | cut -c1-12 | tr \"\\n\" \" \"'"; echo; for f in "$@"; do shasum -a 256 $f | cut -c1-12; done | tr '\n' ' '; echo
