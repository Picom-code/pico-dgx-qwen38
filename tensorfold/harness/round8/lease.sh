#!/bin/bash
# lease.sh acquire NAME MINUTES | renew NAME MINUTES | release NAME | status
# Shared Qwen test window for several agents. DeepSeek is DOWN while any lease is live and comes back by itself when none is.
# acquire: registers the lease, opens a window if none is open (or waits for a closing one to finish, then opens a new one),
#          returns when DeepSeek is stopped and the boxes are free ("WINDOW OPEN until HH:MM:SS UTC"). Max 120 min a call; renew before expiry.
# A window lives at most 300 min (guard); when it ends every Qwen container on all four boxes is killed. Release as soon as your runs are in.
S=$HOME/q38/shared; mkdir -p $S/leases; A=$1; N=$2; M=${3:-30}; [ "$M" -gt 120 ] && M=120
st() { echo "unit=$(systemctl --user is-active q38-win) open=$([ -f $S/OPEN ] && echo yes || echo no) closing=$([ -f $S/CLOSING ] && echo yes || echo no) now=$(date -u +%T) leases: $(for f in $S/leases/*; do [ -f "$f" ] && echo -n "$(basename $f)->$(date -u -d @$(cat $f) +%T) "; done)"; }
case "$A" in
status) st;;
release) rm -f $S/leases/$N; st;;
acquire|renew)
  for i in $(seq 160); do
    exec 9>$S/lock; flock 9
    if [ -f $S/CLOSING ]; then flock -u 9; [ "$A" = renew ] && { echo "WINDOW CLOSING: your engine has been or will be killed; acquire again"; exit 3; }; sleep 15; continue; fi
    if [ "$A" = renew ] && [ ! -f $S/OPEN ]; then flock -u 9; echo "NO OPEN WINDOW"; exit 3; fi
    echo $(( $(date +%s) + M*60 )) > $S/leases/$N
    if ! systemctl --user is-active -q q38-win; then systemctl --user reset-failed q38-win 2>/dev/null; rm -f $S/OPEN
      systemd-run --user --unit=q38-win /bin/bash $S/qws.sh S$(date -u +%d%H%M) 300 $S/w_shared.sh >/dev/null 2>&1; echo "opened a new window"; fi
    flock -u 9; break
  done
  for i in $(seq 90); do
    [ -f $S/OPEN ] && [ ! -f $S/CLOSING ] && { echo "WINDOW OPEN until $(date -u -d @$(cat $S/leases/$N) +%T) UTC for $N"; st; exit 0; }
    # the old window's unit can still be active for a moment after it dropped CLOSING: then nobody opened a new one; do it here
    if [ "$A" = acquire ] && [ ! -f $S/CLOSING ] && [ ! -f $S/OPEN ] && ! systemctl --user is-active -q q38-win; then
      exec 9>$S/lock; flock 9
      if ! systemctl --user is-active -q q38-win; then systemctl --user reset-failed q38-win 2>/dev/null
        echo $(( $(date +%s) + M*60 )) > $S/leases/$N
        systemd-run --user --unit=q38-win /bin/bash $S/qws.sh S$(date -u +%d%H%M) 300 $S/w_shared.sh >/dev/null 2>&1; echo "opened a new window (late)"; fi
      flock -u 9
    fi
    sleep 4
  done
  rm -f $S/leases/$N; echo "window did not open in 6 min; lease dropped"; st; exit 2;;
*) echo "usage: lease.sh acquire|renew NAME MIN | release NAME | status"; exit 1;;
esac
