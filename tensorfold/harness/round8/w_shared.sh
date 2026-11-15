# w_shared.sh : body of a SHARED test window (sourced by qws.sh after DeepSeek is stopped). Stays open while any lease is live.
S=$HOME/q38/shared; mkdir -p $S/leases; rm -f $S/CLOSING; date +%s > $S/OPEN
echo "shared window open $(date -u +%T)"
while :; do
  sleep 20
  exec 9>$S/lock; flock 9
  now=$(date +%s); live=0
  for f in $S/leases/*; do [ -f "$f" ] || continue; e=$(cat "$f" 2>/dev/null); if [ "${e:-0}" -gt "$now" ]; then live=1; else echo "lease $(basename $f) expired $(date -u +%T)"; rm -f "$f"; fi; done
  if [ $live = 0 ]; then touch $S/CLOSING; rm -f $S/OPEN; flock -u 9; break; fi
  flock -u 9
done
echo "no live lease, closing $(date -u +%T)"
