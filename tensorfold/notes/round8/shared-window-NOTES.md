# Shared window harness notes (owner: builder A)
- 17:08 UTC lease.sh fixed (head + Mac copy): after a window closed, an `acquire` that had been waiting could register its lease while the old
  unit was still active for a moment; no new window was started and the call ended "window did not open" leaving a stale lease.
  Now the wait loop starts the window itself when no unit is active, and drops the lease if the window does not open.
  Usage unchanged: acquire NAME MIN | renew NAME MIN | release NAME | status.
- Engines: the window's stop-all kills only containers whose name starts with q38-run. Build containers must NOT use that prefix.
