#!/bin/bash
# watch_arm.sh PLANFILE [PLANFILE...]
#
# Watch every run of the given plans and print ONE line when there is something
# to act on, then exit:
#   exit 0  "ALL N RUNS EXITED CLEANLY"  every daemon logged a clean exit
#   exit 1  "FAILURE: <tag>[<signatures>] ..."  at least one run shows a failure
#   exit 2  "WATCH_TIMEOUT ..."          neither, after WATCH_MAX_POLLS polls
# A clean daemon exit is NOT the end of the run: post-processing writes the
# per-frame outputs afterwards, in one late batch. Follow this with
# wait_static.sh before reading any result.
#
# Environment: WATCH_INTERVAL (s, default 60), WATCH_MAX_POLLS (default 240).
#
# Reads only the shared filesystem (run/<TAG>/results/ and logs/ under each
# plan's WORKDIR), so it runs anywhere that sees it -- no ssh.
#
# Failure signatures, and what each has meant:
#   GPUassert             the daemon died on a CUDA call. After "res = 11" it is a
#                         CONSEQUENCE of thread exhaustion, not a GPU fault.
#   res = 11              OpenCV could not spawn a thread (EAGAIN): the per-user
#                         process limit is exhausted -- preprocessing pool too big.
#   Address already in use / did not open port
#                         port collision, or the daemon never got to bind
#                         (often NFS saturated by simultaneous launches).
#   CUDA error, Traceback, Post-processing failed
#                         the orchestrator logs "Pipeline complete" even after
#                         a failed post-processing step, so look for this too.
#   fork: / Resource temporarily unavailable (in logs/dispatch_*.out)
#                         the dispatcher itself could not fork: later shards
#                         never launched.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=lib.sh
. "$HERE/lib.sh"
[ $# -gt 0 ] || { sed -n '2,3p' "$0" >&2; exit 2; }
read_plans "$@" || exit 2
N=${#PLAN_LINES[@]}
INTERVAL=${WATCH_INTERVAL:-60}; MAXP=${WATCH_MAX_POLLS:-240}
SIG='GPUassert|Address already in use|did not open port|CUDA error|res = 11|Traceback|Post-processing failed'

for _ in $(seq 1 "$MAXP"); do
  clean=0; msg=""
  for ln in "${PLAN_LINES[@]}"; do
    # shellcheck disable=SC2086
    set -- $ln
    W=$4; TAG=$5; R="$W/run/$TAG/results"
    logs=$(find "$R" -maxdepth 2 \( -name daemon.log -o -name '*.launch.log' \) 2>/dev/null)
    extra=$(find "$W/logs" -maxdepth 1 \( -name "dispatch_$TAG.out" -o -name "${TAG}_*.launch" \) 2>/dev/null)
    hits=""
    if [ -n "$logs$extra" ]; then
      # shellcheck disable=SC2086
      hits=$(grep -hoE "$SIG" $logs $extra 2>/dev/null | sort -u | tr '\n' ',' || true)
      # shellcheck disable=SC2086
      # (guarded: grep with no file arguments would read stdin and hang)
      if [ -n "$extra" ] && grep -qsE 'fork: |Resource temporarily unavailable' $extra </dev/null; then
        hits="${hits}DISPATCHER_CANNOT_FORK,"
      fi
    fi
    [ -n "$hits" ] && msg="$msg ${TAG}[${hits%,}]"
    dl=$(find "$R" -maxdepth 2 -name daemon.log 2>/dev/null | sort | tail -n 1)
    [ -n "$dl" ] && grep -q "exited cleanly" "$dl" 2>/dev/null && clean=$((clean + 1))
  done
  if [ -n "$msg" ]; then echo "FAILURE:$msg"; exit 1; fi
  if [ "$clean" -ge "$N" ]; then echo "ALL $N RUNS EXITED CLEANLY (now run wait_static.sh)"; exit 0; fi
  sleep "$INTERVAL"
done
echo "WATCH_TIMEOUT after $MAXP polls: $clean/$N runs exited cleanly"
exit 2
