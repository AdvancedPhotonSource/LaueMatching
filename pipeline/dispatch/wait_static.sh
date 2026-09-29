#!/bin/bash
# wait_static.sh EXPECTED_FRAMES PLANFILE [PLANFILE...]
#
# Wait until the per-frame outputs of all runs in the plans are COMPLETE and
# STATIC, then print one line and exit:
#   exit 0  "OUTPUTS COMPLETE AND STATIC: n/N"  count reached N and held across
#           two consecutive checks WAIT_INTERVAL apart
#   exit 1  "OUTPUTS STATIC BUT SHORT: n/N"     every orchestrator has logged
#           "Pipeline complete" and the count held across two checks, below N
#   exit 2  "WAIT_TIMEOUT ..."
#
# Why both conditions: "exited cleanly" in daemon.log means the daemon has seen
# every frame; the output.h5 tree is written AFTERWARDS by post-processing, in
# one late batch that is largely single-threaded (tens of minutes for a large
# solutions.txt). Counts were once still climbing thousands of frames short at
# the moment every daemon reported clean. And a count that has stopped moving is
# not finished while post-processing is still running, so "short" is only
# declared once each run's orchestrator has logged its summary.
#
# EXPECTED_FRAMES is the number of frames in the shards, summed over the plans.
# Post-processing writes one image_*.output.h5 per frame the image server saw,
# including a stub (n_filtered 0, skip_reason set) for a frame it skipped (no
# spots, beam off the specimen) or the daemon found no solution for, so the
# count reaches EXPECTED_FRAMES on a complete run and SHORT means frames are
# missing. (Runs post-processed before 0.8 wrote no stubs: there, SHORT is only
# a prompt to look.)
#
# Environment: WAIT_INTERVAL (s, default 30), WAIT_MAX_POLLS (default 480).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=lib.sh
. "$HERE/lib.sh"
[ $# -ge 2 ] || { sed -n '2,3p' "$0" >&2; exit 2; }
EXPECT=$1; shift
read_plans "$@" || exit 2
INTERVAL=${WAIT_INTERVAL:-30}; MAXP=${WAIT_MAX_POLLS:-480}

prev=-1; stable=0; tot=0
for _ in $(seq 1 "$MAXP"); do
  tot=0; finished=0
  for ln in "${PLAN_LINES[@]}"; do
    # shellcheck disable=SC2086
    set -- $ln
    R="$4/run/$5/results"
    # find, not ls: a glob over tens of thousands of files exceeds ARG_MAX and
    # `ls ... | wc -l` then reports 0.
    n=$(find "$R" -path '*/results/image_*.output.h5' 2>/dev/null | wc -l | tr -d ' ')
    tot=$((tot + n))
    if find "$R" -maxdepth 1 -name '*.launch.log' -exec grep -l "Pipeline complete" {} + 2>/dev/null | grep -q .; then
      finished=$((finished + 1))
    fi
  done
  if [ "$tot" -eq "$prev" ]; then stable=$((stable + 1)); else stable=0; fi
  prev=$tot
  if [ "$stable" -ge 1 ]; then       # the same count on two checks INTERVAL apart
    if [ "$tot" -ge "$EXPECT" ]; then echo "OUTPUTS COMPLETE AND STATIC: $tot/$EXPECT"; exit 0; fi
    if [ "$finished" -ge "${#PLAN_LINES[@]}" ]; then echo "OUTPUTS STATIC BUT SHORT: $tot/$EXPECT"; exit 1; fi
  fi
  sleep "$INTERVAL"
done
echo "WAIT_TIMEOUT after $MAXP polls: last count $tot/$EXPECT"
exit 2
