#!/bin/bash
# preflight.sh PLANFILE [PLANFILE...]
#
# Assert, BEFORE dispatch, every condition that has previously produced a wrong
# or empty result without failing. Exits non-zero on the first failure. Pass
# EVERY plan that will run at the same time: uniqueness is checked across all
# of them together, not per plan.
#
# Environment:
#   PY        python with laue-index installed, as seen ON THE SHARD HOSTS (REQUIRED)
#   SCRIPTS   directory holding laue_orchestrator.py (default: this checkout's scripts/)
#   LOCAL_PY  python with numpy on THIS host, for the background check (default: $PY)
#   BG_MEDIAN_MIN  smallest plausible background median (default 50)
#
# What each check exists to catch:
#  * Two shards sharing a PORT: the second daemon binds nothing and sits there
#    fully initialised, holding its GPU, having received zero images.
#  * Two shards sharing a ResultDir: both append to one solutions.txt and
#    interleave. It fails late, silently, and nothing is recoverable.
#  * Duplicate TAGs: logs and run directories overwrite each other.
#  * Two shards on one HOST:GPU: each daemon holds its full orientation set on
#    the card; two of them compete for memory and time. Set ALLOW_SHARED_GPU=1
#    if that is really intended.
#  * A missing BackgroundFile does NOT fail: the image server silently computes
#    one from the FIRST FRAME. A background built from dead or blank frames has
#    a median of a few counts and subtracts nothing; hence the size and median
#    checks (size = NrPxX * NrPxY float64 read from each params file, e.g.
#    33554432 bytes for 2048 x 2048).
#  * __SET_ME__ left in a params file: the parsers fall back to defaults.
#  * A port already bound on the host (another user's job, a leftover daemon).
#  * The indexer binaries: asked of the installed package ON EACH HOST, never
#    inferred from `ls bin/` -- a macOS binary once sat in a Linux host's bin/
#    looking present and ready.
# Remote checks go through lib.sh on_host (`ssh HOST bash -s`, never `ssh -n`)
# and each prints a sentinel; no sentinel means ssh broke, which is reported as
# such rather than read as a verdict.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=lib.sh
. "$HERE/lib.sh"
fail() { echo "PREFLIGHT FAIL: $*" >&2; exit 1; }
[ $# -gt 0 ] || fail "usage: preflight.sh PLAN [PLAN...]"
: "${PY:?PY must name the python with laue-index installed, as seen on the shard hosts}"
if [ -z "${SCRIPTS:-}" ]; then
  [ -d "$HERE/../../scripts" ] || fail "SCRIPTS is not set and $HERE/../../scripts does not exist; set SCRIPTS to the directory holding laue_orchestrator.py"
  SCRIPTS="$(cd "$HERE/../../scripts" && pwd)"
fi
LOCAL_PY=${LOCAL_PY:-$PY}
BG_MEDIAN_MIN=${BG_MEDIAN_MIN:-50}

read_plans "$@" || fail "could not read plans"
[ ${#PLAN_LINES[@]} -gt 0 ] || fail "no runs in: $*"

param() { awk -v k="$1" '$1==k {print $2; exit}' "$2"; }

PORTS=(); TAGS=(); RDS=(); BGS=(); SLOTS=()
for ln in "${PLAN_LINES[@]}"; do
  # shellcheck disable=SC2086
  set -- $ln
  H=$1; G=$2; PORT=$3; TAG=$5; PAR=$6; SHARD=$7
  PORTS+=("$PORT"); TAGS+=("$TAG"); SLOTS+=("$H:$G")
  [ -f "$PAR" ] || fail "$TAG: params missing: $PAR"
  if grep -nE '^[^#]*__SET_ME__' "$PAR" >&2; then fail "$TAG: $PAR still has __SET_ME__ placeholders (above)"; fi
  [ -d "$SHARD" ] || fail "$TAG: shard dir missing: $SHARD"
  # find, not ls: `ls dir/* | wc -l` returns 0 past ARG_MAX (~40k files).
  n=$(find "$SHARD" -mindepth 1 -maxdepth 1 -name '*.h5' | wc -l | tr -d ' ')
  [ "$n" -gt 0 ] || fail "$TAG: shard dir has no .h5 frames: $SHARD"
  RD=$(param ResultDir "$PAR"); [ -n "$RD" ] || fail "$TAG: no ResultDir in $PAR"
  RDS+=("$RD")
  BG=$(param BackgroundFile "$PAR")
  [ -n "$BG" ] || fail "$TAG: no BackgroundFile -- the server would compute one from frame 1"
  [ -s "$BG" ] || fail "$TAG: BackgroundFile missing or empty: $BG"
  # The detector size comes from THIS params file; no 2048 x 2048 default.
  nx=$(param NrPxX "$PAR"); ny=$(param NrPxY "$PAR")
  [[ ${nx:-} =~ ^[1-9][0-9]*$ ]] || fail "$TAG: NrPxX missing or not a positive integer in $PAR (got '${nx:-}')"
  [[ ${ny:-} =~ ^[1-9][0-9]*$ ]] || fail "$TAG: NrPxY missing or not a positive integer in $PAR (got '${ny:-}')"
  want=$(( nx * ny * 8 ))
  sz=$(wc -c < "$BG" | tr -d ' ')
  [ "$sz" -eq "$want" ] || fail "$TAG: background is $sz bytes, expected $want (NrPxX*NrPxY float64)"
  BGS+=("$BG")
  printf "  %-20s %-10s gpu%-2s port%-6s %6s frames  bg=%s\n" "$TAG" "$H" "$G" "$PORT" "$n" "$(basename "$BG")"
done

# --- uniqueness across ALL plans passed together ----------------------------
# Ports are compared as bare numbers, not host:port: the same port on two hosts
# works today and collides the day one shard moves host.
dup() { printf '%s\n' "$@" | sort | uniq -d; }
d=$(dup "${PORTS[@]}"); [ -z "$d" ] || fail "DUPLICATE PORTS: $d"
d=$(dup "${TAGS[@]}");      [ -z "$d" ] || fail "DUPLICATE TAGS: $d"
d=$(dup "${RDS[@]}");       [ -z "$d" ] || fail "DUPLICATE ResultDirs: $d"
if [ -z "${ALLOW_SHARED_GPU:-}" ]; then
  d=$(dup "${SLOTS[@]}"); [ -z "$d" ] || fail "TWO SHARDS ON ONE GPU: $d (run those plans one after the other, or set ALLOW_SHARED_GPU=1)"
fi
echo "  unique: ${#PORTS[@]} ports, ${#TAGS[@]} tags, ${#RDS[@]} ResultDirs"

# --- background plausibility ------------------------------------------------
while IFS= read -r BG; do
  med=$("$LOCAL_PY" -c 'import sys, numpy as np; print("%.1f" % float(np.median(np.fromfile(sys.argv[1], dtype=np.float64))))' "$BG") \
    || fail "could not read $BG with LOCAL_PY=$LOCAL_PY (needs numpy)"
  awk -v m="$med" -v t="$BG_MEDIAN_MIN" 'BEGIN{exit !(m>t)}' \
    || fail "background $BG median $med <= $BG_MEDIAN_MIN -- implausible (built from dead frames?)"
  echo "  background $(basename "$BG") median $med  OK"
done < <(printf '%s\n' "${BGS[@]}" | sort -u)

# --- per host: ports free, install answers ----------------------------------
# Hosts go into an array first; nothing that runs ssh iterates a `while read`.
HOSTS=()
while IFS= read -r h; do HOSTS+=("$h"); done < <(plan_hosts)
for h in "${HOSTS[@]}"; do
  plist=""
  for ln in "${PLAN_LINES[@]}"; do
    # shellcheck disable=SC2086
    set -- $ln; [ "$1" = "$h" ] && plist="$plist $3"
  done
  # Everything host-specific is expanded HERE; the remote side is plain bash.
  out=$(on_host "$h" 2>&1 <<EOS || true
busy=""
if command -v ss >/dev/null 2>&1; then lst="ss -ltn"
elif command -v netstat >/dev/null 2>&1; then lst="netstat -ltn"
else lst=""; busy=" (no ss or netstat on host: cannot check)"; fi
for p in $plist; do
  [ -n "\$lst" ] && \$lst 2>/dev/null | grep -q ":\$p " && busy="\$busy \$p"
done
echo "PORTS_BUSY:\$busy"
[ -f $(printf %q "$SCRIPTS/laue_orchestrator.py") ] && echo "SCRIPTS_OK" || echo "SCRIPTS_MISSING"
[ -f $(printf %q "$HERE/launch_run.sh") ] && echo "LAUNCHER_OK" || echo "LAUNCHER_MISSING"
$(printf %q "$PY") - <<'PYEOF' 2>&1 || echo "PROBE BAD python failed"
import os
from laue_index import indexer
d = os.path.dirname(indexer.binary_path())
need = ["LaueMatchingGPUStream", "LaueMatchingGPU", "LaueMatchingCPU"]
missing = [b for b in need if not os.path.exists(os.path.join(d, b))]
ok = indexer.available() and not missing
print("PROBE OK" if ok else "PROBE BAD available=%s missing=%s" % (indexer.available(), missing))
PYEOF
EOS
)
  line=$(printf '%s\n' "$out" | grep '^PORTS_BUSY:' | tail -n 1 || true)
  [ -n "$line" ] || fail "$h: no answer over ssh (not a verdict -- fix ssh first). Output: ${out:-<empty>}"
  [ "$line" = "PORTS_BUSY:" ] || fail "$h: ports already bound:${line#PORTS_BUSY:}"
  printf '%s\n' "$out" | grep -q '^SCRIPTS_OK' || fail "$h: no laue_orchestrator.py in $SCRIPTS as seen from $h"
  printf '%s\n' "$out" | grep -q '^LAUNCHER_OK' || fail "$h: $HERE/launch_run.sh not visible from $h (shared filesystem?)"
  probe=$(printf '%s\n' "$out" | grep '^PROBE ' | tail -n 1 || true)
  [ -n "$probe" ] || fail "$h: binary probe printed NOTHING (ssh/quoting problem, not a verdict). Output: $out"
  [ "$probe" = "PROBE OK" ] || fail "$h: $probe (PY=$PY)"
  echo "  $h: ports free:$plist; orchestrator, launcher and all three indexer binaries present"
done
echo "PREFLIGHT OK"
