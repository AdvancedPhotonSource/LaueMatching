#!/bin/bash
# =============================================================================
# run_laue.sh -- turnkey dual-phase (or single-phase) real-time LaueMatching.
#
# Launches one GPU indexer per crystallographic phase in WATCH mode: every new
# .h5 frame written into DATA_FOLDER is indexed as it lands. For an existing
# dataset it processes all frames then exits (see WATCH below).
#
#   ./run_laue.sh  DATA_FOLDER  [H5_LOCATION]
#
# Stop a watch-mode run cleanly at any time with:
#   touch DATA_FOLDER/STOP_LAUE
#
# Check what a run WOULD use without launching anything:
#   DRY_RUN=1 ./run_laue.sh DATA_FOLDER
#
# EVERYTHING you need to change is in the CONFIG block below. Nothing here is
# specific to any one experiment -- point WORK/PY/SCRIPTS at your install and
# ALPHA_CONFIG/BETA_CONFIG at your parameter files (see params_*.template.txt).
# For many shards across several hosts use pipeline/dispatch/ (see its README).
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# ----------------------------- CONFIG (edit me) ------------------------------
WORK=${WORK:-"$HOME/laue_run"}                       # working dir: results land here
PY=${PY:-"python"}                                   # python with laue-index installed
# SCRIPTS = the directory holding laue_orchestrator.py. Resolved below when unset:
#   1. this checkout's scripts/ (run_laue.sh lives in pipeline/, so ../scripts);
#   2. otherwise the installed package's laue_index/pipeline/, asked of $PY.
# The old default, $(dirname $0)/.., was right when this script lived in scripts/
# and resolved to the repo ROOT once it moved to pipeline/. The launch below is
# detached, so "can't open file" went to a log nobody read while this script
# printed a pid as if the run had started.
SCRIPTS=${SCRIPTS:-}
ALPHA_CONFIG=${ALPHA_CONFIG:-"$WORK/params/params_alpha.txt"}
# "-" not ":-", for the same reason as WATCH below: ${BETA_CONFIG:-...} substitutes
# the default for an *empty* BETA_CONFIG too, so the documented BETA_CONFIG=""
# (single-phase material) never took effect. It fell through to the
# default path, failed the -f check, and exited 1 -- after alpha had already been
# launched, so the run appeared to work while reporting an error.
BETA_CONFIG=${BETA_CONFIG-"$WORK/params/params_beta.txt"}   # set BETA_CONFIG="" to skip beta
ALPHA_GPU=${ALPHA_GPU:-0};  ALPHA_PORT=${ALPHA_PORT:-60517}
BETA_GPU=${BETA_GPU:-1};    BETA_PORT=${BETA_PORT:-60518}
NCPUS=${NCPUS:-32}                                   # CPU cores for the refinement stage
# NB: "-" not ":-" — ${WATCH:-...} would substitute the default for an *empty*
# WATCH too, so the documented WATCH="" (batch an existing folder) never took
# effect and every run silently stayed in watch mode.
WATCH=${WATCH-"--watch"}                             # set WATCH="" to batch an existing folder
LIVENESS_WAIT=${LIVENESS_WAIT:-8}                    # seconds before checking each launch is alive
# -----------------------------------------------------------------------------

FOLDER=${1:?usage: run_laue.sh DATA_FOLDER [H5_LOCATION]}
H5LOC=${2:-/entry1/data/data}

# --- resolve and CHECK everything before launching anything -----------------
if [ -z "$SCRIPTS" ]; then
  if [ -f "$HERE/../scripts/laue_orchestrator.py" ]; then
    SCRIPTS="$(cd "$HERE/../scripts" && pwd)"
  else
    # No checkout beside us (e.g. this file was copied out): use the installed
    # package, which ships the same modules in laue_index/pipeline/.
    SCRIPTS=$("$PY" -c 'from laue_index.pipeline import PIPELINE_DIR; print(PIPELINE_DIR)' 2>/dev/null || true)
  fi
fi
[ -n "$SCRIPTS" ] && [ -f "$SCRIPTS/laue_orchestrator.py" ] || {
  echo "ERROR: laue_orchestrator.py not found in SCRIPTS='$SCRIPTS'." >&2
  echo "  Set SCRIPTS to <checkout>/scripts, or install laue-index into PY='$PY'." >&2
  exit 1
}
command -v "$PY" >/dev/null 2>&1 || { echo "ERROR: PY='$PY' is not an executable python." >&2; exit 1; }
[ -d "$FOLDER" ] || { echo "ERROR: DATA_FOLDER not found: $FOLDER" >&2; exit 1; }
for c in "$ALPHA_CONFIG" "$BETA_CONFIG"; do
  [ -z "$c" ] && continue
  [ -f "$c" ] || { echo "ERROR: config not found: $c" >&2; exit 1; }
  # The templates carry __SET_ME__ where an experiment-specific value belongs.
  # The parsers do not reject it (they fall back to a default), so refuse here.
  if grep -qE '^[^#]*__SET_ME__' "$c"; then
    echo "ERROR: $c still contains __SET_ME__ placeholders:" >&2
    grep -nE '^[^#]*__SET_ME__' "$c" >&2
    exit 1
  fi
done

TS=$(date +%Y%m%d_%H%M%S)
echo "LaueMatching run $TS   data=$FOLDER   h5=$H5LOC"
echo "  WORK=$WORK  PY=$PY  SCRIPTS=$SCRIPTS"
if [ -n "${DRY_RUN:-}" ]; then
  echo "  alpha config: ${ALPHA_CONFIG:-<none>}"
  echo "  beta  config: ${BETA_CONFIG:-<none>}"
  echo "DRY_RUN set: nothing launched."
  exit 0
fi
mkdir -p "$WORK/results"
cd "$WORK"

launch() {   # launch PHASE CONFIG GPU PORT
  local phase=$1 config=$2 gpu=$3 port=$4
  [ -z "$config" ] && return 0
  local out="$WORK/results/${phase}_$TS"
  local log="$WORK/results/${phase}_$TS.launch.log"
  # CUDA_DEVICE_ORDER=PCI_BUS_ID is required for $gpu to mean the same card that
  # nvidia-smi calls $gpu. CUDA defaults to FASTEST_FIRST, so on a mixed-GPU host
  # "GPU 0" can land on a completely different physical card -- observed on a shared
  # machine where the daemon landed on another user's GPU while the intended cards
  # sat idle.
  # $WATCH is deliberately unquoted: empty must vanish, not become an "" argument.
  # shellcheck disable=SC2086
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu \
      setsid nohup "$PY" "$SCRIPTS/laue_orchestrator.py" \
      --config "$config" --folder "$FOLDER" --h5-location "$H5LOC" \
      --ncpus "$NCPUS" --port "$port" $WATCH --output-dir "$out" \
      > "$log" 2>&1 < /dev/null &
  local pid=$!
  # The launch is detached, so a crash on startup (bad path, import error, bad
  # config) lands in the log and nowhere else. Check the process is still
  # there before reporting it. $! is the orchestrator itself: a background job of
  # a non-interactive shell is not a process-group leader, so setsid does not
  # fork and exec's in place.
  sleep "$LIVENESS_WAIT"
  if ! kill -0 "$pid" 2>/dev/null; then
    local rc=0
    wait "$pid" || rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "ERROR: $phase (pid $pid) exited with status $rc within ${LIVENESS_WAIT}s. Last lines of $log:" >&2
      tail -n 20 "$log" >&2 || true
      exit 1
    fi
    echo "  $phase: pid $pid finished with status 0 within ${LIVENESS_WAIT}s (tiny batch?) -- check $log"
    return 0
  fi
  echo "  $phase: pid $pid on GPU$gpu port $port  ->  results/${phase}_$TS/  (alive after ${LIVENESS_WAIT}s)"
}

launch alpha "$ALPHA_CONFIG" "$ALPHA_GPU" "$ALPHA_PORT"
launch beta  "$BETA_CONFIG"  "$BETA_GPU"  "$BETA_PORT"
echo "STOP (watch mode):  touch $FOLDER/STOP_LAUE"
echo "Per-frame results (indexed orientations + assigned spots) appear under results/*_$TS/."
echo "Alive at launch is not success: count results/*_$TS/results/image_*.output.h5 when it ends."
