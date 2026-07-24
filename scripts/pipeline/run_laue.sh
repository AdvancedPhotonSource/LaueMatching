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
# EVERYTHING you need to change is in the CONFIG block below. Nothing here is
# specific to any one experiment -- point WORK/PY/SCRIPTS at your install and
# ALPHA_CONFIG/BETA_CONFIG at your parameter files (see params_*.template.txt).
# =============================================================================
set -euo pipefail

# ----------------------------- CONFIG (edit me) ------------------------------
WORK=${WORK:-"$HOME/laue_run"}                       # working dir: results land here
PY=${PY:-"python"}                                   # python with LaueMatching installed
SCRIPTS=${SCRIPTS:-"$(cd "$(dirname "$0")/.." && pwd)"}   # LaueMatching/scripts (auto)
ALPHA_CONFIG=${ALPHA_CONFIG:-"$WORK/params/params_alpha.txt"}
# "-" not ":-", for the same reason as WATCH below: ${BETA_CONFIG:-...} substitutes
# the default for an *empty* BETA_CONFIG too, so the documented BETA_CONFIG=""
# (single-phase material, e.g. Zn) never took effect. It fell through to the
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
# -----------------------------------------------------------------------------

FOLDER=${1:?usage: run_laue.sh DATA_FOLDER [H5_LOCATION]}
H5LOC=${2:-/entry1/data/data}
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$WORK/results"
cd "$WORK"

launch() {   # launch PHASE CONFIG GPU PORT
  local phase=$1 config=$2 gpu=$3 port=$4
  [ -z "$config" ] && return 0
  [ -f "$config" ] || { echo "ERROR: config not found: $config" >&2; exit 1; }
  local out="$WORK/results/${phase}_$TS"
  # CUDA_DEVICE_ORDER=PCI_BUS_ID is required for $gpu to mean the same card that
  # nvidia-smi calls $gpu. CUDA defaults to FASTEST_FIRST, so on a mixed-GPU host
  # "GPU 0" can land on a completely different physical card -- observed on a shared
  # machine where the daemon landed on another user's GPU while the intended cards
  # sat idle.
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu \
      setsid nohup "$PY" "$SCRIPTS/laue_orchestrator.py" \
      --config "$config" --folder "$FOLDER" --h5-location "$H5LOC" \
      --ncpus "$NCPUS" --port "$port" $WATCH --output-dir "$out" \
      > "$WORK/results/${phase}_$TS.launch.log" 2>&1 < /dev/null &
  echo "  $phase: pid $! on GPU$gpu port $port  ->  results/${phase}_$TS/"
}

echo "LaueMatching run $TS   data=$FOLDER   h5=$H5LOC"
launch alpha "$ALPHA_CONFIG" "$ALPHA_GPU" "$ALPHA_PORT"
launch beta  "$BETA_CONFIG"  "$BETA_GPU"  "$BETA_PORT"
echo "STOP (watch mode):  touch $FOLDER/STOP_LAUE"
echo "Per-frame results (indexed orientations + assigned spots) appear under results/*_$TS/."
