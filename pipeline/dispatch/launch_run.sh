#!/bin/bash
# launch_run.sh WORK TAG PARAMS SHARD_DIR GPU PORT [NCPUS] [PREPROC_WORKERS]
#
# One single-phase orchestrator for one shard, on whatever host this runs on.
# Normally started by dispatch.sh; safe to run by hand for a single shard.
#
# Environment:
#   PY        python with laue-index installed (REQUIRED -- no default: a login
#             shell over ssh does not have your conda env on PATH, and "python"
#             resolving to some other interpreter fails late and quietly)
#   SCRIPTS   directory holding laue_orchestrator.py (default: this checkout's
#             scripts/, i.e. pipeline/dispatch/../../scripts; checked either way)
#   H5_LOCATION   dataset path inside each frame (default /entry1/data/data)
#   LAUE_SHARDS_PER_HOST  how many shards share this host (set by dispatch.sh)
#
# Each run gets its own WORK/run/TAG so results never collide, and its own PORT
# so two daemons on one host cannot bind the same socket (the second one binds
# nothing and sits there holding its GPU having received zero images).
#
# PREPROCESSING WORKERS -- cap them, and cap the thread pools too.
# laue-index <= 0.7.1 sizes the image server's preprocessing pool from THE WHOLE
# MACHINE and knows nothing about sibling shards on the same host. Four shards on
# a 112-core host each asked for 112 workers, every worker fanned out its own
# OpenCV thread pool, and the per-user process limit (`ulimit -u`) ran out.
# Symptoms, in order: OpenCV "Can't spawn new thread: res = 11", the daemon dying
# with "GPUassert: CUDA-capable device(s) is/are busy or unavailable" (a
# CONSEQUENCE of the thread exhaustion, not a GPU or cubin problem), and finally
# the dispatcher itself failing to fork, so the remaining shards never launched.
# LAUE_PREPROCESS_WORKERS wins over the package's heuristic; OPENCV_NUM_THREADS=1
# (and the BLAS/OpenMP equivalents) stop each worker fanning out again, which is
# right anyway because the workers are already parallel across processes.
set -euo pipefail
usage() { sed -n '2,3p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
[ $# -ge 6 ] || usage
W=$1; TAG=$2; PAR=$3; SHARD=$4; GPU=$5; PORT=$6; NC=${7:-16}; PW=${8:-0}
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN_LAUE="$HERE/../run_laue.sh"

: "${PY:?PY must name the python that has laue-index installed (full path)}"
SCRIPTS=${SCRIPTS:-"$HERE/../../scripts"}
[ -x "$PY" ] || command -v "$PY" >/dev/null 2>&1 || { echo "ERROR: PY='$PY' is not executable on $(hostname)" >&2; exit 1; }
[ -f "$SCRIPTS/laue_orchestrator.py" ] || { echo "ERROR: no laue_orchestrator.py in SCRIPTS='$SCRIPTS' on $(hostname)" >&2; exit 1; }
[ -f "$RUN_LAUE" ] || { echo "ERROR: $RUN_LAUE not found" >&2; exit 1; }
[ -f "$PAR" ] || { echo "ERROR: params not found: $PAR" >&2; exit 1; }
[ -d "$SHARD" ] || { echo "ERROR: shard dir not found: $SHARD" >&2; exit 1; }

cpus() { nproc 2>/dev/null || getconf _NPROCESSORS_ONLN; }
if [ "$PW" -le 0 ]; then
  # No explicit size: this host's CPUs, shared by its shards, 3/4 kept for the
  # pool (the rest covers fitting threads and the orchestrator).
  PW=$(( $(cpus) * 3 / (4 * ${LAUE_SHARDS_PER_HOST:-1}) ))
fi
[ "$PW" -lt 4 ] && PW=4

H=$(hostname -s 2>/dev/null || hostname)
SW=$W/run/$TAG
mkdir -p "$SW/results" "$W/logs"
cd "$SW"
echo "  $TAG on $H: gpu=$GPU port=$PORT fit_threads=$NC preproc_workers=$PW shards_on_host=${LAUE_SHARDS_PER_HOST:-1}"
rc=0
WORK=$SW PY=$PY SCRIPTS=$SCRIPTS WATCH="" NCPUS=$NC \
  LAUE_PREPROCESS_WORKERS=$PW LAUE_SHARDS_PER_HOST=${LAUE_SHARDS_PER_HOST:-1} \
  OPENCV_NUM_THREADS=1 OMP_NUM_THREADS=$NC MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  ALPHA_CONFIG=$PAR BETA_CONFIG="" \
  ALPHA_GPU=$GPU ALPHA_PORT=$PORT \
  bash "$RUN_LAUE" "$SHARD" "${H5_LOCATION:-/entry1/data/data}" \
  > "$W/logs/${TAG}_${H}.launch" 2>&1 || rc=$?
echo "  host=$H tag=$TAG gpu=$GPU port=$PORT rc=$rc"
tail -n 3 "$W/logs/${TAG}_${H}.launch"
exit "$rc"
