#!/bin/bash
# run_analysis_chain.sh — full validation/analysis pipeline for one indexed scan,
# runnable on any host. All paths come from the environment:
#
#   LAUE_PHASES          phases to analyse, comma- or space-separated (REQUIRED), e.g.
#                        "alpha,beta" or "zn". Exported comma-separated, which is
#                        the form laue_material / null_model / empirical_gate parse.
#   LAUE_WORK            work root (peel_map/, figures/ live here)
#   LAUE_SCAN_DATA       raw frames
#   LAUE_SCAN_<PHASE>    that phase's indexing-run dir (e.g. LAUE_SCAN_ALPHA)
#   LAUE_PARAMS_<PHASE>  that phase's params_*.txt; with ONE phase, LAUE_PARAMS may
#                        stand in (laue_material refuses the generic file for two)
#   LAUE_OUT_PREFIX      output basename prefix
#   LAUE_MOUNT_DEG       sample mount angle, degrees in [0, 90) -- needed by
#                        variant_coherence.py and validated_figures.py
#   PY                   python with numpy, scipy, h5py, matplotlib (default: python)
#   SCRIPTDIR            the analysis scripts (default: the directory holding this file)
#   NW                   worker processes (default 16)
#
# Steps, in order. Per phase: validate each phase, MEASURE the null on this scan
# (never inherit another scan's), gate against it. Only when BOTH alpha and beta
# are listed (the steps load Phase("alpha") and Phase("beta") and test one
# against the other): the beta-exclusion census and its null, Burgers parent-beta
# reconstruction, the retained-beta anchor null, the variant spatial-coherence
# check. validated_figures.py draws phases named alpha/beta only, so it runs when
# every listed phase is one of those. Skipped steps are announced.
#
# Every variable the steps that WILL run need is checked before the first step,
# and all missing ones are reported together. Every step must succeed or the
# chain stops (set -e): an import crash in one script used to be followed by the
# next step and "analysis chain finished".
set -euo pipefail
NW=${NW:-16}
# The scripts live next to this file. (The old default, $LAUE_WORK/scripts, only
# existed in a campaign directory that had copied them there.)
S=${SCRIPTDIR:-"$(cd "$(dirname "$0")" && pwd)"}
PY=${PY:-python}

log() { echo "[$(date '+%F %T')] $*"; }
run() { log "--> $*"; "$@" || { rc=$?; log "STEP FAILED (exit $rc): $*"; exit "$rc"; }; }
upper() { echo "$1" | tr '[:lower:]' '[:upper:]'; }

# --- phases -------------------------------------------------------------------
: "${LAUE_PHASES:?LAUE_PHASES is not set: list the phases to analyse, e.g. LAUE_PHASES=alpha,beta or LAUE_PHASES=zn}"
read -r -a PHASES <<< "$(echo "$LAUE_PHASES" | tr ',' ' ')"
[ ${#PHASES[@]} -gt 0 ] || { echo "ERROR: LAUE_PHASES='$LAUE_PHASES' names no phase" >&2; exit 1; }
for p in "${PHASES[@]}"; do
  [[ $p =~ ^[A-Za-z][A-Za-z0-9_]*$ ]] || { echo "ERROR: phase name '$p' must be letters, digits and _ (it becomes part of LAUE_SCAN_<PHASE>)" >&2; exit 1; }
done
dups=$(printf '%s\n' "${PHASES[@]}" | sort | uniq -d)
[ -z "$dups" ] || { echo "ERROR: LAUE_PHASES lists a phase twice: $dups" >&2; exit 1; }
LAUE_PHASES=$(IFS=,; echo "${PHASES[*]}")
has() { local p; for p in "${PHASES[@]}"; do [ "$p" = "$1" ] && return 0; done; return 1; }
TWO_PHASE=0; has alpha && has beta && TWO_PHASE=1
FIGURES=1; for p in "${PHASES[@]}"; do case $p in alpha|beta) ;; *) FIGURES=0 ;; esac; done

# --- everything the steps that will run need, checked before any of them ------
missing=()
need() { [ -n "${!1:-}" ] || missing+=("$1${2:+ ($2)}"); }
need LAUE_WORK "work root"
need LAUE_OUT_PREFIX "output basename prefix"
need LAUE_SCAN_DATA "raw-frame folder"
for p in "${PHASES[@]}"; do
  P=$(upper "$p")
  need "LAUE_SCAN_$P" "indexing-run dir for $p"
  if [ ${#PHASES[@]} -eq 1 ]; then
    v="LAUE_PARAMS_$P"
    [ -n "${!v:-}" ] || [ -n "${LAUE_PARAMS:-}" ] \
      || missing+=("LAUE_PARAMS_$P or LAUE_PARAMS (params file for $p)")
  else
    need "LAUE_PARAMS_$P" "params file for $p; LAUE_PARAMS cannot serve two phases"
  fi
done
if [ "$TWO_PHASE" = 1 ] || [ "$FIGURES" = 1 ]; then
  need LAUE_MOUNT_DEG "sample mount angle in degrees, [0, 90): variant_coherence.py / validated_figures.py"
fi
if [ ${#missing[@]} -gt 0 ]; then
  echo "ERROR: run_analysis_chain.sh: not set, needed by the steps this run would execute:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 1
fi
bad=()
[ -d "$LAUE_SCAN_DATA" ] || bad+=("LAUE_SCAN_DATA=$LAUE_SCAN_DATA is not a directory")
for p in "${PHASES[@]}"; do
  P=$(upper "$p"); v="LAUE_SCAN_$P"; [ -d "${!v}" ] || bad+=("$v=${!v} is not a directory")
  v="LAUE_PARAMS_$P"; f=${!v:-${LAUE_PARAMS:-}}; [ -f "$f" ] || bad+=("params file for $p not found: $f")
done
if [ -n "${LAUE_MOUNT_DEG:-}" ]; then
  awk -v m="$LAUE_MOUNT_DEG" 'BEGIN{exit !(m ~ /^[0-9.]+$/ && m+0 >= 0 && m+0 < 90)}' \
    || bad+=("LAUE_MOUNT_DEG=$LAUE_MOUNT_DEG is not a number in [0, 90)")
fi
[ -f "$S/null_model.py" ] || bad+=("SCRIPTDIR=$S does not hold the analysis scripts")
"$PY" -c 'import numpy, scipy, h5py, matplotlib' 2>/dev/null \
  || bad+=("PY=$PY cannot import numpy, scipy, h5py and matplotlib")
if [ ${#bad[@]} -gt 0 ]; then
  echo "ERROR: run_analysis_chain.sh:" >&2
  printf '  %s\n' "${bad[@]}" >&2
  exit 1
fi
export LAUE_WORK LAUE_SCAN_DATA LAUE_OUT_PREFIX LAUE_PHASES

mkdir -p "$LAUE_WORK"/{peel_map,figures}
log "=== analysis chain for $LAUE_OUT_PREFIX (phases: $LAUE_PHASES, NW=$NW) ==="
log "  data:  $LAUE_SCAN_DATA"
for p in "${PHASES[@]}"; do v="LAUE_SCAN_$(upper "$p")"; log "  $p: ${!v}"; done

for p in "${PHASES[@]}"; do
  run "$PY" "$S/parentbeta_validate.py" "$p" "$NW" env
done

# the null MUST be measured on this scan: lambda depends on its own peak and
# reflection counts, and the analytic Poisson gate under-rejects on clustered fields.
#
# Contract with null_model.py: it writes
#   $LAUE_WORK/peel_map/${LAUE_OUT_PREFIX}_null.json
#   {"phases": {"<phase>": {"nhit": {"max": N, ...}, "nhit_distinct": {"max": N, ...}}}}
# (path from frame_peaks.null_json_path). For every phase in LAUE_PHASES we read
# the max of the statistic the gates use (frame_peaks.gate_statistic(), i.e.
# LAUE_GATE_STAT, default nhit) and export it as LAUE_NULLMAX_<PHASE>. The gates
# read the same json through frame_peaks.load_null, where LAUE_NULLMAX_<PHASE>
# overrides the max -- so a stale value inherited from the caller's environment
# would win over this scan's measurement. Exporting the measured value closes
# that door. A missing file or phase stops the chain.
run "$PY" "$S/null_model.py" 120 150 "$NW"
for ph in "${PHASES[@]}"; do
  var="LAUE_NULLMAX_$(upper "$ph")"
  val=$("$PY" - "$S" "$LAUE_WORK" "$LAUE_OUT_PREFIX" "$ph" <<'PYEOF'
import json, os, sys
sys.path.insert(0, sys.argv[1])
from frame_peaks import gate_statistic, null_json_path
work, prefix, phase = sys.argv[2:5]
path = null_json_path(work, prefix)
if not os.path.isfile(path):
    sys.exit(f"null_model.py wrote no {path}")
stat = gate_statistic()
try:
    print(int(json.load(open(path))["phases"][phase][stat]["max"]))
except (KeyError, TypeError, ValueError) as e:
    sys.exit(f"{path} has no {phase}/{stat}/max ({e!r})")
PYEOF
) || { log "STEP FAILED: no measured null for phase $ph"; exit 1; }
  if [ -n "${!var:-}" ] && [ "${!var}" != "$val" ]; then
    log "  $var was $(printf %s "${!var}") in the environment; replaced by the value measured on this scan"
  fi
  export "$var=$val"
  log "  $var=$val (measured on this scan, statistic ${LAUE_GATE_STAT:-nhit})"
done

run "$PY" "$S/empirical_gate.py"

if [ "$TWO_PHASE" = 1 ]; then
  run "$PY" "$S/beta_alpha_exclusion_census.py" env "$NW"
  run "$PY" "$S/exclusion_null.py" 120 150 "$NW"
  run "$PY" "$S/parentbeta_reconstruct.py" 30 "$LAUE_OUT_PREFIX"
  run "$PY" "$S/anchor_null.py"
  run "$PY" "$S/variant_coherence.py"
else
  log "SKIPPED (need both alpha and beta in LAUE_PHASES): beta_alpha_exclusion_census.py," \
      "exclusion_null.py, parentbeta_reconstruct.py, anchor_null.py, variant_coherence.py"
fi

if [ "$FIGURES" = 1 ]; then
  run "$PY" "$S/validated_figures.py"
else
  log "SKIPPED: validated_figures.py (it draws only phases named alpha/beta; LAUE_PHASES=$LAUE_PHASES)"
fi
log "=== analysis chain finished for $LAUE_OUT_PREFIX ==="
