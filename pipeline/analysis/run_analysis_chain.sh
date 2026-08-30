#!/bin/bash
# run_analysis_chain.sh — full validation/analysis pipeline for one indexed scan,
# runnable on any host. All paths come from the environment (see the patched scripts):
#
#   LAUE_WORK        work root (params/, peel_map/, figures/ live here)
#   LAUE_SCAN_DATA   raw frames
#   LAUE_SCAN_ALPHA  alpha indexing-run dir
#   LAUE_SCAN_BETA   beta indexing-run dir
#   LAUE_OUT_PREFIX  output basename prefix
#   NW               worker processes (default 16)
#
# Order matters: validate both phases, MEASURE the null on this scan (never inherit
# another scan's), then gate against it, then the beta-exclusion census, then Burgers
# plus the spatial-coherence check that actually confirms the reconstruction.
set -uo pipefail
: "${LAUE_WORK:?}" "${LAUE_SCAN_DATA:?}" "${LAUE_SCAN_ALPHA:?}" "${LAUE_SCAN_BETA:?}" "${LAUE_OUT_PREFIX:?}"
NW=${NW:-16}
S=${SCRIPTDIR:-$LAUE_WORK/scripts}
PY=${PY:-/home/beams/EPIX34ID/conda-envs/laue_rt/bin/python}
export LAUE_WORK LAUE_SCAN_DATA LAUE_SCAN_ALPHA LAUE_SCAN_BETA LAUE_OUT_PREFIX

log() { echo "[$(date '+%F %T')] $*"; }
run() { log "--> $*"; "$@" || { log "STEP FAILED: $*"; return 1; }; }

mkdir -p "$LAUE_WORK"/{peel_map,figures}
log "=== analysis chain for $LAUE_OUT_PREFIX (NW=$NW) ==="
log "  data:  $LAUE_SCAN_DATA"
log "  alpha: $LAUE_SCAN_ALPHA"
log "  beta:  $LAUE_SCAN_BETA"

run $PY "$S/parentbeta_validate.py" alpha "$NW" env || exit 1
run $PY "$S/parentbeta_validate.py" beta  "$NW" env || exit 1

# the null MUST be measured on this scan: lambda depends on its own peak and
# reflection counts, and the analytic Poisson gate under-rejects on clustered fields
run $PY "$S/null_model.py" 120 150 "$NW"
run $PY "$S/empirical_gate.py"

run $PY "$S/beta_alpha_exclusion_census.py" env "$NW"
run $PY "$S/exclusion_null.py" 120 150 "$NW"

run $PY "$S/parentbeta_reconstruct.py" 30 "$LAUE_OUT_PREFIX"
run $PY "$S/anchor_null.py"
run $PY "$S/variant_coherence.py"

run $PY "$S/validated_figures.py"
log "=== analysis chain finished for $LAUE_OUT_PREFIX ==="
