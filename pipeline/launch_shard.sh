#!/bin/bash
# launch_shard.sh -- RETIRED. Kept only so an old invocation fails with directions
# instead of "No such file".
#
# It was broken four ways: it hard-coded an install directory and a conda env
# that no longer exist, set no SCRIPTS (so run_laue.sh looked for the
# orchestrator in the repo root), left the preprocessing pool sized for the whole
# host on every shard (several shards on one host then exhaust the per-user
# thread limit and die with a misleading "GPUassert: device busy"), and was wired
# to one campaign's file naming. Its liveness check, `pgrep -af <pattern>`, also
# matches its own command line when run through a remote tcsh, so it reported
# dead shards as alive.
#
# Use the multi-host tooling instead -- see pipeline/dispatch/README.md:
#   pipeline/dispatch/mkrun.py      shards, per-shard params and a plan file
#   pipeline/dispatch/preflight.sh  checks the plan before anything starts
#   pipeline/dispatch/dispatch.sh   launches the plan, one shard per GPU
# For one orchestrator on this host, call pipeline/dispatch/launch_run.sh
# (or pipeline/run_laue.sh with BETA_CONFIG="") directly.
echo "launch_shard.sh is retired: use pipeline/dispatch/ (see pipeline/dispatch/README.md)." >&2
echo "  one shard on this host:  PY=... SCRIPTS=... pipeline/dispatch/launch_run.sh WORK TAG PARAMS SHARD_DIR GPU PORT" >&2
exit 1
