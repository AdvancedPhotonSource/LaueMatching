# shellcheck shell=bash
# lib.sh -- helpers shared by the pipeline/dispatch scripts. Source, do not run.
#
# Plan file: one shard per line, whitespace separated, '#' comments allowed:
#   HOST GPU PORT WORKDIR TAG PARAMS SHARDDIR
# WORKDIR is the run root: logs/ and run/<TAG>/ are created under it.
# mkrun.py writes these; nothing else about a campaign lives in the scripts.

# Read every non-comment plan line of the given files into the array PLAN_LINES.
# Deliberately NOT `while read ...; do ssh ...; done < plan`: ssh inherits the
# loop's stdin and swallows the remaining plan lines, so a driver once logged
# "all 7 launched" with 3 running. Read first, then iterate the array with `for`.
# (bash 3.2-compatible: no mapfile.)
read_plans() {
  PLAN_LINES=()
  local p l
  for p in "$@"; do
    [ -f "$p" ] || { echo "ERROR: no plan file $p" >&2; return 1; }
    while IFS= read -r l || [ -n "$l" ]; do
      l="${l%%#*}"
      # shellcheck disable=SC2086
      set -- $l
      [ $# -eq 0 ] && continue
      [ $# -eq 7 ] || { echo "ERROR: $p: expected 7 fields, got $#: $l" >&2; return 1; }
      PLAN_LINES+=("$*")
    done < "$p"
  done
}

# The short hostname of this machine, to run "remote" steps locally when a plan
# names the host we are on.
this_host() { hostname -s 2>/dev/null || hostname; }

# on_host HOST -- run the bash script arriving on STDIN on HOST.
#
# The login shell on the beamline hosts is tcsh, so a remote step is always
# `ssh HOST bash -s` with the script on stdin: tcsh then only ever sees the two
# words "bash -s", never our quoting, `$(...)` or redirections. (tcsh's
# noclobber also silently refuses `>` onto an existing file; inside bash -s that
# cannot bite.)
#
# NEVER add `-n` here. `ssh -n` redirects stdin from /dev/null, so `bash -s`
# receives an EMPTY script, runs nothing and exits 0 -- which reads as a failed
# or empty check ("no binary", "no process") rather than a broken one. Every
# caller therefore also prints a sentinel line and treats its absence as an
# ssh failure, not as a verdict.
#
# Extra arguments become the script's "$@" (`bash -s -- ARGS`). They pass through
# the remote tcsh as plain words, so give only simple tokens (numbers, names).
on_host() {
  local h=$1; shift
  if [ "$h" = "$(this_host)" ] || [ "$h" = localhost ]; then
    bash -s -- "$@"
  else
    ssh -o BatchMode=yes -o ConnectTimeout="${SSH_TIMEOUT:-20}" "$h" bash -s -- "$@"
  fi
}

# Space-separated field $2 (1-based) of every plan line on host $1.
plan_field_on() {
  local want=$1 f=$2 l a
  for l in "${PLAN_LINES[@]}"; do
    read -r -a a <<< "$l"
    if [ "${a[0]}" = "$want" ]; then printf '%s ' "${a[$((f - 1))]}"; fi
  done
}

# Distinct hosts named by PLAN_LINES, one per line.
plan_hosts() {
  local l
  for l in "${PLAN_LINES[@]}"; do
    # shellcheck disable=SC2086
    set -- $l; echo "$1"
  done | sort -u
}

# Number of plan lines on host $1.
plan_count_on() {
  local want=$1 l n=0
  for l in "${PLAN_LINES[@]}"; do
    [ "${l%% *}" = "$want" ] && n=$((n + 1))
  done
  echo "$n"
}
