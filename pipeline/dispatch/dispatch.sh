#!/bin/bash
# dispatch.sh PLANFILE [STAGGER_SECONDS]
#
# Launch every shard of a plan (see lib.sh for the format; mkrun.py writes it),
# one detached launch_run.sh per line, on the host that line names. Run
# preflight.sh on the same plan first.
#
# Environment:
#   PY           python with laue-index installed, as seen ON THE SHARD HOSTS
#                (REQUIRED; passed through to launch_run.sh)
#   SCRIPTS      directory holding laue_orchestrator.py (default: this checkout's
#                scripts/). The checkout must be on a filesystem every host sees.
#   FIT_THREADS  refinement threads per shard (default 16)
#   DRY_RUN=1    print what would be launched, with the worker sizing, and stop
#   DISPATCH_NCPU  pretend every host has this many CPUs (for DRY_RUN checks)
#
# Hard-won rules encoded here:
#  * The plan is read into an array before any ssh runs (lib.sh read_plans):
#    `ssh` inside `while read` eats the loop's stdin and silently drops the
#    remaining plan lines.
#  * Every remote step is `ssh HOST bash -s` with the script on stdin (lib.sh
#    on_host), because the remote login shell is tcsh. Never `ssh -n` with it.
#  * SIZE THE PREPROCESSING POOL PER SHARD, NOT PER MACHINE: 3/4 x cpus divided
#    by the shards this plan puts on that host PLUS your orchestrators already
#    running there (see launch_run.sh for what
#    happens otherwise). LAUE_SHARDS_PER_HOST is exported too, for package
#    versions that size the pool themselves.
#  * Refuse a host only on a REAL conflict: a daemon (any user) already on a GPU
#    this plan assigns there, or one of this plan's ports already bound. Another
#    plan of yours on other GPUs (the other phase of the same run) is fine, and
#    counts toward the pool sizing below.
#  * Stagger launches (default 90 s; keep it in 60-120): each daemon reads the
#    orientation database and forward cache (tens of GB) before it binds its
#    port, and simultaneous reads saturate NFS until they all abort with
#    "Daemon did not open port in time".
# This script's last line is not evidence that anything ran: use watch_arm.sh
# and wait_static.sh, which look at each run's own logs and outputs.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=lib.sh
. "$HERE/lib.sh"

[ $# -ge 1 ] || { sed -n '2,3p' "$0" >&2; exit 2; }
PLAN=$1; STAG=${2:-90}
: "${PY:?PY must name the python with laue-index installed, as seen on the shard hosts}"
if [ -z "${SCRIPTS:-}" ]; then
  [ -d "$HERE/../../scripts" ] || { echo "ERROR: SCRIPTS is not set and $HERE/../../scripts does not exist; set SCRIPTS to the directory holding laue_orchestrator.py" >&2; exit 1; }
  SCRIPTS="$(cd "$HERE/../../scripts" && pwd)"
fi
[ -f "$SCRIPTS/laue_orchestrator.py" ] || { echo "ERROR: no laue_orchestrator.py in SCRIPTS=$SCRIPTS" >&2; exit 1; }
FIT_THREADS=${FIT_THREADS:-16}
LAUNCH="$HERE/launch_run.sh"

read_plans "$PLAN"
N=${#PLAN_LINES[@]}
[ "$N" -gt 0 ] || { echo "ERROR: $PLAN has no runs" >&2; exit 1; }
echo "$N runs to launch from $PLAN, stagger ${STAG}s"

HOSTS=()
while IFS= read -r h; do HOSTS+=("$h"); done < <(plan_hosts)

# --- per-host probe: CPUs, what is already running, real conflicts -------------
# One remote call per host. Refuse only on a REAL conflict:
#   * a LaueMatchingGPUStream daemon (any user) already on a GPU index this plan
#     assigns on that host (nvidia-smi compute apps, uuid -> index; indices are
#     PCI-bus order, the same order CUDA_DEVICE_ORDER=PCI_BUS_ID gives the shards);
#   * a port this plan uses already bound on that host.
# Other users' daemons on other GPUs, and this user's other plans (e.g. the
# other phase of the same run on other slots), are fine. They still share the
# CPUs, so the pool is sized from ALL of this user's orchestrators already on the
# host plus this plan's shards there. Other users' CPU use is not counted.
# Parallel arrays rather than `declare -A` so this also parses on bash 3.2.
# (read -d '' rather than $(cat <<EOS): bash 3.2 misparses a case pattern's ')'
# inside a heredoc inside $(...).
read -r -d '' PROBE_SCRIPT <<'EOS' || true
me=$(id -un)
echo "PROBE_CPUS $(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)"
echo "PROBE_ORCH $(ps -u "$me" -o args= 2>/dev/null | grep -c '[l]aue_orchestrator\.py' || true)"
if command -v nvidia-smi >/dev/null 2>&1; then
  idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null || true)
  apps=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null || true)
  while IFS=', ' read -r pid uuid; do
    [ -n "$pid" ] || continue
    i=$(printf '%s\n' "$idx" | awk -F', *' -v u="$uuid" '$2==u {print $1}')
    info=$(ps -o user=,args= -p "$pid" 2>/dev/null | head -n 1)
    case $info in
      *LaueMatchingGPUStream*) echo "PROBE_GPU_DAEMON ${i:-?} $info" ;;
      *) echo "PROBE_GPU_OTHER ${i:-?} ${info:-pid $pid}" ;;
    esac
  done <<< "$apps"
else
  echo "PROBE_NOGPU"
fi
if command -v ss >/dev/null 2>&1; then lst="ss -ltn"
elif command -v netstat >/dev/null 2>&1; then lst="netstat -ltn"
else lst=""; echo "PROBE_NOPORTTOOL"; fi
for p in "$@"; do
  [ -n "$lst" ] && $lst 2>/dev/null | grep -q ":$p " && echo "PROBE_PORT_BUSY $p"
done
echo "PROBE_END"
EOS
NCPU=(); NEXIST=()
for i in "${!HOSTS[@]}"; do
  h=${HOSTS[$i]}
  if [ -n "${DISPATCH_NCPU:-}" ]; then
    NCPU[i]=$DISPATCH_NCPU; NEXIST[i]=${DISPATCH_NEXIST:-0}
    echo "  $h: $(plan_count_on "$h") shard(s) in this plan, ${NCPU[$i]} cpus -- NOT PROBED (DISPATCH_NCPU)"
    continue
  fi
  gpus=$(plan_field_on "$h" 2); ports=$(plan_field_on "$h" 3)
  # shellcheck disable=SC2086
  out=$(printf '%s\n' "$PROBE_SCRIPT" | on_host "$h" $ports 2>/dev/null || true)
  printf '%s\n' "$out" | grep -q '^PROBE_END' \
    || { echo "ERROR: $h: no answer over ssh (not a verdict -- fix ssh first)" >&2; exit 1; }
  NCPU[i]=$(printf '%s\n' "$out" | sed -n 's/^PROBE_CPUS //p')
  NEXIST[i]=$(printf '%s\n' "$out" | sed -n 's/^PROBE_ORCH //p')
  conflict=""; gpu_conflict=""
  if printf '%s\n' "$out" | grep -q '^PROBE_NOGPU'; then
    conflict="$conflict; no nvidia-smi on $h: cannot see which GPUs are in use"
  fi
  if printf '%s\n' "$out" | grep -q '^PROBE_NOPORTTOOL'; then
    conflict="$conflict; neither ss nor netstat on $h: cannot check ports"
  fi
  while read -r _ gi rest; do
    [ -n "${gi:-}" ] || continue
    for g in $gpus; do
      if [ "$gi" = "$g" ] || [ "$gi" = "?" ]; then
        gpu_conflict="$gpu_conflict; GPU $gi already runs a daemon ($rest)"; break
      fi
    done
  done < <(printf '%s\n' "$out" | grep '^PROBE_GPU_DAEMON ' || true)
  while read -r _ gi rest; do
    [ -n "${gi:-}" ] || continue
    for g in $gpus; do
      if [ "$gi" = "$g" ]; then echo "  WARNING: $h GPU $gi also runs another compute process: $rest" >&2; fi
    done
  done < <(printf '%s\n' "$out" | grep '^PROBE_GPU_OTHER ' || true)
  busy=$(printf '%s\n' "$out" | sed -n 's/^PROBE_PORT_BUSY //p' | tr '\n' ' ')
  [ -z "$busy" ] || conflict="$conflict; port(s) already bound: $busy"
  # ALLOW_SHARED_GPU=1 (as in preflight.sh) accepts a daemon already on a plan
  # GPU; nothing overrides a bound port or a host that cannot be checked.
  [ -n "${ALLOW_SHARED_GPU:-}" ] || conflict="$conflict$gpu_conflict"
  if [ -n "$conflict" ]; then echo "REFUSING $h:${conflict#;}" >&2; exit 1; fi
  gstate="have no daemon"; [ -z "$gpu_conflict" ] || gstate="SHARED with a running daemon (ALLOW_SHARED_GPU)"
  echo "  $h: $(plan_count_on "$h") shard(s) in this plan + ${NEXIST[$i]} orchestrator(s) of yours" \
       "already running = $(( $(plan_count_on "$h") + NEXIST[i] )) sharing ${NCPU[$i]} cpus;" \
       "plan GPUs [${gpus% }] $gstate"
done

host_index() { local i; for i in "${!HOSTS[@]}"; do [ "${HOSTS[$i]}" = "$1" ] && { echo "$i"; return; }; done; }

i=0
for ln in "${PLAN_LINES[@]}"; do
  i=$((i + 1))
  # shellcheck disable=SC2086
  set -- $ln
  H=$1; GPU=$2; PORT=$3; W=$4; TAG=$5; PAR=$6; SHARD=$7
  k=$(host_index "$H"); NS=$(( $(plan_count_on "$H") + NEXIST[k] ))
  # 3/4 of this host's fair share over EVERY orchestrator of yours on it (this
  # plan's + those already running); the rest covers fitting threads + orchestrator
  PW=$(( NCPU[k] * 3 / (4 * NS) )); [ "$PW" -lt 4 ] && PW=4
  echo "[$i/$N] $H gpu$GPU port$PORT workers$PW shards_on_host$NS  $TAG"
  if [ -n "${DRY_RUN:-}" ]; then continue; fi
  # Values are expanded HERE, into a script that runs under bash on the host.
  # launch_run.sh runs in the FOREGROUND of that script: run_laue.sh inside it
  # detaches the orchestrator (setsid nohup) and then checks it is still alive,
  # so waiting for it costs a few seconds and returns a real exit status instead
  # of "a background job was started". Its stdin MUST be /dev/null: under
  # `bash -s` every command inherits the script itself as stdin and would eat
  # the lines after it.
  out=$(on_host "$H" <<EOS 2>&1 || true
mkdir -p $(printf %q "$W/logs")
export PY=$(printf %q "$PY") SCRIPTS=$(printf %q "$SCRIPTS") LAUE_SHARDS_PER_HOST=$NS
bash $(printf %q "$LAUNCH") $(printf %q "$W") $(printf %q "$TAG") $(printf %q "$PAR") $(printf %q "$SHARD") $GPU $PORT $FIT_THREADS $PW \\
  > $(printf %q "$W/logs/dispatch_$TAG.out") 2>&1 < /dev/null
echo "RC $TAG \$?"
EOS
)
  rc=$(printf '%s\n' "$out" | sed -n "s/^RC $TAG //p" | tail -n 1)
  if [ "$rc" != 0 ]; then
    echo "ERROR: $H: $TAG did not start (rc=${rc:-<no answer over ssh>}); see $W/logs/dispatch_$TAG.out" \
         "and $W/logs/${TAG}_*.launch. Runs 1..$((i - 1)) were started." >&2
    exit 1
  fi
  echo "    started: $TAG (orchestrator alive after its liveness wait)"
  if [ "$i" -lt "$N" ]; then sleep "$STAG"; fi
done
if [ -n "${DRY_RUN:-}" ]; then echo "DRY_RUN: nothing launched."; exit 0; fi
echo "ALL_DISPATCHED (verify per run with watch_arm.sh / wait_static.sh, not from this line)"
