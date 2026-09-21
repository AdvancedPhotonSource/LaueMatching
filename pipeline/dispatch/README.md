# Multi-host, multi-GPU indexing

`pipeline/run_laue.sh` runs one orchestrator per phase on the host you are on. A large raster is
faster split into **shards**, one orchestrator per GPU, across several hosts. These scripts do
that, and each encodes a failure that once produced a wrong, partial or empty result while
looking fine.

| script | does | exits non-zero when |
|---|---|---|
| `mkrun.py` | row-aligned shards (symlinks), one params file and one ResultDir per shard, a plan per phase | a frame is missing, a template still has `__SET_ME__`, a port / tag / ResultDir collides with any plan already in WORK |
| `preflight.sh PLAN...` | checks every plan that will run together, before anything starts | duplicate port / tag / ResultDir / HOST:GPU; missing or implausible background; port already bound; orchestrator, launcher or indexer binaries not visible on a host |
| `dispatch.sh PLAN [STAGGER]` | launches each shard on its host, staggered, pool sized per shard | a daemon already holds a GPU the plan assigns, or a plan port is bound; a launch did not survive its liveness check |
| `launch_run.sh ...` | one shard on this host (called by `dispatch.sh`; usable by hand) | `PY` / `SCRIPTS` / params / shard dir missing; `run_laue.sh` failed |
| `watch_arm.sh PLAN...` | waits for every daemon's clean exit, or the first failure signature | any failure signature (exit 1); timeout (exit 2) |
| `wait_static.sh N PLAN...` | waits until the output count reaches N and holds across two checks | outputs static but short after every orchestrator finished (1); timeout (2) |

`lib.sh` holds the shared helpers (plan reading, running a script on a host).

## The flow

```bash
CHECKOUT=/path/to/LaueMatching; D=$CHECKOUT/pipeline/dispatch
export PY=/path/to/env/bin/python          # as seen ON THE SHARD HOSTS; full path
export SCRIPTS=$CHECKOUT/scripts           # optional: this is the default

# 1. shards, params, plan (one plan per phase)
$PY $D/mkrun.py --work /path/to/work --run scanA \
    --raw-dir /path/to/raw/scan1 --prefix scan1_ --ncol 100 --nrow 50 \
    --slots hostA:0,hostA:1,hostB:0,hostB:1 --port-base 61200 \
    --phase alpha=/path/to/work/params/params_alpha.txt \
    --set BackgroundFile=/path/to/work/db/background_s1.bin

# 2. check EVERY plan that will run at the same time, together
bash $D/preflight.sh /path/to/work/plan_scanA_alpha.txt

# 3. launch (DRY_RUN=1 first shows the hosts, CPU counts and worker sizing)
bash $D/dispatch.sh /path/to/work/plan_scanA_alpha.txt 90 > /path/to/work/logs/dispatch.log 2>&1

# 4. wait: daemons first, then the late batch of outputs
bash $D/watch_arm.sh  /path/to/work/plan_scanA_alpha.txt
bash $D/wait_static.sh 5000 /path/to/work/plan_scanA_alpha.txt
```

Everything lands under the work root: `shards/<tag>/`, `params/params_<tag>.txt`,
`run/<tag>/results/<phase>_<timestamp>/` (orchestrator output, `daemon.log`),
`run/<tag>/resultdir/` (the daemon's `solutions.txt`), and `logs/`. The checkout and the work root
must be on a filesystem every host sees.

A plan is plain text, one shard per line: `HOST GPU PORT WORKDIR TAG PARAMS SHARDDIR`. No field
may contain whitespace. `HOST` is an ssh alias; a line naming the host you are on runs locally.

## Traps these scripts encode

- **Remote login shells are tcsh.** Every remote step is `ssh HOST bash -s` with the script on
  stdin; nothing is ever quoted into the outer ssh command string (no `$(...)`, no redirections --
  tcsh's `noclobber` silently refuses `>` onto an existing file).
- **Never `ssh -n` with `bash -s`.** `-n` feeds `/dev/null` as stdin, so the remote bash gets an
  empty script and exits 0. The check then reads as "failed" or "nothing there" when it never ran.
  Every remote check here prints a sentinel line and treats its absence as an ssh failure.
- **`ssh` inside `while read` eats the loop's stdin**, dropping the remaining lines (a driver once
  logged "all 7 launched" with 3 running). Plans and host lists are read into arrays first.
  Likewise a command run inside a `bash -s` script inherits the script as stdin: launches get
  `< /dev/null`.
- **Size the preprocessing pool per shard, not per machine.** laue-index 0.7.1 sizes it from the
  whole host with no knowledge of sibling shards: several shards on a many-core host each spawn a
  full pool, each worker fans out an OpenCV thread pool, and the per-user process limit runs out.
  The symptoms arrive in a misleading order: OpenCV `res = 11`, then the daemon dies with
  `GPUassert: ... busy or unavailable` (a consequence, not a GPU problem), then the dispatcher
  cannot fork and later shards never launch. `dispatch.sh` counts, per host, the plan's shards plus
  your `laue_orchestrator.py` processes already running there (it prints the count) and exports
  that as `LAUE_SHARDS_PER_HOST`; `launch_run.sh` then sets `LAUE_PREPROCESS_WORKERS` to
  3/4 x cpus / `LAUE_SHARDS_PER_HOST` (minimum 4), `OPENCV_NUM_THREADS=1` and single-threaded
  BLAS. From laue-index 0.7.2 the package sizes the
  pool itself: it divides the host by `LAUE_SHARDS_PER_HOST`, subtracts a `LAUE_DAEMON_NCPUS`
  you set (never below half the per-shard share), caps by `RLIMIT_NPROC` and runs pool workers
  single-threaded. Each shard still gets both variables: `LAUE_SHARDS_PER_HOST` is what
  0.7.2 needs, and the explicit `LAUE_PREPROCESS_WORKERS` (which overrides the package's choice)
  is what protects a host still on 0.7.1.
- **Stagger launches** (60-120 s). Each daemon reads the orientation database and forward cache
  (tens of GB) before binding its port; simultaneous launches saturate NFS and abort with
  "Daemon did not open port in time".
- **Refuse a host only on a real conflict.** `dispatch.sh` asks `nvidia-smi` which processes
  hold each GPU and refuses when a `LaueMatchingGPUStream` daemon (any user) is already on a GPU
  index the plan assigns on that host, or when one of the plan's ports is bound
  (`ALLOW_SHARED_GPU=1` accepts the first, nothing overrides the second). Other users' daemons
  on other GPUs, and your own other plans (e.g. the other phase on other slots), do not block.
- **One port, one ResultDir per shard, unique across every plan running together.** A shared port
  leaves the second daemon bound to nothing, holding its GPU, receiving zero images. A shared
  ResultDir interleaves `solutions.txt` unrecoverably.
- **A missing BackgroundFile does not fail**: the server builds one from the first frame. The
  preflight requires the file, its size (`NrPxX * NrPxY` float64 from each params file, e.g. 33554432 bytes at 2048 x 2048)
  and a median above `BG_MEDIAN_MIN` (default 50).
- **Ask the package on each host whether the indexer is there**; never `ls bin/` (a binary for the
  wrong OS once sat there looking ready).
- **Row-aligned shards, zero-padded links.** Row-wise analyses never straddle two shards, and
  lexical order is acquisition order (an unpadded shard was enumerated 1, 10, 100, 1000, ...).
- **Liveness is checked, not assumed.** `run_laue.sh` checks the detached orchestrator is still
  alive a few seconds after launch; `dispatch.sh` waits for that answer per shard; `watch_arm.sh`
  looks for failure signatures in each run's own logs rather than trusting the dispatcher's
  output; `pgrep -f` is not used (it matches its own command line through a remote tcsh).
- **"Exited cleanly" is not "done".** Post-processing writes the per-frame outputs afterwards, in
  one late batch. `wait_static.sh` waits for the count to reach N and hold, and only calls a
  static count "short" once every orchestrator has logged its summary. Frames with the beam off
  the specimen legitimately produce no output, so "short" is a prompt to look, not proof of loss.
