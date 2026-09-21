# Phase 3 — Index

> Part of the **Laue doc set**. The spine — invariants, done-means and the phase
> order — is [`README.md`](README.md).

---

## Phase 3 — Index

```bash
export SCRIPTS=<repo>/scripts          # the directory holding laue_orchestrator.py -- see below
export PY=/full/path/to/env/bin/python # python with laue-index; full path (ssh has no conda on PATH)
WATCH=""  ./run_laue.sh  <SCAN_FOLDER>  [<h5 dataset path>]     # batch an existing scan
./run_laue.sh <SCAN_FOLDER>                                     # live, stop with STOP_LAUE
```

**Set `SCRIPTS` explicitly on 0.7.1.** Its default there resolves to the repository root, not
`scripts/`; the orchestrator launch is backgrounded, so the "can't open file" lands in
`results/*.launch.log` while `run_laue.sh` prints a pid as if the run had started. From 0.7.2
the default is `<repo>/scripts` and the script refuses to launch if `laue_orchestrator.py` is
missing there. Either way, confirm the launch log is not a Python error before walking away.

For many scans, a batch runner that takes one run at a time, smallest first, defers folders
still growing (`--settle 120`) and uses `.laue_done` / `.laue_skip` markers for resume has been
used on past campaigns. It is campaign-local and **not in this repo**, and neither is the
"RUN_PROCESS_REPORT" handoff that described its multi-machine layout; for multi-host runs
use `pipeline/dispatch/` (below).

### Sharding a single big scan across GPUs and hosts

For one large raster (40k+ frames) there is only one phase to index, so GPUs split **frames**
rather than phases. Three hard constraints, all learned the expensive way on the Zn scan:

**1. Budget ~19 GB of HOST RAM per daemon, and do not stack them.** Each daemon holds the 7.2 GB
orientation database *and* the 12.2 GB forward cache in host memory. Three daemons on a 128 GB box
plus their image servers filled RAM, drove swap to 100%, and two of the three shards died with
`Send/save error for image_num=N: timed out` (the 30 s socket send timeout) while appearing
"running". Per-image time went 1 s -> 4.5 s before they stopped entirely.

**2. Size the image server's preprocessing pool PER SHARD, not per host.** (This item used to
say the server had no backpressure. It has: the producer blocks on a futures queue
`FUTURES_QUEUE_DEPTH` = 512 frames deep (`laue_image_server.py`). At 2048² that queue is
~8.6 GB of dense frames, most of the ~17 GB parent RSS once seen on a 13,467-frame shard, and
it is charged to the worker-count budget.) The live hazard is the worker **count**. In
laue-index 0.7.1 `laue_index.workers.choose_preprocess_workers` sizes the pool from the
**whole host** and knows nothing about sibling shards, and every pool worker starts its own
OpenCV thread pool. Two shards on a 112-core host with `ulimit -u` = 8192 exhausted the
per-user thread limit. The symptoms arrive in a misleading order: OpenCV
`Can't spawn new thread: res = 11` → the daemon dies with
`GPUassert: CUDA-capable device(s) is/are busy or unavailable` (a consequence, not a GPU
fault) → the dispatcher cannot fork, so the remaining shards never launch → the host refuses
new ssh logins. `DIAGNOSIS.md` has the discriminating test.

- **0.7.1:** when several shards share a host, set `LAUE_PREPROCESS_WORKERS` for EACH shard
  (about ¾ × host CPUs ÷ shards on that host) and `OPENCV_NUM_THREADS=1` in its environment.
  The `PreprocessWorkers` parameter-file key is not read on the streaming path in 0.7.1; only
  the environment variable works.
- **0.7.2 and later:** the sizing divides the host's CPUs and memory by `LAUE_SHARDS_PER_HOST`,
  subtracts a `LAUE_DAEMON_NCPUS` you set (never below half the per-shard share; nothing sets it for you), and
  is capped by the `RLIMIT_NPROC` thread budget; pool workers run with thread counts of 1;
  `LAUE_PREPROCESS_WORKERS` remains the override; and the streaming config parses `PreprocessWorkers` and `RobustFilter`
  (`MinGoodSpots` was parsed in 0.7.1 but not applied; see "Streaming post-processing"). `pipeline/dispatch/` sets all of this per host from the plan.

**3. EVERY concurrent shard needs its OWN `ResultDir`, i.e. its own params file.** The daemon writes
its raw `solutions.txt` and `spots.txt` into the **params** `ResultDir`. `--output-dir` overrides
only the *orchestrator's* per-frame `output.h5` tree, **not** this — so the widely-repeated note
that "ResultDir in params is ignored" is true only of the per-frame outputs and is a trap if you
generalise it. Point two shards at one params file and both daemons append to the same
`solutions.txt`/`spots.txt` and interleave. It then fails **late and silently**: indexing runs to
completion and reports `Pipeline complete`, and only post-processing dies, on torn lines
(`got 19 columns instead of 12` from two collided records, `got 2 columns` from a truncated one).
Nothing is recoverable, because each orchestrator numbers its images `1..N` independently, so the
interleaved rows cannot be attributed back to a shard. This cost three 3,400-frame shards on the
bt_34ide_jul26 campaign (`LAB_NOTEBOOK.md` §2d records "~30 min GPU, 5 shards" for the same
trap; whether these are one event or two is unreconciled). Generate params per shard (the sampleH campaign's `params_*_run_s1..s7.txt` exist for
exactly this reason) and **assert the `ResultDir` set is unique before dispatching**.

Machines that can see `$LAUE_ROOT` and the beamline account's install. As of 2026-09 that
install is a **pip environment with laue-index** in the beamline account's shared home (plus a
separate environment on an editable checkout for development); the older `laue_rt` environment
and the old non-canonical checkout are gone (the checkout was archived on 2026-08-30), so any script or note naming
them is stale. Check what a host will actually run with the install gate in `README.md`, not
from a path in a note.

| host | RAM | cores | GPUs | notes |
|---|---|---|---|---|
| copland | 2015 GB | 96 | 2x A6000 48 GB | **cannot even READ** the analysis host (not merely write) -- unusable for indexing this data, despite the RAM |
| alleppey | 502 GB | 112 | 4x H100 80-96 GB | usually shared; check `nvidia-smi` first |
| sentosa | 250 GB | 64 | 2x H200 144 GB + 2x Blackwell | Blackwell cards (2,3) are **sm_120**, often in use. The beamline's 0.7.1 build was reported as built with nvcc 12.1.105 (PTX sm_90, no sm_120 cubin), so these cards run it by PTX JIT; unmeasured on this pipeline. `laue-index doctor` shows what the installed build covers |
| shannon | 125 GB | 40 | 3x A4500 20 GB | 34-ID-E box; smallest RAM, budget 2 daemons max. Unreachable 2026-08-12; **reachable again 2026-09-20** |
| chutoro | — | 64 | 2x A6000 48 GB | added 2026-08-12: has the install, sees `/gdata/dm/34IDE`, and `epix34id@chutoro` is **directly key-authorised** — no shannon hop |

**Log in as `epix34id` on every host** (not s1iduser): the data, the DB and the caches are all
owned by epix34id, and the s1iduser LaueMatching build is older -- it ignores `LAUE_STREAM_PORT`
and silently binds 60517, so two daemons on one host collide.

**Reachability — try the direct route first.** This section used to say the only route was
`copland(s1iduser) -> epix34id@shannon -> epix34id@<host>` because "epix34id keys live on
shannon". That is **false**: as of 2026-09-20 `ssh epix34id@<host>` is directly
key-authorised for alleppey, sentosa and chutoro. With shannon down on 2026-08-12 the documented chain was a dead end and the
direct route worked immediately. Try `epix34id@<host>` first; fall back to the hop only if it
refuses.

Note also that the data path differs by mount: this section is written around
`$LAUE_ROOT`, while the campaigns are reachable on copland and chutoro as
`/gdata/dm/34IDE/<Run>/<Campaign>`. Whether these are the same underlying store has **not**
been established — check before assuming a path from one works on the other.

Every remote shell is **tcsh**: pipe scripts to `bash -s`, and never use `$(...)` in the outer
ssh command. **Never combine that with `ssh -n`:** `-n` points stdin at `/dev/null`, so
`ssh -n host bash -s < script.sh` runs an empty script, prints nothing and reads as a failed
check. `ssh -n` is for a single quoted command (and for any `ssh` inside a loop, README
invariant 6); a stdin-fed script needs plain `ssh host bash -s < script.sh`.

**Multi-host launch: use [`pipeline/dispatch/`](../../pipeline/dispatch/README.md)**
(`mkrun.py`, `preflight.sh`, `dispatch.sh`, `launch_run.sh`, `watch_arm.sh`, `wait_static.sh`,
shared helpers in `lib.sh`). `pipeline/launch_shard.sh` is retired: in 0.7.2 it is a stub that
prints that pointer and exits 1. (In 0.7.1 it did not work as shipped: it hard-coded the removed
install path and environment, set no `SCRIPTS`, and capped no preprocessing workers.) The
dispatch tooling: a plan is read into an array (no
`while read` + `ssh`), workers are sized per host, a host already running a daemon is
refused, ports / tags / `ResultDir`s are asserted unique across plans before dispatch, and
completion means a static output count (INVARIANTS.md invariant 21b). On 0.7.1, do the same by hand:
explicit `SCRIPTS`, per-shard `ResultDir` and port, `LAUE_PREPROCESS_WORKERS` and
`OPENCV_NUM_THREADS=1` per shard. Stagger launches by ~60 s: each daemon reads 19 GB before
binding its port.

Also: files written by one account are not automatically readable by another. `forward_*.bin` is
created mode `600`; `chmod 644` it before another host's account can load it.

### Streaming post-processing (0.7.2 and later)

After the daemon has seen every frame, the orchestrator post-processes the solutions into the
per-frame `.output.h5` files. In 0.7.2 this honours the parameter file the way `RunImage.py`
does, with one deliberate exception:

- **Explicit keys are honoured.** `RobustFilter`, `MinGoodSpots`, `MinNrSpots` and `MaxAngle`
  mean the same on the streaming and single-image paths.
- **An ABSENT `RobustFilter` keeps 0.7.1 streaming behaviour** (the legacy filter) and logs one
  warning line saying so. `RunImage.py`'s absent default is the robust filter; set
  `RobustFilter 1` explicitly to get it on the streaming path.
- **`MinGoodSpots` is now honoured on streaming.** 0.7.1 passed `--min-unique 2` whatever the
  file said. On a config with `MinGoodSpots 4` the exclusive-label floor rises 2 → 4 in 0.7.2;
  `MinGoodSpots 2` reproduces 0.7.1.
- **The robust filter's `MinNrSpots` floor** is the orientation's own winner-take-all pixel
  count, so it is monotonic (0.7.1 fell back to the total `NMatches` exactly when that count was
  0, keeping a fully-shared Σ3 twin and dropping one with 1–4 own spots). With the robust filter,
  hexagonal near-duplicates within `MaxAngle` are now removed too.
- **`MaxAngle` is a MISORIENTATION in degrees**, not a pixel radius: the C merges orientations
  within it (`mergeDuplicateOrientations`) and Python uses it as the near-duplicate angle.
  Refinement is always Nelder-Mead: an `Optimizer` line is still parsed so old files work,
  and `Optimizer BOBYQA` prints a notice and runs Nelder-Mead anyway.
- **A post-processing failure fails the run.** The orchestrator exits non-zero and no longer
  prints `Pipeline complete` after a post-processing error.
- Worker sizing, as item 2 above: `LAUE_SHARDS_PER_HOST`, `LAUE_DAEMON_NCPUS`, the
  `RLIMIT_NPROC` cap, `LAUE_PREPROCESS_WORKERS` override.

Spot counts in the outputs: `INVARIANTS.md` invariant 15b.

Sanity while it runs:

- **orientations/frame is scan-dependent** — 10–100 on a sparse scan, 275–1043 on a dense one, both
  healthy. Judge by the flood signature (thousands, `MAX_MATCHES`, >100 s/frame), not an absolute.
- **`output.h5` files appear in one late batch** after largely single-threaded post-processing
  (a 1.9 GB `solutions.txt` took ~45 min). "0 outputs after N frames" is not a stall.
- **Frames with the beam off the specimen legitimately produce nothing.** One 40,401-frame scan had
  a genuine 1,947-frame blank band. Verification tolerances must allow ~10%, or you will reject a
  good run.
- **Never stack launches.** Each daemon reads a ~7 GB database before binding its port; two at once
  saturate NFS and both abort with "Daemon did not open port in time".

---
