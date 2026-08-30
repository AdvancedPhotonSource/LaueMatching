# Phase 3 — Index

> Part of the **Laue doc set**. The spine — invariants, done-means and the phase
> order — is [`README.md`](README.md).

---

## Phase 3 — Index

```bash
WATCH=""  ./run_laue.sh  <SCAN_FOLDER>  [<h5 dataset path>]     # batch an existing scan
./run_laue.sh <SCAN_FOLDER>                                     # live, stop with STOP_LAUE
```

For many scans, use the batch runner: one run at a time, smallest first, `--settle 120` to defer
folders still growing, `.laue_done` / `.laue_skip` markers for resume and for splitting work across
machines. See §6 of the RUN_PROCESS_REPORT handoff for the multi-machine layout.

### Sharding a single big scan across GPUs and hosts

For one large raster (40k+ frames) there is only one phase to index, so GPUs split **frames**
rather than phases. Two hard constraints, both learned the expensive way on the Zn scan:

**1. Budget ~19 GB of HOST RAM per daemon, and do not stack them.** Each daemon holds the 7.2 GB
orientation database *and* the 12.2 GB forward cache in host memory. Three daemons on a 128 GB box
plus their image servers filled RAM, drove swap to 100%, and two of the three shards died with
`Send/save error for image_num=N: timed out` (the 30 s socket send timeout) while appearing
"running". Per-image time went 1 s -> 4.5 s before they stopped entirely.

**2. `laue_image_server` has no backpressure.** It enumerates the whole folder and its preprocessing
pool buffers results faster than the daemon consumes them (~1.5 s/image). Parent RSS reached
**17 GB on a 13,467-frame shard**; at 201 frames this cannot manifest. Either give the host enough
RAM or keep shards small.

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
bt_34ide_jul26 campaign. Generate params per shard (the sampleH campaign's `params_*_run_s1..s7.txt` exist for
exactly this reason) and **assert the `ResultDir` set is unique before dispatching**.

Machines that can see `$LAUE_ROOT` and have the epix34id LaueMatching install
(`/home/beams/EPIX34ID/opt/LaueMatching`, shared home, has the `LAUE_STREAM_PORT` fix):

| host | RAM | cores | GPUs | notes |
|---|---|---|---|---|
| copland | 2015 GB | 96 | 2x A6000 48 GB | **cannot even READ** the analysis host (not merely write) -- unusable for indexing this data, despite the RAM |
| alleppey | 502 GB | 112 | 4x H100 80-96 GB | usually shared; check `nvidia-smi` first |
| sentosa | 250 GB | 64 | 2x H200 144 GB + 2x Blackwell | Blackwell cards (2,3) are **sm_120**, often in use |
| shannon | 125 GB | 40 | 3x A4500 20 GB | 34-ID-E box; smallest RAM, budget 2 daemons max. **Was unreachable 2026-08-12** (no route, including via copland) |
| chutoro | — | 64 | 2x A6000 48 GB | added 2026-08-12: has the install, sees `/gdata/dm/34IDE`, and `epix34id@chutoro` is **directly key-authorised** — no shannon hop |

**Log in as `epix34id` on every host** (not s1iduser): the data, the DB and the caches are all
owned by epix34id, and the s1iduser LaueMatching build is older -- it ignores `LAUE_STREAM_PORT`
and silently binds 60517, so two daemons on one host collide.

**Reachability — try the direct route first.** This section used to say the only route was
`copland(s1iduser) -> epix34id@shannon -> epix34id@<host>` because "epix34id keys live on
shannon". That is **false for at least chutoro**, where `ssh epix34id@chutoro` is directly
key-authorised. With shannon down on 2026-08-12 the documented chain was a dead end and the
direct route worked immediately. Try `epix34id@<host>` first; fall back to the hop only if it
refuses.

Note also that the data path differs by mount: this section is written around
`$LAUE_ROOT`, while the campaigns are reachable on copland and chutoro as
`/gdata/dm/34IDE/<Run>/<Campaign>`. Whether these are the same underlying store has **not**
been established — check before assuming a path from one works on the other.

Every remote shell is **tcsh**: pipe scripts to `bash -s`, and never use `$(...)` in the outer
ssh command.

`pipeline/launch_shard.sh SHARD GPU PORT NCPUS` runs one orchestrator on whatever host it
is invoked on. Stagger launches by ~60 s: each daemon reads 19 GB before binding its port.

Also: files written by one account are not automatically readable by another. `forward_*.bin` is
created mode `600`; `chmod 644` it before another host's account can load it.

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
