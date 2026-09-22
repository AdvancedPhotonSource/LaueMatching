# Changelog

Version history for the LaueMatching C/CUDA indexer, its pipeline, and the Python
distributions in `packages/`, which version independently and are released to PyPI.

**From laue-index 0.7.0 on, entries are headed by the PyPI version that shipped
them.** The `v2.2` section below collects what shipped in laue-index 0.3.0 to 0.6.1
under the pipeline's internal version string. `v2.2` was never a release: it is the
value of `laue_index/pipeline/_version.py`, which older provenance records wrote as
`laue_version` and which does not identify a build. Provenance schema 2 (laue-index
0.7.2) records the package version and source/binary hashes instead.


## laue-index 0.7.3 (unreleased)

- **Run-level provenance records the filter a streaming run actually applies.** The
  orchestrator stamps `provenance.json` from `ConfigurationManager`, which fills an absent
  `RobustFilter` with RunImage's default (1), so a streaming run that applied the legacy
  filter was recorded as `robust_filter: true` (0.7.2 known issue). The record now also
  carries `extra.streaming_postprocess` (`robust_filter_effective`, whether the key was
  present, and the exclusive-label floor in force with its source) and a
  `config_notes.robust_filter` line saying which field to read.
- **The post-processor's output is kept.** It was captured and dropped on success, so its
  startup warnings (including the absent-`RobustFilter` notice) reached no log. It is now
  written to `<output_dir>/postprocess.log` on every run.

## laue-index 0.7.2 (2026-09-21)

A correctness release. Several items change results; each says how to reproduce
0.7.1.

### Results change

- **Streaming post-processing honours the filter keys in the config.** 0.7.1
  streaming was hard-wired to the legacy filter and passed `--min-unique 2`
  whatever `MinGoodSpots` said. Now an EXPLICIT `RobustFilter`, `MinGoodSpots`,
  `MinNrSpots` or `MaxAngle` is honoured exactly as `RunImage` honours it. An
  ABSENT `RobustFilter` keeps the 0.7.1 streaming behaviour (legacy filter) and
  says so once at startup, because `RunImage`'s default for an absent key is 1.
  **On a config carrying `MinGoodSpots 4` the exclusive-label floor rises from 2
  to 4**; `MinGoodSpots 2` reproduces 0.7.1. Measured on one 2,613-frame shard of
  the texture-case sample: filtered orientations 8,870 -> 7,340 (−17%), 1,528 of the
  1,548 dropped having 2 or 3 own labels. That config has no `RobustFilter` line and
  `DoFwd 0`, so it does not exercise the robust filter, hexagonal near-duplicate
  removal or the forward-cache write. Raw C output (six runs, three per version):
  per-frame disagreement between versions (10-39 frames) lay within the range between
  runs of one version (21-35); a per-solution check found one frame of 2,613 that
  differed in every 0.7.1 run and no 0.7.2 run (a pair of candidates within MaxAngle merged
  differently). It is not attributable to the version: 0.7.1 and 0.7.2 sources built
  with the same toolchain give the same solutions on that frame in isolation, the
  forward cache has no off-detector entries (so the new bounds check is inert here),
  and the six runs confound version with GPU load (the 0.7.1 runs shared GPUs with
  seven sibling shards, the 0.7.2 runs did not), which can reorder the candidates the
  greedy duplicate merge sees. (An
  earlier line here, "raw C output unchanged within run-to-run variation", rested on
  a single repeat pair and was withdrawn after adversarial review.) **A config written by laue-index
  itself** (`RunImage config`, `write_config`) has always carried an explicit
  `RobustFilter 1`, so streaming runs from such a file now use the robust filter;
  set `RobustFilter 0` to reproduce 0.7.1.
- **Robust filter floor is monotonic.** It counted the orientation's own
  winner-take-all pixels but fell back to total `NMatches` exactly when that count
  was 0, so a Σ3 twin with every spot claimed was kept while one with 1-4 own
  pixels was dropped. The floor is now own pixels throughout; a solution with no
  own pixels is dropped. (Flooring on `NMatches` instead, as first proposed, kept
  four extra Σ3 relatives with 0-1 own labels on the test frame and was not used.)
- **Hexagonal near-duplicates** within `MaxAngle` are now removed by the robust
  filter (6/mmm operators); the CSL exemption stays cubic-only.
- **`RunImage` honours `GaussSigmaMax`**, which only the streaming path applied.
- **The analysis chain** (`pipeline/analysis/`, not in the wheel) counts hits
  without stacking harmonics (`nhit_distinct` beside `nhit`), picks symmetry from
  the space group rather than the phase name (any hex phase not named `alpha` had been
  given 24 cubic operators in two scripts), reads its null from the measured JSON
  instead of a hard-coded Ti null, labels the hex a-axis texture rows correctly
  (they had measured ⟨10-10⟩), and iterates the kept grains in `within_grain.py`.
  Grain connectivity is one shared setting, 8 (what `regrain.py` always used);
  scripts that had used 4 change unless `LAUE_CONNECTIVITY=4`.

### Fails loudly where it used to be silent

- **Crystal-fit tolerances are fractions, and the C checks them.** `tol_LatC` and
  `tol_c_over_a` are fractional bounds (0.001 = 0.1%); 0.7.1's usage text said
  percent. ≥ 1 or NaN is fatal, > 0.1 warns. When `tol_c_over_a` is set,
  `tol_LatC` is ignored with a NOTE instead of being validated.
- **Unfilled or malformed geometry stops the run.** The Python parsers refuse a
  malformed `SpaceGroup`, `Symmetry`, `LatticeParameter`, `P_Array` or `R_Array`
  (0.7.1 logged it and kept a default, e.g. SpaceGroup 225); the C refuses a short
  `LatticeParameter`/`P_Array`/`R_Array` line and a zero detector distance. The
  parameter templates now carry `__SET_ME__` placeholders instead of another
  experiment's fitted geometry.
- **A zero `R_Array` is the identity rotation.** It divided 0 by 0 and every
  predicted spot became NaN.
- **Forward-cache writes are atomic.** The cache is written to
  `<ForwardFile>.partial.<host>.<pid>` and renamed only after it is complete and flushed;
  a write, fsync or close failure is fatal and deletes the partial file. A killed
  run leaves only a `.partial.<host>.<pid>` file, which is never read. The GPU
  binary had printed and carried on after a failed write, and the CPU loop could
  spin on a zero-byte write.
- **Shards that share a forward cache take turns building it.** An `fcntl` lock on
  `<ForwardFile>.lock` (works over NFS) lets one process simulate while siblings
  wait and then reuse the published file, so a cold start needs one copy of the
  cache, not one per shard. The lock holder deletes stale partial files. If locking
  is unavailable the run warns and proceeds unlocked. A symlinked `ForwardFile` is
  replaced at its target; a new cache is created 0644 (it was 0600, unreadable to
  other beamline accounts) and a replaced one keeps its mode.
- **`Elo`/`Ehi` are checked.** An unparseable value silently kept 5 / 30 keV in all
  three binaries; it is now fatal, and so is a band that fails 0 < Elo < Ehi.
- **The `.cu` files build with gcc 14 and later**: `<omp.h>` and `<math.h>` are
  included before the `extern "C"` block (a hard error on gcc ≥ 14; CI's gcc 13
  passed it).
- **The stream daemon** bounds-checks its kernel like the GPU binary, initialises
  and validates its geometry, and checks `numProcs`.
- **The orchestrator exits non-zero when post-processing fails**, instead of
  logging "Pipeline complete".
- **`pipeline/run_laue.sh`** found its orchestrator at the repo root, where it is
  not, and reported a launch that had died. It now resolves `scripts/` (or the
  installed package), checks every input before launching anything, and confirms
  each launch is alive. `pipeline/launch_shard.sh` (deleted paths, campaign
  naming) is retired in favour of `pipeline/dispatch/`.
- **Analysis scripts exit naming the variable** instead of falling back to another
  campaign's data paths, a Ti null, a 201-column raster or a 1 µm step; five
  scripts that raised `NameError` on import now run.

### Known issues in 0.7.2

- **Provenance misreports the streaming filter.** The orchestrator stamps provenance
  from `ConfigurationManager`, which fills an absent `RobustFilter` with RunImage's
  default (1), so `provenance.json` says `robust_filter: true` for a streaming run that
  applied the legacy filter (absent key). Read the params file, not provenance, for
  the filter a streaming run used. (Fixed in 0.7.3.)
- A forward cache of the right size built for a different geometry is still accepted
  (see `manuals/laue/DIAGNOSIS.md`).

### Worker sizing (the 0.7.0 defect)

- `LAUE_SHARDS_PER_HOST` divides the CPU, memory and thread budgets; a
  `LAUE_DAEMON_NCPUS` you set is subtracted, never below half the per-shard share
  (nothing sets it for you, so a plain run sizes as 0.7.1 did);
  `RLIMIT_NPROC` caps the pool; a fractional cgroup quota gives 1 worker.
  `LAUE_PREPROCESS_WORKERS` still overrides. Pool workers run their
  OpenMP/MKL/OpenBLAS/OpenCV/diplib thread pools at 1. The pinning is done in the
  PARENT just before the pool forks, never inside a forked worker: OpenCV and
  libgomp are not fork-safe, and a version that pinned from the worker
  initializer hung a Linux worker indefinitely. A spawned worker (macOS) pins
  its own pools.
- The streaming config now parses `PreprocessWorkers` and `RobustFilter`.

### Added

- **`pipeline/dispatch/`**: multi-host sharding (`mkrun.py`, `preflight.sh`,
  `dispatch.sh`, `launch_run.sh`, `watch_arm.sh`, `wait_static.sh`), each encoding
  a failure that cost a run.
- **Provenance schema 2** records the `laue_index` version, the SHA-256 of the C
  sources and of each binary, and the binary that actually ran. Schema 1's
  `laue_version` ("2.1.0"/"2.2.0") did not identify a build.
- **Config schema rows** for `tol_LatC`, `tol_c_over_a`, `MinSpotIntensity`,
  `GaussSigmaMax`, so they are validated and survive a config rewrite.
- IndexFile `NiData` is filled (it was always 0); its `err(deg)` is documented as
  integer-pixel quantisation, not a fit residual.
- `.unique_spot_counts.txt` now reaches the output HDF5 (the reader looked for
  `.bin.unique_spot_counts.txt`, which nothing writes).

### Docs

- The three spot counts are defined once (handbook invariant 15b):
  `NMatches` does not stack harmonics (the C dedups by q-hat);
  `unique_spots_per_orientation` is winner-take-all across a frame's orientations,
  not "distinct"; the analysis `nhit` stacked harmonics (1.186x on the hcp
  deposit), `nhit_distinct` does not. Descriptions across the package corrected.
- `Optimizer` is parsed and ignored: refinement is always Nelder-Mead (the README
  said BOBYQA was the default).
- Handbook invariant 38: laue_torch renders `[X, Y]`, real frames are `[row, col]`.

## laue-torch 0.1.4 (2026-09-21)

### Fixed (results change)

- `realdata.LaueScanLoader` read the orientation seed from solution columns 1-9
  (intensity, scores, NMatches), never the matrix. It now reads the matrix columns
  by layout: 34 columns (RunImage, 22-30) or 35 (stream, 23-31); other layouts
  raise unless `orientation_columns` is given.
- `realdata.VoxelODFRefiner` compared a stored `[row, col]` frame with the model's
  `[X, Y]` render: transposed on a square detector, a broadcast error on a
  non-square one. Frames stay as stored with `VoxelMeasurement.axis_order`, and
  both refiners convert on entry with a shape check. **Every earlier
  `VoxelODFRefiner` result on a square detector changes.**
- **Breaking: `axis_order` is required where a frame meets a refiner.**
  `VoxelMeasurement.axis_order` and `MultiGrainVoxelRefiner.refine(axis_order=)`
  have no default; the refiners raise until it is `"YX"` (a real frame,
  `image[row, col]`) or `"XY"` (a laue_torch render). `LaueScanLoader` always sets
  it (the file's marker, else `"YX"`). No default could be right both for 0.1.3
  callers and for real frames, and on a square detector a wrong guess is silent.
  `MultiGrainVoxelRefiner.refine` no longer transposes implicitly.
  `TwoSourceMeasurement` and marker-less coded-aperture files keep `"XY"`, being
  laue_torch-written by construction.
- Both refiners rendered in a fixed 5-30 keV band whatever the parameter file said;
  they now use `Elo`..`Ehi` and refuse a missing or defaulted band.
- `MultiGrainVoxelRefiner(mode="strain_deviatoric")` was an alias of
  `strain_voigt`, fitting 6 components including the unobservable hydrostatic one.
  It now fits the 5 trace-free components.
- The multi-grain fallback target (no indexer spots) used the window sum against a
  peak-normalised splat, gave every co-located harmonic the full amplitude, and
  summed grains into one vector. It is now the window peak, one reflection per
  predicted pixel, per grain.
- `laue_torch.io.load_orientations` never detected a `%`/`#` header, so a
  solutions file gave columns 0-8 as the matrix.
- The `VoxelODFRefiner` Laplace posterior used a different random seed from its
  fit, a per-pixel-MSE curvature (σ inflated by about √(N/2)), and turned every
  exception into NaN. It now shares the fit's seed, uses 0.5·SSR with the plug-in
  noise variance, and catches only `torch.linalg.LinAlgError`.

### Added

- `MultiGrainVoxelRefiner(compute_posterior=True)`, previously accepted and
  ignored, returns a Laplace posterior; `LaplacePosterior.is_positive_definite`
  and `n_negative_eigvals` expose a saddle point. Read `eigvals`, `cond_number`
  and `rank_eff` before any σ.
- `/entry1/axis_order = b"XY"` in the CLI's HDF5 output; the coded-aperture HDF5
  writes and honours `/entry/axis_order`.
- `laue_torch.jointfit.fault_rod`: a continuous-L forward model for one (h, k)
  reciprocal-lattice row (stacking-fault rods), cross-checked against
  `LaueForwardModel` at integer L.

## laue-index 0.7.1 (2026-09-09)

- **The streaming daemon frees its host copy of the forward simulation after the
  GPU upload.** In unchunked mode that copy is read once, by the upload, and was
  then held for the daemon's life: 12.2 GB at 100M orientations. Idle RSS on 40
  frames against the 100M cache went 18.35 -> 6.99 GB; solutions and spots were
  identical (GrainNr excluded, it is nondeterministic). The PEAK is unchanged
  (~18.2 GB, the cache must still be read to be uploaded); what goes is the holding,
  which competed with the preprocessing pool for memory.
- **The sdist pre-flight measures what it ships.** Its size guard summed the
  compressed tarball, so a 32.93 MB working file compressed to 320 KB and passed;
  it now sums uncompressed members and names the ten largest. `*.bin`, `build` and
  `dist` are excluded from the sdist. PyPI was not affected.

## laue-index 0.7.0 (2026-09-09)

- **Preprocessing worker count chosen from the machine.** It was
  `min(os.cpu_count(), 8)`; 8 sits at the end of the linear region (measured on a
  40-core host: 17.8 frames/s at 8 workers, 34.7 at 40). The pool is now sized from
  the CPUs this process may use (affinity and cgroup quota) and a fitted per-worker
  memory model. **Known defect, fixed in 0.7.2:** the size does not account for
  sibling shards on the same host, and each worker also starts its own OpenCV
  thread pool. Two shards on a 112-core host with `ulimit -u` 8192 exhausted
  threads; the visible symptom was a misleading `GPUassert: device busy or
  unavailable`. On 0.7.0/0.7.1 set `LAUE_PREPROCESS_WORKERS` per shard and
  `OPENCV_NUM_THREADS=1` when several shards share a host.
- **Preprocessing centroids only the lit pixels** (`_centers_of_mass_sparse`), bit-
  identical to `ndimage.center_of_mass`; `preprocess_image` 486 -> 291 ms/frame.
- **The coarse-fit blur is parallel over rows**, bit-identical at 1/16/64 threads;
  it had been 449 of 464 ms per image in the streaming fit.
- **Four exact rearrangements of the forward-simulation loop**, 27% off it with no
  spot moved: a per-thread duplicate-pixel mask replaced by a scan of the spots
  already written (`pixelClaimed`), one shared stage-1 comparison instead of three
  drifted copies, no sqrt in the inner loop, cheaper rejection order.

## v2.2 (laue-index 0.3.0 to 0.6.1)

- **`LAUEMATCHING_CUDA=1` means ATTEMPT again (0.6.1).** 0.6.0 redefined it to
  "require", which was a gratuitous break: every README and script from the
  opt-in era says `LAUEMATCHING_CUDA=1 pip install laue-index`, so on any
  machine without nvcc the install began *failing* where it used to succeed
  CPU-only. The new capability moves to a new name, `LAUEMATCHING_CUDA=require`.
  Values now: unset or `1` attempt and degrade with a warning, `0` skips,
  `require` fails the install if the GPU binaries cannot be built.
  The `pip-cuda` CI job is updated to the post-0.6.0 contract it had been
  pinning the opposite of — a plain install in a CUDA container must now
  *produce* the GPU binaries — and gains two steps: `=0` stays CPU-only, and
  `=1` with nvcc hidden must still succeed, which is exactly the case 0.6.0
  broke.

- **The CUDA binaries are now built BY DEFAULT, and the install records what it
  did.** Auto-detecting nvcc used to be rejected for a good reason: once a CUDA
  target is added to a project, CMake cannot try-and-continue, so a toolkit that
  cannot compile these sources killed the whole `pip install` and took the
  working CPU binary with it. The CUDA targets now live in their own project
  under `cmake/cuda/`, configured and built through `execute_process()`, so a
  failure is an exit code the parent catches — the CPU binary cannot be
  affected, by construction. `LAUEMATCHING_CUDA` becomes a tri-state: unset
  attempts and degrades with a warning, `0` skips, and **`1` now REQUIRES** the
  GPU build and fails the install if it cannot be done (previously `1` only
  meant "try", and degraded silently).
- **`laue-index doctor`, and a build manifest behind it.** Every install writes
  `laue_index/_build_info.json`: which binaries were built, by which nvcc, for
  which architectures, the SHA-256 of the compiled sources, and — when the CUDA
  binaries were not built — *why not*. `doctor` reads it and catches the two
  states that are otherwise silent: a CUDA device present with no GPU binary
  (measured 2026-09-06: `pip install --upgrade` removed the GPU binaries from
  two beamline environments and nothing said so), and a GPU binary that cannot
  launch on the card in front of it, which prints `Unique Orientations: 0` and
  exits 0. `--json` for deployment checks. The recorded source hash is what
  lets a binary salvaged from a previous install be shown equivalent to what
  this install would have produced, rather than assumed to be.

- **`laue_index.xmas`: XMAS detector calibrations to `P_Array` / `R_Array`.**
  XMAS describes a detector with a distance, a "center channel" pixel and
  roll/pitch/yaw; LaueMatching wants a translation in metres and a rotation
  vector in radians. Nothing bridged the two, so an XMAS-calibrated dataset
  could not be indexed without hand-deriving the pose — and that derivation has
  three traps that each yield a confident wrong answer rather than a failure:
  the pixel origin (XMAS x counts from the far end of the long axis and its y is
  1-based), `P` being the *inverse* of the projection rather than a fit, and
  `R_Array` being **radians** where `GenerateHKLs --help` and
  `params_alpha.template.txt` both say degrees. `enumerate_candidates` yields the
  32 physically distinct poses for the indexer to discriminate rather than
  picking one, having first removed the 180°-about-beam duplicates (an exact
  gauge, verified at 2.0e-12 px). Detector handedness is carried as
  `LaueGeometry.row_parity` because it cannot live in a proper rotation matrix.
  Validated at TPS 21A (NSRRC): 46 Si(100) reflections at **0.42 px** median
  residual, and 6 of 8 candidate mounts return `Initial solutions: 0`.

- **`MinSpotIntensity`: a floor on what counts as a matched spot.** The match
  test was a bare `image[px] > 0`, so a pixel carrying 4e-06 counted as evidence
  for a candidate orientation. The objective is `NMatches * sqrt(Intensity)` —
  it multiplies by the *count* — so such a pixel adds nothing to the intensity
  sum, inflates the score anyway, and increments the `NMatches` that
  `MinNrSpots` gates on. Measured on real 34-ID-E frames, **35.3% (Zn) and 28.8%
  (Si) of matched spots carry intensity below 1.0**. The new
  `MinSpotIntensity` parameter is the pixel value a predicted reflection must
  exceed; it is applied at all nine match sites — candidate matching in all
  three binaries (including both device kernels), the refinement objective, and
  the reported `NMatches` — so the optimised and reported counts cannot describe
  different things.

  **The default is `0.0`, which is exactly the historical `> 0`.** Verified on
  real data rather than asserted: on a Si frame and a Zn frame, indexed on both
  the CPU and the GPU path, the new binaries reproduce the old ones' solution
  set exactly. A positive control at `MinSpotIntensity 1.0` moves Si from 77 to
  24 candidate matches and Zn from 140 to 55, so the no-op default is not a
  parameter that fails to arrive.

  > **⚠️ Leave this at the default unless you have measured, on your own data,
  > that raising it helps. No setting of it has been shown to improve a result.**
  > Follow-up work (2026-09-03) found: on one real frame (34-ID-E `Si-white_1`)
  > `MinSpotIntensity >= 5` **deleted the correct crystal** — the highest-scoring
  > solution in the frame, 49 matched reflections, independently confirmed by a
  > separate campaign to 5e-5° — leaving its Σ3/Σ5 coincidence ghosts ranked top.
  > It is removed from the *stage-1 candidate set*, before refinement can recover
  > it, so no later stage can put it back.
  >
  > That failure did **not** generalise: across 10 frames of a Zn raster, no grain
  > was lost at any tested value (0 losses in 30 frame×floor cells), and on a
  > synthetic image with 19 known orientations all 19 were recovered at every
  > value from 0 to 20. So this is a demonstrated hazard on at least one real
  > frame, not a general defect — and the parameter has no demonstrated upside to
  > weigh against it. The default `0.0` is exactly the historical `> 0` and was
  > verified byte-identical on real Si and Zn frames, on both the CPU and GPU
  > paths.

- **Repository layout rearranged (2026-08-29).** No behaviour change; every move
  was a `git mv`, so history follows the files.
  - The C/CUDA source now lives in **one** place, `packages/laue_index/c_src/`.
    The repo-root `src/` copy is gone, and with it `utils/sync_vendored_c.py` and
    its CI step: the root `CMakeLists.txt` reaches down into the package, exactly
    as it already did for `cmake/LaueCudaArch.cmake`. The duplication existed
    because a pip sdist cannot reach up to a repo-root `src/`; putting the single
    copy inside the package satisfies that without making a one-copy edit
    possible.
  - The doc set moved `scripts/pipeline/laue/` → **`manuals/laue/`**, matching
    MIDAS's `manuals/ff-hedm/` and `manuals/nf-hedm/`.
  - The campaign launcher and its analysis scripts moved
    `scripts/pipeline/` → **`pipeline/`**. **If you have a deployed checkout, the
    launcher is now `pipeline/run_laue.sh`** — the old path no longer exists.
  - `scripts/` now holds only the entry-point shims onto
    `laue_index/pipeline/`, which is all it had left to be.
  - Version history moved out of `README.md` into this file.

- **Fixed: SIGSEGV at the end of every cold-cache run.** A run that *wrote* the
  forward-simulation cache — the first run on any machine, with the orientation
  database under `/dev/shm` — exited with a segmentation fault **after** writing
  complete and correct output, and the pipeline reported the image as failed.
  The cleanup's `if (orientsMapped) munmap(…) else free(orients)` pair had a
  second `if` inserted between its halves (by the C hardening below, ironically),
  re-parenting the `else`: writing the cache leaves `outArr` NULL, so `free()`
  ran on memory `munmap`'d one line earlier. Runs that *read* an existing cache
  took the other branch, which is why every test stayed green — they all
  supplied a prebuilt cache. Now covered by `test_cold_forward_cache.py`, which
  fails against the unfixed binary.
- **`pip install laue-index` is now the whole thing.** The orchestrators moved
  into the package (`laue_index/pipeline/`), so `laue-index run process -c … -i …`
  indexes a frame with no checkout; `laue-index fetch-db` downloads the 6.7 GB
  orientation database; `scripts/` keeps a shim for each entry point so existing
  invocations are unchanged. `LAUEMATCHING_CUDA=1 pip install laue-index` also
  compiles `LaueMatchingGPU` and `LaueMatchingGPUStream` — opt-in, because a
  toolkit that cannot build them would otherwise fail the whole install.
- **Optimizer: BOBYQA and NLopt removed.** Refinement is a vendored Nelder–Mead
  simplex; `Optimizer BOBYQA` in a config is accepted, noted, and ignored. On
  198 paired synthetic seeds Nelder–Mead was better on every statistic (median
  0.0041° vs 0.0054°, p95 3.3× tighter, max 25× tighter) at identical
  wall-clock. With no external optimizer to fetch, the C compiles at
  `pip install` time and builds offline.
- **CUDA builds for every architecture the toolkit supports, plus PTX.** The old
  hardcoded `70;80;86;90` failed outright on CUDA 13, which dropped Volta
  (`nvcc fatal : Unsupported gpu architecture 'compute_70'`), and covered
  nothing newer than Hopper. Building for the *local* card is not the fix
  either: PTX JIT works forward, never backward, so a binary built on a newer
  GPU than it runs on finds **zero** orientations and exits **0**. The build now
  asks `nvcc --list-gpu-arch` and covers all of them, with PTX for the newest.
- **Fixed: an arch mismatch was silent.** `cudaErrorNoKernelImageForDevice` is
  reported by the kernel *launch*, and only the following synchronize was
  checked — so the kernel never ran, nothing complained, and the run reported no
  grains. Both CUDA binaries now check the launch; the streaming daemon would
  otherwise have served zero matches for every frame of a scan.
- **Fixed: the forward cache was validated only by the CPU binary.** The CUDA
  binaries accepted any file that existed, so a 0-byte leftover — which the C
  itself creates when a run is interrupted mid-write — was mapped as a 12.2 GB
  cache and took SIGBUS on first touch. One check, in the shared header, called
  by all three.
- **Provenance tracking**: every generated artifact (HKL CSV, simulation HDF5,
  per-image indexing HDF5, orchestrator run directory) now carries a git
  commit, config snapshot, and weak fingerprints of its input files.
  See [docs/provenance.md](docs/provenance.md).
- **IndexFile text output**: on by default — each indexed image emits a
  Tischler-style `.indexing.txt` alongside the HDF5 (`--no-indexfile` to
  disable). See [docs/indexfile-format.md](docs/indexfile-format.md).
- **`scripts/GenerateOrientations.py`**: reproduce the orientation database
  at any spacing / crystal system using `orix`. Emits the **full SO(3)**,
  not the fundamental zone — the oversampling is load-bearing for the
  indexer's spurious-match filter. Writes a `.meta.json` sidecar with full
  provenance.
- **`scripts/annotate_orientation_db.py`**: writes a retroactive sidecar
  next to the existing `100MilOrients.bin`. Hooked into `build.sh`.
- **`GenerateHKLs.py` -Ehi flag**: the max-energy cutoff is no longer
  silently hardcoded to 30 keV.
- **`laue_index` package**: the Python orchestration is restructured into typed
  pipeline stages (records / geometry / filtering / thresholds / preprocess /
  indexer / postprocess / output / config_schema / cli).  Positional column
  "magic numbers" are replaced by a typed `Solution` record; the orientation
  filter (incl. the twin/CSL-aware variant) lives in one place; thresholding is
  pluggable; one declarative schema drives config parse **and** write.
  `laue_stream_utils.py` is now a thin re-export shim and `RunImage.py` is a
  thin orchestrator over the stages.  pip-installable
  (`pip install -e packages/laue_index`, `laue-index` CLI).  Behaviour-preserving,
  guarded by a golden-anchored characterization test suite under
  `packages/laue_index/tests/`.
- **C / CUDA hardening**: `size_t` indexing (large-DB overflow safety), full
  malloc/`mmap`/`fread` checks, single end-of-run `fsync` (was per-write
  `O_SYNC`), `snprintf` bounds, kernel pixel-bounds + spot-count clamps, and a
  cap on the O(N²) duplicate-merge.  The GPU now scores on full-precision
  intensity (uint8 quantization dropped) for parity with the CPU.  Streaming
  daemon: clean-shutdown thread join, `recv` timeout, `sigaction`.  Built and
  functionally validated on H200 (GPU and streaming-daemon runs reproduce the
  CPU result).
- **Robustness fixes**: twin/CSL-aware orientation filter (keeps real Σ3 twins),
  adaptive noise-floor threshold (recovers faint frames), and a fixed
  stream/RunImage column-format heuristic.

## v2.1 (2026-03-03)

- **GPU Kernel Optimizations**: 2.3× faster GPU matching:
  - Float32 kernel with `__ldg()` texture cache reads.
  - `atomicAdd` compact output: eliminates 800 MB D2H transfer.
  - **uint8 image quantization**: image shrinks from 16 MB to 4 MB, fits in L2 cache. Kernel time drops from 273 ms to 108 ms.
  - Nonzero-preserving quantization ensures spot counts remain exact.
- **CPU uint8 Matching**: `LaueMatchingCPU.c` uses uint8 quantized image for the `doFwd=0` matching path, improving L3 cache sharing across 96 threads.
- **Parallel Preprocessing**: `laue_image_server.py` uses `ProcessPoolExecutor` (up to 8 workers) for multi-process frame preprocessing.
- **Async Pipeline**: 3-stage architecture (submit → consumer → sender) fully decouples preprocessing from TCP sending.
- **KDTree Sigma**: `calculate_gaussian_sigma` uses `scipy.spatial.cKDTree` (O(n log n)) instead of O(n²) brute-force.
- **Reduced Log Verbosity**: Orchestrator result listing replaced with single-line summary.

## v2.0 (2026-02-18)

- **Streaming Pipeline**: New `LaueMatchingGPUStream` CUDA daemon + Python orchestrator for multi-image processing over TCP.
- **Float32 Wire Protocol**: Image transfer uses float32 (16 MB/frame for 2048×2048) instead of float64, halving bandwidth with no precision loss in GPU matching.
- **Pipelined Image Server**: Producer-consumer threading overlaps H5 loading/preprocessing with TCP sending.
- **Progress Bar**: Real-time tqdm progress bar with throughput (img/s) and ETA.
- **Graceful Daemon Shutdown**: Handles unresponsive GPU processes without crashing the pipeline.
- **Scripts Reorganization**: All Python scripts moved to `scripts/` directory with comprehensive `scripts/README.md`.
- **Module Decomposition**: Decomposed `RunImage.py` (3,553 → 1,673 lines) into reusable modules:
  - `laue_config.py` (782 lines) — configuration dataclasses and parameter file parser.
  - `laue_stream_utils.py` (1,108 lines) — image I/O, preprocessing, TCP wire protocol, orientation sorting/filtering.
  - `laue_visualization.py` (937 lines) — 8 standalone visualization functions (Plotly interactive, simulation comparison, reports, etc.).
- **Post-Processing**: `laue_postprocess.py` now sorts filtered orientations by quality and supports optional per-image interactive visualization.
- **Streaming Utilities**: `laue_image_server.py` for TCP image sending with live progress tracking; `laue_orchestrator.py` for full pipeline management.

## v1.0 (2026-02-17)

- **Code Refactor**: Consolidated ~700 lines of duplicated code into shared `LaueMatchingHeaders.h`.
- **Bug Fixes**:
  - Fixed c/a ratio fitting (was integer division `1/3`).
  - Fixed negative pixel handling (uint16_t underflow).
  - Fixed trigonal symmetry definition (consistent between CPU/GPU).
  - Fixed memory leaks and file descriptor handling.
  - Fixed GPU unique-solution indexing bug.
- **Build System**: Improved CMake configuration with working strict warning flags.
- **Performance**: Hoisted memory allocations out of critical loops; added `gpuErrchk` macro for CUDA error handling.
