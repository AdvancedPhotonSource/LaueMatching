# Laue diagnosis reference

Symptom → discriminating test → cause → lever. Read by `beamreport`; each entry attaches
to a symptom the generic diagnostics detect.

**Every entry carries a test that can come back the other way.** An entry that cannot
exonerate the cause it names does not belong here.

Fourteen entries. Provenance is the campaign notebooks and the handbook (`README.md`)
invariants, cited per entry. Where an entry uses a spot count, the meaning is the glossary in
`INVARIANTS.md` invariant 15b.

---

## Local symptoms

Emitted by **this technique's own procedure**, not by `beamreport`'s generic diagnostics,
which key off per-observation residuals against declared coordinates. A run that exits 0 with
nothing indexed, a synthetic control indexing at a fraction of the real rate, or a host that
runs out of threads are real and useful, and nothing generic will detect them, so they are
declared here rather than renamed into the wrong shape.

Every row names where the check lives. A symptom nothing produces is dead text that reads as
coverage.

| symptom | emitted by |
|---|---|
| `run.completed_zero_grains` | the indexer printing `Initial solutions: 0` / `Unique Orientations: 0` and exiting 0 on every frame of a run; `laue-index doctor` for the architecture half |
| `compute.thread_oversubscribed` | `res = 11` (OpenCV cannot spawn a thread) or `GPUassert: ... busy or unavailable` in a shard's logs, or `fork: retry` from the dispatcher; `pipeline/dispatch/watch_arm.sh` reports all three (0.7.2 and later) |
| `map.coincidence_falls_with_distance` | the neighbour spot-coincidence check: fraction of one frame's brightest spots within N px of another frame's, against raster separation |
| `solutions.csl_related` | pairwise disorientation of one frame's accepted solutions landing on a CSL angle (60.0° Σ3, 38.9° Σ9, ...) |
| `map.smooth_on_fine_raster` | a label-shuffle smoothness null passing on a raster whose step moves the pattern by only a few px |
| `background.structured` | the per-frame median sampled across the raster (invariant 20) rising well above the scan's background over a band of positions |
| `cache.forward_geometry_mismatch` | a one-frame run with `DoFwd 1` (forward cache rebuilt) disagreeing with the same frame run on the existing `ForwardFile`; or `ForwardFile` older than the last edit to the geometry or lattice lines of its parameter file |
| `control.partial_yield_on_simulated_frames` | the index rate of a synthetic-control batch against the rate on real frames of the same geometry |

---

## An indexing run that finishes cleanly and finds nothing

symptom: run.completed_zero_grains

**Test.** Run the **CPU** binary on the same frame with the same parameter file. If the
CPU finds grains where the GPU found none, the data is not the problem and no amount of
threshold tuning will help. Then ask the binary and the card whether they agree:

```bash
laue-index doctor                                  # build manifest's architectures vs this card
CUOBJDUMP=$(ls /usr/local/cuda-*/bin/cuobjdump | tail -1)   # NOT on PATH on the beamline hosts
for t in GPU GPUStream; do
  b=$(python -c "from laue_index import indexer; print(indexer.binary_path('$t'))")
  echo "$t $b"
  "$CUOBJDUMP" --list-elf "$b" | grep -oE 'sm_[0-9]+' | sort -u
  "$CUOBJDUMP" -all --list-ptx "$b" | head -3     # -all is REQUIRED or it reports none
done
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

Check **both** binaries: a sharded run uses `LaueMatchingGPUStream`, not `LaueMatchingGPU`.
**Empty output is a FAILED check, not "no cubins".** A bare `cuobjdump` is not on PATH on
the beamline hosts; the shell prints `command not found` to stderr, the pipe hands `grep`
nothing, and the discriminating test for a silent failure is itself silent. If
`$CUOBJDUMP` is empty or a binary path does not exist, stop and fix that first.

A card whose compute capability has no matching cubin, and no PTX at or below it, cannot
run the kernel. The entry exonerates itself: if the architectures cover the card, this is
not your problem.

**Cause.** `cudaErrorNoKernelImageForDevice` is reported by the kernel **launch**, and only
the following `cudaDeviceSynchronize` was checked — so the kernel never ran, nothing
complained, and zero matches is a perfectly ordinary outcome for a frame with no grains.
Measured: an sm_120-only build on an sm_90 card printed `Initial solutions: 0 Unique
Orientations: 0` and **exited 0**. PTX JIT rescues the older→newer direction and never the
reverse, so a binary built on a newer GPU than it runs on fails this way — and with a
home directory shared across hosts, binaries travel between hosts constantly.

**Lever.** Fixed in `3b2abcb` (laue-index 0.3.1): the launch result is now checked, so this
fails loudly (`GPUassert: named symbol not found`, exit 244; message and exit code: source not
in repo) instead of returning an empty map. Do not
narrow `CMAKE_CUDA_ARCHITECTURES` for a binary you will move — the default builds every
architecture the toolkit supports plus PTX for the newest. A `laue-index` older than 0.3.1
has neither the check nor the multi-architecture default.

## Detector artefact reproducing as a persistent signal

symptom: systematic.common_offset

**Test.** A bloom or a hot column sits at a **fixed detector position**, so it reproduces
frame to frame exactly. Persistence alone therefore cannot separate it from a real
single-crystal reflection. Test in **orientation space** instead: does the solution
derived with those detections agree across many frames, and does it survive discarding
them? On the bare-Cu reference the indexer's own solution stayed clean — 16 of its 18
assigned spots off-streak, orientation moving **0.011°** when the other 2 were dropped.
If the orientation moves substantially, the detections were load-bearing and the artefact
is driving the fit.

Compare the two detectors before concluding anything: the analysis-side peak count was
inflated **2×** on that frame while the indexer's own detection was unaffected.

**Cause.** Charge bloom out of a saturated reflection, or permanently hot pixels, entering
an unfiltered local-maximum detector. Identical spurious spots on every frame hand the
indexer the same wrong orientation every time — a fake result with a beautiful null.

**Lever.** Filter blooms by **shape** in the analysis detector (`frame_peaks.py`): a
morphological opening that erases structures thin in one axis and long in the other cannot
touch a compact reflection. Never delete a detection — flag it; position stays valid for a
clipped peak even when intensity does not. `INVARIANTS.md` invariant 13.

## A correlation that cannot pin the direction it appears to

symptom: null.not_cleared

**Test.** Recompute the correlation under the opposite registration flip. If **both** flips
reach a comparable |corr|, the magnitude carries no directional information and this entry
applies — on Zn/Zn the identity flip gave footprint↔deposit −0.5 and the both-flip gave
+0.45. If one flip collapses toward zero, the direction is genuinely determined and this
entry does not apply.

Then check the null: adjacent scan positions are strongly autocorrelated, so a plain
permutation null is optimistic by orders of magnitude — effective n was ~70, not 40,357,
making the naive p ~600× too small. Use a toroidal-shift or block null.

**Cause.** The sign of a map↔image correlation is set by the flip chosen, and choosing the
flip with the largest |corr| is circular.

**Lever.** Anchor the direction on something **independent** of the correlation: the
experimenter's direct observation, SEM/optical morphology, or two different maps that must
agree. The correlation refines the fit; it does not decide the direction. This cost three
re-flips before it was learned. `INVARIANTS.md` invariants 10–11.

## A threshold-defined split, contrasted on the quantity that defined it

symptom: split.bimodal

**Test.** Ask what defined the two groups. If the split was made by a cutoff on X, then
"A differs from B in X" is guaranteed and means nothing. The test that settles it is an
**independently measured** signature: does a quantity not used in the split also separate
the groups? If it does not, the split is an artefact of the cutoff.

**Cause.** A cutoff applied to form the groups, then used as evidence about them.

**Lever.** Contrast on an independent signature, and store the threshold's statistic beside
any threshold-defined split so no reader mistakes the split for a result. `README.md`
invariant 9.

## A diffuse fan that does not come from the direct beam

symptom: background.structured

**Test.** Fit the local streak directions (structure tensor) for a common convergence point,
and compare that point against **where the transmitted beam meets the detector plane** —
computed from the geometry, *not* taken to be the PONI. On a tilted panel the two differ:
measured 751 px apart at 30° tilt. Control the fit against random directions at the same
pixels; a real convergence gave 29.5 px residual against 86.9 ± 0.3 px for the control.

If the streaks converge on the beam, it is small-angle scattering. If each streak instead
follows its own direction anchored to a Bragg reflection, it is asterism — but prove that
with a control matched for **spatial support** (invariant 28), because predicted reflections
occupy only part of the panel and an unmatched control will hand you 28σ for nothing.

**Cause.** Small-angle scattering off the direct beam from sharp density contrast in the
specimen — a crack, scribe line or scratch. It raises the frame median (5 → 40 counts here),
which raises a percentile threshold, which **drowns weak Laue spots**: those positions show
*more* total intensity and *fewer* indexable peaks.

**Lever.** Subtract a per-frame smooth background before thresholding; do not use a single
shared background, because the feature is present at only a narrow band of positions
(invariant 16 in reverse). To identify the feature itself, ask for an optical or SEM look —
SAXS says sharp contrast, not which kind. `LAB_NOTEBOOK_16BMD_Si.md` §5; `README.md`
invariants 24 and 28.

## Two codes agree suspiciously well, or a residual will not go below a floor

symptom: systematic.common_offset

**Test.** Before quoting any residual that follows a fitted transform, print **`mean(dx)` and
`mean(dy)` separately**. A median hides a constant offset completely. If the offsets are
constant to many decimals across spots spanning the panel, the two models are identical and
what you are quoting is fit residual, not accuracy. Then re-predict with the **unfitted**
geometry and see whether the residual collapses to a pure translation.

**Cause.** Orthogonal Procrustes / Kabsch has three rotational DOF and no translation, so a
rigid pixel-origin offset is absorbed as a spurious rotation. Measured: 0.667 px became
0.0423° of crystal rotation, larger than the 0.0188° agreement it produced. Common origin
offsets: pixel-centre vs pixel-corner (`(N−1)/2` vs `N/2`, exactly 0.5 px), and a documented
beam shift between the calibration beam and the experiment beam.

**Lever.** Fix the origin convention and re-predict with no fit at all — here that took the
residual to 4e-5 px. And remember what agreement between two codes can and cannot test: it
tests their algebra, never anything they both read from the same calibration (invariant 26).
`LAB_NOTEBOOK_16BMD_Si.md` §3–4; `INVARIANTS.md` invariants 25 and 26.


## One phase indexes to nothing while another, same station and geometry, indexes fine

symptom: run.completed_zero_grains

**Symptom.** `Initial solutions: 0 / Unique Orientations: 0`, exit 0, on every frame of one
dataset, while a different phase on the same detector and the same converted geometry returns
dozens of solutions per frame.

**Test.** Do not touch the code. Project a few hundred random orientations
through the geometry and histogram the **on-panel reflections by energy**, against the
detector's `Threshold_setting` and the beam's declared band:

```python
# ~1 min. For each random U: project all allowed hkl, keep on-panel, bin the energies.
```

If the count inside the usable window is near or below `MinNrSpots`, the measurement cannot
index that phase and no parameter will make it.

**Cause.** A small unit cell puts the strong low-index reflections at low energy. Measured at
TPS 21A: Ni's (111) through (400) all sit **below** the 8.74 keV discriminator at every 2θ the
panel covers, leaving **18.6** weak high-index reflections per orientation against 47 for Si
and 52 for Ti α. Ni indexed 27.9 % of voxels; Ti indexed 99 %.

**Lever.** None at analysis time — and prove that rather than assuming it. Widening Elo
8.74 → 6.0 keV bought +0.54 reflections/orientation; widening Ehi 26 → 45 keV bought +65.75
*predicted* and **−0.01 matched**, on exactly the same 135 voxels. That second number is also
a measurement of the beam: there is no usable flux above 26 keV. The fix is a detector
threshold that reaches the strong reflections, flux at those energies, or a different
detector distance — i.e. a beamline change, reportable as such.

## The same solution table read from two files disagrees

symptom: scale.inflated

**Symptom.** A match count that is plausible but wrong — e.g. NMatches 280 where the physics
caps it at 30.

**Test.** Count the columns. `solutions_filtered.txt` has **34**;
`/entry/results/filtered_orientations` in the `.output.h5` has **35**, because the HDF5 writer
prepends `image_nr`.

| quantity | .txt | .h5 |
|---|---|---|
| NMatches | `[5]` | **`[6]`** |
| NSpotsCalc | `[6]` | `[7]` |
| OrientMatrix | `[22:31]` | **`[23:32]`** |

**Cause.** Text-file offsets applied to the HDF5 array return `NMatches*sqrt(Intensity)`.

**Lever.** Assert the column count before indexing into either, and take the frame number from
the `source_file` attribute on `/entry/results` rather than parsing the output filename. What
caught this was a **physical ceiling**, not inspection: the phase could only put ~18.6
reflections in the window and `MaxNrLaueSpots` capped at 30, so 280 was impossible by
construction. Keep a physical bound in mind for every count you read back.

## A spot-coincidence test says neighbouring voxels are different crystals

symptom: map.coincidence_falls_with_distance

**Symptom.** The fraction of one voxel's detected spots landing within N px of the next
voxel's falls off fast with raster distance — 0.70 at one step, 0.30 at two, ~0 beyond — and
the obvious reading is that the illuminated volume changed grain.

**Test.** Stop counting coincidences and **track individual reflections**. Take
the two or three brightest blobs per frame along one raster line and plot their positions.

**Cause.** A fixed-tolerance coincidence test cannot distinguish a grain boundary from a
smooth orientation *gradient*. At TPS 21A the reflections never disappeared — they swept
**56 and 46 px** along smooth monotonic paths across a 1.05 µm row, about 2.7 px per 50 nm
step, which is exactly why a 3 px test fails after one step. It is one crystal rotating.

**Lever.** For a gradient, the right validation is **spatial smoothness** against a
label-shuffled null (which preserves every orientation and destroys only their arrangement),
plus monotonic growth of misorientation with distance — not neighbour agreement. Measured:
neighbour misorientation 0.086° against a shuffled null of 0.236–0.266°, 100/100 shuffles
worse.


## Several accepted orientations on one frame, at Σ3 / Σ9 to each other

symptom: solutions.csl_related

**Symptom.** The indexer returns N solutions whose pairwise disorientations are 60.0°
(Σ3), 38.9° (Σ9) or another CSL angle, each with a respectable `NMatches`, all clearing the
scrambled-image null.

**Test.** For each solution, the set of **reflections** it explains — not
pixels, not watershed labels. Then ask how many are explained by no other accepted
solution.

```python
d, idx = cKDTree(detected_blobs).query(predicted_xy)   # one entry per REFLECTION
hits_i = set(idx[d < tol])
unique_i = hits_i - set().union(*(hits_j for j != i))
```

**Cause.** A CSL-related orientation shares a fixed fraction of the reciprocal lattice —
1/3 for Σ3 — so it re-explains that fraction of the parent's reflections for free. It is
not a random orientation, so a random-orientation null says nothing about it: **that null
gates chance, not redundancy.**

**Lever.** Require ≥3 reflections no other accepted orientation explains, across all phases
at once. Do not count in a finer unit: matched pixels of the blurred indexer image (722 px
per blob) reported these artifacts as *disjoint*, and `--min-unique 2` passed them because
it counts watershed labels (3540 regions for 50 reflections).

## An orientation map that is smooth, and wrong

symptom: map.smooth_on_fine_raster

**Symptom.** A per-voxel orientation field over a finely-stepped raster looks convincingly
smooth and beats a label-shuffle null comfortably.

**Test.** Ignore the map. Per frame, ask what fraction of the observed
reflection intensity the accepted orientation explains, and how many of the five brightest
reflections it accounts for.

**Cause.** On a fine raster, neighbouring frames are near-identical images, so any solution
driven by them varies smoothly whether or not it is right. The shuffle destroys the spatial
arrangement, so the real field wins by construction.

**Lever.** Gate per frame on explanatory power before assembling any map. Measured on an A5
Ni scan: median **7.8 %** of intensity explained, median **1 of the 5 brightest** (36 of 135
frames explained none of them) — while the map passed smoothness at 100/100 shuffles.

## A synthetic control that indexes ~30% of its frames

symptom: control.partial_yield_on_simulated_frames

**Symptom.** Frames generated by `laue-torch` (or any `laue_torch` forward render) index at
a fraction of the rate real frames do — ~30% here — with no error anywhere. Peak *detection*
looks fine (median 60 detected spots per control frame against 100 on real data, none
skipped), so the frames plainly contain signal.

**Test.** Do not test the indexer. Project the frame's **own known truth
orientation** through the pipeline's `Phase.project()` and count predicted reflections
landing within the match radius of a detected peak — then repeat on the transposed image:

```python
pr = ph.project(U_truth)
xs, ys, _ = detect_peaks(img, NPX)
hits = (cKDTree(np.c_[xs, ys]).query(pr)[0] < 8.0).sum()   # then again on img.T
```

Measured on 34-ID-E Zn, six frames: **0, 0, 1, 1, 2, 0** hits as written against **32, 17,
23, 61, 18, 59** transposed, of ~60 predicted. A truth orientation scoring at chance on its
own frame is decisive; no indexer statistic is.

**Cause.** `laue_torch` splats into `img[X, Y]`; real frames, backgrounds and the indexer's
reader are `image[row, col]`. `laue-torch` writes `/entry1/data/data`, the indexer's own
dataset name, so nothing signals the mismatch. The surviving ~30% are spurious solutions
found on a transposed pattern — and a noise floor measured from those is an axis convention
wearing the costume of a measurement.

**Lever.** Transpose before writing (`im = im.T`), and check `/entry1/axis_order` if present.
Standing rule, beyond this bug: **a synthetic control must be verified against its own ground
truth before it is used to calibrate anything.** Invariant 38.

## `GPUassert: device busy or unavailable` when several shards share a host

symptom: compute.thread_oversubscribed

**Symptom.** With two or more shards on one host, a daemon dies with
`GPUassert: CUDA-capable device(s) is/are busy or unavailable`, the remaining shards of the
plan never start (the dispatcher logs `fork: retry` / `Resource temporarily unavailable`),
and the host may then refuse new ssh logins. It reads as a GPU fault.

**Test.** Look for the thread limit before the GPU. Grep the shard's image-server and daemon
logs for `res = 11` (OpenCV `Can't spawn new thread`) and compare the user's thread count with
the limit on that host:

```bash
grep -l "res = 11" <shard>/results/*/*.log
ps -L -u "$USER" --no-headers | wc -l ; ulimit -u
nvidia-smi --query-compute-apps=pid,used_memory --format=csv    # is the card really in use?
```

If `res = 11` is present and the thread count sits at `ulimit -u`, the GPU error is a
consequence. If there is no `res = 11`, the thread count is well below the limit and
`nvidia-smi` shows another process on the card, the device really is busy and this entry
does not apply.

**Cause.** In laue-index 0.7.1 the image server sizes its preprocessing pool from the WHOLE
host (`laue_index.workers.choose_preprocess_workers`) and knows nothing about sibling shards,
and each pool worker fans out its own OpenCV thread pool. On a 112-core host with
`ulimit -u` = 8192, two shards exhausted the per-user thread limit: OpenCV `res = 11`, then
the daemon's `GPUassert` (a consequence of the thread exhaustion, not a cubin or device
fault), then the dispatcher itself could not fork. (Campaign dispatch logs; source not in repo.)

**Lever.** On 0.7.1: set `LAUE_PREPROCESS_WORKERS` per shard (about ¾ of the host's CPUs
divided by the shards on that host) and `OPENCV_NUM_THREADS=1` in each shard's environment,
and stagger launches. On 0.7.2 and later the sizing also honours `LAUE_SHARDS_PER_HOST` and
`RLIMIT_NPROC`, pool workers run with thread counts of 1, and `LAUE_PREPROCESS_WORKERS` remains
the override; `pipeline/dispatch/` sets these for a multi-host plan. Phase 3 has the
operational detail.

## A run that indexes badly after a geometry or lattice change, with no error

symptom: cache.forward_geometry_mismatch

**Symptom.** After editing `P_Array`, `R_Array`, the lattice or the reflection list, a rerun
finds fewer or different orientations than expected, or none, and every log is clean.

**Test.** Run one representative frame twice with the same parameter file: once as is, and
once with `DoFwd 1` (or the old `ForwardFile` moved aside) so the forward cache is rebuilt from
the current geometry. If the two agree, the cache is not the problem and this entry does not
apply. If the rebuilt run finds the expected solutions and the cached one does not, the cache
was built for another geometry.

**Cause.** The daemon reuses `ForwardFile` whenever it has the right SIZE for the orientation
count and `MaxNrLaueSpots` (`forwardCacheUsable`). Nothing records which geometry or lattice it
was built from, so a right-sized cache from a DIFFERENT geometry is accepted and its predicted
spots are scored against frames they do not describe. 0.7.2 fixed the other two ways a cache
went wrong (a killed run's partial file, now written as `<ForwardFile>.partial.<host>.<pid>` and never
read; a failed write, now fatal and cleaned up); this one is a known open limit.

**Lever.** Delete (or rename) `ForwardFile` whenever the geometry, lattice or reflection list
changes, and give each geometry its own cache path. `*.partial.<host>.<pid>` files from killed runs
are never read and can be deleted.
