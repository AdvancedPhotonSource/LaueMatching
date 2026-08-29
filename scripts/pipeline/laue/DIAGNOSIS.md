# Laue diagnosis reference

Symptom → discriminating test → cause → lever. Read by `beamreport`; each entry attaches
to a symptom the generic diagnostics detect.

**Every entry carries a test that can come back the other way.** An entry that cannot
exonerate the cause it names does not belong here.

Four entries — a working start. Provenance is the campaign notebooks, cited per entry.

---

## An indexing run that finishes cleanly and finds nothing

symptom: run.completed_zero_grains

**Test.** Run the **CPU** binary on the same frame with the same parameter file. If the
CPU finds grains where the GPU found none, the data is not the problem and no amount of
threshold tuning will help. Then ask the binary and the card whether they agree:

```bash
cuobjdump --list-elf $(python -c "from laue_index import indexer; print(indexer.binary_path('GPU'))") | grep -oE 'sm_[0-9]+'
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
cuobjdump -all --list-ptx <binary> | head -3      # -all is REQUIRED or it reports none
```

A card whose compute capability has no matching cubin, and no PTX at or below it, cannot
run the kernel. The entry exonerates itself: if the architectures cover the card, this is
not your problem.

**Cause.** `cudaErrorNoKernelImageForDevice` is reported by the kernel **launch**, and only
the following `cudaDeviceSynchronize` was checked — so the kernel never ran, nothing
complained, and zero matches is a perfectly ordinary outcome for a frame with no grains.
Measured: an sm_120-only build on an sm_90 card printed `Initial solutions: 0 Unique
Orientations: 0` and **exited 0**. PTX JIT rescues the older→newer direction and never the
reverse, so a binary built on a newer GPU than it runs on fails this way — and
`/home/beams` being shared, binaries travel between hosts constantly.

**Lever.** Fixed in `3b2abcb`: the launch result is now checked, so this fails loudly
(`GPUassert: named symbol not found`, exit 244) instead of returning an empty map. Do not
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
clipped peak even when intensity does not. Lab Notebook, invariant 13.

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
re-flips before it was learned. Lab Notebook, invariants 10–11.

## A threshold-defined split, contrasted on the quantity that defined it

symptom: split.bimodal

**Test.** Ask what defined the two groups. If the split was made by a cutoff on X, then
"A differs from B in X" is guaranteed and means nothing. The test that settles it is an
**independently measured** signature: does a quantity not used in the split also separate
the groups? If it does not, the split is an artefact of the cutoff.

**Cause.** A cutoff applied to form the groups, then used as evidence about them.

**Lever.** Contrast on an independent signature, and store the threshold's statistic beside
any threshold-defined split so no reader mistakes the split for a result. Lab Notebook,
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
SAXS says sharp contrast, not which kind. Lab Notebook 16BMD §5, invariants 24 and 28.

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
Lab Notebook 16BMD §3–4, invariants 25 and 26.
