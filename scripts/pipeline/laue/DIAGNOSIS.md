# Laue diagnosis reference

Symptom → discriminating test → cause → lever. Read by `beamreport`; each entry attaches
to a symptom the generic diagnostics detect.

**Every entry carries a test that can come back the other way.** An entry that cannot
exonerate the cause it names does not belong here.

Three entries — a working start. Provenance is the campaign notebooks, cited per entry.

---

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
