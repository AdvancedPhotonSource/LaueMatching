# Phase 4 — Analyse

> Part of the **Laue doc set**. The spine — invariants, done-means and the phase
> order — is [`README.md`](README.md).

---

## Phase 4 — Analyse

```bash
env LAUE_WORK=$WORK/analysis/<scan> \
    LAUE_SCAN_DATA=<SCAN_FOLDER> \
    LAUE_SCAN_ALPHA=<results/alpha_TS> \
    LAUE_SCAN_BETA=<results/beta_TS> \
    LAUE_OUT_PREFIX=<scan> NW=16 \
    bash analysis/run_analysis_chain.sh
```

Steps 1–5 are the material-agnostic core; 6–8 run **only** if phase 1 said an orientation
relationship applies; 9 is figures.

| # | script | applies to |
|---|---|---|
| 1 | `parentbeta_validate.py {phase} {nw} env` | any (per-frame spot test) |
| 2 | `null_model.py` | **any — never skip, never inherit** |
| 3 | `empirical_gate.py` | any |
| 4 | `beta_alpha_exclusion_census.py` | two-phase only |
| 5 | `exclusion_null.py` | two-phase only |
| 6 | `parentbeta_reconstruct.py {minsize} {prefix}` | OR-related phase pair only |
| 7 | `anchor_null.py` | only if step 6 ran |
| 8 | `variant_coherence.py` | only if step 6 ran |
| 9 | `validated_figures.py` | any |

Then, always: `regrain.py` (contiguity-aware counts — pass the scan's own measured null maximum via
`LAUE_NULLMAX_<PHASE>`; it now **exits** rather than falling back to the Ti values),
`tolerance_sensitivity.py`, and `collect_scan_metrics.py` for the cross-scan JSON.

**Clustering does not scale past a test scan.** The greedy loop inside `parentbeta_validate.py` is
O(n_clusters x n_instances x n_sym): fine for the ~1e3–1e4 instances a test scan produces, but a
full 201x201 raster gives ~2e5 and it never finishes. For a full raster:

```bash
LAUE_SKIP_CLUSTER=1 python parentbeta_validate.py <phase> <nw> env    # stops after the npz
python cluster_orientations.py <validated.npz> <clustered.npz> 1.0 <phase>
```

`cluster_orientations.py` is KD-tree based (quaternions; a misorientation cut theta becomes a radius
`sqrt(2-2cos(theta/2))`), and clusters are **connected components**, which — unlike greedy
assignment — do not depend on iteration order. 1,076 instances in 0.22 s.

**But connected components CHAIN, and the price is real.** "Within tol of *some* neighbour" links
A–B and B–C at 0.9° each even when A–C are 1.8° apart, so a "1.0° cluster" can span several degrees.
On sampleH a 924-position cluster had **2.65° median internal spread (6.17° max)** and failed the
raw-image test at chance (peak at the predicted position in 3 of 12 cells, 1.1 expected) while each
of its positions indexed perfectly on *its own* orientation (17–73% vs a 0–7% random null). Per-instance
indexing succeeding while one shared orientation fails is exactly what over-merging looks like.

Use `--diameter` for a **diameter criterion** — complete linkage, so every member is within tol of
every *other* member, not merely of some neighbour. It removes chaining by construction, is
deterministic, and on sampleH **halved the tolerance sensitivity of the grain count, 4.28× → 2.02×**,
while cutting the largest "grain" from 1,833 positions to 1,298 (the 1,833 was the chain). Two
warnings: `regrain.py`'s tolerance sweep silently *changes algorithm* at the 1.0° boundary (supplied
labels at ≥1.0, a greedy **leader** loop below it), so its sweep varies definition as well as cut;
and any residual spread after de-chaining is physics — sampleH's grain population spans 0.08°–2.7°
internal spread, so no single cut is "correct" and every count must carry its tolerance.

> **Trap, and it cost real time: the symmetry operator multiplies on the RIGHT.** The pipeline's
> misorientation is `min_S angle(A^T B S)`. The left-handed form `A^T S B` is a *different*
> quantity — verified numerically, they differ by up to **78.7 deg** — and using it silently split
> 120 of 392 real grains while every synthetic test passed. Gate any new orientation code against
> `laue_material.misorientation` / `midas_stress` directly, never against your own reimplementation:
> a test written from the same misunderstanding as the code agrees with it perfectly.

Merging shards: each shard's orchestrator numbers its images `1..N` **independently**, so image
numbers collide across shards. Merge on the stored `frames` field (the source `.h5` filename),
which is unique map-wide.

**Three lessons that cost real time, and generalize to any material:**

1. **The analytic Poisson `p<1e-4` gate under-rejects on clustered peak fields.** Measured nulls
   reached 16 hits where Poisson forbids it. Across nine scans the measured α null maximum ranged
   14–17 and β 11–16 — one inherited value misstates the rest. On one dataset this changed the
   defensible count by two orders of magnitude.
2. **A grain is a *contiguous* region of consistent orientation.** Orientation-only clustering
   merges disjoint regions. On one scan, splitting into connected components moved α from 325 to
   614 and β from 40 to 27 — the two phases moving in *opposite* directions, which is what shows it
   is a definitional fix and not a tuning knob.
3. **"Corroboration" must beat chance.** With 2,537 candidate clusters, a random orientation lands
   within 1.74° of one 9% of the time. The same statistic was genuinely strong at 767 clusters.
   Measure it before quoting it.
4. **Tune the threshold on VALIDATED orientations, not raw ones.** A looser threshold always yields
   more raw "unique orientations" and they are overwhelmingly noise. On Zn: 99.5 gave 3,023 raw
   orientations/frame at 0.4 img/s (the flood), 99.8 gave 7 and 99.9 gave 4 — but after the
   per-frame Poisson test, 99.8 gave **2.07x more validated** orientations than 99.9 *and* lost no
   frames, while 99.9 dropped 9 of 201 frames below `MinNrSpots` entirely.
   **And re-gate against the MEASURED null before you compare — the analytic gate is not enough.**
   On sampleH the analytic-Poisson counts made 99.5 look 1.34× better than 99.8 (794 vs 592), but after
   re-gating at the measured null max the two were **equal** (495 vs 486) for 2.3× the compute, and
   99.5's purity was far worse (62% of its "validated" instances survived, against 82%). The gate
   admits instances down to nhit 5 while the measured null reaches 10–11, so the analytic-validated
   comparison is dominated by exactly the marginal instances the null rejects. Compare after
   `empirical_gate.py`, never before.
5. **A texture null must be indexability-matched.** Detector coverage, the energy window and the
   reflection list all make some orientations easier to index than others, so a peaked pole figure
   can be an artefact of what is *indexable*. Compare against random orientations passed through the
   same "at least MinNrSpots reflections on the detector" filter, not a flat sphere — and use one
   representative per grain, since one grain contributes many positions and instances are not
   independent samples.
6. **Max MRD is binning-dependent — never quote it raw.** The peak of a pole/IPF density grows with
   the number of cells (fewer grains per cell → higher peak from sampling): on Zn/Zn the substrate
   c-axis "max MRD" ran 1.9→10.7 as cells went 128→4608. Only the peak measured against a null binned
   *identically* is meaningful, and a real texture must clear that null at *every* binning — if the
   p-value flickers around chance as you rebin, there is no texture (just sampling noise).
   **Identical binning is not sufficient — the null must also be SUBSAMPLED to the measured n.**
   Max MRD rises as sample size falls, so 1,132 measured orientations against a 20,000-orientation
   null on the same grid is meaningless even at matched cells: it produced a spurious "126.9 MRD vs
   4.3" on sampleH, where almost every cell held 0.12 poles and one cell catching a handful read >100.
   Coarsen the bins to suit n, smooth, subsample the null to the same count (median over draws), and
   quote the **ratio**: sampleH's basal texture is 8.34/1.98 = 4.2, which independently matches
   `texture_null.py`'s 29.51/6.11 = 4.8. Agreeing ratios from two implementations is the check;
   the absolute MRD is not comparable across methods.
7. **Texture needs enough INDEPENDENT grains — and a 1-µm map of coarse grains has very few.** A dense
   step oversamples each grain many times; the independent-orientation count is set by *area / grain
   area*, not by the number of points. Zn/Zn: a 200×200 µm map at 1 µm held only ~350 independent
   substrate grains (8-µm grains oversampled ~8×) — far short of the ~5,000–10,000 for an ODF, so a
   flat texture result there is undersampling, not absence. For a texture survey, set the **step ≈ the
   grain size** (each point a fresh grain) and spend the points on *area*: the same grid at 10 µm
   instead of 1 µm covers 100× the area and ~20,000 grains at the same beamtime. Max step ≈ 2–3× the
   grain size before you skip the fine tail and bias toward the coarsest grains.

---
