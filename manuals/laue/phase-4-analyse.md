# Phase 4 — Analyse

> Part of the **Laue doc set**. The spine — invariants, done-means and the phase
> order — is [`README.md`](README.md).

---

## Phase 4 — Analyse

```bash
env LAUE_WORK=$WORK/analysis/<scan> \
    LAUE_PHASES=alpha,beta \
    LAUE_PARAMS_ALPHA=<params_alpha.txt> LAUE_PARAMS_BETA=<params_beta.txt> \
    LAUE_SCAN_DATA=<SCAN_FOLDER> \
    LAUE_SCAN_ALPHA=<results/alpha_TS> \
    LAUE_SCAN_BETA=<results/beta_TS> \
    LAUE_OUT_PREFIX=<scan> LAUE_MOUNT_DEG=<mount angle, deg> NW=16 \
    PY=<full path to the env's python> \
    bash analysis/run_analysis_chain.sh
```

Every variable the analysis scripts read is in the table at the end of this phase.
`run_analysis_chain.sh` (0.7.2 and later):

- needs `PY` (a python that imports numpy, scipy, h5py and matplotlib; it checks) and refuses
  to start without `LAUE_WORK`, `LAUE_SCAN_DATA`, `LAUE_SCAN_ALPHA`, `LAUE_SCAN_BETA` and
  `LAUE_OUT_PREFIX`. It runs both `alpha` and `beta` steps, so as written it is a two-phase
  chain; for a single phase run the scripts one by one.
- **stops at the first failing step** (`STEP FAILED (exit N)`), instead of carrying on to
  "analysis chain finished" as 0.7.1 did after an import crash.
- after `null_model.py`, reads the measured null from `peel_map/<prefix>_null.json` (prefix =
  `LAUE_OUT_PREFIX`; the scripts' own default is `scan`) and **exports
  `LAUE_NULLMAX_<PHASE>`** for each phase in `LAUE_PHASES`, for the statistic in force
  (`LAUE_GATE_STAT`). A value already in your environment is replaced by the measured one, and
  the log says so.
- step 1 (`parentbeta_validate.py`) **exits without writing** when nothing validates, rather
  than printing "VALIDATED 0" and writing an empty npz for the next step to read.

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

Then, always: `regrain.py` (contiguity-aware counts; it reads this scan's measured null from
`peel_map/<prefix>_null.json`, `LAUE_NULLMAX_<PHASE>` only overriding it, and **exits** rather than
falling back to the Ti values),
`tolerance_sensitivity.py`, and `collect_scan_metrics.py` for the cross-scan JSON.

**Every gate needs a null measured on this scan, and says which count it gates.** In 0.7.1,
`empirical_gate.py` and `validated_figures.py` still carry a hard-coded null from an earlier Ti
campaign and do not read `LAUE_NULLMAX_<PHASE>`: on 0.7.1, do not use their gate for any other
scan. From 0.7.2 `regrain.py`, `empirical_gate.py` and `validated_figures.py` read the measured
null from `peel_map/<prefix>_null.json` (written by `null_model.py` on this scan) and exit if
neither that file nor the override below exists; there is no built-in fallback. `LAUE_NULLMAX_<PHASE>` is an **override** and must
be the maximum of the statistic in force: a value equal to the OTHER statistic's maximum in the
JSON is refused. Symmetry comes from the phase's **space group**, never its name.

**Which count.** The validated npz carries `nhit` and `nhit_distinct` side by side (glossary:
`INVARIANTS.md` invariant 15b). The gates default to **`nhit`**; `LAUE_GATE_STAT=nhit_distinct`
switches, also selects the matching null, and every gate prints which it used.
`nhit_distinct` is available but **not yet validated as a gate**: whether the extra grains it
admits are real is untested (raw-image hit test on the newly admitted grains pending).
Per-frame analytic Poisson gates stay on `nhit`, since the distinct count has no closed-form
null, and say so at run time.

**Winner per position.** Figures that keep one orientation per raster position rank by
`nhit_distinct` (`raster.winner_per_position`), because `nhit` favours harmonic-rich low-index
orientations; on an npz written before `nhit_distinct` existed they fall back to `nhit` with a
warning. Ties are broken by `nhit`, then orientation, then instance index, so the pick no
longer depends on iteration order.

**Connectivity.** Grain and component counts use one shared connectivity,
`pipeline/analysis/raster.py` (`LAUE_CONNECTIVITY`, default **8**). Before 0.7.2 the scripts
disagreed; any quoted `variant_coherence` z or `big_grain_*` component count was computed with
**4**-connectivity, and `LAUE_CONNECTIVITY=4` reproduces it.

**Texture.** `texture_null.py` now builds directions from the direct lattice by space group. Any
earlier "a-axis" texture it reported was a ⟨10-10⟩ pole density (both "a-axis" rows measured
that family); c-axis rows are unaffected.

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
   defensible count by two orders of magnitude. (Nine-scan figures: source not in repo.)
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
   admits instances down to nhit 5 while the measured null reaches 10–11 (sampleH's full-raster
   `nhit` null is max 10 in 60,000 draws, INVARIANTS.md 15b; the "max 9" in the README worked example
   is sampleG's), so the analytic-validated
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

### Environment variables of the analysis scripts

**The one table.** Read from `os.environ` in `pipeline/analysis/*.py`, `raster.py`,
`frame_peaks.py` and `laue_material.py` (0.7.2). "Required" means the script exits naming the
variable when it is unset; nothing below silently falls back to another campaign's data.

| variable | meaning | required / default | read by |
|---|---|---|---|
| `LAUE_WORK` | work root; `peel_map/`, `figures/`, `params/` under it | required | nearly every script; the chain |
| `LAUE_PHASES` | comma-separated phases the null is measured and gated for | default `alpha,beta` (`null_model.py`, `empirical_gate.py`, the chain) | `laue_material.py`, `null_model.py`, `empirical_gate.py`, chain |
| `LAUE_PARAMS_<PHASE>` | the indexing parameter file for that phase: lattice, hkl list, geometry, energy window, space group | required for any script that loads a `Phase` (or `LAUE_PARAMS`) | `laue_material.py` (every `Phase.load`) |
| `LAUE_PARAMS` | generic parameter file | accepted **only when one phase is in use**; two phase names resolving to one file are refused (0.7.2) | `laue_material.py` |
| `LAUE_PHASE` | the phase for single-phase tools | default `alpha` (`batch_peel_driver`, `map_validate_cluster`, `grain_extent_backfill`) or `zn` (`cluster_orientations`, `ipf_map`, `spot_energy`, `texture_null`) | those scripts |
| `LAUE_SCAN_DATA` | folder of this scan's raw frames | required | validate, null, census, exclusion null, `beta_map_validate`, `map_validate_cluster`, `batch_peel_driver`, `grain_extent_backfill`, `exposure_signal_check`, `fullped`, `parentbeta_backfill`, `scan_map`, chain |
| `LAUE_SCAN_<PHASE>` | indexing-run directory for that phase (holds `frame_mapping.json`) | required by `null_model.py`, `scan_map.py`, `parentbeta_validate.py` (scan `env`); `parentbeta_backfill.py` defaults to `$LAUE_WORK/results/parentbeta_<phase>`; the chain requires `_ALPHA` and `_BETA` | those |
| `LAUE_SCAN_BETA` | the beta run directory, as above | required | `beta_map_validate.py`, chain |
| `LAUE_REF_DATA` | raw frames of the longer-exposure reference scan | required | `exposure_signal_check.py` |
| `LAUE_OUT_PREFIX` | basename prefix of every output, incl. `peel_map/<prefix>_null.json` | default `scan` (`frame_peaks.out_prefix`); the chain requires it set | `frame_peaks.py`, `null_model.py`, `empirical_gate.py`, `regrain.py`, `ipf_map.py`, `scan_map.py`, chain |
| `LAUE_GATE_STAT` | `nhit` or `nhit_distinct`; selects the matching null too | default `nhit` | every gate, `null_model.py`, `parentbeta_validate.py`, `collect_scan_metrics.py`, chain |
| `LAUE_NULLMAX_<PHASE>` | override of the measured null max; must be the max of the statistic in force | optional (the chain exports it from the JSON) | `frame_peaks.load_null` via `regrain`, `empirical_gate`, `validated_figures` (`collect_scan_metrics` deliberately ignores it: per-scan nulls only) |
| `LAUE_CONNECTIVITY` | 4 or 8 for grain / component labelling | default `8`; `4` reproduces pre-0.7.2 numbers | `raster.py` via `regrain`, `variant_coherence`, `big_grain_*`, `collect_scan_metrics`, `substrate_deposit` |
| `LAUE_NR` | frames per raster row = number of COLUMNS (fast axis) | required wherever positions come from frame numbers | `raster.py` via `ipf_map`, `optical_overlay`, `reg_refine`, `render_registered`, `separate_layers`, `substrate_deposit`, `zn_report_figures`, `hardening_fullmap`, `drift_control`, `fullped`, `within_grain` |
| `LAUE_NROWS` | number of raster rows (slow axis) | required, as `LAUE_NR` | as `LAUE_NR` |
| `LAUE_STEP_UM` | raster step, µm | required where used | `drift_control`, `ipf_map`, `optical_overlay`, `reg_refine`, `render_registered`, `zn_report_figures` |
| `LAUE_MOUNT_DEG` | sample mount angle, degrees, in [0, 90) | required where used, **including by the chain** (`variant_coherence`, `validated_figures`) | also `catalog_figures`, `big_grain_*`, `collect_scan_metrics` |
| `LAUE_OPTICAL_CX`, `_CY` | scan centre in optical-image pixels | required where used | `optical_overlay`, `reg_refine` |
| `LAUE_OPTICAL_PX_PER_UM` | optical image scale | required where used | `optical_overlay`, `reg_refine` |
| `LAUE_OPTICAL_FLIP_Y` | `+1` optical image vertically flipped vs the scan, `-1` not | required where used | `optical_overlay`, `reg_refine`, `render_registered` |
| `LAUE_SCAN_LABEL` | label for the background maps | optional | `zn_report_figures.py` |
| `LAUE_SHARD_GLOB` | glob matching the indexer's shard result directories | required | `separate_layers.py` |
| `LAUE_SHARD_PROV_MATCH` | keep only shards whose `provenance.json` contains this string | optional | `separate_layers.py` |
| `LAUE_FRAME_PREFIX` | only frames whose name starts with this | optional | `fullped.py` |
| `LAUE_TESTSCANS` | root of the legacy named test scans | required only for those legacy scan keys | `parentbeta_validate.py`, `beta_alpha_exclusion_census.py` |
| `LAUE_LM` | a LaueMatching checkout | default: the checkout holding the script | `batch_peel_driver.py` |
| `LAUE_H5LOC` | image dataset inside each frame | default `/entry1/data/data` | `batch_peel_driver.py`, `map_validate_cluster.py` |
| `LAUE_IN_NPZ` | explicit input npz | optional | `ipf_map.py` |
| `LAUE_SKIP_CLUSTER` | `1` = stop after the validated npz | optional | `parentbeta_validate.py` |
| `PY`, `SCRIPTDIR`, `NW` | python; the analysis scripts; worker count | `PY` default `python` (set a full path); `SCRIPTDIR` default the chain's own directory; `NW` default 16 | `run_analysis_chain.sh` |

---

