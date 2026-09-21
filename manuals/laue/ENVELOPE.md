# Laue microdiffraction — measurement envelope

**Instruments:** polychromatic (pink/white-beam) Laue microdiffraction at 34-ID-E (reflection,
Perkin Elmer panel), TPS 21A (reflection, PILATUS3 6M) and 16-BM-D (transmission, Pilatus3 X
CdTe 1M). Rows written for one station say so.
**Last checked:** 2026-09-21 for the header, §1 detector distance, §2 crystal-fit tolerances,
§3 strain and §4 orientation / strain / depth rows; the other rows were last checked
2026-08-12 and predate the TPS 21A campaign, 0.7.x and invariants 33–38.
**Owner:** Hemant Sharma (hsharma@anl.gov)

> Part of the **Laue doc set**. Spine: [`README.md`](README.md). Contract: `~/opt/beamreport/DOCS_SPEC.md` §6 (separate repo, not under `$MIDAS`).

What this measurement can and cannot determine, and which of those is changeable. Read it
before promising an answer, and before suggesting a different measurement.

> **Not the scope gate.** Scope says whether this doc set applies to your data. This says
> whether the measurement can answer the question at all. A scan can be squarely in scope
> and still unable to support what is being asked of it.

Most rows below are restatements of content already in
[`phase-1-science.md`](phase-1-science.md); this file exists so they are in one place, sorted
by whether anything can be done about them.

---

## 1. Fixed — cannot change this cycle

No suggestions here. State the consequence and the substitute.

| Property | Value | Provenance | What it makes unobtainable | Substitute |
|---|---|---|---|---|
| Beam / detector angle | 90°, panel **edge-on** to the beam; detector normal is lab **+Y** | station geometry, `phase-1-science.md` §3b | Any declination quoted "from Z" is declination from the **beam**, an instrument direction with no sample meaning. c-axis-along-Z does **not** mean c-axis-along-growth. | The specimen surface normal, from **measured stage motion**: both raster axes lie in the surface, so their cross product is the normal. Never a convention. |
| Detector distance | 34-ID-E **513 mm**; TPS 21A **~522 mm**; 16-BM-D **223.381 mm** (30.038° tilt) | 34-ID-E: station geometry, `phase-1-science.md` §3b; TPS 21A: `LAB_NOTEBOOK_TPS21A.md` §1 and ledger (`P_Array` z 0.522115 / 0.521652 m); 16-BM-D: `LAB_NOTEBOOK_16BMD_Si.md` ledger (`poly.poni`) | — | — |
| Incident beam axis | lab **Z**; `Phase.project` computes `kf = ki - 2*qh[:,2]*qh`, valid only for `ki = (0,0,1)` | `phase-1-science.md` §3b (confirm with `ph.ki`) | A geometry with a different `ki` is not describable by this forward model. | none — stop and ask |
| Panel orientation check | — | — | The usual validation (rotate a crystal about the beam, check the pattern rotates rigidly) **needs a detector perpendicular to the beam**. Edge-on, all three axes fail and prove nothing. | Validate instead by forward-model prediction of observed peaks against a random-orientation null. |
| **Rotation about the beam axis** | **an exact, unmeasured gauge freedom** | measured: at φ = 90° the largest change in any predicted pixel is **2.3e-13 px** and in any energy **0**; the CeO₂ rings move 1.9e-14° in 2θ while χ sweeps 30° | **Absolute orientation in the laboratory frame.** Every orientation this chain produces is correct only up to an unknown rotation about the beam. No cross-check *internal* to the diffraction can fix it — not a rotation series, not agreement with a second code on the same calibration. | Metrology from outside the pattern: a **surveyed rotation-axis direction** (cheapest, and nearly fully informative when the axis is ⊥ beam), a **surveyed detector translation**, a knife-edge on a surveyed lab axis, or a plumb/fiducial reading of the panel column direction. **Sample translation will not work** — the source is the beam–sample intersection and the beam is lab-fixed. |
| **Detector handedness** (readout row direction vs the panel's own axis) | **not recoverable from the pattern** | measured at TPS 21A: with the tilts zeroed a row-mirror about the PONI is exact to **6.7e-16** in q̂; at the real −0.388°/+0.308° tilts it breaks only linearly, and on real data both parities index **identically** — 46 reflections each, **0.42 vs 0.43 px** median | The **sense** of any rotation axis, and absolute orientation. Symmetry-reduced misorientation *angles* survive (the mirror conjugates them by point-group operations); axes do not. | Metrology, exactly as for the beam azimuth: the station's own detector drawing, a χ or azimuth sign convention, or a surveyed panel direction. A `tth` annotation constrains the COLUMN direction only — the parity is the ROW direction. |

**Consequence worth stating on any report:** every angle this pipeline produces is relative
to an instrument frame until the surface normal is supplied. On sampleH that distinction turned
a meaningless "69.7° from Z" into "**c-axis avoids the growth direction by 8×**".

**A near-gauge is more dangerous than a gauge.** The beam azimuth (row 5) is exactly
unmeasurable and everyone treats it that way. Handedness is *nearly* unmeasurable — a
synthetic test with the orientation held to the 24 symmetry images makes it look decidable at
~34 px — and a free 100M-orientation search absorbs almost all of it. Measure a claimed
discriminator against the search the pipeline actually runs, not against a constrained one.

**And the instrument frame itself is only fixed up to a rotation about the beam** (row 5).
Relative quantities — misorientation, texture, grain boundaries, a rotation series — are
unaffected, because the gauge cancels. Anything absolute is not. Say which kind you are
quoting.

> **Transmission geometry (16-BM-D).** Rows 1–3 above are written for the 34-ID-E edge-on
> panel. In transmission the panel sits downstream and the pattern is radial about the beam,
> so: the direct beam lands **on or just off the panel** and is **not** the point of normal
> incidence — at 30° tilt they were 751 px apart (invariant 24); the largest 2θ is at a
> **corner**, not the top edge (invariant 23); and the low-energy cutoff may be a **measured**
> detector discriminator setting rather than a guess (`Threshold_setting` in the Pilatus
> header). Row 5 applies identically to both geometries.

## 2. Configured — set per run, changeable next time

The only tier where "what could be observed differently" has an answer.

| Parameter | Used | Achievable range | Limited by | What changing it would buy |
|---|---|---|---|---|
| **Depth resolution optic** | wire / coded aperture **present or absent** | either | what was on the table that week | **Present:** each frame is one depth. **Absent:** the whole illuminated column superimposes — hundreds of grains per frame and *no per-grain depth at all*. This changes what may be claimed, not how hard it is. |
| Raster step and extent | per run, see the scan folder | stage-limited | stage travel | Spatial sampling, and the smallest feature that can be separated from its neighbour |
| Energy window | per run (keV) | source + optics | undulator and monochromator configuration | Which reflections are accessible, and the per-spot energy assignment |
| Exposure per frame | per run | detector-limited | detector frame rate and readout | Counting statistics per spot, which sets how weak a grain can be and still index |
| Detector geometry refinement | `geoN_*.xml` from **this** run | — | must be re-derived per run | Nothing — see the hard rule. Geometry from another run is the single fastest way to get a confident, wrong answer. |
| Crystal-fit tolerances `tol_LatC`, `tol_c_over_a` | **FRACTIONS** (0.01 = ±1 %); 0 disables the fit | 0 ≤ tol < 1 | the C fit forms bounds as `value·(1 ∓ tol)`. The 0.7.1 usage text says "in %", which is wrong: 1.0 meant as 1 % is read as ±100 %. From 0.7.2 the binaries reject tol ≥ 1 or NaN, warn above 0.1, and with `tol_c_over_a` non-zero ignore `tol_LatC` with a NOTE (`validateCrystalFitTolerances`, `LaueMatchingHeaders.h`) | A wider or narrower lattice bound. It does **not** buy strain resolution below the integer-pixel floor (§4). |

**Rows deliberately blank.** Detector maximum frame rate, stage travel limits, and
the dose at which a given sample starts to damage are not recorded anywhere in this doc set
and are not in the parameter files either. Until they are filled in, a report **will not**
propose changing exposure, step size, or total dwell — an undeclared bound produces no
counterfactual, by design.

## 2b. In scope for the station, out of reach for the PHASE

A station being covered does not mean every material at it can be indexed. Whether a phase
is reachable is set by how many of its reflections land on the panel inside the recorded
energy window, and a small unit cell can put all the strong ones below the detector's
discriminator. Measured at TPS 21A, same geometry and same frames: Si gets 47
reflections per orientation and Ti α gets 52, while **Ni gets 18.6, all weak high-index**,
because (111) through (400) sit below the 8.74 keV threshold at every 2θ the panel covers.
Si and Ti indexed; **Ni did not** — its accepted solutions explained a median 7.8 % of each
frame's reflection intensity and were withdrawn.

**Count this before promising a phase**, not after: a few hundred random orientations
projected through the geometry, histogrammed by energy, takes a minute. If the count inside
the window is near `MinNrSpots`, say so up front.

## 3. Intrinsic — the sample or the physics forbids it

No configuration helps.

| Question | Why it is not answerable | Distinguish from |
|---|---|---|
| Which side of a **same-phase interface** does this grain belong to? (Zn on Zn, weld and parent, homoepitaxial deposit) | Nothing crystallographic separates them. Both sides produce the same reflection set. | Laue-footprint fragmentation, flat-background scattering and per-spot energy are still available and *do* carry information — see the same-phase section of `phase-1-science.md`. Absence of a crystallographic separator is not absence of all evidence. |
| Parent-phase reconstruction in a **single-phase** system | There is no parent to reconstruct. | Twin relationships (Σ3 for FCC) and texture remain available. Do not run the parent machinery "to see". |
| Parent reconstruction across **two unrelated phases** (matrix + precipitate) | No orientation relationship connects them. | Phase fraction and exclusion census are the applicable analyses. |
| Per-grain depth without a depth-resolving optic | §2 row 1 — the column superimposes. | This is *configured*, not intrinsic, for any future run. It is intrinsic only for data already taken. |
| **Hydrostatic strain, absolute lattice parameter, cell volume** from white-beam Laue | Spot **positions** fix only the directions of the scattering vectors, so a uniform scaling of the cell moves nothing. Laue determines 8 of the 9 components of the orientation-times-lattice matrix (3 orientation + 5 **deviatoric** strain); the scale needs **spot energy**. | The deviatoric shape (e.g. c/a) IS observable in principle; see §4 for what this pipeline actually resolves. Scale needs an energy measurement (a monochromatic / energy scan of chosen reflections, or an energy-resolving detector). |

## 4. Derived limits

What follows arithmetically from §1–2. A report may quote these directly.

| Quantity | Limit | From |
|---|---|---|
| Smallest separable feature | ≈ raster step | §2 raster step; two positions closer than one step are not distinguishable |
| Angular quantities relative to the sample | **undefined** until the surface normal is supplied | §1 row 1 |
| Number of grains per frame, no depth optic | the whole illuminated column | §2 row 1 |
| **Orientation resolution of the C fit** | **~0.004°**; nothing finer is resolvable by it | The objective (`calcOverlap`) samples observed intensity at the **integer** pixel of each predicted reflection, so it is a staircase in orientation. Measured on 198 synthetic seeds scored against a known truth: Nelder-Mead median **0.0041°** (p95 0.0091°). Source: the comment above the Nelder-Mead call in `FitOrientation`, `LaueMatchingHeaders.h`. |
| **Deviatoric strain (c/a) from the C fit** (`tol_c_over_a`) | **not resolvable at elastic magnitudes.** On sampleH geometry the fit recovered **0.237** (bootstrap 95 % CI [0.17, 0.39]) of an injected c/a change, against a noise floor sd **2.07e-4**; a c/a spread counts as detected only above 3 × 2.07e-4 = 6.2e-4, i.e. a true strain of ~2.6e-3. The preregistered strain test was **REFUTED**. | The binding constraint is **integer pixel positions**, not physics: `spots.txt` X, Y are whole pixels and **1 px = 5.7e-4 in c/a** at this geometry, and the orientation absorbs what c/a cannot resolve. A sub-pixel objective or spot energies would change this. Provenance: synthetic-control addendum of the campaign preregistration and its control analysis (`$ANALYSIS/<sampleH re-run>/PREREGISTER.md`, `analyse_control.py` → `analysis_out/control_ca.json`; campaign-local, not in this repo). |
| **Depth projection, 45° mount** (reflection geometry, no depth optic) | a scatterer at depth *d* along the beam appears displaced **0.70711·d along the SLOW raster axis** and **0 along the fast axis**: the probe's footprint is a line, and grain extents along slow are inflated | Geometry: the slow axis lies in the surface plane containing the beam, at 45° to it (surface normal `(0, −0.7071, 0.7071)`, `phase-1-science.md` §3b). On sampleH grains are **1.62× longer along slow** (12.0 vs 8.0 µm); **UNTESTED** whether that is this projection or columnar growth: the discriminating test (does the slow-axis excess scale with mean spot energy?) has not been run. Campaign checkpoint, not in this repo. |

## 5. Did not versus cannot

Things skipped on a given run that are perfectly possible, and read identically to hard
limits in a parameter file.

- **No depth optic on a run** is a choice, not a limit of the instrument. Report it as
  "not measured", never as "not measurable".
- **Phase not identified** is testable, not unknowable: index with a candidate phase and
  compare the validated fraction against the measured null. That is an experiment, not a
  lookup, but it is available.
- **Single scan rather than a series.** Cross-scan comparison needs multiple scans of the
  same specimen; a set of unrelated test scans does not support it. This is a scheduling
  choice.

---

**Checklist before this file is trusted**

- [x] Every row has a unit or is explicitly dimensionless
- [ ] Every bound in §2 names what imposes it — **three rows are still blank** (frame rate, stage travel, damage dose)
- [x] Nothing in §1 or §3 is phrased as a suggestion
- [ ] `Last checked` is within the current run cycle -- **only for the rows named in the header**; the rest predate TPS 21A and 0.7.x

**Open.** The three blank bounds in §2 are the highest-value thing to fill in here. They are
the difference between a report that can say "a shorter exposure would reach the fast
process" and one that has to stay silent about exposure entirely.
