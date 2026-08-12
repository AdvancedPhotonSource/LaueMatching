# Laue microdiffraction — measurement envelope

**Instrument:** 34-ID-E, polychromatic (pink-beam) Laue microdiffraction
**Last checked:** 2026-08-12 · **Owner:** Hemant Sharma (hsharma@anl.gov)

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
| Detector distance | 513 mm | station geometry | — | — |
| Incident beam axis | lab **Z**; `Phase.project` computes `kf = ki - 2*qh[:,2]*qh`, valid only for `ki = (0,0,1)` | `phase-1-science.md` §3b (confirm with `ph.ki`) | A geometry with a different `ki` is not describable by this forward model. | none — stop and ask |
| Panel orientation check | — | — | The usual validation (rotate a crystal about the beam, check the pattern rotates rigidly) **needs a detector perpendicular to the beam**. Edge-on, all three axes fail and prove nothing. | Validate instead by forward-model prediction of observed peaks against a random-orientation null. |

**Consequence worth stating on any report:** every angle this pipeline produces is relative
to an instrument frame until the surface normal is supplied. On sampleH that distinction turned
a meaningless "69.7° from Z" into "**c-axis avoids the growth direction by 8×**".

## 2. Configured — set per run, changeable next time

The only tier where "what could be observed differently" has an answer.

| Parameter | Used | Achievable range | Limited by | What changing it would buy |
|---|---|---|---|---|
| **Depth resolution optic** | wire / coded aperture **present or absent** | either | what was on the table that week | **Present:** each frame is one depth. **Absent:** the whole illuminated column superimposes — hundreds of grains per frame and *no per-grain depth at all*. This changes what may be claimed, not how hard it is. |
| Raster step and extent | per run, see the scan folder | stage-limited | stage travel | Spatial sampling, and the smallest feature that can be separated from its neighbour |
| Energy window | per run (keV) | source + optics | undulator and monochromator configuration | Which reflections are accessible, and the per-spot energy assignment |
| Exposure per frame | per run | detector-limited | detector frame rate and readout | Counting statistics per spot, which sets how weak a grain can be and still index |
| Detector geometry refinement | `geoN_*.xml` from **this** run | — | must be re-derived per run | Nothing — see the hard rule. Geometry from another run is the single fastest way to get a confident, wrong answer. |

**Rows deliberately blank.** Detector maximum frame rate, stage travel limits, and
the dose at which a given sample starts to damage are not recorded anywhere in this doc set
and are not in the parameter files either. Until they are filled in, a report **will not**
propose changing exposure, step size, or total dwell — an undeclared bound produces no
counterfactual, by design.

## 3. Intrinsic — the sample or the physics forbids it

No configuration helps.

| Question | Why it is not answerable | Distinguish from |
|---|---|---|
| Which side of a **same-phase interface** does this grain belong to? (Zn on Zn, weld and parent, homoepitaxial deposit) | Nothing crystallographic separates them. Both sides produce the same reflection set. | Laue-footprint fragmentation, flat-background scattering and per-spot energy are still available and *do* carry information — see the same-phase section of `phase-1-science.md`. Absence of a crystallographic separator is not absence of all evidence. |
| Parent-phase reconstruction in a **single-phase** system | There is no parent to reconstruct. | Twin relationships (Σ3 for FCC) and texture remain available. Do not run the parent machinery "to see". |
| Parent reconstruction across **two unrelated phases** (matrix + precipitate) | No orientation relationship connects them. | Phase fraction and exclusion census are the applicable analyses. |
| Per-grain depth without a depth-resolving optic | §2 row 1 — the column superimposes. | This is *configured*, not intrinsic, for any future run. It is intrinsic only for data already taken. |

## 4. Derived limits

What follows arithmetically from §1–2. A report may quote these directly.

| Quantity | Limit | From |
|---|---|---|
| Smallest separable feature | ≈ raster step | §2 raster step; two positions closer than one step are not distinguishable |
| Angular quantities relative to the sample | **undefined** until the surface normal is supplied | §1 row 1 |
| Number of grains per frame, no depth optic | the whole illuminated column | §2 row 1 |

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
- [x] `Last checked` is within the current run cycle

**Open.** The three blank bounds in §2 are the highest-value thing to fill in here. They are
the difference between a report that can say "a shorter exposure would reach the fast
process" and one that has to stay silent about exposure entirely.
