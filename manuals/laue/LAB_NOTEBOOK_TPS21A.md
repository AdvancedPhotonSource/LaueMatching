# Lab notebook — TPS 21A (NSRRC) Laue

Campaign started 2026-09-06. Question from TPS beamline staff: **can LaueMatching
index these patterns?** Data: `A5_fine` (Ni, 484 frames, 2024-10-24) and `Nb1_3`
(Ti α + β, 961 frames, 2023-06-06), plus one Si(100) frame (2025-04-09).

Handbook = [`README.md`](README.md). This file is the campaign
record: what was found, including what turned out to be wrong.

---

## 1. What this campaign has established

| # | Finding | Status |
|---|---|---|
| 1 | Geometry is the **34-ID-E reflection class** at a third station: 2θ = 90°, k_in = (0,0,1), sample at 45°, panel above the sample. PILATUS3 6M, 172 µm, 2527 rows × 2463 cols, ~522 mm. | established, from the XMAS panel + Pilatus headers |
| 2 | **XMAS pixel origin**: TIFF 0-based `row = 2527 − x_XMAS`, `col = y_XMAS − 1`. | established — exact to both decimals on all four center-channel values |
| 3 | The four `flip_*` mounts are the four `34IDE_*` mounts **rotated 180° about the beam** — the documented exact gauge. 8 candidate mountings collapse to **4**. | established, 2.0e-12 px / 1.4e-14 keV |
| 4 | An **image row-mirror (detector handedness) is an exact gauge when the tilts are zero**, and is broken *only* by the sub-degree tilts. | established, see §3a |
| 5 | Pairwise-interplanar-angle matching **has no power** as a convention discriminator for a cubic phase. | negative result, see §4 |
| 6 | `A5_fine` is **one crystal carrying a continuous orientation gradient**, not a single uniform crystal and not a multi-grain region. Reflections persist across the whole raster and sweep smoothly by 46-56 px over 1.05 um. | established, see §3c |

| 7 | **LaueMatching indexes TPS 21A patterns.** Si(100): 46 reflections, hkl-assigned, 8.87-25.91 keV, **median residual 0.42 px** (0.0079 deg). | established, see §5a |
| 8 | The geometry convention is `mount=34IDE`, settled **three independent ways**. Handedness (row parity) is **NOT** settled and cannot be, from these patterns. | see §5a, §3a-correction |
| 9 | A search-adjusted null gates CHANCE, not REDUNDANCY — see the §5c retraction. It is **0 solutions** for Si and Ni, but **8 matched spots** for the dense Ti frames — so `MinNrSpots = 8` is exactly ON the Ti null and too loose. | established, see §5b, §5h |
| 10 | A5's CPU/GPU difference is **marginality at the ±1-spot level**, not a search fault: both paths return the same orientation to 0.0185°. | established, see §5g |

---

## 2. Operational

- Work: `$ANALYSIS/tps21a_laue/` (analysis host), `$SCRATCH/tps21a_laue/` (compute host).
- Python: `/home/beams/EPIX34ID/conda-envs/lauematching/bin/python` — laue-index 0.4.0,
  has `LaueMatchingCPU` + `LaueMatchingGPU` + `LaueMatchingGPUStream`.
  The S1IDUSER shared env has the **CPU binary only**.
- Orientation DB: `/home/beams/EPIX34ID/opt/LaueMatching_canonical/100MilOrients.bin`.
- Detector traps, measured on the raw frames:
  - negative sentinels (−1, −2) over **8.46 %** of the panel (module gaps)
  - **one module suppressed to 7.3 %**: rows 0–194 × cols 988–1474, median 9 counts
    against 124 in its neighbours, identical on Nb frames 1 / 481 / 961
  - `Threshold_setting: 8740 eV` on every frame ⇒ the real low-energy cutoff is
    **8.74 keV**, not the 6 keV the `Condition.txt` files declare

---

## 3. Method findings

### 3a. Detector handedness — CORRECTED: not recoverable from these patterns

> **This section's original conclusion was wrong and is corrected below.** The
> synthetic measurement stands; the inference from it did not.

**Original (synthetic, orientation CONSTRAINED to the 24 symmetry images):**

An image row-mirror about the PONI is the lab mirror `diag(−1,1,1)`, which fixes
`ki = (0,0,1)`. Reinterpreting the **same observed pixels** under a mirrored readout
therefore yields the pattern of a mirrored crystal, and for a centrosymmetric lattice
that mirrored crystal is itself a valid proper orientation `M·U·S` inside the search
space. So the indexer would index a mirrored readout happily and return a mirror-image
orientation — a silent-success failure of exactly the class the handbook warns about.

Measured, on synthetic Si patterns through `calibrate.project`, as the residual `max|Δq̂|`
between the mirrored readout and the best of the 24 proper `M·U·S` candidates:

| tilts (pitch, yaw) | max abs Δq̂ | verdict |
|---|---|---|
| 0.000°, 0.000° | **6.7e-16** | exact gauge |
| −0.050°, +0.040° | 1.43e-03 | distinguishable |
| −0.388°, +0.308° (the real A5 values) | **1.11e-02** | distinguishable |

The break is **linear in the tilt** (7.8× tilt → 7.8× residual). At the real tilts,
1.11e-2 rad over 522 mm is ~34 px of systematic misfit — large, and easy to see.

**Consequences.** (i) The candidate set must include both image parities: 4 mounts ×
2 parities × 4 tilt conventions = 32. Omitting the parity risks a solution that indexes
with a ~34 px residual, which is easy to misread as a bad conversion. (ii) The
discriminating signal for handedness is *entirely* the sub-degree tilts. Had this panel
been mounted square, handedness would have been unrecoverable from the pattern and would
have needed external metrology, exactly like the beam azimuth. Do not generalise "the
indexer settles handedness" to another station without repeating this measurement.
(iii) Because R_Array must be a proper rotation, the parity cannot be expressed in the
parameter file — it lives in the **image**, and is applied at HDF5-conversion time.

**CORRECTION, from real data.** Consequence (ii) above is wrong. Run on the Si frame, the
two parities are indistinguishable:

| | NMatches | NSpotsCalc | median residual | energies |
|---|---|---|---|---|
| `34IDE` parity **+1** | 46 | 56 | **0.42 px** (0.00794°) | 8.87–25.91 keV |
| `34IDE` parity **−1** | 46 | 56 | **0.43 px** (0.00806°) | 8.87–25.90 keV |

(Verified the two are genuinely different inputs: the images satisfy `b == flipud(a)` and
P1 is negated, `+0.005851` vs `−0.005851`.) The indexed hkl sets are related by swapping
which index is dominant — mean |h|,|k|,|l| = (1.37, 1.37, 11.28) versus (1.37, 11.28, 1.37).

**Why the synthetic test over-promised.** It asked the mirrored readout to be explained by
one of the 24 orientations `M·U·S`. The indexer is not so constrained: it searches 100
million orientations and re-fits, and a free orientation absorbs almost all of the
tilt-induced distortion. The 1.11e-2 residual is what survives *at fixed orientation*, not
what survives a search.

**So handedness at TPS 21A must come from outside the diffraction**, exactly like the beam
azimuth. TPS's own `Detector Geometry.pdf` fixes the panel's in-plane mount (§5a) but not
its handedness: the `tth` annotation constrains the COLUMN direction, and the parity is the
ROW direction. The `chi` sign convention would settle it. **Ask TPS for it.** Until then
both parities must be reported as an open two-fold choice.

**What it does and does not change.** Under a global mirror every orientation becomes
`M·U·S`, so a misorientation `U1ᵀU2` becomes `S1ᵀ(U1ᵀU2)S2` — conjugated by point-group
operations. Symmetry-reduced **misorientation angles are invariant**; **rotation axes and
absolute orientations are not**. The A5 gradient magnitude is safe; its sense is not.

### 3b. The 180°-about-beam gauge shows up as duplicate mounts

Constructing mountings as "panel normal along ±lab Y, right-handed" produces 8, but the
four with normal −Y are the four with +Y turned 180° about the beam. Verified two ways:
`Rz(180) @ MOUNTS[a] == MOUNTS[b]` to 1.2e-16 for all four pairs, and identical predicted
patterns (47 spots, 2.0e-12 px, 1.4e-14 keV) from `(34IDE, U)` and `(flip, Rz(180)·U)`.
This is the same gauge `calibrate.py` records at 4.5e-13 px; seeing it reproduced here is
a check on the converter, not a new fact.

### 3c. A5 is one crystal with a gradient -- and the obvious test says the opposite

Comparing detected spot POSITIONS between A5 frames says the pattern decorrelates within
2-3 raster steps: fraction of frame 1's spots within 3 px of another frame's, top-10
brightest blobs only --

| separation | 1 step (50 nm) | 2 | 3 | 11 | 21 | random-list null |
|---|---|---|---|---|---|---|
| fraction matched | 0.70 | 0.30 | 0.10 | 0.00 | 0.00 | mean 0.000, max 0.010 |

Read naively that says "not a single crystal", and the folder name says it is. Both
readings are wrong. **Tracking the two brightest reflections across a raster row settles
it**: they never disappear, and they move *smoothly and monotonically* --

    frame  1 -> 21 :  brightest (758.6, 2376.5) -> (786.3, 2327.5)   [+27.8, -49.0] px = 56.3
                      second    (1316.3, 1196.7) -> (1346.7, 1162.0)  [+30.4, -34.7] px = 46.1

**CORRECTED 2026-09-06.** These first read 45 and 40 px, from a background subtraction
that did NOT clip the residual at zero. The negative tails pull the centroids, and the
error grows along the row (0.4 px at frame 1, 10.1 px at frame 21). Clipping at zero is
what the pipeline's own preprocessing does (`np.maximum(background_subtracted, 0)` in
RunImage), so the clipped numbers above are the measurement. The correction moves the
raw-frame estimate of the lattice rotation from ~0.85 deg to ~1.06 deg, which agrees
*better* with the 1.206 deg the indexed field reaches -- the two lines of evidence were
closer than the first pass suggested.

56 px of travel over 1.05 um is ~2.7 px per 50 nm step, which is why a 3 px coincidence
test fails after about one step. **A fixed-tolerance position-coincidence test cannot tell a gradient from a
grain boundary; only tracking individual reflections can.**

Consequence for the plan: the verification "adjacent A5 positions must return the same
orientation" as originally written would have FAILED, and for the wrong reason. The
correct version is that adjacent positions must return orientations differing by a small,
*spatially smooth* rotation, with the misorientation growing monotonically with raster
distance. That is now the test -- and the gradient is the A5 result worth reporting.

---

### 3d. OPERATIONAL TRAP: `ResultDir` is where the daemon writes, and it must be unique

The streaming daemon writes `solutions.txt` and `spots.txt` into the **parameter file's
`ResultDir`** — *not* into the orchestrator's `--output-dir`. Two runs of the same phase
therefore share one `solutions.txt` even when their `--output-dir`s are different, and each
one's post-processing reads whatever the other has written so far.

Hit on 2026-09-06. A 10-frame GPU check derived its config from the same
`params_TPS_Ti_alpha__34IDE_pp_rpy_zxy_pos.txt` as the 961-frame production run and
inherited its `ResultDir`. Result:

| run | frames asked for | outputs |
|---|---|---|
| GPU check | 10 | **770** |
| Ti_alpha production | 961 | **770** |
| Ti_beta production (no collision) | 961 | 961 |

Both runs logged `Pipeline complete` and exited **0**. The production α map was silently
truncated to 80 %, and nothing in either log said so. The only signal was 770 outputs from a
10-frame job being arithmetically impossible.

**Fix, now in `run_production.sh`:** `ResultDir` is rewritten per run to
`prod/scratch_<phase>_<timestamp>`. **Never** run two jobs off configs that share a
`ResultDir`, and never assume `--output-dir` isolates a run. Ti_alpha was re-run alone.

---

## 4. Retracted / negative — read before re-arguing

**"Pairwise interplanar angles can pick the convention without indexing."** NEGATIVE, and
it was my first attempt. The construction is sound — `q̂ ∝ kf − ki` is not rigidly rotated
when the detector turns, because `ki` is fixed — but the cubic angle set is far too dense
to carry the test. Measured on the Si frame, 46 spots, space group 227:

- best of 32 candidates: match fraction **0.4700**
- null, random detector poses (300 draws): mean 0.5211, **max 0.7227**
- null, spots scrambled inside the panel (300 draws): mean 0.4354, max 0.4835

The null sits **above** every candidate. Chance coverage explains it exactly: at hmax = 6
there are 572 distinct allowed angles in [0°, 90°], so a ±0.10° window covers ~25 % of the
range by construction, and ~64 % at the hmax = 8 the first run used. Tightening the
tolerance does not rescue it — the whole-pattern constraint that gives indexing its power
is a *simultaneous* hkl assignment, which pairwise angles discard. **Use the indexer.**

Two lesser errors in the same attempt, both caught by controls rather than by inspection:
the synthetic control used `hmax = 8` and produced **6-spot** patterns (too few to score
anything; hmax = 12 gives 25–47, matching the 46 observed), and the first mirror test
compared *panel-windowed* predictions, which cannot work — the panel is a window on
directions and mirroring the readout does not mirror the window. Its residual was constant
to three digits across trials, which is the signature that said "systematic, not mismatch".

---

## 5. Results

### 5a. The geometry convention, settled three ways

**(i) The 8-candidate sweep on the Si frame** (4 mounts × 2 parities, full 100M search,
`LaueMatchingCPU`, so a zero here is not the GPU-no-cubin signature). All eight exited 0:

| candidate | Initial solutions | Unique | best NMatches |
|---|---|---|---|
| `34IDE` parity +1 | **83** | **7** | **46** |
| `34IDE` parity −1 | **40** | **6** | **46** |
| `34IDE_rot90` ±1 | 0 | 0 | 0 |
| `34IDE_rot180` ±1 | 0 | 0 | 0 |
| `34IDE_rot270` ±1 | 0 | 0 | 0 |

**(ii) TPS's own `Detector Geometry.pdf`, with no indexing at all.** It annotates the Albula
display `tth_high` LEFT, `tth_low` RIGHT, `chi_pos` TOP, `chi_neg` BOTTOM. Computing 2θ at
the panel edges per candidate:

| mount | 2θ @ left | @ right | @ top | @ bottom | 2θ runs |
|---|---|---|---|---|---|
| **`34IDE`** | **111.32°** | **67.18°** | 89.24° | 89.24° | **across columns, low on the right — matches** |
| `34IDE_rot90` | 90.96° | 90.95° | 68.35° | 113.52° | across rows — contradicts |
| `34IDE_rot180` | 68.68° | 112.82° | 90.76° | 90.76° | columns, low on the **left** — contradicts |
| `34IDE_rot270` | 89.04° | 89.05° | 111.65° | 66.48° | across rows — contradicts |

**(iii) The physics of the indexed solution.** The 46 assigned reflections are [001]-centred
(mean |h|,|k|,|l| = 1.37, 1.37, 11.28), which is what a Si(100) wafer must give and was not
an input to the fit. Energies span 8.87–25.91 keV; the lowest sits just above the measured
8.74 keV discriminator, an independent consistency check on `Elo`.

### 5b. The null — and why the obvious one is not enough

**Single-draw null**, reproducing the indexer's own `NMatches` exactly (the reimplementation
returns 46/33355, 18/20629, 20/12017, 17/15269 against the four reported solutions — exact):
30,000 random orientations on the same geometry and the same `.bin` give **mean 0.35, p99 2,
max 7**. Observed 46: 0/30,000 draws reach it.

**That null is not sufficient, and saying so matters.** It bounds ONE random orientation. The
indexer reports the best of 100 million, and the Si frame carried three secondary solutions at
17–20 matches that a single-draw null cannot speak to.

**Search-adjusted null.** The same binary, same 100M database, same 12690 hkls, same 60
threads, same **36098 lit pixels in 50 components at 100 % of the original intensity** — only
the spot positions scrambled (and never onto a module gap):

| image | Pixels with intensity | Initial solutions | Unique |
|---|---|---|---|
| real Si frame | 36098 | **83** | **7** |
| spot-scrambled control | 36098 | **0** | **0** |

Zero. Every reported solution clears it.

### 5c. RETRACTED — "the Si calibrant is not a single crystal"

**This section claimed the Si(100) piece carried three Σ3 twins. It does not. It is one
crystal, and the claim was wrong for a reason worth keeping.**

The indexer returns four solutions, at 60.0° (Σ3) from the first and 38.9° (Σ9) from each
other. The obvious suspicion was a coincidence-site artifact: a Σ3-related orientation
shares a third of the reciprocal lattice, so it re-explains ~46/3 ≈ 15 of the parent's
spots for free, and 17–20 were observed. **I tested that, got "disjoint", and refuted it.
The test was wrong.**

It compared the sets of matched **PIXELS** in the indexer's `.bin`. That image is blurred:
36 098 lit pixels in **50 connected components, 722 px per blob**. Two predicted spots
20 px apart land on different *pixels* of the *same reflection*, and the test scored them
as explaining different data. Redone at the level of the reflection:

| | shared with sol1 | |
|---|---|---|
| sol2 | **18 of 18** | 1.00 |
| sol3 | **20 of 20** | 1.00 |
| sol4 | **17 of 17** | 1.00 |
| union of all four | **46 blobs** | exactly sol1's set |

They explain nothing the parent does not. The original suspicion was right and the
refutation was the error.

**Two lessons, both general.**

1. **Test redundancy at the level of the physical entity — a reflection — not pixels, and
   not sub-regions.** Any unit finer than the thing being counted inflates uniqueness. This
   is why the pipeline's own `--min-unique 2` did not filter these either: it counts
   *watershed* labels, and watershed found **3540 regions** on a frame with 50 reflections,
   so a redundant orientation clears a 2-unique-label gate trivially.

2. **A random-orientation null cannot gate a solution that is symmetry- or CSL-related to a
   true one.** The scrambled-image null for this frame was **0 solutions**, and all three
   artifacts cleared it — because they are not random draws. Clearing a random null shows a
   solution is not chance; it says nothing about whether it is *independent* of another
   accepted solution. Those need the unique-reflection test above.

**Reach of the error.** Si is the worst case (1 of 4 survive) because there is only one
crystal. On Nb1_3, measured over 5 frames, **49 of 67** orientations (73 %) have ≥3
reflections no other accepted orientation explains — so the two-phase counts in §5i are
inflated and need re-gating, but the qualitative structure is not in question.

### 5f. RETRACTED — "A5_fine / Ni: a 50 nm-resolution lattice-rotation map"

**The indexed orientation field is withdrawn. The gradient measured from the raw frames is
not.** Across all 135 frames that produced an accepted Ni orientation:

| | |
|---|---|
| median share of blob intensity the solution explains | **7.8 %** (mean 21.9 %, max 73.1 %) |
| frames explaining under 10 % of the intensity | **88 / 135** |
| of the 5 brightest blobs, how many explained | **median 1**; 36 frames explain **0**, one frame explains 5 |

Frame 132 is typical: 55 blobs, 4 explained, 4 % of the intensity, and both strong
reflections (1455 and 659 counts, 2857 and 2047 px) missed.

**The smoothness gate was confounded, and this is the transferable part.** I accepted the
field because neighbour misorientation (0.086°) beat a label-shuffle null (0.236–0.266°,
100/100 shuffles worse). But adjacent A5 frames are near-identical images — the pattern
moves ~2.7 px per 50 nm step — so *any* solution driven by those inputs, correct or not,
varies smoothly between neighbours. **A label shuffle tests whether a field is spatially
coherent; it cannot test whether the values are right when the inputs themselves are
spatially coherent.** For a finely-stepped raster, smoothness is nearly free.

The test that would have caught it, and did: does the solution explain the pattern —
specifically the STRONGEST reflections? It does not.

This is consistent with §5d rather than in tension with it: Ni's accessible reflections in
this window are the weak high-index ones, so the bright reflections on these frames cannot
be Ni in 8.74–26 keV at all. The indexer, restricted to that weak set, fits a few weak
blobs. **What the bright reflections are is now an open question worth putting to TPS.**

#### Superseded: the original section

Full 484-frame run, GPU, `Pipeline complete`, no errors, 4 min 12 s (1.92 img/s).

**135 of 484 frames indexed (27.9 %)**, exactly one orientation each (median 1, max 1 —
a single crystal, as the raw frames said), NMatches median 8 (min 6, max 10) against
NSpotsCalc ~17. Marginal, at the `MinNrSpots` floor, exactly as the energy-window count
predicts.

**Marginal per frame, but the FIELD is not marginal.** The gate that matters here is spatial
coherence, because noise cannot be smooth. Null: shuffle the voxel labels, which preserves
every orientation and destroys only their arrangement.

| | median misorientation between raster neighbours |
|---|---|
| real field | **0.0862°** (p90 0.2791, max 0.7826) |
| label-shuffled null, 100 draws | 0.2662 mean, **minimum 0.2361** |

Every one of 100 shuffles is worse than the real field, by a factor of ~3.

**And the gradient is monotonic in distance** — misorientation from the centre voxel:

| raster distance | 0–0.15 µm | 0.15–0.30 | 0.30–0.45 | 0.45–0.60 | 0.60–0.75 | 0.75–0.90 |
|---|---|---|---|---|---|---|
| n voxels | 18 | 45 | 32 | 17 | 16 | 6 |
| median misorientation | 0.061° | 0.145° | 0.151° | 0.257° | 0.243° | **1.206°** |

This is the same object the raw frames showed before any indexing (§3c, the two brightest
reflections sweeping 56 and 46 px across the row) arrived at independently, and it is the A5 result
worth reporting: ~1.2° of lattice rotation across 1.1 µm, sampled at 50 nm.

Caveat carried to the report: the map is **sparse** (27.9 % of voxels), and the raster is
nominal — no stage readback, sample at 45°, so µm figures are the commanded grid, not measured.

### 5i. Nb1_3 / Ti alpha + beta: a two-phase grain map over 961 voxels

**RE-GATED 2026-09-07 after the Si retraction (§5c).** NMatches alone cannot tell a real
orientation from one related to it by a coincidence-site lattice, which re-explains a third
of its reflections for free. Adding a second gate -- **>= 3 reflections that no other
accepted orientation explains, computed across BOTH phases together** -- removed
**599 of 5113** orientations (12 %). Cross-phase matters: a beta orientation that only
re-explains alpha's reflections is not independent evidence of a beta grain, and the
"both phases" count depends on that.

| | before | after |
|---|---|---|
| orientations | 5113 | **4514** |
| Ti alpha | 952 voxels / 3488 orient | **948 / 2948** |
| Ti beta | 959 / 1625 | **958 / 1566** |
| both phases | 950 | **945** (3 alpha-only, 13 beta-only) |
| alpha grains | 1464, largest 88 | **1069, largest 79** (707 singletons) |
| beta grains | 77, largest 530 | **42, largest 529** (27 singletons) |

Nulls re-run on the surviving set, 30 position shuffles, both monotonic statistics:

| | statistic | real | shuffled | verdict |
|---|---|---|---|---|
| Ti alpha | grain count | **1069** | mean 2689, min 2661 | fewer than any shuffle |
| Ti alpha | largest grain | **79** | mean 8.7, max 20 | ~4x any shuffle |
| Ti beta | grain count | **42** | mean 586.6, min 561 | fewer than any shuffle |
| Ti beta | largest grain | **529** | mean 120.5, max 187 | ~2.8x any shuffle |

**The result is unchanged in kind and sharper in degree**: beta's top four grains now hold
1315 of 1566 orientations (84 %). The coarse prior-beta matrix with fine alpha laths
survives the stricter gate -- which is the point of applying it.

#### Superseded: the original section

Both phases indexed over the full 31 x 31 raster, gated at the MEASURED null (NMatches >= 9,
§5h) rather than the shipped `MinNrSpots = 8`:

| | voxels | accepted orientations | per voxel | NMatches |
|---|---|---|---|---|
| Ti α | 952 / 961 (99.1 %) | 3488 (from 5667 before the gate) | median 3, max 16 | median 19 of 46 predicted |
| Ti β | 959 / 961 (99.8 %) | 1625 (from 2933) | median 2, max 6 | median 14 of 26 |

**950 voxels carry both phases** (98.9 %); 2 α-only, 9 β-only. The gate removed 2179 α and
1308 β orientations — a third to a half of the raw output, which is what measuring the null
instead of inheriting it buys.

**Grain structure, contiguous at 1.0°:**

| | grains | largest grains (voxel-orientations) | singletons |
|---|---|---|---|
| Ti α | 1464 | 88, 71, 55, 49, 43, 42, 42, 40, 37, 32 | 1046 |
| Ti β | **77** | **530, 326, 239, 228**, 80, 60, 51, 10, 9, 8 | 58 |

α is many medium grains; β is a few very large ones — 1323 of β's 1625 orientations sit in
just four grains. That is the coarse prior-β matrix with fine α laths inside it, which is
what a Ti α/β alloy should look like, and it was not put in.

**Null: same orientations, positions shuffled** (destroys contiguity, preserves every
orientation), 50 draws:

| | statistic | real | shuffled null | verdict |
|---|---|---|---|---|
| Ti α | grain count | **1464** | mean 3200.1, **min 3154** | fewer than any shuffle |
| Ti α | largest grain | **88** | mean 9.5, **max 19** | 4.6x any shuffle |
| Ti β | grain count | **77** | mean 641.2, **min 597** | fewer than any shuffle |
| Ti β | largest grain | **530** | mean 121.1, **max 290** | 1.8x any shuffle |

**A null statistic must be monotonic in the thing it tests.** The first version of this used
"number of multi-voxel grains", which is not: α scored 418 against a null max of 222 and
read as PASS, while β scored 19 against 210 and read as FAIL — when β is the *more* clustered
of the two. A field of a few huge grains has few multi-voxel grains, exactly like a field
with no structure at all, and the statistic cannot separate them. Grain count and
largest-grain size are both monotonic and both give the same verdict.

### 5g. CPU vs GPU on A5: marginality, not a discrepancy — RESOLVED

A5 frame 9: the GPU stream accepted it (NMatches 8 of NSpotsCalc 17); a CPU run with the
same parameter file returned `Initial solutions: 0`. Two readings — a real search
disagreement, or the solution sitting exactly on the `MinNrSpots = 8` gate with the two
preprocessing paths differing by about one spot.

Discriminating test: drop `MinNrSpots` to 5 on the CPU path and compare the ORIENTATION.

| | NMatches | NSpotsCalc | orientation |
|---|---|---|---|
| GPU stream | **8** | 17 | — |
| CPU, MinNrSpots=5 | **7** | 17 | **0.0185° from the GPU's** |

Same orientation to 0.0185° — comparable to the 0.011° run-to-run reproducibility recorded
for this code. The paths differ by **one matched spot**, and 8 is the gate. **Marginality
confirmed; not a GPU fault.**

Consequence, and it must reach the report: the A5/Ni orientations are real (two independent
paths find the same one), but the per-voxel **accept/reject is fragile at ±1 spot**. The
135/484 figure is a property of the threshold as much as of the sample. What is *not*
fragile is the field: 135 voxels arranged into a smooth monotonic rotation field cannot be
produced by a boundary artefact.

### 5h. The Ti null is NOT zero, and the default gate is too loose

Unlike Si and Ni (scrambled control: 0 solutions), the Ti frames are dense — 123 720 lit
pixels in 126 components against Si's 36 098 in 50 — so chance matching is real. Same
search, same spots, positions scrambled:

| phase | real frame, unfiltered NMatches | scrambled control |
|---|---|---|
| Ti α | 23, 19, 8, 8, 8, 8, 8 | **no solutions written** |
| Ti β | 12, 12 | **one, at NMatches 8** |

So the measured search-adjusted ceiling is **8 matched spots**, and `MinNrSpots = 8` — the
value the parameter file ships — sits exactly ON it. **Five of Ti α's seven real solutions
are at 8 and are therefore inside the null.** The honest acceptance floor is
**NMatches >= 9**, and the maps below are gated there rather than at the default.

Noted inconsistency, not resolved: the α scrambled run's stdout reported `Initial solutions:
3, Unique Orientations: 3` while writing an empty solutions file. The written count is what
is used; the 3 did not survive refinement. Worth a look before this null is reused.

### 5e. Column trap: the same solution table has 34 columns as text and 35 as HDF5

`solutions_filtered.txt` and `/entry/results/filtered_orientations` carry the same rows, but
the HDF5 array **prepends `image_nr`**. So every column index shifts by one:

| quantity | in the .txt | in the .h5 |
|---|---|---|
| NMatches | `[5]` | **`[6]`** |
| NSpotsCalc | `[6]` | `[7]` |
| OrientMatrix | `[22:31]` | **`[23:32]`** |

Using the text offsets on the HDF5 returns `NMatches*sqrt(Intensity)` in place of NMatches.
It cost me a wrong reading of the first GPU output — **280, 281, 232** where the truth was
**8, 8, 8** — and 280 is not obviously absurd, so nothing flagged it. What caught it was a
physical bound: Ni can only put ~18.6 reflections in the window and `MaxNrLaueSpots` is 30,
so any NMatches above 30 is impossible by construction.

Guards now in `a5_gradient_map.py`: assert 35 columns, and take the frame number from the
`source_file` attribute on `/entry/results` rather than parsing the output filename.

## 6. Measurement ledger

| Quantity | Value | Where from |
|---|---|---|
| PONI, A5 (col, row) | 1203.76, 1228.98 | `A5_fine_Condition.txt` via `xmas_geometry.poni_pixel` |
| PONI, Nb1_3 (col, row) | 1211.62, 1228.73 | `Nb1_3_Condition.txt`, same |
| P_Array, A5 (m) | 0.0046853, 0.0058514, 0.522115 | `xmas_geometry.convert` |
| P_Array, Nb1_3 (m) | 0.0033334, 0.0058944, 0.521652 | same |
| R_Array seed (nominal mount, no tilts) | −1.20920, −1.20920, −1.20920 | 120° about −(1,1,1)/√3 |
| Peaks/frame, A5 | 39–68 | `$ANALYSIS/tps21a_survey/peaks.py` |
| Peaks/frame, Nb1_3 | 217–249 | same |
| Spot FWHM, Si calibrant | 0.057° median, single-lobe 46/46 | `tps21a_survey/extent.py` |
| Background, Nb1_3 | 121 counts/px in 0.1 s (σ ≈ 11) | `tps21a_survey/probe.py` |
| Background, A5 | 3 counts/px in 0.05 s | same |
| A5 spot travel across the raster row | 56.3 and 46.1 px over 1.05 um (~2.7 px/step) | section 3c, corrected |
| Detection optimum, Si | ThresholdPercentile 99.95 -> 47 centers (46 measured) | `tune_detection.py` |
| Detection optimum, A5 | 99.8 -> 36 centers (39 measured) | same |
| Detection optimum, Nb1_3 | ~99.75 -> ~229 centers (229 measured) | same |
| Gap dilation needed | **4 px** (2x detection excess at 0-4 px, gone beyond) | measured, 11 frames |

