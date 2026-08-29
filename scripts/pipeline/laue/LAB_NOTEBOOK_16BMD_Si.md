# Lab notebook — 16-BM-D white-beam TRANSMISSION Laue, Si wafer

> Part of the **Laue doc set**. Procedure is [`README.md`](README.md); this file is what was
> *found*. First campaign outside 34-ID-E reflection geometry.

**Beamline** HPCAT 16-BM-D · **Detector** Pilatus3 X CdTe 1M · **Specimen** Si wafer
**Data** `Commissioning_2026_1/WB_scaning/Si/series1`, collected 2026-02-05
**Analysed** 2026-08-25 · **Full record** `~/Desktop/analysis/16IDB_Laue/RUNNING_LOG.md`

---

## 1. What this campaign established

| # | Finding | Status |
|---|---|---|
| 1 | The LaueMatching forward model works unchanged in **transmission**. `kf = ki − 2(q̂·ki)q̂` is the general Bragg mirror, valid at any 2θ; only the detector moves. | **ESTABLISHED** |
| 2 | Geometry converted from a pyFAI poni reproduces the beamline's own PolyLaue predictions to **4e-5 px / 0.0007 keV with nothing fitted**, once a pixel-origin convention and a 0.01 mm white-beam shift are accounted for. The two forward models are identical. | **ESTABLISHED** |
| 3 | The chain recovers PolyLaue's crystal to **0.026°** at all 1,243 indexed positions of one scan (median 0.027°, worst 0.213°). | **ESTABLISHED** |
| 4 | Six ω settings are mutually consistent to **0.0012°** after per-scan offsets, about an axis 0.16° from lab +X. **5 independent comparisons, not 15.** | **ESTABLISHED** |
| 5 | **Absolute orientation is NOT recoverable from this dataset.** The beam-azimuth gauge is exact and unbroken. | **ESTABLISHED (negative)** |
| 6 | The specimen is a single crystal over the whole mapped area; a ~40 µm filament crossing it produces small-angle scattering off the direct beam. | **ESTABLISHED** |
| 7 | The filament is a crack / scribe / scratch — SAXS cannot separate them. | **OPEN** |

## 2. Defects fixed

**`GenerateHKLs.py` truncated the reflection list in transmission.** θ_max was taken from the
**top-centre pixel** in the Y–Z plane only — right for a panel edge-on above the sample, wrong
when the beam lands near the panel centre, where the largest 2θ is at a *corner* and carries
the X component `atan2(Y,Z)` discards. Measured: hmax **35 → 14**, cutting off at d ≈ 0.39 Å
when the data demonstrably contains d = 0.334 Å. Fixed to a four-corner maximum of the full
3-D angle to the beam; for an edge-on panel it returns the same or slightly larger, so it can
never drop reflections at 34-ID-E.

**The reflection list needed the structure factor, not just the centering rule.**
`GenerateHKLs.py` applies only the Bravais condition. For Si (Fd-3m) the two-atom basis kills
every all-even reflection with h+k+l = 4n+2 — **24.7 %** of the list. Note this is a
**structure-factor cancellation, not a space-group systematic absence** (those are 2.0 %),
which is why `is_systematically_absent` correctly keeps the "forbidden" (222). Physics from
`midas_hkls`, never hand-written.

**Sort order is load-bearing.** `LaueMatchingCPU`'s own usage asks for the list "sorted on
f2", and `calcOverlap` stops at `MaxNrLaueSpots` distinct directions — so the order decides
*which* predictions get scored.

**The canonical tree on the compute host had never been usable.**
`~s1iduser/opt/LaueMatching_canonical` is an rsync copy from a Mac, and `bin/LaueMatchingCPU`
was an **arm64 Mach-O binary** on a Linux x86-64 host, with a macOS `libnlopt.a` beside it. A
`ls bin/` check reports it present and ready. Same class as invariant 6a.

> **Both halves of that are now fixed, and the rsync habit is retired.**
> `pip install laue-index` compiles the C on the machine it will run on, so a
> foreign-architecture binary cannot be staged by copying a tree; check with
> `python -c "from laue_index import indexer; print(indexer.binary_path())"`,
> which reports the binary that will actually be used.
> The `compute_70` request is gone too — CUDA 13 dropped Volta. The build now
> asks `nvcc --list-gpu-arch` and covers **every** architecture that toolkit
> supports plus PTX for the newest, rather than the card in the build machine:
> PTX JIT works forward and never backward, so a binary built on a newer GPU
> than it runs on finds zero grains and exits 0 — the same silent-success class
> as this section's arm64-binary trap. `LAUEMATCHING_CUDA=1 pip install
> laue-index` builds the GPU binaries the same way. NLopt is gone entirely.

## 3. Method findings — the transmission-specific ones

**On a tilted detector the PONI is not the beam.** At 30° tilt they are **751 px apart**: the
point of normal incidence at (502.2, 525.8), the transmitted beam at (−248.8, 525.3), off the
panel. Every "distance from the direct beam" computed against the PONI is wrong. This cost a
wrong interpretation of the streak fan (§5).

**Two codes can share a pixel-origin convention error invisibly.** LaueMatching uses
`px = xp/pxX + 0.5*(NrPxX−1)` — integer indices are pixel **centres**. pyFAI's
`calc_pos_zyx(0,0)` returns the pixel **corner**. The difference is exactly 0.5 px per axis.
PolyLaue additionally applies `WhiteBeamShift = 0.01 mm` (0.0581 px) to y, because the poni is
a **mono** calibration and the white beam sits off it at the sample. Predicted residual
−0.441860465 px; observed −0.441860465 px.

**`R_Array` is a rotation vector in RADIANS**, θ·axis — not Rodrigues and not degrees,
whatever `GenerateHKLs --help` and `params_alpha.template.txt` say. Both
`DetectorType.__init__` and `LaueMatchingCPU.c:239` take `rotang = norm(R)` and feed it to
`cos()`. Taking it from the docs puts the detector 30° out, silently.

**`Elo` can be measured, not guessed.** The Pilatus header carries
`Threshold_setting: 10000 eV` on every frame — the discriminator, a real low-energy cutoff.
`Count_cutoff` bounds saturation. `Detector_distance` is the Pilatus **default** (1.000 m) and
must not be used.

## 4. Retracted claims — read before re-arguing any of these

**"The streaks are asterism (lattice curvature), 28σ."** RETRACTED. The control was not
matched for spatial support: the true orientation's reflections lie in the half of the panel
where the streaks are, while random orientations spread over the whole panel including the
empty half. The comparison measured *"are the reflections in the left half"*. Matched controls
— scrambling radius or angle about the convergence point, preserving support — give
**0.5–2.0σ**.

**"Geometry validated to 0.23 px median against PolyLaue."** RETRACTED. That was the
**in-sample residual of a 3-parameter Procrustes fit to the same 99 points**. Kabsch has no
translation DOF, so it absorbed a rigid 0.667 px origin offset as a spurious **0.0423°**
crystal rotation — larger than the 0.0188° agreement it then produced. Unfitted, the residual
is a pure constant with zero scatter.

**"Disorientation 0.0188° from PolyLaue."** CORRECTED to **0.0257°**. The smaller value only
appears when the fit has absorbed the origin error. It is also finer than the pipeline's own
run-to-run reproducibility (0.011° between two runs of the same code on the same frame), so
only the first digit was ever real.

**"Common rotation axis [+0.98141, +0.18577, +0.04826] in the lab frame."** RETRACTED.
`midas_stress.misorientation_om` returns the axis in the **crystal** frame, folded into the
cubic fundamental sector — components sorted descending, all positive, which is the signature
to read. The lab axis is **[+1.00000, +0.00014, +0.00277]**, 0.16° from +X. A noise-free
synthetic whose true axis is exactly lab +X returns the same 11.08° "from +X".

**"ω residual 0.0185° RMS, 0 % outliers, 15 independent pairs."** CORRECTED. `[:400]` took the
400 **lowest raster indices** — one contiguous edge strip, not a sample. All positions give
**0.0084°**, and there are **31 outliers past 0.2°** (worst 53.3°). The 15 pairs regress onto
6 per-scan offsets at R² 0.97 — **5 degrees of freedom**.

**"Random-orientation null: max 2 hits in 120,001 draws."** CORRECTED for the `NMatches`
statistic. The null used a 2 px centroid tolerance on thresholded frames; the indexer uses
**zero-pixel tolerance** (`image[(int)py*nrPxX+(int)px] > 0`) on its own `.bin`, which has
**11.5× more acceptance area**, and `maxNrSpots *= 3` in refinement. Re-measured with the
indexer's criterion: **mean 2.06, max 10 in 20,000 draws** — 118× the published mean.

**"The hot-pixel screen found one pixel and nothing else."** RETRACTED. The screen tested raw
counts at 250 while the indexer keeps everything above the per-frame 99.8 percentile, median
**9 counts** — a **9–250 count blind band**. Rebuilt on occupancy in the *thresholded* frames:
**22 pixels**, twelve of them on the first column after the module gap, in contiguous runs
long enough to pass `MinArea 4` as a fixed-position "spot" on every frame. Masking them
removed **821 spurious orientations** and improved stacking 1.050 → 1.021.

## 5. The streak fan — what it took to get right

A ~40 µm filament of raster positions produces a bright fan that lifts the frame median from
5 to 40 counts and *reduces* indexable spots (it raises the percentile threshold and drowns
weak reflections). Structure-tensor fit over 38,154 streak pixels gives a convergence point
**11 px from the transmitted beam position** (2.0 mm on the panel, 0.50° from the sample);
random directions at the same pixels give 86.9 ± 0.3 px residual against 29.5 px measured.

It is **small-angle scattering off the direct beam**, not diffraction. Fixed stage coordinates
through 37.45° of ω, so it is in the specimen. Cause open.

The wrong answer came first only because "the streaks do not come from the direct beam" was
computed against the **PONI**, 751 px from the actual beam.

## 6. Measurement ledger

| quantity | value | source |
|---|---|---|
| raster | 51 × 50 at 10 µm, 500 × 490 µm | `fScan_*.txt` stage records |
| ω settings | 0, 2.78, 7.56, 16.21, 27.31, 37.45° | `Si/log.odp` `content.xml` |
| detector | 981 × 1043 at 172 µm, 223.381 mm, tilt 30.038° | `poly.poni` |
| 2θ range | 0.40 – 53.29° | four-corner computation |
| dead fraction | 7.41 % (module gaps + 327 bad px) | frame read + Pilatus header |
| sharp spots / frame | median 16, max 93; 40 % of positions off-sample | `peak_count.csv` |
| distinct observed / orientation | median 45 | `summary_v2.npy` |
| stacking ratio | **1.021** | `summary_v2.npy` |
| frames indexed | 8,108 of 15,300 | six shard logs |
| index rate | 0.188 s/frame, one GPU | `daemon.log` |
| forward cache | 12.2 GB = 1e8 × 122 B | `db/forward_Si.bin` |
