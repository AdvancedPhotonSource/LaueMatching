# Laue Lab Notebook — method and defect record

**Companion to `Laue_Handbook.md`.** The runbook says what to do; this records what was
actually found running the chain on real campaigns — how it was measured, and what turned
out to be wrong. They are kept apart on purpose: the runbook has to stay short enough to
follow, and this has to stay honest enough to stop a refuted idea coming back.

**Scope of this public record.** This file carries the *transferable* content: tool defects,
their symptoms, the operational traps, and the method findings that hold regardless of
material. Campaign-specific scientific results, the materials they were measured on, and the
measurement ledger are **not** in the public tree — they belong to the users who ran those
experiments. See `BEAMTIME_KEY.md` (git-excluded) for where the full records live.

Setting for everything below: APS 34-ID-E, Perkin Elmer PE1621, 5–30 keV, sample at 45° to
the beam, 2-D rasters of 2,600–20,300 frames at 1–10 µm step. Samples are referred to by
pseudonym (`sampleA`, `sampleD`, …); the substrate/deposit systems are described
structurally.

---

## 1. What these campaigns established, method-side

| # | Finding | Status | Where |
|---|---|---|---|
| 1 | Three detector artefacts inflate peak counts 35–45% | FIXED (8146d31, c5a12e3) | §2a |
| 2 | Masking a bright reflection MANUFACTURES peaks; image-space peeling cannot work | ESTABLISHED | §3a |
| 3 | Substrate removal must happen in SCORE space, via stage-invariant peak positions | ESTABLISHED | §3b |
| 4 | A silent single-frame background can substitute one frame's diffraction for a whole scan's background | FIXED | §2b |
| 5 | The support metric that was gated on counted the wrong thing, and the obvious join returns zeros silently | FIXED | §2c |

**Read §4 before re-opening any question here** — the retracted claims are recorded with the
measurement that killed each one.

---

## 2. Defects fixed

### 2a. Detector artefacts (committed: 8146d31, c5a12e3)

Three, all on saturated reflections, all inflating counts:

- **Flat-top plateaus.** A clipped spot has a flat top, so every pixel equals the local
  maximum and a max-filter flags all of them. One 117 px saturated substrate spot produced
  **58 detections at identical intensity**; collapsing plateaus cut whole-frame counts
  **35–45%**. Position the single reflection from its unsaturated 20–90%-of-clip shoulders.
- **Isotropic halo** around intense spots (1480–2142 ADU at r=15 px against σ≈50).
- **Vertical blooming**, real but rarer than expected.

Hot pixels must be excluded by *area and shoulder* tests: an early version reported 36–38
"saturated reflections" per frame on a panel with 34 permanent hot pixels.

### 2b. The silent single-frame background

`mkbg_gen.py` builds frame paths as `{prefix}{i}.h5` with **plain integers**; derived frames
(substrate-peeled) were written zero-padded, so it raised `FileNotFoundError`. **The pipeline
does not fail on a missing `BackgroundFile`** — the image server logs
`Computing background from first frame...` at INFO, writes it to the expected path, and every
later shard then reports "used supplied background". One frame's own diffraction became the
background for 10,201 frames.

Bad background: min 121 / med 263 / max 567. Correct: 93 / 214.5 / 441.5. All 32 runs in the
campaign were audited — only one re-index was affected, and it was re-run. `mkbg_gen.py` now
accepts both naming conventions.

### 2c. The wrong support metric, and the wrong join

`entry/results/unique_spots_per_orientation` — distinct **observed** peaks — is written to
every output file and was never used. The analysis gated on `filtered_orientations[:,6]`,
which counts matched **predicted** reflections.

**Join on `filtered_orientations[:,1]` ↔ `unique_spots_per_orientation[:,0]`** (orientation id
within the frame). Column 0 of the orientation rows is the **image number**; joining on it
returns zeros silently. That is how the mistake surfaced — a known strong single crystal
appearing to have 0 distinct spots.

Measured stacking (matched ÷ distinct) ran **1.1–1.5×** across four sample/phase
combinations. Modest in general — the sevenfold stacking of a substrate-overlaying
orientation (§4) does **not** generalise to ordinary solutions.

### 2d. Operational traps

- **`pgrep -f <pattern>` over `ssh -n host "..."` self-matches** when the login shell is tcsh:
  the wrapper's own command line contains the pattern, so every finished run looks alive and
  an unattended watcher hangs forever. Require the binary name in the matched line.
- **Output-file count is not a completion test** — a frame yielding no solution writes no
  `.output.h5`, so a finished run legitimately shows fewer outputs than frames (2662/3400).
  Use the orchestrator's `Streaming: 100%` line plus a settled non-zero count.
- **Concurrent shards need per-shard `ResultDir`** or their `solutions.txt` interleave and
  post-processing dies on torn lines. Cost when missed: ~30 min GPU, 5 shards.
- **Concurrent runs also need a unique PORT, and the check must run on the host.** Two plan
  files written days apart both allocated 61101; the second daemon logged
  `bind: Address already in use` mid-startup, then sat fully initialised holding 12 GB of GPU
  with **zero** images received for 20 minutes, indistinguishable from a slow run.
  `grep -h '' plan_*.txt | awk '{print $3}' | sort | uniq -c` before dispatch; and
  `ssh -n <host> "bash -lc 'ss -ltn | grep -q \":$PORT \"'"` as the gate. Handbook 18.
- **A scan's frames are not necessarily in that scan's directory.** One sample's nominal
  scan directory was EMPTY and its 10,201 frames lived in a *different sample's* directory,
  distinguished only by filename prefix — two experiments in one folder. The shards here were
  built by file prefix and are clean, but a directory glob would silently merge them. The
  same trap appeared in reverse on another sample, whose scan directory was empty while its
  frames sat in the parent. **Select by prefix, print the count per prefix, never trust the
  folder.** Handbook 19.
- **Streaming completion is NOT output completion — do not launch validation on the
  `Streaming: 100%` line.** One validator started the moment streaming finished and found
  **275** `.output.h5`; the run's post-processing then wrote the rest, ending at **2,498**. It
  produced 686 validated instances against ~6,000 expected, and nothing in the log said
  anything was wrong — it validated exactly what existed. Gate on the output count being
  *unchanged* across two checks 30 s apart before validating.
- **Reading `h5["/entry1/data/data"][0]` gives the first ROW, not the first frame.** These
  files store one 2048 × 2048 image per file as a 2-D dataset, so `[0]` is 2,048 pixels of
  detector edge — all zeros. A diagnostic written that way reported "min 0 med 0 max 0" for
  every frame of a perfectly good scan and nearly triggered a hunt for corrupt data. Use
  `[:]`, and sanity-check the median against the background before concluding anything.

---

## 3. Method findings

### 3a. Image-space substrate removal cannot work

Four approaches, measured on 60 matched frames, detections/frame at r=15–20 px from a
substrate reflection (raw rate 0.9):

| approach | 15–20 px ring | note |
|---|---|---|
| mask the core with background | **12.1–19.7** | the ring |
| halo-subtract, then mask | 8.3 | annulus median flat at −0.7 ADU, residual is *azimuthal* |
| fill core with local level + noise | 7.8 (and **31.7** inside 0–15 px) | synthetic noise makes its own maxima |
| **reconstruct core as a Gaussian, subtract whole reflection** | **0.87 = raw** | ring solved; far field 95% preserved |

**The ring is not in the image.** With the core masked, pixels at 15–20 px are
*bit-identical to raw* (9.6 ADU both) yet detections there go 1.9 → 19.7. The halo rises
inward, so a rim pixel is never the maximum in its 9-px window until the core is deleted;
deleting it promotes the whole halo ridge to local maxima. **Detector-agnostic** — a plain
SNR local-max finder with all halo machinery OFF shows it *more* strongly (19.7) than the
full detector (10.9). `frame_peaks.py` is not at fault.

Even the Gaussian reconstruction leaves the substrate detected 26.6×/frame while the image
is flat (−2.7 ADU): a clipped core held ≥65,535 counts, so its shot noise alone is ~256 ADU,
far above an 8×MAD threshold. **The signal is removable; its noise is not.**

A leak test using radii 15/25/40/60 px **could not see** the ring, because its bins straddle
the rim. It reported "0.0% within 15 px" and that was read as success when it meant the
substrate had moved 15 px outward. Every peeled result was built on it.

### 3b. Score-space peeling, via stage-invariant peaks

Detector positions carrying a peak *regardless of stage position* cannot be deposit grains —
the beam moves ~1 µm between frames and a small grain leaves the probe volume. Occupancy
across the raster identifies the substrate with **no orientation, no lattice, and no
single-crystal assumption**. (Suggested by the beamline scientist, from noticing that some
spots never move.)

- **Threshold matters, bin size does not.** Coarsening bins 6→24 px changes nothing;
  occupancy 80% → 3/29 substrate reflections recovered, 50% → 13/29, **30% → 25/29**. The
  substrate's reflections are simply not detected on every frame — most likely because the
  percentile threshold rises where the deposit diffracts strongly.
- One sample at ≥30%: 84 bins — **29 near a predicted substrate reflection, 66 on
  persistently-bright pixels (overlapping), 16 unexplained**. The panel has **5,261 pixels
  bright in the pixelwise minimum** across the scan, far more persistent structure than the
  29 substrate reflections.
- Another: 50 invariant positions carrying **11.2%** of peaks; **88.8% moves with the stage**.

---

## 4. Retracted claims, and what killed them

The scientific claims themselves are campaign-specific and are not reproduced here. The
*failure modes* are general, and each is worth knowing before trusting a similar result.

**R1 — an epitaxial orientation relationship at a large null ratio.** Killed by 3 of 4
adversarial lenses. The validator scored *predicted* reflections, and the proposed
relationship put the deposit's c\* along a substrate direction, **stacking all seven
harmonics of one deposit reflection family onto a single substrate pixel** (separation
0.00 px). Over half the deposit's in-plane vectors fell within 0.1° of a substrate vector.
*Lesson: when two lattices are related by the relationship you are testing for, a
predicted-reflection score cannot distinguish them — the test is circular.*

**R2 — the reversal of R1.** Measured on peeled frames carrying the manufactured ring (§3a).
**89.3% of instances were ONE orientation present at 93.3% of raster positions**; the 11
largest clusters all lay within 1.54° of each other; **effective n ≈ 1.25**. The three
largest "grains" drew 47–52% of their matched peaks from the ring and spanned the whole
raster. *Lesson: a grain count is not a sample size. Compute effective n before quoting any
ratio over grains.*

**R3 — an occupancy figure quoted from a sweep subset.** A 510-frame threshold-tuning subset
gave ~45%; the full 2,601-position raster gave **98.1%**. *Lesson: a sweep subset is chosen
for threshold tuning, not sampled uniformly — never quote an occupancy from one.*

Also withdrawn: a "re-gating doubles the accepted count" claim that compared against a
threshold never actually applied.

### Diagnostics that could not see what they were built to check

Recorded because the pattern repeated three times in one session, and each time the tell was
**a number that should have sat near a known value and did not**:

- the leak test binned 15/25/40/60 px and straddled the ring (§3a) — "0.0%" read as success
- a von Mises–Fisher sampler dropped the `(1-u)e^{-2κ}` term, turning weak fibres into tight
  *antipodal* clusters; the tell was a **non-monotonic** power curve (κ=0.5 → MRD 2189,
  κ=4 → 57)
- peak MRD over 24×24 bins with ~4.75 counts/bin gave a **null median of 16.4** for a
  quantity that must sit near 1

**Every diagnostic needs a case where the answer is known in advance.** A confirmed single
crystal in the field of view is the anchor that caught §2c.

---

## 5. Open method questions

1. **Iterative indexing** for the large unexplained peak fraction. Index, subtract the peaks
   the accepted orientation explains, re-index the residual, repeat until nothing clears a
   re-measured gate.
2. The **`nhit > 11` gate is not calibrated for a best-of-10⁸ argmax search** and was
   inherited on the peeled runs — no null-model log exists for those frames.
3. Substrate-mimicking phases must be re-scored on **substrate-free peaks** before any
   apparent match is believed (see R1's mechanism).
