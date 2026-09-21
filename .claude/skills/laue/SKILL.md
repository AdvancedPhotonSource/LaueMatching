---
name: laue
description: >-
  Take a Laue microdiffraction dataset from an experiment folder to an indexed,
  null-gated grain map and a report: survey the raster, decide what science is
  askable, configure the material, index (sharded across GPUs), analyse against
  a measured null, and report. Use when asked to index, analyse or diagnose a
  Laue / polychromatic / white-beam microdiffraction scan, when handed a raster
  of Laue frames, or when a Laue grain map, orientation or correlation looks
  wrong. Covers both REFLECTION geometry (34-ID-E and TPS 21A, panel edge-on
  above the sample) and TRANSMISSION geometry (16-BM-D / HPCAT, panel
  downstream and centred near the direct beam), through the LaueMatching chain,
  including converting an XMAS detector calibration to the pose the indexer
  wants. Detector handedness is NOT recoverable from a Laue pattern, absolute
  lattice scale / hydrostatic strain is not obtainable from white-beam spot
  positions, and deviatoric strain (c/a) is not resolvable by the C fit at
  elastic magnitudes; all three are gated, not delivered.
---

# Laue microdiffraction

**This skill is a pointer, not the procedure.** The procedure is a doc set in this
repository so it lives beside the code it cites and stays usable without this skill.

## Start here

Read **`manuals/laue/README.md`** — the spine: scope gate, install gate, halt
conditions, the invariant index (full text in `INVARIANTS.md`), a worked example and done-means. The seven phases open as you
reach them.

Then give, or work out:

```
Experiment folder: <ABSOLUTE PATH>
Material:          <e.g. Ni superalloy / 316L / Zr-4 / unknown, tell me from the data>
```

## Getting the code

`pip install laue-index` installs the pipeline **and compiles the C indexer on the machine
it will run on**. The GPU and streaming binaries are attempted by default when `nvcc` is on
PATH (`LAUEMATCHING_CUDA=require` fails the install if they cannot be built); or set
`LAUEMATCHING_BIN` to binaries you already have. `laue-index run process -c … -i …` drives a
frame without a checkout; `laue-index fetch-db` gets the orientation database
(7,200,000,000 bytes: 7.2 GB, which is 6.7 GiB; the two figures in the docs are the same file).

On the beamline hosts, use the beamline account's pip environment with laue-index (or its
editable-checkout environment for development). The old `laue_rt` environment and the old
checkout are gone (the checkout was archived on 2026-08-30); a note or script naming them is stale.

Never confirm the binary by listing `bin/` — a Mach-O arm64 binary once sat there on a
Linux host, looking present and ready. Ask for **all three** by name, because
`indexer.binary_path()` with no argument answers for the CPU binary only:

```bash
python -c "from laue_index import indexer
for t in ('CPU', 'GPU', 'GPUStream'):
    print(t, indexer.available(t), indexer.binary_path(t))"
laue-index doctor        # the build's GPU architectures against this card
```

For the by-hand architecture check use the full path, `/usr/local/cuda-*/bin/cuobjdump`: a
bare `cuobjdump` is not on PATH on the beamline hosts and prints nothing, which reads as "no
cubins". Empty output is a failed check (`DIAGNOSIS.md`, first entry).

**Current release is 0.7.1; 0.7.2 is prepared and not yet published.** Anything older than
0.3.1 built CUDA for the build machine's own card and did not check the kernel launch, so a
binary moved between hosts could index nothing and report success. `laue-index --version`.
The beamline's 0.7.1 build was reported as nvcc 12.1 with PTX for sm_90 as its newest
architecture, so sm_120 (Blackwell) cards run it by PTX JIT, unmeasured on this pipeline;
`laue-index doctor` shows what an install actually covers. **On 0.7.1, set
`LAUE_PREPROCESS_WORKERS` per shard (and `OPENCV_NUM_THREADS=1`) whenever several shards share
a host** (Phase 3 item 2); 0.7.2 and later also honour `LAUE_SHARDS_PER_HOST` and the
process limit.

## Which geometry?

The spine opens with a scope table. Establish this before anything else, because it changes
where the pattern sits on the panel and therefore what every downstream assumption means:

- **Reflection** (34-ID-E, TPS 21A) — panel edge-on above the sample, pattern runs vertically.
- **Transmission** (16-BM-D) — panel downstream, pattern radial about the beam, and the
  **direct beam is not at the point of normal incidence**: at 30° tilt they were 751 px
  apart. See `LAB_NOTEBOOK_16BMD_Si.md`.

The forward model is identical for both. Everything built *around* it is not.

**Calibrated in XMAS rather than a geoN XML?** `laue_index.xmas` converts it:
`from laue_index import XmasCalibration, xmas_to_laue, xmas_candidates`, then
`xmas_to_laue(XmasCalibration(...))`, and `xmas_candidates()` enumerates the 32 candidate
poses for the indexer to discriminate. (Those two are top-level aliases; inside
`laue_index.xmas` the same functions are `convert` and `enumerate_candidates`.) Do not pick one by argument:
at TPS 21A six of eight candidate mountings returned `Initial solutions: 0` and the
station's own detector drawing independently selected the survivor. Three traps the
converter encodes so you do not have to rediscover them: the XMAS pixel origin counts
from the far end of the long axis and is 1-based, `P` is the *inverse* of the projection
rather than a fit, and `R_Array` is **radians** however loudly `GenerateHKLs --help` says
degrees.

## Twelve things to know before you start

0. **Nothing you can measure inside a Laue pattern fixes the rotation about the beam, or the
   detector's handedness.** The rotation about the beam is
   an exact gauge freedom — measured at φ = 90°, no predicted pixel moves by more than
   2.3e-13 px and no energy changes at all. Relative quantities are fine; anything absolute
   needs metrology from outside the pattern. Agreement with a second code on the same
   calibration does **not** count. This was missed twice in one day.

   **Handedness is the same class and is worse, because it looks decidable.** With the
   detector tilts zeroed a readout row-mirror is exact to 6.7e-16 in q̂; nonzero tilts break
   it only linearly, and a synthetic test that holds the orientation to the 24 symmetry
   images makes it look resolvable at ~34 px. On real data both parities index identically
   — 46 reflections, 0.42 vs 0.43 px. Misorientation *angles* survive a wrong choice;
   rotation *axes* and absolute orientations do not. `ENVELOPE.md` §1 carries the row.

## Four more

1. **Phase 1 is not optional and cannot be automated.** What science is askable decides
   which half of the analysis chain runs at all. Answer it in writing, including which
   steps do *not* apply and why.

2. **Measure the null on the scan in hand. Never inherit one.** On sampleG the measured
   random-orientation null was **max 9 hits in 30,000 draws**, where the analytic Poisson
   gate would have accepted down to 5 — and re-gating at nhit > 9 kept 83.2 % of
   previously "validated" instances. Every reported number carries its null; anything else
   is an intermediate.

3. **Measure the raster from stage coordinates, never a folder name.** A folder called
   `10x10um_0p25umStepSize` measured 20.000 µm × 14.142 µm — exactly 1/√2, because the
   sample sits at 45° to the beam. Any area or density taken from the name is wrong by a
   factor you will not notice.

4. **Suspect success.** Most bugs here reported success: a daemon killed while healthy, a
   batch flag silently ignored, a shard driver logging "all 7 launched" with three
   running, and a GPU binary with no cubin for the card printing `Unique Orientations: 0`
   and **exiting 0**. A status check that only greps for the success marker cannot see a
   dead run. `DIAGNOSIS.md` has the discriminating test for the last one.

## Seven more, each of which cost a run or a wrong claim

5. **`run_laue.sh` needs `SCRIPTS` set explicitly** to the directory holding
   `laue_orchestrator.py` (`<repo>/scripts`). On 0.7.1 the default is the repository root and
   the failure goes to a log while the script prints a pid; 0.7.2 and later default correctly
   and refuse to launch without the orchestrator.

6. **Never combine `ssh -n` with a stdin-fed `bash -s`.** `-n` makes stdin `/dev/null`, so
   the remote script is empty, nothing runs, and the silence reads as a failed check.

7. **There are three spot counts, and a fourth file that reuses the word.** `NMatches`
   (distinct observed pixels for this orientation; does **not** stack harmonics),
   `unique_spots_per_orientation` (**winner-take-all** across the frame's orientations) and
   `nhit` (analysis side; **stacks** harmonics). The gates default to `nhit`;
   `nhit_distinct` (0.7.2 and later, `LAUE_GATE_STAT=nhit_distinct`) is available but NOT yet
   validated as a gate, and per-frame analytic Poisson gates stay on `nhit`.
   The glossary is `INVARIANTS.md` invariant 15b. Read the code that writes a column before
   trusting its name.

8. **A simulated `laue_torch` frame is transposed** relative to a real one. Verify any
   synthetic control against its own truth orientation (count hits on the frame as written
   and transposed) before using it to calibrate anything. Invariant 38.

9. **"Exited cleanly" is not "outputs written".** Wait for the output count to hold still
   across two checks ~30 s apart before any post-processing (invariant 21b).

10. **A null maximum is a noisy statistic.** Measure it on the FULL raster, not a subset,
    and prefer the 99.9th percentile to the maximum when a gate has to be stable. Each count
    has its own null.

11. **Check the campaign's `CHECKPOINT` and notes for prior measurements before asserting a
    defect.** Asserting one without that check cost a wrong claim in the 2026-09 campaign.

## When something looks wrong

Go to **`manuals/laue/DIAGNOSIS.md`** — symptom → discriminating test → cause →
lever. The sharpest entry: a detector artefact at a fixed position reproduces perfectly
frame to frame, so **persistence cannot separate it from a real reflection**. Test in
orientation space instead.

Before re-arguing anything, read the lab notebook for your geometry:

- `LAB_NOTEBOOK.md` §3b — where the substrate/deposit direction flipped several times
  before stage-invariance settled it.
- `LAB_NOTEBOOK_16BMD_Si.md` §4 — read before quoting any agreement number.
- `LAB_NOTEBOOK_TPS21A.md` — read before assuming a phase that indexes to **nothing** is a
  code fault. At TPS 21A it was the energy window, measured: the phase's six strongest
  reflections sat below the detector's discriminator, and no parameter could recover them.
  §3a is the one to read before claiming any pattern settles detector handedness.

## Sibling doc sets

In the MIDAS repository: `manuals/ff-hedm/` (skill `ff-hedm`) and `manuals/nf-hedm/`
(skill `nf-hedm`). All three follow `beamreport/DOCS_SPEC.md`.
