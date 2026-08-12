---
name: laue
description: >-
  Take a Laue microdiffraction dataset from an experiment folder to an indexed,
  null-gated grain map and a report: survey the raster, decide what science is
  askable, configure the material, index (sharded across GPUs), analyse against
  a measured null, and report. Use when asked to index, analyse or diagnose a
  Laue / polychromatic microdiffraction scan, when handed a raster of Laue
  frames, or when a Laue grain map or correlation looks wrong. Covers 34-ID-E
  raster acquisition through the LaueMatching chain.
---

# Laue microdiffraction

**This skill is a pointer, not the procedure.** The procedure is a doc set in this
repository so it lives beside the code it cites and stays usable without this skill.

## Start here

Read **`scripts/pipeline/laue/README.md`** — the spine: scope gate, install gate, halt
conditions, the invariants, a worked example and done-means. The seven phases open as you
reach them.

Then give, or work out:

```
Experiment folder: <ABSOLUTE PATH>
Material:          <e.g. Ni superalloy / 316L / Zr-4 / unknown, tell me from the data>
```

## Four things to know before you start

1. **Phase 1 is not optional and cannot be automated.** What science is askable decides
   which half of the analysis chain runs at all. Answer it in writing, including which
   steps do *not* apply and why.

2. **Measure the null on the scan in hand. Never inherit one.** On sampleH the measured
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
   running. A status check that only greps for the success marker cannot see a dead run.

## When something looks wrong

Go to **`scripts/pipeline/laue/DIAGNOSIS.md`** — symptom → discriminating test → cause →
lever. The sharpest entry: a detector artefact at a fixed position reproduces perfectly
frame to frame, so **persistence cannot separate it from a real reflection**. Test in
orientation space instead.

Before re-arguing anything, read the campaign notebooks — `LAB_NOTEBOOK_ZnZn.md` §5 in
particular, where the substrate/deposit direction flipped several times before it settled.

## Sibling doc sets

In the MIDAS repository: `manuals/ff-hedm/` (skill `ff-hedm`) and `manuals/nf-hedm/`
(skill `nf-hedm`). All three follow `beamreport/DOCS_SPEC.md`.
