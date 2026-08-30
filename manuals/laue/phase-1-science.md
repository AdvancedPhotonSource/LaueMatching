# Phase 1 — Decide what science is askable

> Part of the **Laue doc set**. The spine — invariants, done-means and the phase
> order — is [`README.md`](README.md).

---

## Phase 1 — Decide what science is askable (the part that cannot be automated)

> **Check the campaign's lab notebook before halting to ask.** If a notebook exists for this
> experiment folder (`LAB_NOTEBOOK_<campaign>.md`), it answers most of what follows —
> material, phases, geometry provenance, wire status — and re-asking wastes the reader's
> time. Ask only for what the notebook does not settle. Phase 1 reads as if every session
> starts from zero; for a returning campaign it does not.

Ask the user these, in this order. The first three block everything; the rest shape the report.

1. **Material and phases present.** Space group + lattice parameters (nm) for each. If unknown from
   the sample history, it can be *tested* — index with a candidate phase and look at whether the
   validated fraction beats the measured null — but that is an experiment, not a lookup.
2. **Refined detector geometry** — the `geoN_*.xml` from the calibration of *this* run.
   `P_Array` / `R_Array` / pixel size / detector size come from it. Geometry from another run is
   the single fastest way to get a confident, wrong answer.
3. **Energy window** of the incident spectrum (keV). **Do not read it from
   `entry1/sample/incident_energy`** — measured 2026-08-12 across all 10,201 frames of
   `bt_34ide_jul26/sampleD`, that field is a constant **−56.05**, which is not a
   physical energy for a 5–30 keV pink beam. It is a metadata-field problem, not a data
   problem. Get the window from the beamline record or the campaign notebook, and state
   which — an unsourced energy window is exactly what hard rule "never take a number from a
   name" exists to prevent, one level up.
3b. **What frame are the orientations in, and where is the specimen surface?** The indexer's
   orientation matrices live in the **LAB** frame, and **lab Z is the incident beam** —
   `Phase.project` computes `kf = ki - 2*qh[:,2]*qh`, which is only valid for `ki = (0,0,1)`
   (confirm with `ph.ki`). At 34-ID-E the detector normal is lab **+Y** at 513 mm, so beam and
   detector sit at **90°** and the panel is edge-on to the beam. Consequence: *any* "declination
   from Z" is declination from the **beam**, an instrument direction with no sample meaning —
   c-axis-along-Z does **not** mean c-axis-along-growth. Get the specimen surface from the
   **measured stage motion**, never a convention: both raster axes lie in the surface, so their
   cross product is the normal (45° mount → `(0,-0.7071,0.7071)`, exactly 45.00° to the beam).
   Convert every declination to the surface normal before interpreting it. On sampleH that turned a
   meaningless "69.7° from Z" into "**c-axis avoids the growth direction by 8×**".
   *Do not* try to confirm the beam axis by rotating a crystal about it and checking the pattern
   rotates rigidly on the detector — that identity needs a detector *perpendicular* to the beam,
   and here it is edge-on, so all three axes fail and prove nothing. The real validation is that
   the forward model predicts observed peaks far above a random-orientation null.
4. **Is there depth resolution?** With a wire/coded aperture, each frame is one depth. Without one
   (the plain pink-beam case), the whole illuminated column superimposes — hundreds of grains per
   frame, and no per-grain depth. This changes what can be claimed, not just how hard it is.
5. **What is the experiment asking?** Grain map? Phase fraction? Parent-phase reconstruction?
   Deformation/streaking? In-situ evolution across a series?
6. **Is this a series?** Multiple scans of the same specimen (load steps, temperatures, positions)
   support cross-scan comparison; a set of unrelated test scans does not.

### Which analysis applies to which material system

| system | phases | applicable beyond the core | notes |
|---|---|---|---|
| Ti / Zr alloys | β BCC + α HCP | **Burgers OR** parent-β reconstruction (12 variants) | the implemented path; `burgers_Cv()` |
| steels, Fe–Ni | γ FCC + α′ BCC/BCT | parent-γ via **K-S (24)** or **N-W (12)** | swap `burgers_Cv()` (§6); re-derive the accept threshold |
| single-phase FCC/BCC/HCP | one | twin relationships (Σ3 for FCC), texture | **no parent reconstruction** — skip steps 6–8 of the chain |
| two unrelated phases (e.g. matrix + precipitate) | two | phase fraction, exclusion census | the parent machinery does not apply; do not run it "to see" |
| **same phase either side** (Zn on Zn, weld/parent, epitaxial deposit) | one | Laue-footprint fragmentation, flat-background scattering, per-spot energy | **nothing crystallographic separates them** — see §Same-phase problems |
| unknown / mixed | — | phase identification first | index each candidate phase separately, compare validated-vs-null |

**Always applies, regardless of material** (this is the core, and it is where the pipeline's
credibility lives):

- measure the **random-orientation null on this scan** (`null_model.py`) and re-gate against it
  (`empirical_gate.py`);
- count grains under the **contiguity-aware definition** (`regrain.py`);
- report **tolerance sensitivity** (`tolerance_sensitivity.py`);
- check that any "corroborating" statistic **beats chance** before quoting it (`anchor_null.py`
  is the template: with enough candidates, a random orientation matches one by luck).

---
