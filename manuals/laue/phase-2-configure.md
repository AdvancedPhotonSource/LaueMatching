# Phase 2 — Configure

> Part of the **Laue doc set**. The spine — invariants, done-means and the phase
> order — is [`README.md`](README.md).

---

## Phase 2 — Configure

> **`$WORK` first — it is used throughout Phases 2, 3, 4 and 6 and is defined nowhere
> else.** It is the campaign's own working directory, holding `params/`, `db/` and
> `results/`. It is **not** the read-only experiment folder. Pick one you own on a host
> that can see the data, export it, and keep using the same one:
>
> ```bash
> export WORK=/gdata/dm/34IDE/<Run>/<Campaign>/laue_matching_results   # yours to write
> mkdir -p $WORK/params $WORK/db $WORK/results
> ```
>
> The orientation database (`db/100MilOrients.bin`) is shared across every phase and
> campaign — point at an existing one rather than regenerating it. Never `/tmp`.

Per phase, once per material:

```bash
# 1. params: copy the template and replace EVERY __SET_ME__ (crystal, geometry, energy, paths)
cp params_alpha.template.txt  $WORK/params/params_<mat>_<phase>.txt
grep -n __SET_ME__ $WORK/params/params_<mat>_<phase>.txt    # must print nothing

# 2. build the per-material inputs (the 100M-orientation DB is shared across all phases)
#    From pipeline/ the generators are ../scripts/*.py (shims onto laue_index.pipeline);
#    ../Generate*.py resolves to the repository ROOT, where they are not.
python ../scripts/GenerateOrientations.py            # -> db/100MilOrients.bin      (once, ever)
python ../scripts/GenerateHKLs.py       <params>     # -> params/valid_hkls_<phase>.csv
python ../scripts/GenerateSimulation.py <params>     # -> db/forward_<phase>.bin
```

> Spot intensities: `GenerateSimulation.py` gives every spot the same intensity unless the
> parameter file declares a `PhaseAtom`/`PhaseCIF` basis, in which case it uses |F(hkl)|²
> from `midas_hkls` (`-intensityModel`, `-spectrumFile` for I0(E)). Before 2026-08-29 it
> assigned a *uniform random* intensity per spot and floored every spot to an integer
> pixel — a mean 0.42 px displacement that put a 0.0077° floor under any sub-pixel fit
> made against those images. Regenerate anything synthetic produced earlier than that.

**Replace every `__SET_ME__`.** The templates carry it wherever an experiment-specific value
belongs (`SpaceGroup`, `Symmetry`, `LatticeParameter`, `P_Array`, `R_Array`, the file paths,
`ResultDir`). From 0.7.2 a leftover placeholder is refused: the Python parsers raise on it in
`SpaceGroup`, `Symmetry`, `LatticeParameter`, `P_Array` and `R_Array`, the C binaries in
`LatticeParameter`, `P_Array` and `R_Array` (and on `P_Array[2]` = 0), and `run_laue.sh` /
`dispatch/mkrun.py` refuse a file that still holds one. In 0.7.1 a placeholder fell back to a
built-in default (another experiment's lattice and a 0.513 m detector) and the run went ahead.
`R_Array` is a rotation vector with the angle in **radians**; `tol_LatC` / `tol_c_over_a` are
**fractions** (`ENVELOPE.md` §2).

Then point `run_laue.sh`'s CONFIG block (or the environment) at `WORK`, `PY`, `SCRIPTS` and
the two param files. **`PY` is the full path of the python with laue-index installed** (an ssh
login shell does not have your conda environment on PATH). **`SCRIPTS` is the directory
holding `laue_orchestrator.py`** (`<repo>/scripts` in a checkout). On 0.7.1 the default
resolves to the repository root and the launch fails into a log while the script reports a
pid; from 0.7.2, when unset, it uses `../scripts` next to `run_laue.sh`, else the installed
package's `laue_index/pipeline/` (asked of `PY`), and refuses to launch if neither holds the
orchestrator. `DRY_RUN=1` prints the resolved paths without launching.

**What the C binaries refuse or change in 0.7.2** (0.7.1 accepted all of these silently):

- an incomplete numeric line in `LatticeParameter`, `P_Array` or `R_Array` (e.g. a leftover
  `__SET_ME__`), and `P_Array[2]` = 0 (it collapses every spot onto one point): fatal;
- a crystal-fit tolerance ≥ 1 or NaN: fatal; above 0.1: a warning giving the percent it will be
  used as; `tol_c_over_a` non-zero: `tol_LatC` is ignored, with a NOTE;
- an all-zero `R_Array` is now the identity rotation (it used to produce NaN spots);
- the forward cache is written to `<ForwardFile>.partial.<host>.<pid>` and renamed onto `ForwardFile`
  only after every write, fsync and close succeeded; a write/fsync/close failure is fatal and
  deletes the partial. A `*.partial.<host>.<pid>` left by a killed run is never read and can be
  deleted. **Still open:** a right-sized `ForwardFile` built for a DIFFERENT geometry or
  lattice is accepted; after changing either, delete the old cache (`DIAGNOSIS.md`).

**Detection settings are the difference between 1 s/frame and 170 s/frame.** The validated set for
34-ID-E Ti: `ThresholdPercentile 99.8`, `MinNrSpots 8`, `MinIntensity 50`, `MinArea 4`,
`GaussSigmaMax 2.5`. Loosening to `99.0` or `MinNrSpots 6` produced 7 M coarse matches and 13 k
spurious orientations per frame. Re-tune for a new detector/material, but treat a run that shows
`WARNING: match count … exceeded MAX_MATCHES` plus >100 s/frame as a **configuration fault, not a
slow computer**.

---
