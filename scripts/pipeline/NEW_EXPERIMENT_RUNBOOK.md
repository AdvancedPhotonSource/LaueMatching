# New experiment — survey → index → analyse → report (chat handoff)

**Use this doc to start a fresh chat on a Laue dataset this pipeline has never seen.**
Paste it in, then give one line:

```
Experiment folder: <ABSOLUTE PATH>          # e.g. $LAUE_DATA-X/<PI>_<YYYYMM>/
Material:          <e.g. Ni superalloy / 316L / Zr-4 / unknown, tell me from the data>
```

Everything else the agent works out or asks for. The order below is not optional: each phase
produces the inputs the next one needs, and phase 1 (what science is even askable) decides which
half of the analysis chain runs at all.

> **This runbook is material-agnostic; the *code* is not yet.** The indexer and the statistical
> core are general. Several analysis scripts still carry hard-coded a two-phase hcp/bcc alloy lattice constants
> and `valid_hkls_Ti_*.csv` filenames. §6 lists every one of them, by file and line. Do that port
> **before** running the chain on a non-Ti material, or the nulls will be computed against the
> wrong reflection list and will look perfectly healthy while being meaningless.

Companion docs: [`README.md`](README.md) (the pipeline itself), and — outside this repo — the
34-ID-E operational runbook `laue_torch/report/RUN_PROCESS_REPORT_HANDOFF.md` (beamline access,
the a two-phase hcp/bcc alloy campaign, and its current state).

---

## Phase 0 — Survey the experiment folder (do this before promising anything)

Goal: a written `SURVEY.md` in the work directory answering *what is actually here*, with numbers
read from the files, never from folder names.

```bash
E=<EXPERIMENT_FOLDER>

# 1. what scan folders exist, and how big is each (ls|wc -l lies past ~40k files: ARG_MAX)
for d in "$E"/*/; do
  n=$(find "$d" -maxdepth 1 -name '*.h5' | wc -l)
  [ "$n" -gt 0 ] && printf '%8d  %s\n' "$n" "$d"
done | sort -rn

# 2. the HDF5 layout of ONE frame -- do not assume /entry1/data/data
python - <<'PY'
import h5py, sys, glob
f = sorted(glob.glob(sys.argv[1] + "/*.h5"))[0] if len(sys.argv) > 1 else None
def show(n, o):
    if isinstance(o, h5py.Dataset): print(f"{n:60s} {str(o.shape):20s} {o.dtype}")
h5py.File(f).visititems(show)
PY
```

Record, per scan folder:

| field | how to get it | why it matters |
|---|---|---|
| frame count | `find … -name '*.h5' \| wc -l` | queue order, runtime estimate |
| image dataset path + shape | `visititems` above | goes in the params / `run_laue.sh` arg |
| **stage coordinates** | `entry1/sample/sampleX`, `sampleZ` (or the local equivalent) | **the real raster** |
| step size and span | `np.unique(np.diff(sorted(X)))` | see the 45° trap below |
| exposure | frame header / folder name, then confirm against counts | 0.25 s vs 1 s changes what is detectable |
| still growing? | frame count twice, 120 s apart | never index a scan still being collected |
| peaks on one frame | detect on a background-subtracted frame, SNR>8, **area ≥ 4 px** | density regime (below) |

**The 45° trap.** A folder called `10x10um_0p25umStepSize` measured 20.000 µm in X at 0.2500 µm and
14.142 µm in Z at 0.1768 µm — exactly 1/√2, because the sample sits at 45° to the beam. It is a
20×20 µm region in the sample frame, 400 µm² not 100 µm². **Measure the raster from the stage
coordinates for every scan and quote that.** Any area, density, or grains-per-µm² taken from a
folder name is wrong by a factor you will not notice.

**Peak-count check, and why the area filter is not optional.** A bare SNR>8 local-maximum test on a
weak frame returns single-pixel noise spikes as "peaks": on one Si calibration frame it gave 159,
of which 47 were extended reflections. Requiring a connected area of ≥4 px above 4σ is what makes
the count mean something. `pattern_complexity_figure.py` in the report scripts does exactly this
for both frames it compares.

**Density regime** — sets expectations and the peel depth:

| peaks/frame | regime | consequence |
|---|---|---|
| ≲ 50 | single crystal / few grains | classical indexing works; this pipeline is overkill but fine |
| 100–500 | many grains | the design case; nulls matter |
| ≳ 900 | dense/streaky | expect the iterative peel (`batch_peel_driver.py`) and several s/frame |

---

## Phase 1 — Decide what science is askable (the part that cannot be automated)

Ask the user these, in this order. The first three block everything; the rest shape the report.

1. **Material and phases present.** Space group + lattice parameters (nm) for each. If unknown from
   the sample history, it can be *tested* — index with a candidate phase and look at whether the
   validated fraction beats the measured null — but that is an experiment, not a lookup.
2. **Refined detector geometry** — the `geoN_*.xml` from the calibration of *this* run.
   `P_Array` / `R_Array` / pixel size / detector size come from it. Geometry from another run is
   the single fastest way to get a confident, wrong answer.
3. **Energy window** of the incident spectrum (keV).
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

## Phase 2 — Configure

Per phase, once per material:

```bash
# 1. params: copy the template, fill in crystal + geometry + energy + paths
cp params_alpha.template.txt  $WORK/params/params_<mat>_<phase>.txt

# 2. build the per-material inputs (the 100M-orientation DB is shared across all phases)
python ../GenerateOrientations.py            # -> db/100MilOrients.bin      (once, ever)
python ../GenerateHKLs.py       <params>     # -> params/valid_hkls_<phase>.csv
python ../GenerateSimulation.py <params>     # -> db/forward_<phase>.bin
```

Then point `run_laue.sh`'s CONFIG block at `WORK`, `PY`, and the two param files.

**Detection settings are the difference between 1 s/frame and 170 s/frame.** The validated set for
34-ID-E Ti: `ThresholdPercentile 99.8`, `MinNrSpots 8`, `MinIntensity 50`, `MinArea 4`,
`GaussSigmaMax 2.5`. Loosening to `99.0` or `MinNrSpots 6` produced 7 M coarse matches and 13 k
spurious orientations per frame. Re-tune for a new detector/material, but treat a run that shows
`WARNING: match count … exceeded MAX_MATCHES` plus >100 s/frame as a **configuration fault, not a
slow computer**.

---

## Phase 3 — Index

```bash
WATCH=""  ./run_laue.sh  <SCAN_FOLDER>  [<h5 dataset path>]     # batch an existing scan
./run_laue.sh <SCAN_FOLDER>                                     # live, stop with STOP_LAUE
```

For many scans, use the batch runner: one run at a time, smallest first, `--settle 120` to defer
folders still growing, `.laue_done` / `.laue_skip` markers for resume and for splitting work across
machines. See §6 of the RUN_PROCESS_REPORT handoff for the multi-machine layout.

Sanity while it runs:

- **orientations/frame is scan-dependent** — 10–100 on a sparse scan, 275–1043 on a dense one, both
  healthy. Judge by the flood signature (thousands, `MAX_MATCHES`, >100 s/frame), not an absolute.
- **`output.h5` files appear in one late batch** after largely single-threaded post-processing
  (a 1.9 GB `solutions.txt` took ~45 min). "0 outputs after N frames" is not a stall.
- **Frames with the beam off the specimen legitimately produce nothing.** One 40,401-frame scan had
  a genuine 1,947-frame blank band. Verification tolerances must allow ~10%, or you will reject a
  good run.
- **Never stack launches.** Each daemon reads a ~7 GB database before binding its port; two at once
  saturate NFS and both abort with "Daemon did not open port in time".

---

## Phase 4 — Analyse

```bash
env LAUE_WORK=$WORK/analysis/<scan> \
    LAUE_SCAN_DATA=<SCAN_FOLDER> \
    LAUE_SCAN_ALPHA=<results/alpha_TS> \
    LAUE_SCAN_BETA=<results/beta_TS> \
    LAUE_OUT_PREFIX=<scan> NW=16 \
    bash analysis/run_analysis_chain.sh
```

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

Then, always: `regrain.py` (contiguity-aware counts — pass the scan's own measured null maxima via
`LAUE_NULLMAX_ALPHA` / `LAUE_NULLMAX_BETA`, or it falls back to the Ti values and warns),
`tolerance_sensitivity.py`, and `collect_scan_metrics.py` for the cross-scan JSON.

**Three lessons that cost real time, and generalize to any material:**

1. **The analytic Poisson `p<1e-4` gate under-rejects on clustered peak fields.** Measured nulls
   reached 16 hits where Poisson forbids it. Across nine scans the measured α null maximum ranged
   14–17 and β 11–16 — one inherited value misstates the rest. On one dataset this changed the
   defensible count by two orders of magnitude.
2. **A grain is a *contiguous* region of consistent orientation.** Orientation-only clustering
   merges disjoint regions. On one scan, splitting into connected components moved α from 325 to
   614 and β from 40 to 27 — the two phases moving in *opposite* directions, which is what shows it
   is a definitional fix and not a tuning knob.
3. **"Corroboration" must beat chance.** With 2,537 candidate clusters, a random orientation lands
   within 1.74° of one 9% of the time. The same statistic was genuinely strong at 767 clusters.
   Measure it before quoting it.

---

## Phase 5 — Report

Two deliverables, from the same numbers:

- **PDF** — LaTeX, one per scan or one per campaign. Every number reported with its null. State the
  measured raster (not the folder name), the null maxima, the grain definition used, and what was
  *not* measurable (e.g. no depth resolution → no per-grain depth).
- **HTML artifact** — the shareable version. Keep slides to one screen each; embed figures as
  data-URI JPEGs (a strict CSP blocks every external request). To update an existing artifact, pass
  its URL back — do not mint a new one for the same deliverable.

Reusable figure generators live in the report scripts: `validated_figures.py` (report plates),
`catalog_figures.py`, `scan_map.py` (quick-look map), and the survey figure that puts a
single-crystal frame beside a many-grain frame **under identical detection and scaling** — the
honest way to show why a dataset is hard.

Report only validated, recurring quantities. Cluster catalogs (e.g. "28,063 orientation clusters")
are pipeline intermediates; on a 400 µm² region half of them were single-position. Quote the
doubly-supported subset: beyond the measured null **and** recurring across positions.

Deliverable layout that worked (13 scans, ~293k result files):

```
DELIVERABLE_<campaign>/
  MANIFEST.md          contents, key numbers, method caveats
  indexing_output/     per-frame output.h5 (hardlinked), indexing.txt, frame_mapping.json, provenance.json
  per_scan_analysis/   per scan: peel_map/*.npz, figures/, logs
  cross_series/        metrics.json + summary
  reports/             PDFs, HTML artifacts, LaTeX sources
  scripts/             the exact analysis scripts used
```

Hardlink the per-frame outputs rather than copying (same inode, zero extra disk, becomes an
independent copy when tarred) — but only within one filesystem; check `stat -c %d` on both paths.

---

## Phase 6 — Porting to a non-Ti material (do this first, or the nulls lie)

Each of these reads a Ti lattice or a `valid_hkls_Ti_*.csv` by name. Nothing errors if you leave
them — the reflection list simply belongs to another material, and every downstream null is
computed against it.

| file (in `analysis/`) | lines | hard-coded |
|---|---|---|
| `null_model.py` | 38–46 | HCP a/c, BCC a, `valid_hkls_Ti_{alpha,beta}.csv` |
| `parentbeta_validate.py` | 49–65 | same |
| `exclusion_null.py` | 39–47 | same |
| `beta_alpha_exclusion_census.py` | 57–65 | same |
| `map_validate_cluster.py` | 28–34 | HCP + α hkls |
| `beta_map_validate.py` | 17–18 | BCC + β hkls |
| `grain_extent_backfill.py` | 33–38 | HCP + α hkls |
| `parentbeta_backfill.py` | 33–40 | both |
| `batch_peel_driver.py` | 30–70 | HCP + α hkls + `params_Ti_alpha.txt` |
| `exposure_signal_check.py` | 24–25 | BCC + β hkls |

Also:

- **Symmetry operators** are selected by the *phase name* `"alpha"`/`"beta"` → hex-12 / cubic-24
  (`regrain.py`, `collect_scan_metrics.py`, `parentbeta_validate.py`). For a material whose phases
  are not one hexagonal and one cubic, that mapping must change too.
- **The orientation relationship** lives in one function: `burgers_Cv()` in
  `parentbeta_reconstruct.py:49`, returning a `(12,3,3)` variant set. Everything downstream
  (`pred_alpha`, `cand_parents`, `variants_matched`, the parent search) is generic in that array.
  Swapping in K-S gives `(24,3,3)`; the **accept threshold must be re-derived** — "11 of 12" is an
  empirical cut for Burgers, not a law, and the synthetic gate at the top of the script is how you
  re-derive it.
- The 100M-orientation database is **material-independent** (it is an SO(3) grid). Only the hkl
  list and forward cache are per-phase.

A clean port is: parameterize lattice + hkl path + symmetry per phase in one small module, import
it in the ten files above. That is a contained change and worth doing on the first non-Ti dataset
rather than the second.

---

## Invariants (violate these and the result is wrong but looks fine)

1. Measure the null **on the scan in hand**. Never inherit one.
2. Measure the raster from the **stage coordinates**. Never trust a folder name.
3. A grain is **contiguous** and consistent in orientation.
4. Every reported number carries its null; anything else is an intermediate.
5. Detect on the aggressive threshold, **verify on the full background-subtracted frame** — no
   signal is discarded from the evidence.
6. **Suspect success.** Most of the bugs in this pipeline reported success: a daemon killed while
   healthy, a batch flag silently ignored, a drain that stopped before the file finished writing, a
   dict mutated during serialization, `>` refused by tcsh `noclobber`. `scripts/tests/test_streaming_regressions.py`
   pins the fixes; run it after touching the streaming path.

## Done means

- [ ] `SURVEY.md` exists, with measured raster + frame counts + peak density per scan
- [ ] Phase 1 answered in writing, including which chain steps do **not** apply and why
- [ ] Material port (§6) done, or explicitly not needed because the material is a two-phase hcp/bcc alloy
- [ ] Indexing complete: `output.h5` count ≈ frame count (allow ~10% for genuine blank bands)
- [ ] Null measured on **each** scan; counts re-gated against it
- [ ] Grain counts from `regrain.py`, with tolerance sensitivity
- [ ] Any corroborating statistic tested against its own chance null
- [ ] PDF + artifact, every number with its null, caveats stated
- [ ] Deliverable folder assembled with a MANIFEST; scripts used are the repo's, not a copy
