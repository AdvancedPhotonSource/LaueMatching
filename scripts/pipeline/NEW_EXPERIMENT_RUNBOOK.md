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

> **The material port is DONE (2026-07-24, on the Zn/Zn dataset).** `analysis/laue_material.py`
> now reads lattice, reflection list, detector geometry, energy window and symmetry from *the
> indexing parameter file itself*, so the analysis cannot disagree with the run it describes.
> All 13 previously Ti-hard-coded scripts import it. Selecting a material is an environment
> variable, not an edit:
>
> ```bash
> export LAUE_PHASES=zn                                  # comma-separated; single-phase is fine
> export LAUE_PARAMS_ZN=$WORK/params/params_Zn.txt       # LAUE_PARAMS_<PHASE>, upper-case
> ```
>
> §6 is now a *verification* step, not a porting step. Symmetry follows the **space group**, not
> the phase name -- the old rule silently handed cubic-24 operators to any phase not called
> `"alpha"`.

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
| hot pixels | pixels saturated in ≥90% of ~60 frames spread over the scan | Zn scan: **36 permanent** hot pixels; only ~8% of saturated pixels were real reflections. Every frame's `max` looked like signal and was not. |
| background, decomposed | median of four detector **corners** (flat) vs a central box (halo) | the flat part is isotropic emission (fluorescence); it is the part that tracks path length |
| spot shape | blob aspect ratio, median **and** p95 | Zn *looked* heavily streaked; measured median AR was 1.6 with only ~10% above 3. The eye reads the p95 tail. |

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
| **same phase either side** (Zn on Zn, weld/parent, epitaxial deposit) | one | orientation persistence, fluorescence pedestal, per-spot energy | **nothing crystallographic separates them** — see §6 |
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

### Sharding a single big scan across GPUs and hosts

For one large raster (40k+ frames) there is only one phase to index, so GPUs split **frames**
rather than phases. Two hard constraints, both learned the expensive way on the Zn scan:

**1. Budget ~19 GB of HOST RAM per daemon, and do not stack them.** Each daemon holds the 7.2 GB
orientation database *and* the 12.2 GB forward cache in host memory. Three daemons on a 128 GB box
plus their image servers filled RAM, drove swap to 100%, and two of the three shards died with
`Send/save error for image_num=N: timed out` (the 30 s socket send timeout) while appearing
"running". Per-image time went 1 s -> 4.5 s before they stopped entirely.

**2. `laue_image_server` has no backpressure.** It enumerates the whole folder and its preprocessing
pool buffers results faster than the daemon consumes them (~1.5 s/image). Parent RSS reached
**17 GB on a 13,467-frame shard**; at 201 frames this cannot manifest. Either give the host enough
RAM or keep shards small.

Machines that can see `$LAUE_ROOT` and have the epix34id LaueMatching install
(`/home/beams/EPIX34ID/opt/LaueMatching`, shared home, has the `LAUE_STREAM_PORT` fix):

| host | RAM | cores | GPUs | notes |
|---|---|---|---|---|
| copland | 2015 GB | 96 | 2x A6000 48 GB | **cannot write** to the-analysis-host -- point ResultDir elsewhere |
| alleppey | 502 GB | 112 | 4x H100 80-96 GB | usually shared; check `nvidia-smi` first |
| sentosa | 250 GB | 64 | 2x H200 144 GB + 2x Blackwell | Blackwell cards (2,3) are **sm_120**, often in use |
| shannon | 125 GB | 40 | 3x A4500 20 GB | 34-ID-E box; smallest RAM, budget 2 daemons max |

**Log in as `epix34id` on every host** (not s1iduser): the data, the DB and the caches are all
owned by epix34id, and the s1iduser LaueMatching build is older -- it ignores `LAUE_STREAM_PORT`
and silently binds 60517, so two daemons on one host collide. Reachability: epix34id keys live on
shannon, so the route is `copland(s1iduser) -> epix34id@shannon -> epix34id@<host>`. Every remote
shell is **tcsh**: pipe scripts to `bash -s`, and never use `$(...)` in the outer ssh command.

`scripts/pipeline/launch_shard.sh SHARD GPU PORT NCPUS` runs one orchestrator on whatever host it
is invoked on. Stagger launches by ~60 s: each daemon reads 19 GB before binding its port.

Also: files written by one account are not automatically readable by another. `forward_*.bin` is
created mode `600`; `chmod 644` it before another host's account can load it.

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

Then, always: `regrain.py` (contiguity-aware counts — pass the scan's own measured null maximum via
`LAUE_NULLMAX_<PHASE>`; it now **exits** rather than falling back to the Ti values),
`tolerance_sensitivity.py`, and `collect_scan_metrics.py` for the cross-scan JSON.

**Clustering does not scale past a test scan.** The greedy loop inside `parentbeta_validate.py` is
O(n_clusters x n_instances x n_sym): fine for the ~1e3–1e4 instances a test scan produces, but a
full 201x201 raster gives ~2e5 and it never finishes. For a full raster:

```bash
LAUE_SKIP_CLUSTER=1 python parentbeta_validate.py <phase> <nw> env    # stops after the npz
python cluster_orientations.py <validated.npz> <clustered.npz> 1.0 <phase>
```

`cluster_orientations.py` is KD-tree based (quaternions; a misorientation cut theta becomes a radius
`sqrt(2-2cos(theta/2))`), and clusters are **connected components**, which — unlike greedy
assignment — do not depend on iteration order. 1,076 instances in 0.22 s.

> **Trap, and it cost real time: the symmetry operator multiplies on the RIGHT.** The pipeline's
> misorientation is `min_S angle(A^T B S)`. The left-handed form `A^T S B` is a *different*
> quantity — verified numerically, they differ by up to **78.7 deg** — and using it silently split
> 120 of 392 real grains while every synthetic test passed. Gate any new orientation code against
> `laue_material.misorientation` / `midas_stress` directly, never against your own reimplementation:
> a test written from the same misunderstanding as the code agrees with it perfectly.

Merging shards: each shard's orchestrator numbers its images `1..N` **independently**, so image
numbers collide across shards. Merge on the stored `frames` field (the source `.h5` filename),
which is unique map-wide.

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
4. **Tune the threshold on VALIDATED orientations, not raw ones.** A looser threshold always yields
   more raw "unique orientations" and they are overwhelmingly noise. On Zn: 99.5 gave 3,023 raw
   orientations/frame at 0.4 img/s (the flood), 99.8 gave 7 and 99.9 gave 4 — but after the
   per-frame Poisson test, 99.8 gave **2.07x more validated** orientations than 99.9 *and* lost no
   frames, while 99.9 dropped 9 of 201 frames below `MinNrSpots` entirely.
5. **A texture null must be indexability-matched.** Detector coverage, the energy window and the
   reflection list all make some orientations easier to index than others, so a peaked pole figure
   can be an artefact of what is *indexable*. Compare against random orientations passed through the
   same "at least MinNrSpots reflections on the detector" filter, not a flat sphere — and use one
   representative per grain, since one grain contributes many positions and instances are not
   independent samples.

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

## Phase 6 — Material configuration (the port is done; verify it)

`analysis/laue_material.py` is the single source of truth. It parses the **indexing parameter
file** — the same file the indexer consumed — for lattice, space group, symmetry, detector
geometry, energy window and the reflection-list path, so the analysis physically cannot be run
against a different material's reflections than the indexing was.

```python
from laue_material import Phase
ph = Phase.load("zn")            # -> $LAUE_PARAMS_ZN, else $LAUE_PARAMS
ph.B, ph.hkls, ph.sym_ops        # reciprocal matrix, reflections, proper rotations
ph.project(OM, with_energy=True) # (n,3): px, py, E_keV
ph.misorientation(A, Bs)         # DEGREES, symmetry-reduced
```

There is **no built-in default material**: failing to resolve `LAUE_PARAMS_<PHASE>` raises rather
than falling back, because silently analysing one material with another's reflection list is the
exact failure this module exists to prevent.

**Orientation maths comes from `midas_stress`**, the canonical MIDAS implementation (a
byte-for-byte port of the C `GetMisorientation.h`): `misorientation_om_batch` for misorientation
(it returns **radians** — `laue_material` converts to degrees) and `make_symmetries(sg)` for the
operator set. Install it with the rest of the deps: `pip install midas-stress midas-hkls`
(both are **torch-free** — `pip install midas-stress` needs no torch as of **0.8.1**; torch is an
opt-in `midas-stress[torch]` extra). `laue_material` keeps a local operator fallback only for
environments where midas_stress is not installed at all. On midas_stress **>= 0.8.1** the operators
are exact and agree with the fallback to machine precision; older (<= 0.8.0) installs stored 5-decimal
symmetry quaternions and agreed to **0.032 deg** worst case (still **0 pairs** across the 1.0 deg
clustering cut, measured on 30,000 real Zn pairs), which is why the selftest tolerance is 1e-4 not
machine epsilon.

Setting up a new material:

```bash
python ../GenerateHKLs.py -resultFileName $WORK/params/valid_hkls_<M>.csv \
   -sym <F|I|C|A|R|P|B> -sgnum <SG> -latticeParameter a b c al be ga \
   -RArray ... -PArray ... -NumPxX 2048 -NumPxY 2048 -dx 200e-6 -dy 200e-6 -Ehi <keV>
# then let the daemon build the forward cache once (~10 min), and
export LAUE_PHASES=<m>  LAUE_PARAMS_<M>=$WORK/params/params_<M>.txt
```

Still per-material and *not* automatic:

- **The orientation relationship**, if one applies: `burgers_Cv()` in
  `parentbeta_reconstruct.py:49`, returning a `(12,3,3)` variant set. Everything downstream is
  generic in that array. K-S gives `(24,3,3)`; the **accept threshold must be re-derived** —
  "11 of 12" is an empirical cut for Burgers, not a law.
- The 100M-orientation database is **material-independent** (an SO(3) grid) — symlink it, never
  copy 7.2 GB. Only the hkl list and forward cache are per-phase.

### Same-phase problems (substrate/deposit, weld/parent, deposit on like substrate)

If the two things you want to separate are the *same phase* — e.g. Zn electroplated on Zn — then
no phase fraction, no exclusion census and no parent reconstruction applies (chain steps 4–8 are
out). What is left:

- **orientation persistence** — a substrate grain is large and continuous, so its orientation
  recurs over a wide *contiguous* area; deposit grains do not;
- **fluorescence pedestal** — more material in the beam path means more of its own K-alpha, which
  lands as a *flat, detector-wide* offset. Separate it from the forward-peaked halo (air scatter +
  thermal diffuse) by measuring detector corners against the centre; only the flat part should
  track path length;
- **per-spot energy** — `Phase.project(OM, with_energy=True)`, or exactly, from the stored per-spot
  `hkl` (spots-table cols 3,4,5) and its grain's orientation matrix. The 1/e sampled depth is a
  strong function of energy (for Zn at 45 deg: 3.4 um at 12 keV, 41 um at 30 keV, from
  `midas_hkls.absorption`), so a thick overlayer preferentially removes *low*-energy reflections.

Use them together: two independent observables that must move in the predicted directions is a far
stronger claim than either alone, and each needs its own null (§ below).

## Invariants (violate these and the result is wrong but looks fine)

1. Measure the null **on the scan in hand**. Never inherit one.
2. Measure the raster from the **stage coordinates**. Never trust a folder name.
3. A grain is **contiguous** and consistent in orientation.
4. Every reported number carries its null; anything else is an intermediate.
5. Detect on the aggressive threshold, **verify on the full background-subtracted frame** — no
   signal is discarded from the evidence.
6. **Suspect success.** Most of the bugs in this pipeline reported success: a daemon killed while
   healthy, a batch flag silently ignored, a drain that stopped before the file finished writing, a
   dict mutated during serialization, `>` refused by tcsh `noclobber`, an image server "running" for
   ten minutes after its socket had timed out, `BETA_CONFIG=""` falling through to the default and
   exiting 1 *after* alpha had already launched. `scripts/tests/test_streaming_regressions.py`
   pins the fixes; run it after touching the streaming path.
7. **Verify a new implementation against the incumbent, on real data, at the level of the decision
   that matters.** Agreement "to 1e-12" is not the claim; "no pair changes side of the 1.0 deg cut"
   and "0/2000 draws change their Poisson verdict" are. Synthetic tests share your assumptions;
   real data does not.
8. **Empty is not zero.** `ls */*.h5 | wc -l` past ~40k files hits ARG_MAX and reports 0. A count of
   zero output files mid-run is normal (post-processing writes them in one late batch). Neither is
   evidence of anything.

## Worked example

The Zn/Zn electroplated dataset (`bt_34ide_jul26/sampleG/scan1_Laue2D`, 40,401 frames) is the first
non-Ti material through this chain and exercised every phase; its `SURVEY.md` in
`$LAUE_WORK/` is a filled-in template for Phase 0. Headline
numbers, all measured: 201x201 at 1.000 um (the 45-deg trap did **not** bite — stage coordinates
agreed with the folder name for once), wire parked so no depth resolution, 96–167 peaks/frame at
99.8, measured random-orientation null **max 9 hits in 30,000 draws** (the analytic Poisson gate
would have accepted down to 5), and re-gating at nhit>9 kept **78.9%** of "validated" instances.

## Done means

- [ ] `SURVEY.md` exists, with measured raster + frame counts + peak density per scan
- [ ] Phase 1 answered in writing, including which chain steps do **not** apply and why
- [ ] `LAUE_PHASES` + `LAUE_PARAMS_<PHASE>` set; §6 verified (`python laue_material.py` selftest passes)
- [ ] Indexing complete: `output.h5` count ≈ frame count (allow ~10% for genuine blank bands)
- [ ] Null measured on **each** scan; counts re-gated against it
- [ ] Grain counts from `regrain.py`, with tolerance sensitivity
- [ ] Any corroborating statistic tested against its own chance null
- [ ] PDF + artifact, every number with its null, caveats stated
- [ ] Deliverable folder assembled with a MANIFEST; scripts used are the repo's, not a copy
