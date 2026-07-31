# Laue Handbook — survey → index → analyse → report

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

### Handbook vs lab notebook

**This file says what to do. The lab notebooks say what was found.** They are kept apart on
purpose: a handbook has to stay short enough to follow, and a campaign record has to stay
honest enough to stop a refuted idea coming back. When a rule below cites a measurement, the
full account — including the controls that killed the competing explanation — is in a notebook.

- [`Laue_Lab_Notebook_bt_34ide_jul26.md`](Laue_Lab_Notebook_bt_34ide_jul26.md) — Zn on
  the fcc substrate and Al on Al. Three retracted claims, the image-peel autopsy, the α-brass
  identifiability limit, and the measurement ledger.

**Write a new lab notebook per campaign, not per dataset**, and start it on day one — the
retractions are the part that decays fastest. Structure that works: what the campaign
established (a table with a status column) → defects fixed → method findings → scientific
findings → **retracted claims and open questions** → measurement ledger.

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
| **position-label integrity** | `np.unique(np.diff(X))` in **acquisition order** — look for `0` and `2*step` | a readback race, see below. Present on **both** Zn scans, so assume it until checked |
| exposure | frame header / folder name, then confirm against counts | 0.25 s vs 1 s changes what is detectable |
| still growing? | frame count twice, 120 s apart | never index a scan still being collected |
| peaks on one frame | detect on a background-subtracted frame, SNR>8, **area ≥ 4 px** | density regime (below) |
| hot pixels | pixels saturated in ≥90% of ~60 frames spread over the scan | Zn scan: **36 permanent** hot pixels; only ~8% of saturated pixels were real reflections. Every frame's `max` looked like signal and was not. |
| background, decomposed | median of four detector **corners** (flat) vs a central box (halo) | the flat part is isotropic — but TEST whether it is fluorescence (tracks amount) or diffuse scattering (tracks grain-size/disorder); on Zn/Zn it was scattering, see §Same-phase and invariant on the background |
| spot shape | blob aspect ratio, median **and** p95 | Zn *looked* heavily streaked; measured median AR was 1.6 with only ~10% above 3. The eye reads the p95 tail. |

**The stage-readback race — mislabels, NOT gaps.** If the acquisition-order diffs of the fast axis
contain `0` and `2*step` as adjacent pairs, the readback is being sampled *after* the move begins and
one frame takes the next position's label. On sampleH this hit 180 of 20,301 frames (0.89%); the same
signature is in sampleG, so it is beamline/macro behaviour, not a one-off.

Decide the mechanism from the **images**, because the two readings have opposite fixes. Labels read
`X, X+2, X+2, X+3`. If the motor genuinely skipped, the two frames sharing a label are *co-located*;
if the readback merely led, they are one step apart. Compare their residual-pattern similarity
against a 1-step and a 2-step control taken **from the same four frames**, so row, time and
brightness are shared and both controls hold under either hypothesis. On sampleH: suspect pair
**+0.8742** against a 1 µm control of **+0.8738** — identical to four decimals — with the 2 µm
control at **+0.7488**, which proves the measure resolves one step from two. 0 of 20 pairs were
bit-identical, so not a duplicated readout either.

**Conclusion: the stage visited every position exactly once. No gaps, no duplicates — only wrong
labels.** True position is therefore index-derived, `X = X0 + ((N-1) mod NR)`. Validate that it
reproduces the recorded value on every unaffected frame and differs by exactly one step on precisely
the flagged ones (sampleH: 20,121 exact, 180 off by exactly +1, nothing unexplained). Apply it with
`analysis/fix_positions.py` **before `regrain.py`** — raw labels inject one false hole and one
doubled pixel per affected frame directly into the contiguity test the grain count rests on.

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

**The area filter is necessary but NOT sufficient — a very bright spot breaks detection three
different ways.** All three were found on the 34-ID-E Perkin Elmer panel and all three are handled
in `analysis/frame_peaks.py`, the shared detector that `null_model.py` and `parentbeta_validate.py`
both import (so a count and the null it is gated against cannot drift apart).

1. **FLAT-TOP PLATEAUS — the big one, and the least obvious.** A clipped reflection has a *flat*
   top, so every pixel on it equals the local maximum and `sub == maximum_filter(sub)` flags all of
   them: one reflection is reported as dozens of peaks. One 117 px saturated Cu spot produced **58
   detections at identical intensity**; collapsing plateaus cut whole-frame counts by **35–45%**
   (197→111 on sampleD, 189→62 on the bare-Cu reference, 228→145 on a sampleA scan). This inflates the peak
   list, inflates any null built from it, and buries weak neighbours. One connected saturated region
   is ONE reflection — and position it from the **unsaturated shoulders** (the 20–90%-of-clip band),
   which is also *less* biased than the plateau centroid whenever the spot is asymmetric.
2. **ISOTROPIC HALO.** The wing of an intense reflection, measured in ADU above background at
   15/25/40/60/100 px: 1480/508/74/34/14 along the column and 2142/363/79/20/4 along the row, with
   frame noise σ = 50. Vertical and horizontal decay *together*, so it is a halo, not a streak. The
   standing background (a 25 px median on a 4× downsample, ~100 px scale) cannot follow something
   decaying over 40 px, so it leaks into the residual, raises the local bar and manufactures maxima.
   Subtract an azimuthal radial profile per bright spot. This is what lets a weak neighbour survive:
   on that frame the nearer neighbour (I = 579) sat on ~500 ADU of halo.
3. **VERTICAL BLOOMING.** Charge overflow running a bright column hundreds of rows from its source.
   Real but **rarer than expected** — present on the bare-Cu reference frame, absent from every sampleD
   frame sampled. Remove it by **shape**: a morphological opening that erases structures thin in one
   axis and long in the other cannot touch a compact reflection wherever it sits.

**Never delete a detection; flag it.** Position stays valid for a clipped peak even when intensity
does not, so indexing can use it and intensity analysis can skip it. Two filters that *did* delete
were both wrong, and only measurement revealed it: one keyed on tall-and-narrow columns with no
saturation test and removed ~40 real reflections per frame from ordinary crowded columns; another
fabricated ~34 reflections per frame by treating the panel's permanently hot pixels as clipped
reflections. **A clipped reflection has unsaturated shoulders; a hot pixel jumps straight from full
scale to background** — that, plus a minimum area, is what separates them.

This matters beyond bookkeeping. On deposit-on-substrate(111) the substrate reflection saturates and the
reflections sitting a few tens of pixels from it are exactly the ones carrying the orientation
relationship, so a filter that clears the neighbourhood of a bright spot destroys the measurement it
was meant to protect. Verify explicitly that near-neighbours survive: on sampleD they sit at r = 22.0,
29.8 and 53.3 px and must come through unchanged.

The indexer is *not* affected by any of this — it runs its own percentile + `MinArea` + watershed
detection. Two separate code paths; check which detector produced a peak count before comparing.

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

**3. EVERY concurrent shard needs its OWN `ResultDir`, i.e. its own params file.** The daemon writes
its raw `solutions.txt` and `spots.txt` into the **params** `ResultDir`. `--output-dir` overrides
only the *orchestrator's* per-frame `output.h5` tree, **not** this — so the widely-repeated note
that "ResultDir in params is ignored" is true only of the per-frame outputs and is a trap if you
generalise it. Point two shards at one params file and both daemons append to the same
`solutions.txt`/`spots.txt` and interleave. It then fails **late and silently**: indexing runs to
completion and reports `Pipeline complete`, and only post-processing dies, on torn lines
(`got 19 columns instead of 12` from two collided records, `got 2 columns` from a truncated one).
Nothing is recoverable, because each orchestrator numbers its images `1..N` independently, so the
interleaved rows cannot be attributed back to a shard. This cost three 3,400-frame shards on the
bt_34ide_jul26 campaign. Generate params per shard (the sampleH campaign's `params_*_run_s1..s7.txt` exist for
exactly this reason) and **assert the `ResultDir` set is unique before dispatching**.

Machines that can see `$LAUE_ROOT` and have the epix34id LaueMatching install
(`/home/beams/EPIX34ID/opt/LaueMatching`, shared home, has the `LAUE_STREAM_PORT` fix):

| host | RAM | cores | GPUs | notes |
|---|---|---|---|---|
| copland | 2015 GB | 96 | 2x A6000 48 GB | **cannot even READ** the-analysis-host (not merely write) -- unusable for indexing this data, despite the RAM |
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

**But connected components CHAIN, and the price is real.** "Within tol of *some* neighbour" links
A–B and B–C at 0.9° each even when A–C are 1.8° apart, so a "1.0° cluster" can span several degrees.
On sampleH a 924-position cluster had **2.65° median internal spread (6.17° max)** and failed the
raw-image test at chance (peak at the predicted position in 3 of 12 cells, 1.1 expected) while each
of its positions indexed perfectly on *its own* orientation (17–73% vs a 0–7% random null). Per-instance
indexing succeeding while one shared orientation fails is exactly what over-merging looks like.

Use `--diameter` for a **diameter criterion** — complete linkage, so every member is within tol of
every *other* member, not merely of some neighbour. It removes chaining by construction, is
deterministic, and on sampleH **halved the tolerance sensitivity of the grain count, 4.28× → 2.02×**,
while cutting the largest "grain" from 1,833 positions to 1,298 (the 1,833 was the chain). Two
warnings: `regrain.py`'s tolerance sweep silently *changes algorithm* at the 1.0° boundary (supplied
labels at ≥1.0, a greedy **leader** loop below it), so its sweep varies definition as well as cut;
and any residual spread after de-chaining is physics — sampleH's grain population spans 0.08°–2.7°
internal spread, so no single cut is "correct" and every count must carry its tolerance.

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
   **And re-gate against the MEASURED null before you compare — the analytic gate is not enough.**
   On sampleH the analytic-Poisson counts made 99.5 look 1.34× better than 99.8 (794 vs 592), but after
   re-gating at the measured null max the two were **equal** (495 vs 486) for 2.3× the compute, and
   99.5's purity was far worse (62% of its "validated" instances survived, against 82%). The gate
   admits instances down to nhit 5 while the measured null reaches 10–11, so the analytic-validated
   comparison is dominated by exactly the marginal instances the null rejects. Compare after
   `empirical_gate.py`, never before.
5. **A texture null must be indexability-matched.** Detector coverage, the energy window and the
   reflection list all make some orientations easier to index than others, so a peaked pole figure
   can be an artefact of what is *indexable*. Compare against random orientations passed through the
   same "at least MinNrSpots reflections on the detector" filter, not a flat sphere — and use one
   representative per grain, since one grain contributes many positions and instances are not
   independent samples.
6. **Max MRD is binning-dependent — never quote it raw.** The peak of a pole/IPF density grows with
   the number of cells (fewer grains per cell → higher peak from sampling): on Zn/Zn the substrate
   c-axis "max MRD" ran 1.9→10.7 as cells went 128→4608. Only the peak measured against a null binned
   *identically* is meaningful, and a real texture must clear that null at *every* binning — if the
   p-value flickers around chance as you rebin, there is no texture (just sampling noise).
   **Identical binning is not sufficient — the null must also be SUBSAMPLED to the measured n.**
   Max MRD rises as sample size falls, so 1,132 measured orientations against a 20,000-orientation
   null on the same grid is meaningless even at matched cells: it produced a spurious "126.9 MRD vs
   4.3" on sampleH, where almost every cell held 0.12 poles and one cell catching a handful read >100.
   Coarsen the bins to suit n, smooth, subsample the null to the same count (median over draws), and
   quote the **ratio**: sampleH's basal texture is 8.34/1.98 = 4.2, which independently matches
   `texture_null.py`'s 29.51/6.11 = 4.8. Agreeing ratios from two implementations is the check;
   the absolute MRD is not comparable across methods.
7. **Texture needs enough INDEPENDENT grains — and a 1-µm map of coarse grains has very few.** A dense
   step oversamples each grain many times; the independent-orientation count is set by *area / grain
   area*, not by the number of points. Zn/Zn: a 200×200 µm map at 1 µm held only ~350 independent
   substrate grains (8-µm grains oversampled ~8×) — far short of the ~5,000–10,000 for an ODF, so a
   flat texture result there is undersampling, not absence. For a texture survey, set the **step ≈ the
   grain size** (each point a fresh grain) and spend the points on *area*: the same grid at 10 µm
   instead of 1 µm covers 100× the area and ~20,000 grains at the same beamtime. Max step ≈ 2–3× the
   grain size before you skip the fine tail and bias toward the coarsest grains.

---

## Phase 5 — Report

Two deliverables, from the same numbers:

- **PDF** — LaTeX, one per scan or one per campaign. Every number reported with its null. State the
  measured raster (not the folder name), the null maxima, the grain definition used, and what was
  *not* measurable (e.g. no depth resolution → no per-grain depth).
- **HTML artifact** — the shareable version. Keep slides to one screen each; embed figures as
  data-URI JPEGs (a strict CSP blocks every external request). To update an existing artifact, pass
  its URL back — do not mint a new one for the same deliverable.

### Artifact structure: ONE overview, linking OUT to one page per sample

A single artifact that grows with the campaign becomes a *chronology of the analysis* rather than a
description of the samples, and the experimenter cannot find their own sample in it. On
bt_34ide_jul26 the single page reached 2.3 MB and had to be split under protest from the reader.
**Split it at the second dataset, not when it hurts.**

```
OVERVIEW  (the URL you share first; keep this one stable and re-publish in place)
  ├── what the campaign is, one table of samples with links
  ├── findings that SPAN samples (method problems, cross-sample comparisons)
  └── what is still open
        ├──> per-sample page: sampleA scan 1        ├──> per-sample page: sampleB (deposit + bare)
        ├──> per-sample page: sampleA scan 2        └──> per-sample page: sampleD
```

Rules that make it work:

- **One page per SCAN, not per specimen**, when scans differ in raster or condition. Two scans of
  one specimen get two pages and are compared *in the overview*, never silently averaged.
- **Combine only what the reader treats as one question** — deposit and its bare substrate belong
  on one page because they are read against each other.
- **Each page is self-contained**: it repeats the method section and the caveats. Readers arrive
  from a link, not from the overview, and a page that assumes the overview was read will be
  misread.
- **Keep the per-sample pages descriptive**: what orientations are there and where. Cross-cutting
  interpretation (relationships between phases, comparisons between samples) lives in the overview.
  When the experimenter says "just the orientations, we don't need the relationship" — that is
  exactly this split, and it is the right instinct.
- **Every per-sample page carries the same three diagnostics**, because they travel: the tolerance
  sweep, the effective (Kish) n beside the nominal grain count, and the count of objects spanning
  more than half the map. A reader comparing two samples needs to know that one has effective n=51
  and the other n=7.5.
- **Link back** from each page to the overview; put the sample links in a grid near the top of the
  overview, not buried at the bottom.
- **Export the numbers next to the pictures** — a `<key>_grains.csv` per sample with grain id,
  position, size and the full orientation matrix. "Orientations extracted" usually means the reader
  wants the table, not only the map.
- Generate all pages from ONE builder with a shared stylesheet (`build_reports.py` pattern:
  `dataset_page(key, ...)` reading a per-dataset `_stats.json`), so a fix to the method text or the
  palette lands everywhere at once.
- Publish the per-sample pages FIRST, collect their URLs, then build the overview with the links in
  it. The overview is re-published in place afterwards whenever a sample page changes.
- **Every spatial map is drawn to TRUE SCALE — `aspect="equal"`, never `aspect="auto"`.** A
  stretched map misrepresents grain shape and elongation, which is exactly what the reader is
  looking at. On bt_34ide_jul26 a 100 × 150 µm scan was rendered nearly square and a beamline scientist
  caught it before we did. Size the figure from the map's own aspect ratio so equal-scale panels
  do not leave a band of whitespace; where two panels share coordinates, give them the same
  extent and the same aspect.

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

- **Laue footprint (orientation coherence)** — the size of a contiguous single-orientation cluster;
  the cleanest same-phase discriminator (Zn/Zn: substrate = large coherent grains ~213 positions,
  deposit = fine ~36). Footprint measures crystallographic coherence, not necessarily physical grain
  size — a terraced/twinned crystal can over-segment — so if you have an SEM, check whether "fine
  footprint" means genuinely fine grains or a fragmented large crystal. (On Zn/Zn the deposit read as
  fine, and the off-region SEM could not settle which morphology it was; the assignment came from the
  co-registered maps + IPF, not the SEM.)
- **the flat detector background** — do NOT assume it is fluorescence tracking "how much material".
  Split it from the forward-peaked halo by corners-vs-centre, then TEST what it tracks: if it scales
  with grain size / Laue footprint it is **diffuse scattering** from disordered material (fluorescence
  from a thick sample is escape-depth *saturated* and therefore blind to grain size). On Zn/Zn the flat
  floor tracked footprint (corr −0.33, spatial p=0) and is scattering — HIGHER over the rough,
  fragmented deposit — not a path-length/fluorescence gauge. Decide which mechanism by the grain-size
  dependence, not by assumption.
- **per-spot energy** — `Phase.project(OM, with_energy=True)`, or from the stored per-spot `hkl`
  (spots-table cols 3,4,5) and its grain's orientation. In reflection geometry a reflection from under
  an overlayer round-trips through it and hardens; but this is a SEPARATE effect from the background
  and was too small to detect on Zn/Zn (thin deposit, wire parked).

Use them together: two independent observables that must move in the predicted directions is a far
stronger claim than either alone, and each needs its own null (§ below).

**Do not let a threshold define the groups you then test.** It is tempting to *label* each
orientation substrate/deposit by a footprint-or-energy cutoff and then compare the two labels'
energies or footprints — but that contrast is **circular**: the label was built from the very
quantity being contrasted (on Zn/Zn the circular split reads a spectacular −0.64 keV / −0.73, an
artefact of the cutoff, not a measurement). The honest test asks whether two *independently
measured* signatures **agree**: `corr(log footprint, median energy)`, footprint from the clustering
geometry and energy from the spot wavelengths, nothing shared. On Zn/Zn that is −0.10 (wrong sign,
r²~1%): the layers are real on the *map* but not separable *per position*. Keep any threshold-defined
split only as map colouring, and store the independent-test statistic (and a warning) next to it —
see `separate_layers.py`.

**State the aggregation of every map correlation, and keep it fixed.** "footprint vs pedestal" is
two different numbers: *per grain* (one point per cluster, −0.12 on Zn/Zn) and *per scan position*
(one point per occupied pixel, −0.33). Both are legitimate and point the same way, but they are not
interchangeable — the per-position measure is the one comparable to the optical/ground-truth map. Pick
one aggregation for the narrative claim and use it everywhere, labelled. (The *sign* of the
optical-comparison also depends on the registration flip — see invariant #11; anchor direction on
ground truth, not on which flip maximises |corr|.)

### Deposit on a SINGLE-CRYSTAL substrate — peel before you ask about epitaxy

The most natural question about a deposit on a single-crystal substrate ("is there an orientation
relationship?") is the one this pipeline answers *wrongly by construction*, and it answers it with
a large, clean-looking effect. **Validation scores PREDICTED reflections**, not distinct observed
peaks. So when a candidate orientation of the deposit can be rotated to overlay the substrate, it
collects the substrate's peaks as evidence for itself.

Two mechanisms, both measured on bt_34ide_jul26 sampleD (Zn electrodeposited on the fcc substrate):

- **Harmonic stacking.** A Laue spot's position depends only on the *direction* of **g**. Putting
  Zn's c\* on Cu[111] — i.e. exactly the epitaxial relationship under test — sends all seven Zn
  (000ℓ) harmonics inside the 5–30 keV window onto the **single** the fcc substrate pixel (separation
  0.00 px). One observed substrate peak then scores ~7 for the Zn candidate.
- **Generic vector coincidence.** 52.8% of Zn *hki*0 vectors sit within 0.1° of some Cu vector, so
  the overlay is rewarded far beyond the harmonics.

The result: the deposit orientation that best *overlays* the substrate wins on score with no
deposit diffraction present at all. On sampleD this produced "26.4% of 53 grains within 5° of Cu⟨111⟩
at 18.3× the null", which met a pre-registered bar and was **retracted**: removing Cu-explained
peaks dropped the epitaxial grain's pass rate 72%→20%, and the largest apparent Zn grain (56% of
all validated Zn) went 94%→0%. Honest ratio 2.3×.

**Also**: a (111)-polished substrate is parallel to (111) *by construction*, so out-of-plane
alignment of the deposit with the surface normal carries **no lattice information**. Only the
in-plane relationship (e.g. Zn⟨11-20⟩‖Cu⟨110⟩) tests epitaxy.

**Procedure — peel the substrate, then re-index:**

1. Measure the substrate orientation from its own gated instances (dominant cluster).
2. **Measure how far its predicted spots wander across the raster** — do not assume a mask radius.
   Project the cluster's orientations, nearest-neighbour match to a reference projection, and take
   the p99 of the displacement. It differs wildly between scans that look alike: sampleD 6.2 px median
   (p99 12.4), sampleA scan 1 **19.0 px median (p99 89.1)**. A fixed 15 px disc is right for the first
   and leaks the substrate straight back in for the second — recreating the artefact while looking
   like it removed it. (Ignore the *max*: when a reflection leaves the energy window the
   nearest-neighbour match jumps to a different spot.)
3. Build the mask as the **union** of the discs predicted by the cluster's own orientations, after
   dropping the most deviant ~5% so a few bad fits cannot inflate it. Spots that wander get an
   elongated mask, stable spots stay tight, and the geometry sets the shape. sampleA scan 2: 1.14% of
   the detector.
4. **Fill masked pixels with the local background, never zero** — a field of zeros skews the
   indexer's percentile threshold and manufactures false edges.
5. **Rebuild the background from the peeled frames.** Reusing the unpeeled background leaves the
   substrate in the background model and partly undoes the peel.
6. Re-index the deposit, and report the mask fraction next to the result.

The peel also removes any genuine deposit reflection that coincides with a substrate one. That is
unavoidable and it is the point: the test becomes **stricter**, which is the direction it must err.
A relationship that survives is real; one that vanishes is consistent with either artefact or
over-masking, and the mask fraction is what separates those. The real fix is to score distinct
**observed** peaks.

## Invariants (violate these and the result is wrong but looks fine)

1. Measure the null **on the scan in hand**. Never inherit one.
2. Measure the raster from the **stage coordinates**. Never trust a folder name.
3. A grain is **contiguous** and consistent in orientation.
4. Every reported number carries its null; anything else is an intermediate.
5. Detect on the aggressive threshold, **verify on the full background-subtracted frame** — no
   signal is discarded from the evidence.
6a. **A status check that only greps for the success marker cannot see a dead run.** A daemon that
    died six seconds in looks *identical* to one still working, so a monitor watching only for
    "complete" stays quiet through the whole failure — on the bt_34ide_jul26 campaign a shard sat reported
    as `RUNNING` for 25 minutes after `GPUassert: initialization error` killed it. Check every
    terminal state (daemon exit, CUDA error, post-processing traceback, `Pipeline complete` with
    **zero** result files) and treat an unchanged log older than ~15 min as stale, not running.
6b. **A liveness probe that greps the remote process table self-matches through a tcsh login
    shell.** `ssh -n host "pgrep -f params_foo.txt"` runs `tcsh -c pgrep -f params_foo.txt` on the
    far side, and that wrapper's own command line *contains the pattern*, so pgrep matches itself
    and reports every finished run as still alive — the exact inverse of 6a, and it hangs an
    unattended watcher forever. Require something in the matched line that the wrapper cannot
    contain (the binary name: `ps -eo args= | grep -F params_foo.txt | grep -q LaueMatchingGPUStream`).
    Better still, take completion from the orchestrator's own `Streaming: 100%` line rather than
    from process presence.
6c. **Output-file count is NOT a completion test.** A frame that yields no solution writes no
    `.output.h5`, so a healthy finished run legitimately shows fewer outputs than frames (2766/3400
    peeled, 3189/3232 unpeeled). Waiting for `n == N` hangs forever on a perfect run; waiting for a
    *static* count mistakes a slow run for a dead one. Use the orchestrator's completion line, and
    use the count only to spot a run that ended **short**.
6. **Suspect success.** Most of the bugs in this pipeline reported success: a daemon killed while
   healthy, a batch flag silently ignored, a drain that stopped before the file finished writing, a
   dict mutated during serialization, `>` refused by tcsh `noclobber`, an image server "running" for
   ten minutes after its socket had timed out, `BETA_CONFIG=""` falling through to the default and
   exiting 1 *after* alpha had already launched, and a shard driver logging "all 7 launched" with
   **three** running because `ssh` inside a `while read` loop eats the loop's stdin and swallows the
   remaining plan lines. (Use `ssh -n` and a `for` loop over an array — and verify **per shard**,
   never from the launcher's own log.) `scripts/tests/test_streaming_regressions.py`
   pins the fixes; run it after touching the streaming path.
7. **Verify a new implementation against the incumbent, on real data, at the level of the decision
   that matters.** Agreement "to 1e-12" is not the claim; "no pair changes side of the 1.0 deg cut"
   and "0/2000 draws change their Poisson verdict" are. Synthetic tests share your assumptions;
   real data does not.
8. **Empty is not zero.** `ls */*.h5 | wc -l` past ~40k files hits ARG_MAX and reports 0. A count of
   zero output files mid-run is normal (post-processing writes them in one late batch). Neither is
   evidence of anything.
9. **A threshold that defines a group makes every contrast on that group circular.** If you split
   into A/B by a cutoff on X, "A differs from B in X" is guaranteed and meaningless. Test with an
   *independently measured* signature instead, and store its statistic beside any threshold-defined
   split so no reader mistakes the split for a result.
10. **A map-to-map correlation needs a SPATIAL null, never a plain permutation.** Adjacent scan
    positions are strongly autocorrelated, so the naive SE (and its p) is optimistic by orders of
    magnitude — on the Zn/Zn map the effective n was ~70, not 40,357 (naive p ~600x too small). Use a
    toroidal-shift or block null: shift one map relative to the other and rebuild the correlation
    distribution. Only a correlation that clears that spread is real (Zn/Zn footprint↔background −0.33
    cleared it, p=0).
11. **You CANNOT pin a registration flip / direction from the correlation magnitude.** The sign of a
    map↔image correlation is set by the flip you chose, and BOTH signs are reachable with a strong
    |corr| (Zn/Zn: identity flip gave footprint↔deposit −0.5, the both-flip gave +0.45 — equally
    "strong", opposite meaning). Choosing "the flip with the biggest |corr|" is circular and flip-flops.
    Anchor the direction on something INDEPENDENT of the correlation: the experimenter's direct
    observation, the SEM/optical morphology, or two *different* maps that must agree (Zn/Zn: background
    AND footprint both mark the same deposit). This cost three re-flips before it was learned — the
    correlation refines the *fit*, it does not decide the *direction*.
12. **Don't trust the printed optical scale bar; fit the scan→image transform from the alignment.**
    On Zn/Zn the bar was ~2x off. Solve for scale (allow anisotropy), offset, AND rotation. And know the
    ceiling: correlating a continuous map against a binary optical mask tops out around ~0.5 even at
    perfect registration — a strong *visual* overlay (blobs on blobs) is not the same as a high pixel
    correlation, so report ~0.5 as the real relationship and do not chase it higher (that fits noise).
13. **A detector artefact can pass every statistical test you have.** The panel blooms out of
    saturated reflections; an unfiltered local-maximum detector stacks detections down the bloom,
    and because the bloom sits at a FIXED position for a fixed reflection it reproduces perfectly
    frame to frame. That is indistinguishable from a persistent single-crystal signal by
    persistence alone, and identical spurious spots on every frame can hand an indexer the same
    wrong orientation every time — a fake result with a beautiful null. Look at the raw overlay
    before trusting any "recurs in every frame" claim, and separate the two detectors: on the
    34-ID-E bare-Cu frame the analysis-side peak count was inflated 2x while the indexer's own
    solution was clean (16 of its 18 assigned spots off-streak, orientation moving 0.011° when the
    other 2 were discarded). Filter blooms in analysis code (`frame_peaks.py`), and confirm any
    fixed-position claim against something the artefact cannot fake, such as agreement in
    ORIENTATION space across many frames.
14. **A null result is only as good as its power — state the effect size it can exclude.** "No texture"
    / "no hardening" from p>0.05 means nothing without the detectable-effect-size. The Zn/Zn texture
    test could exclude only peak-MRD excess >~1 (moderate fibre texture would have passed unseen); the
    under-deposit within-grain test had 11% power at the 0.1 keV effect it was used to dismiss, and its
    95% CI *contained* the population signal. Report "excludes effects larger than E", not "no effect".
    A Wilcoxon p=0.90 from an underpowered test is not evidence of absence.
15. **Scoring PREDICTED reflections makes the pipeline reward a phase for peaks it did not
    produce.** Any candidate orientation whose pattern can be rotated to overlay a strong
    single-crystal substrate is systematically favoured, and harmonics along a shared axis stack
    many predicted reflections onto one observed pixel. This fabricated a complete, pre-registered
    epitaxy result on sampleD (see §Deposit on a single-crystal substrate). It is **not** specific to
    Zn/Cu: it applies to any two-phase sample with a dominant single crystal, and it also drives
    plain phase misindexing — on sampleD the largest apparent "Zn grain", 56% of all validated Zn
    instances, was Cu misindexed as Zn. Before any two-phase orientation claim, peel the dominant
    phase and re-index. Until scoring counts distinct **observed** peaks, no
    deposit-on-single-crystal orientation result from this pipeline is trustworthy.
15b. **GATE ON DISTINCT OBSERVED PEAKS, NOT ON MATCHED PREDICTED REFLECTIONS — the pipeline
    already writes both.** `entry/results/unique_spots_per_orientation` is the distinct-observed
    count; `filtered_orientations[:,6]` is the matched-predicted count. **Join them on
    `filtered_orientations[:,1]` ↔ `unique_spots_per_orientation[:,0]`** (the orientation id
    within the frame). `filtered_orientations[:,0]` is the IMAGE number — joining on it silently
    returns zeros, which showed up as a known single crystal having "0 distinct spots" and is how
    the mistake was caught. Sanity-anchor every such analysis on a phase you know is there.
    Measured stacking (matched ÷ distinct) on bt_34ide_jul26: **Cu 1.1×, sampleB Al 1.2×, sampleA Zn 1.2×,
    sampleD Zn 1.5×** — modest in general, so do NOT generalise the extreme harmonic stacking of a
    substrate-overlaying orientation to all solutions. Re-gating on distinct spots against a
    null measured the same way roughly **doubled** the accepted Zn (sampleD 5,665 → 10,493;
    sampleA s1 1,775 → 4,405) while leaving the known Cu single crystal **unchanged at 1.0×, 100% of
    frames** — the correct signature of a better gate. Much of an "unindexed" fraction is a
    gating choice before it is a missing capability: check the gate before building iterative
    indexing.
16. **A missing background file does not fail — the image server silently computes one FROM THE
    FIRST FRAME.** `mkbg_gen.py` builds frame paths as `{prefix}{i}.h5` with *plain* integers,
    which is how raw beamline frames are named; derived frames (substrate-peeled, denoised,
    re-binned) written zero-padded (`g31p_000001.h5`) make it raise `FileNotFoundError`. If that
    traceback scrolls past, the daemon starts, finds no `BackgroundFile`, logs
    `Computing background from first frame...` at INFO, writes it to the expected path, and every
    later shard then loads that file and reports "used supplied background". One frame's own Laue
    spots and diffuse scattering become the background for the whole raster — the exact opposite
    of the position-neutral median the pipeline requires, and nothing anywhere says the word
    "error". Caught on the bt_34ide_jul26 peeled re-index: the bad background read min 121 / med 263 /
    max 567 against the correct 93 / 214.5 / 441.5. **Check `background_*.bin` exists with a
    plausible median BEFORE dispatching, and grep every `server.log` for
    `Computing background from first frame`.** `mkbg_gen.py` now accepts both naming conventions.
17. **A subset is not a raster.** "~45% of sampleA positions have the substrate extinguished" came from
    a 510-frame sweep subset; the full 2,601-position scan shows Cu at 98.1%. Sweep subsets are
    chosen for threshold tuning, not sampled uniformly — never quote an occupancy, fraction or rate
    from one.

## Worked example

The Zn/Zn electroplated dataset (`bt_34ide_jul26/sampleG/scan1_Laue2D`, 40,401 frames) is the first
non-Ti material through this chain and exercised every phase; its `SURVEY.md` in
`$LAUE_WORK/` is a filled-in template for Phase 0. Headline
numbers, all measured: 201x201 at 1.000 um (the 45-deg trap did **not** bite — stage coordinates
agreed with the folder name for once), wire parked so no depth resolution, 96–167 peaks/frame at
99.8, measured random-orientation null **max 9 hits in 30,000 draws** (the analytic Poisson gate
would have accepted down to 5), and re-gating at nhit>9 kept **83.2%** (171,644 / 206,343) of
"validated" instances.

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
