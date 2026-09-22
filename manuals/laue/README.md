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

**Scope.** This doc set covers **Laue microdiffraction** through the `LaueMatching` chain: a
raster of frames, a per-frame indexing pass, and the grain/orientation analysis over the
resulting map. Two geometries are covered and they are **not** interchangeable:

| | **Reflection** | **Transmission** |
|---|---|---|
| station | 34-ID-E · **TPS 21A (NSRRC)** | 16-BM-D (HPCAT) |
| panel | edge-on above the sample, normal along lab **+Y** | downstream, centred near the beam |
| direct beam | not on the panel, far off | **just off the panel edge — and NOT at the PONI** |
| pattern runs | vertically | radially about the beam |
| worked example | `LAB_NOTEBOOK.md` · `LAB_NOTEBOOK_TPS21A.md` | `LAB_NOTEBOOK_16BMD_Si.md` |

**TPS 21A is the first non-APS station and the first calibrated in XMAS.** Its geometry is
the 34-ID-E class — 2θ = 90°, k_in = (0,0,1), sample at 45°, panel above — on a PILATUS3 6M
rather than a Perkin Elmer. `laue_index.xmas` converts an XMAS calibration to
`P_Array`/`R_Array` and enumerates the candidate poses; `LAB_NOTEBOOK_TPS21A.md` is the
campaign record. A *fourth* station in this class needs no new code, but does need the
candidate sweep re-run: nothing about the mounting transfers.

The **forward model is the same** for both — `kf = ki − 2(q̂·ki)q̂` is the general Bragg
mirror and has no hemisphere restriction (invariant 23). What differs is everything that
assumed where the pattern sits on the panel: see invariants 23–26 and Phase 2.

It assumes the material port (Phase 6) — lattice, reflection list, detector geometry, energy
window and symmetry all read from the indexing parameter file. **Outside that — a third
geometry, a non-raster acquisition, or a phase whose parameter file does not exist yet — stop
and ask rather than adapting a phase below.**

### Verify the configuration before you start

**Run everything below from `pipeline/`** — every relative path in this doc set is
written from there, and from the repository root the first command fails with
`No such file or directory`.

The install gate here is the material selftest, because the failure it catches is silent:
symmetry follows the **space group**, not the phase name, and the old rule handed cubic-24
operators to any phase not called `"alpha"`.

```bash
cd pipeline                                      # every path below is relative to here
export LAUE_PHASES=<phase>                       # comma-separated; single-phase is fine
export LAUE_PARAMS_<PHASE>=<path to params.txt>  # upper-case suffix
python analysis/laue_material.py                 # selftest must pass
```

**Check the indexer binary the same way — by asking, not by looking.** `ls bin/` has
reported a binary present and ready that was built for another architecture entirely
(`LAB_NOTEBOOK_16BMD_Si.md` §2). The package answers with the one that will actually run:

```bash
python -c "from laue_index import indexer
for t in ('CPU', 'GPU', 'GPUStream'):
    print(t, indexer.available(t), indexer.binary_path(t))"
laue-index doctor                                # GPU binaries vs THIS card's compute capability
```

Ask for all three by name: `indexer.binary_path()` with no argument answers for the **CPU**
binary only, so a gate written that way passes on an install whose GPU and streaming
binaries (the ones a sharded run actually uses) are missing. `laue-index doctor` checks
the build's recorded architecture list against the card; `DIAGNOSIS.md` §"An indexing run
that finishes cleanly and finds nothing" has the by-hand `cuobjdump` check.

`pip install laue-index` compiles it on this machine. The GPU and streaming binaries are
attempted by default when `nvcc` is on PATH (`LAUEMATCHING_CUDA=require` makes a failed
CUDA build fail the install instead of quietly skipping it); or point `LAUEMATCHING_BIN` at
binaries you already have. The orchestrators ship with the package too, so
`laue-index run process -c … -i …` works without a checkout, and
`python scripts/RunImage.py …` still works inside one.

**`pipeline/run_laue.sh` needs `SCRIPTS` pointing at the directory that holds
`laue_orchestrator.py`**: in a checkout, `<repo>/scripts`. In 0.7.1 its default resolves to
the repository **root**, the launch is backgrounded, and the "can't open file" goes to a log
while the script prints a pid as if the run had started: set `SCRIPTS=<repo>/scripts`
explicitly. From 0.7.2 the default is `<repo>/scripts` and the script refuses to launch if
`laue_orchestrator.py` is not there.

### When to stop and come back with a question

**"Get back to me if you get stuck" does not fire here.** A wrong registration flip
produces a strong correlation with the opposite meaning. A detector artefact at a fixed
position reproduces perfectly frame to frame and hands the indexer the same wrong
orientation every time — a fake result with a beautiful null. Both finish and look right.

**Halt on these named conditions, whether or not anything seems wrong:**

| Condition | Why you cannot decide it yourself |
|---|---|
| the registration direction rests on correlation magnitude | both flips reach a strong \|corr\| with opposite meaning; it needs an *independent* anchor (invariant 11) |
| no null measured **on the scan in hand** | an inherited null is not a null (invariant 1) |
| the raster came from a folder name rather than stage coordinates | a folder named `10x10um` measured 20.000 × 14.142 µm at 45° (Phase 0) |
| a scan is still being written | count frames twice, 120 s apart, before indexing anything |
| a contrast is drawn on the quantity that defined the split | guaranteed and meaningless; needs an independently measured signature (invariant 9) |
| the phase has no parameter file, or the selftest fails | the analysis would silently describe a different material |

When you halt, say which row fired, what you measured, and what you would need in order to
proceed. Finish everything not blocked by it first.

### The doc set — what to read when

The phases below carry the actual commands. This spine names them by number throughout and,
until 2026-08-12, never linked them: a fresh session that trusted this file to be "the one
you keep loaded" read the invariants and the worked example and never learned the procedure
existed in separate files. It only found them by listing the directory.

| file | covers | read |
|---|---|---|
| [`phase-0-survey.md`](phase-0-survey.md) | survey the experiment folder before promising anything | first |
| [`phase-1-science.md`](phase-1-science.md) | what science is askable — the part that cannot be automated | before configuring |
| [`phase-2-configure.md`](phase-2-configure.md) | params, per-material inputs, the orientation database | before indexing |
| [`phase-3-index.md`](phase-3-index.md) | sharded indexing across GPUs and hosts | the long step |
| [`phase-4-analyse.md`](phase-4-analyse.md) | nulls, gating, grain definition, tolerance sweep | after indexing |
| [`phase-5-report.md`](phase-5-report.md) | PDF and artifact structure, overview + per-sample pages | at the end |
| [`phase-6-material.md`](phase-6-material.md) | material configuration — a verification step, not a port | when adding a phase |
| [`INVARIANTS.md`](INVARIANTS.md) | the full text and evidence of every invariant indexed below | before acting on one |
| [`DIAGNOSIS.md`](DIAGNOSIS.md) | symptom → test → cause → lever | when something looks wrong |
| [`ENVELOPE.md`](ENVELOPE.md) | what this measurement can and cannot determine | before promising an answer |
| [`RUNBOOK.md`](RUNBOOK.md) | where it runs, healthy ranges, current pick-up point | on resume |

### Handbook vs lab notebook

**This file says what to do. The lab notebooks say what was found.** They are kept apart on
purpose: a handbook has to stay short enough to follow, and a campaign record has to stay
honest enough to stop a refuted idea coming back. When a rule below cites a measurement, the
full account — including the controls that killed the competing explanation — is in a notebook.

- [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md) — the reflection-geometry record, merged across both
  34-ID-E deposit-on-substrate campaigns and carrying only what transfers: the detector
  artefacts and operational traps (§2), why image-space substrate removal **cannot** work and
  what replaced it (§3a–3b), three retracted claims and what killed them (§4). **Read §3b
  before re-arguing which grains are substrate and which are deposit** — that direction
  flipped several times before stage-invariance settled it.
- [`LAB_NOTEBOOK_TPS21A.md`](LAB_NOTEBOOK_TPS21A.md) — **the XMAS-calibrated reflection
  campaign.** TPS 21A, three datasets and three beamtimes. The XMAS→`P_Array`/`R_Array`
  conversion and its three traps, the two mount degeneracies (one an exact gauge, one a
  near-gauge that looks decidable and is not), the energy window as the binding constraint
  for a small-cell phase, and four measurement errors caught by controls rather than by
  inspection. **Read §3a before claiming any Laue pattern settles detector handedness.**
- [`LAB_NOTEBOOK_16BMD_Si.md`](LAB_NOTEBOOK_16BMD_Si.md) — **the transmission-geometry
  campaign.** Si wafer at 16-BM-D, six ω settings. The PONI-is-not-the-beam trap, the
  pixel-origin offset that a Procrustes fit turned into a crystal rotation, seven retracted
  claims, and the beam-azimuth gauge that makes absolute orientation unrecoverable. **Read §4
  before quoting any agreement number.**

**Write a new lab notebook per campaign, not per dataset**, in the campaign's own directory,
and start it on day one — the
retractions are the part that decays fastest. Structure that works: what the campaign
established (a table with a status column) → defects fixed → method findings → scientific
findings → **retracted claims and open questions** → measurement ledger. The public notebooks
here are the other half of the rule: what transfers from a campaign notebook is merged into
one public notebook per geometry or station (`pipeline/Laue_Handbook.md`), which is why
`LAB_NOTEBOOK.md` covers two 34-ID-E campaigns.

Companion doc: [`RUNBOOK.md`](RUNBOOK.md) — healthy ranges and the current pick-up point.
Site-specific operational detail (beamline access, which host runs the daemon, a campaign's
own state) is deliberately kept out of this public tree; see `RUNBOOK.md` §R1.

---


## Invariants (violate these and the result is wrong but looks fine)

The full text, with the evidence behind each rule, is in **[`INVARIANTS.md`](INVARIANTS.md)**;
read the entry before acting on any rule below that bears on your step. Numbers are stable
and are what the rest of the doc set cites.

| # | rule |
|---|---|
| 1 | Measure the null on the scan in hand. Never inherit one. |
| 2 | Measure the raster from the stage coordinates, never from a folder name. |
| 3 | A grain is contiguous and consistent in orientation. |
| 4 | Every reported number carries its null; anything else is an intermediate. |
| 5 | Detect on the aggressive threshold, verify on the full background-subtracted frame. |
| 6a | A status check that only greps for the success marker cannot see a dead run. |
| 6b | A remote process-table grep self-matches through a tcsh login shell. |
| 6c | Output-file count is not a completion test. |
| 6 | Suspect success: most bugs in this pipeline reported success. |
| 7 | Verify a new implementation against the incumbent, on real data, at the decision that matters. |
| 8 | Empty is not zero (`ls | wc -l` past ARG_MAX reports 0). |
| 9 | A threshold that defines a group makes every contrast on that group circular. |
| 10 | A map-to-map correlation needs a spatial null, never a plain permutation. |
| 11 | A registration flip cannot be pinned from the correlation magnitude. |
| 12 | Fit the scan-to-image transform; do not trust the printed optical scale bar. |
| 13 | A detector artefact can pass every statistical test you have. |
| 14 | A null result is only as good as its power: state the effect it can exclude. |
| 15 | Scoring predicted reflections rewards a phase for peaks it did not produce. |
| 15b | Know which spot count you gate on: there are three (glossary here). |
| 16 | A missing background file is silently computed from the first frame. |
| 17 | A subset is not a raster. |
| 18 | Gate the port as well as the ResultDir, and gate it on the host. |
| 19 | Select frames by prefix, never by directory: two samples can share a folder. |
| 20 | A scan can lose beam partway through and nothing in the file says so. |
| 21 | `h5["/entry1/data/data"][0]` is the first row, not the first frame. |
| 21b | Streaming completion is not output completion. |
| 22 | A second-moment peak width does not escape a flux confound on its own. |
| 23 | The forward model is 2θ-agnostic; the things built around it are not. |
| 24 | On a tilted detector the PONI is not the beam. |
| 25 | A rotation fit cannot represent a translation; it launders origin errors into orientation. |
| 26 | Two codes can share a convention error invisibly. |
| 27 | The beam-azimuth gauge is exact; no amount of Laue data breaks it. |
| 28 | A null must have the same spatial support as the data. |
| 29 | The null's denominator is the search, and its matching criterion must be the indexer's. |
| 30 | Screen detector artefacts on what the indexer sees, not on raw counts. |
| 31 | `midas_stress.misorientation_om` returns the axis in the crystal frame, fundamental sector. |
| 32 | Compare a rotation series per position, never per-scan modal orientation. |
| 33 | Before blaming the indexer, count what the energy window can deliver. |
| 34 | A null statistic must be monotonic in the thing it tests. |
| 35 | `ResultDir` must be unique per run. |
| 36 | `NMatches` is not evidence of independence; the unit of uniqueness is the reflection. |
| 37 | A spatial-coherence null cannot validate a finely-stepped raster. |
| 38 | A simulated laue_torch frame is transposed relative to a real one, silently. |
| 39 | A grain count is a property of the grain definition; quote it with its range. |

## Worked example

The Zn/Zn electroplated dataset (`bt_34ide_jul26/sampleG/scan1_Laue2D`, 40,401 frames) is the first
non-Ti material through this chain and exercised every phase; its `SURVEY.md` in
`$LAUE_WORK/` is a filled-in template for Phase 0. Headline
numbers, all measured: 201x201 at 1.000 um (the 45-deg trap did **not** bite — stage coordinates
agreed with the folder name for once), wire parked so no depth resolution, 96–167 peaks/frame at
99.8, measured random-orientation null **max 9 hits in 30,000 draws** (the analytic Poisson gate
would have accepted down to 5), and re-gating at nhit>9 kept **83.2%** (171,644 / 206,343) of
"validated" instances.

These numbers are **sampleG's** (201 × 201 = 40,401 frames). sampleH, the other Zn/Zn scan
of the campaign and the texture case, is a different raster: scan 1 is 20,301 frames
(Phase 0, the readback race) and scan 2 is 15,251 frames of which 4,053 are live
(invariant 20). Its null is its own: `nhit` max 10 on the full raster in 60,000 draws
(invariant 15b). Do not quote sampleG's null for sampleH or the reverse (invariant 1). The
nulls here are of `nhit`, the stacked count (15b).

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
