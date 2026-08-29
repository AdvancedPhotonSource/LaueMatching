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
| station | 34-ID-E | 16-BM-D (HPCAT) |
| panel | edge-on above the sample, normal along lab **+Y** | downstream, centred near the beam |
| direct beam | not on the panel, far off | **just off the panel edge — and NOT at the PONI** |
| pattern runs | vertically | radially about the beam |
| worked example | `LAB_NOTEBOOK.md` | `LAB_NOTEBOOK_16BMD_Si.md` |

The **forward model is the same** for both — `kf = ki − 2(q̂·ki)q̂` is the general Bragg
mirror and has no hemisphere restriction (invariant 23). What differs is everything that
assumed where the pattern sits on the panel: see invariants 23–26 and Phase 2.

It assumes the material port (Phase 6) — lattice, reflection list, detector geometry, energy
window and symmetry all read from the indexing parameter file. **Outside that — a third
geometry, a non-raster acquisition, or a phase whose parameter file does not exist yet — stop
and ask rather than adapting a phase below.**

### Verify the configuration before you start

**Run everything below from `scripts/pipeline/`** — every relative path in this doc set is
written from there, and from the repository root the first command fails with
`No such file or directory`.

The install gate here is the material selftest, because the failure it catches is silent:
symmetry follows the **space group**, not the phase name, and the old rule handed cubic-24
operators to any phase not called `"alpha"`.

```bash
cd scripts/pipeline                              # every path below is relative to here
export LAUE_PHASES=<phase>                       # comma-separated; single-phase is fine
export LAUE_PARAMS_<PHASE>=<path to params.txt>  # upper-case suffix
python analysis/laue_material.py                 # selftest must pass
```

**Check the indexer binary the same way — by asking, not by looking.** `ls bin/` has
reported a binary present and ready that was built for another architecture entirely
(`LAB_NOTEBOOK_16BMD_Si.md` §2). The package answers with the one that will actually run:

```bash
python -c "from laue_index import indexer; print(indexer.available(), indexer.binary_path())"
```

`pip install laue-index` compiles it on this machine; add `LAUEMATCHING_CUDA=1` for the
GPU and streaming binaries, or point `LAUEMATCHING_BIN` at ones you already have. The
orchestrators ship with the package too, so `laue-index run process -c … -i …` works
without a checkout, and `python scripts/RunImage.py …` still works inside one.

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
| [`DIAGNOSIS.md`](DIAGNOSIS.md) | symptom → test → cause → lever | when something looks wrong |
| [`ENVELOPE.md`](ENVELOPE.md) | what this measurement can and cannot determine | before promising an answer |
| [`RUNBOOK.md`](RUNBOOK.md) | where it runs, healthy ranges, current pick-up point | on resume |

### Handbook vs lab notebook

**This file says what to do. The lab notebooks say what was found.** They are kept apart on
purpose: a handbook has to stay short enough to follow, and a campaign record has to stay
honest enough to stop a refuted idea coming back. When a rule below cites a measurement, the
full account — including the controls that killed the competing explanation — is in a notebook.

- [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md) — method and defect record for
  the fcc substrate and Al on Al. Three retracted claims, the image-peel autopsy, the α-brass
  identifiability limit, and the measurement ledger.
- [`LAB_NOTEBOOK_ZnZn.md`](LAB_NOTEBOOK_ZnZn.md) — Zn on Zn. The
  substrate/deposit direction saga (it flipped several times; **read §5 before re-arguing it**),
  scattering-vs-fluorescence, the readback race, and why the layers separate on the map but not
  per position.
- [`LAB_NOTEBOOK_16BMD_Si.md`](LAB_NOTEBOOK_16BMD_Si.md) — **the transmission-geometry
  campaign.** Si wafer at 16-BM-D, six ω settings. The PONI-is-not-the-beam trap, the
  pixel-origin offset that a Procrustes fit turned into a crystal rotation, seven retracted
  claims, and the beam-azimuth gauge that makes absolute orientation unrecoverable. **Read §4
  before quoting any agreement number.**

**Write a new lab notebook per campaign, not per dataset**, and start it on day one — the
retractions are the part that decays fastest. Structure that works: what the campaign
established (a table with a status column) → defects fixed → method findings → scientific
findings → **retracted claims and open questions** → measurement ledger.

Companion docs: [`README.md`](README.md) (the pipeline itself), and — outside this repo — the
34-ID-E operational runbook `laue_torch/report/RUN_PROCESS_REPORT_HANDOFF.md` (beamline access,
the a two-phase hcp/bcc alloy campaign, and its current state).

---


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
18. **Gate the PORT as well as the ResultDir, and gate it on the host.** Two independently-written
    plan files gave the same port to different work directories; the second daemon logged
    `bind: Address already in use` *in the middle of a normal-looking startup*, then sat there
    fully initialised, holding 12 GB of GPU, having received **zero** images. It looks exactly like
    a slow run. `grep -h '' plan_*.txt | awk '{print $3}' | sort | uniq -c` finds the collision
    before dispatch; `ssh -n host "bash -lc 'ss -ltn | grep -q \":$PORT \"'"` finds it on the host.
    Same class as 6a/6c: the failure is silent and mimics progress.
19. **Two different samples can share one raw folder — select frames by PREFIX, never by
    directory.** In `bt_34ide_jul26`, `sampleA/scan2_Laue2D` holds 12,802 files:
    2,601 named `G30_scan2_*` and **10,201 named `F98_scan1_*`** — an entirely different sample.
    A directory glob silently merges two experiments. (The same folder-vs-content trap in the
    other direction: `sampleK_scanpair` holds two scans at different step sizes under one
    prefix.) Build shards from the file prefix and print the frame count per prefix before
    dispatch.
20. **A scan can lose beam partway through and nothing in the file structure says so.** sampleH's
    second scan wrote all 15,251 frames; frames 1–4,053 have a median of ~213 counts and frames
    4,054–15,251 have **2–7 counts** with only the permanently hot pixels left. The background
    built across that raster came out at median 3.0 and the background gate (16) refused it —
    which is how it was found. **Sample the per-frame median across the raster before building a
    background**, index only the live block, and report the recovered area rather than the
    requested one. A stride-25 sample plus a bisect at the edge locates the cut in about a minute.
21. **`h5["/entry1/data/data"][0]` is the first ROW, not the first frame.** One 2048 × 2048 image
    per file is stored as a **2-D** dataset, so `[0]` returns 2,048 pixels of detector edge — all
    zeros. A quick diagnostic written that way printed `min 0 med 0 max 0` for every frame of a
    healthy scan and looked exactly like corrupt data. Use `[:]`, and anchor any "the frames are
    empty" claim on a median compared against the scan's own background before acting on it.
21b. **Streaming completion is NOT output completion.** The orchestrator's `Streaming: 100%` (or a
    full `Received image` count) means the daemon has *seen* every frame; the per-frame
    `.output.h5` files are written afterwards, in a batch. A validator launched on the streaming
    line ran on **275** of sampleK scan 1's eventual **2,498** outputs and reported 686 validated
    instances instead of ~6,000 — with no error anywhere, because it correctly validated
    everything that existed at the time. Before any post-processing step, require the output
    count to be **unchanged across two checks ~30 s apart** and non-trivially large.
22. **A peak WIDTH measured as a second moment is not intensity-independent, so it does not
    escape a flux confound on its own.** The reasoning "amplitude drifts, shape does not" is
    right about the underlying reflection and wrong about the estimator: a fixed-box second
    moment weights the noisy tails, so a dimmer peak measures *wider*. On the homoepitaxial
    series the median widths ran 1.981 / 2.651 / 2.775 px, apparently monotonic with deposit
    thickness — while the median peak intensities ran 1,491 / 1,358 / 1,087, falling in exact
    lockstep. **Two quantities that move together perfectly are one quantity until proven
    otherwise.** The control is to compare width *within matched intensity bins*; only a
    difference that survives that is a property of the crystal. This generalises: any shape
    statistic (width, ellipticity, kurtosis, profile asymmetry) computed over a fixed window
    inherits the SNR of what is inside it.

23. **The forward model is 2θ-agnostic; the things built around it are not.**
    `kf = ki − 2(q̂·ki)q̂` (`LaueMatchingHeaders.h:449-469`) is the general Bragg mirror and
    works unchanged in transmission — the beam is along +Z either way, only the *detector*
    moves. But `GenerateHKLs.py` took θ_max from the **top-centre pixel** in the Y–Z plane,
    which is right only for a panel edge-on above the sample. In transmission the largest 2θ
    is at a **corner** and carries the X component `atan2(Y,Z)` discards: measured hmax
    **35 → 14**, truncating at d ≈ 0.39 Å on data that demonstrably contains d = 0.334 Å.
    Fixed to a four-corner maximum. Before porting to any new geometry, ask of every step
    *"does this assume where the pattern sits on the panel?"*

24. **ON A TILTED DETECTOR THE PONI IS NOT THE BEAM.** The point of normal incidence and the
    point where the transmitted beam meets the panel are different, and at 30° tilt they were
    **751 px apart** — PONI at (502, 526), beam at (−249, 525), off the panel. Every
    "distance from the direct beam" computed against the PONI is wrong. This produced a
    confident, completely wrong reading of a diffuse feature until the beam position was
    computed properly (`LAB_NOTEBOOK_16BMD_Si.md` §5).

25. **A ROTATION FIT CANNOT REPRESENT A TRANSLATION, SO IT LAUNDERS ORIGIN ERRORS INTO
    ORIENTATION ERRORS.** Orthogonal Procrustes / Kabsch has three DOF, all rotational. Given
    a rigid pixel-origin offset it returns a spurious *rotation* and a plausible-looking
    residual. Here a 0.667 px offset became a **0.0423° crystal rotation** — larger than the
    0.0188° agreement it then produced — and the quoted "0.23 px median" was the in-sample
    residual of a fit to the same 99 points used to score it. **Check `mean(dx)`, `mean(dy)`
    before quoting any post-fit residual**; a median can never expose a constant offset. Here
    they were −0.5000 and −0.4419 px, constant to 1e-10, and both had exact explanations
    (pixel-centre vs corner; a documented 0.01 mm white-beam shift).

26. **Two codes can share a convention error invisibly.** Agreement between packages tests
    their algebra — reciprocal metric, q sign, handedness, hkl list — and **nothing** they
    both take from the same input. Measured shared-input sensitivity here: 1 px beam-centre
    error → 0.089° of orientation, 0.1° of detector tilt → 0.191°, 1 mm of distance →
    0.026°, every one larger than the agreement being celebrated.

27. **THE BEAM-AZIMUTH GAUGE IS EXACT AND NO AMOUNT OF LAUE DATA BREAKS IT.** With
    `ki = (0,0,1)`, rotating detector and crystal together about the beam leaves every
    predicted pixel and every predicted energy identical — measured at **φ = 90°**,
    max|Δpx| = **2.3e-13**, max|ΔE| = **0**. A powder calibration cannot fix it either: the
    CeO₂ rings move by 1.9e-14° in 2θ while χ sweeps 30°. So **every orientation any of this
    produces is correct only up to an unmeasured rotation about the beam**, and no cross-check
    internal to the diffraction — not a rotation series, not agreement with another code —
    can establish otherwise. Breaking it needs metrology outside the pattern: a surveyed
    rotation-axis direction, a surveyed detector translation, a knife-edge on a surveyed axis,
    or a plumb/fiducial reading. **Sample translation will not do it.** Recorded in the ledger
    as `project_eiger_1m_laue_calib.md`; missed twice in one day here, which is why it is now
    an invariant.

28. **A null must have the SAME SPATIAL SUPPORT as the data.** "80 % of streak intensity
    within 40 px of a predicted reflection, against 35.7 % for random orientations, 28σ" was
    an artefact: the true orientation puts its reflections in the half of the panel where the
    streaks are, random ones spread over the whole panel including the empty half. The
    statistic measured *"are the reflections in the left half"*. Controls that preserve the
    support — scramble radius keeping angle, or angle keeping radius, about the relevant
    origin — gave **0.5–2.0σ**.

29. **The null's denominator is the SEARCH, not one draw — and its matching criterion must be
    the indexer's.** A gate safe against a single random orientation is not safe against
    best-of-1e8: at 3 distinct spots, ~1,400 of 1e8 reach it by chance here, at 6 it is 0.01.
    Separately, a null measured with a 2 px centroid tolerance on thresholded frames was
    **118× too permissive** for `NMatches`, which `writeCalcOverlap` scores with **zero-pixel
    tolerance** on the daemon's own image (11.5× more acceptance area) and `maxNrSpots *= 3`.
    Measure the null with the criterion, budget and image the indexer actually uses.

30. **Screen detector artefacts on what the INDEXER sees, not on raw counts.** A screen at
    `max(50 × frame_median, 250)` raw counts left a **9–250 count blind band**, because the
    indexer keeps everything above the per-frame 99.8 percentile, median 9 counts. Twenty-two
    pixels lived there, twelve on the first column after a module gap in contiguous runs long
    enough to pass `MinArea 4` as a fixed-position "spot" on every frame. **The discriminator
    for a rotation series is occupancy in the MINIMUM over all settings**: an artefact is
    present at every ω, a reflection cannot be (measured ceiling for real reflections: 59.5 %
    in one scan, ~0 in the others). Masking them removed **821 spurious orientations**.

31. **`midas_stress.misorientation_om` returns the axis in the CRYSTAL frame, folded into the
    fundamental sector.** The signature to read: components sorted descending, all positive.
    Reporting it as a lab-frame direction put a goniometer axis **11.07°** off; a noise-free
    synthetic whose true axis is exactly lab +X returns the same wrong vector. To get the lab
    axis, remove the symmetry variant explicitly. And note the axis scatter is **zero by
    construction** for rotations about a common axis, so it constrains nothing.

32. **Compare a rotation series PER POSITION, never per-scan modal orientation.** Seeding a
    "modal orientation" from each scan's best frame landed on *different* orientation clusters
    in different scans and produced a clean-looking **refutation** (11.9° RMS, no common axis)
    of a result that is real at 0.008°. And when reporting, remember N scans give **N−1**
    independent comparisons, not N(N−1)/2: fifteen pairs here regress onto six per-scan
    offsets at R² 0.97.

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
