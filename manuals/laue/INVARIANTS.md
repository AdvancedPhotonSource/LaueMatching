# Invariants

Violate these and the result is wrong but looks fine. This file holds the full text,
with the evidence for each; [`README.md`](README.md) carries the one-line index and is the
file you keep loaded. **Numbers are stable**: code comments, `DIAGNOSIS.md`, the phase
files and the `laue` skill cite invariants by number, so a new invariant takes the next
free number (or a letter suffix beside its family) and none is ever renumbered.


1. Measure the null **on the scan in hand**. Never inherit one.
2. Measure the raster from the **stage coordinates**. Never trust a folder name.
3. A grain is **contiguous** and consistent in orientation.
4. Every reported number carries its null; anything else is an intermediate.
5. Detect on the aggressive threshold, **verify on the full background-subtracted frame** — no
   signal is discarded from the evidence.
6a. **A status check that only greps for the success marker cannot see a dead run.** A daemon that
    died six seconds in looks *identical* to one still working, so a monitor watching only for
    "complete" stays quiet through the whole failure — on the bt_34ide_jul26 campaign a shard sat reported
    as `RUNNING` for 25 minutes (source not in repo) after `GPUassert: initialization error` killed it. Check every
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
    peeled, 3189/3232 unpeeled; `LAB_NOTEBOOK.md` §2d records 2662/3400 for what may be the same
    peeled run -- the two are unreconciled and neither run log is in the repo). Waiting for `n == N` hangs forever on a perfect run; waiting for a
    *static* count mistakes a slow run for a dead one. Use the orchestrator's completion line, and
    use the count only to spot a run that ended **short**.
6. **Suspect success.** Most of the bugs in this pipeline reported success: a daemon killed while
   healthy, a batch flag silently ignored, a drain that stopped before the file finished writing, a
   dict mutated during serialization, `>` refused by tcsh `noclobber`, an image server "running" for
   ten minutes after its socket had timed out, `BETA_CONFIG=""` falling through to the default and
   exiting 1 *after* alpha had already launched, and a shard driver logging "all 7 launched" with
   **three** running because `ssh` inside a `while read` loop eats the loop's stdin and swallows the
   remaining plan lines. (Use `ssh -n` and a `for` loop over an array — and verify **per shard**,
   never from the launcher's own log.) **Never combine `ssh -n` with a script fed on stdin
   (`ssh -n host bash -s < script.sh`):** `-n` redirects stdin from `/dev/null`, so the remote
   `bash -s` reads an empty script, runs nothing and prints nothing, which reads as a failed
   check when the check never ran. Use `ssh -n` for a single quoted command, and
   plain `ssh host bash -s < script.sh` (no `-n`, outside any `while read` loop) to run a script.
   `packages/laue_index/tests/test_streaming_regressions.py` pins the fixes; run it after
   touching the streaming path.
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
    cleared it, p=0). (These Zn/Zn numbers: source not in repo.)
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
    phase and re-index.
    **What is fixed and what is not (see 15b).** The harmonic stacking was on the ANALYSIS side
    (`nhit`); the indexer's own `NMatches` never stacked harmonics, and `nhit_distinct`
    (0.7.2 and later) counts each observed peak once. Neither removes the other mechanism: a deposit orientation
    whose reciprocal vectors merely **coincide** with the substrate's (generic vector overlay,
    not harmonics) still collects distinct substrate peaks as evidence for itself. So the peel
    is still required: no deposit-on-single-crystal orientation result is trustworthy until
    the dominant phase has been peeled and the deposit re-indexed.
15b. **KNOW WHICH SPOT COUNT YOU ARE GATING ON — there are THREE (and a fourth file that
    reuses the word "unique"), they measure different things, and this invariant used to name
    the wrong one.** (Rewritten 2026-09-21. The
    earlier text said `unique_spots_per_orientation` "is the distinct-observed count". It is
    not, and a campaign's worth of numbers were read through that label.)

    **This table is the glossary for the whole doc set.** Other files point here rather than
    redefine a count; if a count is described differently elsewhere, this table wins.

    | count | where | what it actually is |
    |---|---|---|
    | `NMatches` | `filtered_orientations[:,6]` (`.h5`; `[5]` in `solutions_filtered.txt`) | distinct observed pixels matched by **this** orientation; **does not stack** harmonics |
    | `unique_spots_per_orientation` | `[:,1]` of that dataset (column `Unique_Spots` in text outputs) | **winner-take-all across the frame's orientations** |
    | `nhit` | analysis side, `parentbeta_validate.py` | predicted reflections within TOL of an analysis peak; **stacks** harmonics. `nhit_distinct` (0.7.2 and later) counts each observed peak once; see "Which count the gates use" below |
    | "Unique_Experimental_Spots" | `<output>.unique_spot_counts.txt` (and `/entry/results/unique_spot_counts_text`), written by `laue_visualization.py` | a **fourth** meaning under the same word: distinct `(x, y)` spot positions per grain, with no winner-take-all across grains |

    **`NMatches` does not stack.** `calcOverlap` (`LaueMatchingHeaders.h`) rejects any
    reflection whose q-hat matches one already recorded (to 1e-6), so harmonics — (001),
    (002), (003), which share a direction exactly and land on one pixel — count once.
    Verified on sampleH, 25,172 solutions: spot rows equal `NMatches` in **100%**, and
    **0.00%** of solutions put two matches on the same pixel; at `NMatches` = 8 exactly,
    all 4,125 have 8 distinct pixels. The dedup is present in 0.7.1 as well: the harmonic
    stacking behind the sampleD epitaxy result (invariant 15) was never in the indexer's
    count; it was in the analysis-side `nhit` below.

    **`unique_spots_per_orientation` is a WINNER-TAKE-ALL count, not a distinct count.**
    `laue_index.filtering.calculate_unique_spots` processes a frame's orientations in
    descending quality; once a pixel or label is claimed, weaker orientations cannot reuse
    it. So a marginal solution with 8 genuine matches, 6 of them shared with a stronger
    grain, reports **2**. That measures *independent evidence* — peaks no better
    orientation explains — which is a legitimate thing to gate on, but it is not stacking,
    and a low value usually means overlapping grains sharing peaks, which is normal.

    **`nhit` DID stack, on the analysis side.** `Phase.project` returns one row per hkl with
    no q-hat dedup, so every harmonic is counted. Measured on sampleH: the Zn hkl list's 7,984
    entries reduce to 6,638 directions, a typical orientation lands 60.1 predicted rows on
    50.7 distinct pixels (**1.186x**), and real solutions stack **1.400x** in total (harmonics
    x ~1.18x genuine coincidences within TOL). The null stacks 1.177x — almost pure
    harmonics, since random orientations rarely coincide. Fixed in
    `frame_peaks.count_matched_peaks`, which `parentbeta_validate.py` and `null_model.py`
    now share: `nhit` keeps its historical meaning (every existing threshold stays valid)
    and `nhit_distinct` is carried alongside (0.7.2 and later). **Each has its own null** — on sampleH, full
    raster, 60,000 draws: `nhit` max 10, `nhit_distinct` max 5. Never gate one statistic
    against the other's null.

    **Which count the gates use (0.7.2 and later).** The scripts default to **`nhit`**;
    `LAUE_GATE_STAT=nhit|nhit_distinct` switches, the chosen statistic also selects the
    matching null from the scan's null JSON, and every gate prints which one it used.
    `nhit_distinct` is available but **not yet validated as a gate**: on sampleH it admits
    more grains than `nhit` (campaign measurement, source not in repo), and whether those extra grains are real is untested (the
    raw-image hit test on the newly admitted grains is pending). Per-frame analytic Poisson
    gates stay on `nhit`, because the distinct count has no closed-form null, and say so at
    run time.

    **Reading older numbers.** The "stacking" quoted here before — Cu 1.1x, sampleB Al 1.2x,
    sampleA Zn 1.2x, sampleD Zn 1.5x — was `NMatches` / `unique_spots_per_orientation`, i.e.
    a **sharing** ratio under winner-take-all, not stacking. The known Cu single crystal
    reading "1.0x" is not evidence that any gate works: the dominant orientation in a frame
    goes first under winner-take-all, keeps every peak, and reads 1.0x by construction.

    **WITHDRAWN, do not restore: "re-gating on distinct spots roughly doubled the accepted
    Zn"** (sampleD 5,665 -> 10,493; sampleA s1 1,775 -> 4,405). That compared against
    `col6 > 11`, a threshold never actually applied. Against the gate actually used, the
    re-gate gives **0.82x (sampleD Zn), 1.13x (sampleA s1), 2.10x (sampleA s2), 0.84x (sampleB)** —
    comparable yield, up in two cases and down in two (campaign notebook §4f and its
    retraction list). An earlier version of this invariant carried the doubling after the
    notebooks had withdrawn it, and the 2026-09-21 rewrite briefly re-asserted it as "real";
    both were wrong.

    **The join trap still stands.** Join `filtered_orientations[:,1]` <->
    `unique_spots_per_orientation[:,0]`, the orientation id within the frame.
    `filtered_orientations[:,0]` is the IMAGE number, and joining on it silently returns zeros.

    **How to apply:** name the count in every gate and every table, and read the code that
    produces a column before trusting its name. Anchor any spot-count analysis on a case with
    a known answer. And do not assume an "unindexed" fraction is a gating choice: on the
    deposit-on-Cu scans it was measured to be **real** — 70.3% of detected peaks unexplained,
    with the Zn matches that do exist at 21.8x chance — and re-gating did not recover it
    (notebook §4f). Check the gate, but expect the residual to survive it.
16. **A missing background file does not fail — the image server silently computes one FROM THE
    FIRST FRAME.** `mkbg_gen.py` (a campaign-local background builder, not in this repo) builds frame paths as `{prefix}{i}.h5` with *plain* integers,
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
    2,601 named `<sampleA prefix>_scan2_*` and **10,201 named `<other sample's prefix>_scan1_*`** — an entirely different sample.
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
    `kf = ki − 2(q̂·ki)q̂` (`calcOverlap` in `LaueMatchingHeaders.h`) is the general Bragg mirror and
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
    or a plumb/fiducial reading. **Sample translation will not do it.** Pinned in the repo by
    `packages/laue_index/tests/test_calibrate.py::test_the_beam_axis_gauge_is_exact` (the
    module docstring of `laue_index/calibrate.py` records the gauge at 4.5e-13 px); the
    φ = 90° and CeO₂ figures above: source not in repo.
    Missed twice in one day here, which is why it is now an invariant.

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
    **The cheap way to do all of that at once: run the SAME search against a spot-SCRAMBLED
    image** — same component count, same pixel intensities, same lit-pixel total, positions
    randomised and never onto a masked region (`scramble_bin.py`, campaign-local, not in this repo). Whatever the best-of-1e8
    reaches on that is the bar, and it is measured rather than modelled. Measured at TPS 21A:
    **0 solutions** on scrambled Si (83 on the real frame) and **0** on scrambled Ni — but
    **NMatches 8** on scrambled Ti, because those frames carry 123,720 lit px against Si's
    36,098. `MinNrSpots = 8` — the value the template ships — therefore sat exactly ON the Ti
    null, and five of seven raw α solutions were inside it. **The null sets the gate; the
    template default is a starting point, not a threshold.**

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

33. **Before blaming the indexer for finding nothing, COUNT what the energy window can
    even deliver.** A phase with a small unit cell puts its strong low-index reflections at
    low energy, and a photon-counting detector's discriminator can sit above all of them. At
    TPS 21A, Ni's (111) through (400) are **all** below the 8.74 keV `Threshold_setting` at
    every 2θ the panel covers, leaving 18.6 weak high-index reflections per orientation
    against 47 for Si and 52 for Ti α on the same station and geometry — and Ni indexed 27.9 %
    of voxels while Ti indexed 99 % (`LAB_NOTEBOOK_TPS21A.md` §5e/§5f/§5i; the 47 and 52:
    source not in repo). Project a few hundred random orientations through the
    geometry and histogram the on-panel reflections by energy; it takes a minute and it
    separates "the code is wrong" from "the measurement cannot see it". Note which end binds:
    widening Elo 8.74 → 6.0 keV bought **+0.54** reflections/orientation, and Ehi 26 → 45 keV
    bought **+65.75 predicted and −0.01 matched** — which is itself a measurement that the
    beam carries nothing above 26 keV. (Window-widening figures: source not in repo.)

34. **A null statistic must be MONOTONIC in the thing it tests.** Gating a grain map on the
    *number of multi-voxel grains* passed Ti α (418 real vs 222 null) and FAILED Ti β
    (19 vs 210) — when β is the **more** clustered of the two. A field that is a few huge
    grains has few multi-voxel grains, exactly like a field with no structure, and the
    statistic cannot tell them apart. Grain **count** (fewer = more clustered) and
    **largest-grain size** are both monotonic and both put β far outside its null
    (77 vs ≥597 grains; largest 530 vs ≤290). Before running a null, ask which direction the
    statistic moves as the effect gets stronger — and if the answer is "both", pick another.

35. **`ResultDir` is where the streaming daemon writes, and it must be unique per run.**
    `solutions.txt` and `spots.txt` go to the **parameter file's** `ResultDir`, not to the
    orchestrator's `--output-dir`. Two runs of the same phase off the same template therefore
    share one solutions file and each one's post-processing reads whatever the other has
    written so far. Measured: a 10-frame check alongside a 961-frame production run left
    **both with 770 outputs**, both logging `Pipeline complete` and exiting **0** — the
    production map silently truncated to 80 %. Rewrite `ResultDir` per run before launching,
    and check output count against frames requested in **both** directions.

36. **`NMatches` is not evidence of INDEPENDENCE, and the unit of uniqueness is the
    REFLECTION.** An orientation related to a true one by a coincidence-site lattice
    re-explains a share of its reflections for free — Σ3 shares a third — so it clears any
    match-count gate and any random-orientation null, because it is not a random draw.
    Measured on a Si(100) calibrant: three accepted solutions at 60.0° (Σ3) and 38.9° (Σ9)
    whose reflections were a **100 % subset** of the parent's (18/18, 20/20, 17/17; union =
    the parent's 46). One crystal reported as four.
    **Count uniqueness in reflections.** Any finer unit inflates it and the artifacts
    survive: matched PIXELS of the blurred indexer image (722 px per blob there) called them
    disjoint, and the pipeline's own `--min-unique 2` passed them too because it counts
    **watershed labels** — watershed found 3540 regions on a frame carrying 50 reflections.
    Gate on "reflections no other accepted orientation explains", computed across **all**
    phases at once: a β orientation that only re-explains α's reflections is not independent
    evidence of a β grain. On Nb1_3 this removed 12 % of accepted orientations (5113 → 4514)
    and left the structure intact; on Si it removed three of four.

37. **A spatial-coherence null cannot validate a finely-stepped raster.** If neighbouring
    frames are near-identical images, *any* input-driven solution — right or wrong — varies
    smoothly between them, so it beats a label shuffle automatically. The null separates
    "coherent" from "random", and coherence was free. Measured: an A5 Ni orientation field
    passed at 0.086° neighbour misorientation against a shuffled 0.236–0.266°, 100/100
    shuffles worse, on a raster whose pattern moves ~2.7 px per 50 nm step — and the same
    solutions explain a **median 7.8 %** of each frame's reflection intensity and a median
    of **1 of the 5 brightest reflections**, 36 frames of 135 explaining none. The field was
    withdrawn.
    **The test that works is per-frame and physical: does the solution explain the pattern,
    starting with the STRONGEST reflections?** A solution that misses the brightest spots
    while picking up faint ones is not describing that frame, whatever the map looks like.

38. **A SIMULATED frame is transposed relative to a real one, and it does not fail loudly.**
    `laue_torch` splats into `img[X, Y]`; every real beamline frame, every background built
    from them, and the indexer's own reader are `image[row, col]`. `laue-torch` writes its
    output to **`/entry1/data/data`** — the exact dataset the indexer reads — so a simulated
    frame handed straight to the indexer is transposed against everything it is compared
    with. Measured on 34-ID-E Zn: projecting the **truth** orientation onto as-written frames
    hit **0–2 of ~60** predicted reflections (chance), and **17–61** once transposed. The
    indexer still "succeeded" on ~30% of them with spurious solutions, and those would have
    become a synthetic control's measured noise floor — a number that looks like a
    measurement and is an axis convention.
    **Before trusting ANY synthetic control, project the known truth orientation onto the
    generated frame and count hits against a chance level.** A control you cannot verify
    against its own ground truth is not a control. In laue-torch 0.1.4 and later the layout is
    carried, not guessed: `laue_torch/cli.py` writes `/entry1/axis_order` = `XY` and
    `coded_aperture` writes `/entry/axis_order`; `LaueScanLoader` and the coded-aperture
    `load_voxel_h5` read that marker. The loader keeps a frame **as stored** and records its
    layout in `VoxelMeasurement.axis_order` (`"YX"`, a real `image[row, col]` frame, unless the
    file's `<entry>/axis_order` says otherwise). A hand-built `VoxelMeasurement`, or a call to
    `MultiGrainVoxelRefiner.refine`, has NO default layout: `axis_order` must be declared
    (`"YX"` real frame, `"XY"` laue_torch render) or the refiner raises, because on a square
    detector a wrong guess is silent. `VoxelODFRefiner` and
    `MultiGrainVoxelRefiner` convert on entry with `laue_torch.io.to_model_layout`, which
    checks the shape against the detector so a non-square frame in the wrong layout fails. The
    refiners also render only in the parameter file's `Elo`..`Ehi` band and refuse a missing
    or defaulted band. In 0.1.3 none of this existed: no marker, `LaueScanLoader` read the
    orientation matrix from the wrong columns and did not transpose. Same family as invariant 21 (`h5[...][0]` is a row, not a
    frame): an axis convention that produces a plausible array rather than an error.
