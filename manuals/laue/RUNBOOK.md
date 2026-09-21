# Laue runbook — operational state

> Part of the **Laue doc set**. The spine is [`README.md`](README.md).
>
> **This is the volatile document.** The handbook is procedure; the notebooks only grow.
> This file describes *right now*. **Update §R3 before you finish.**

---

## R1. Where it runs

Site-specific operational detail — beamline account and access, which host runs the daemon,
how work is sharded across GPUs, and a given campaign's state — is **deliberately not in
this public tree**. It belongs to the facility and the users who ran the experiment, and it
goes stale faster than anything else here. Keep it in the campaign's own directory beside
the data. What follows is the part that transfers.

| | |
|---|---|
| material selection | `LAUE_PHASES=<phase>` and `LAUE_PARAMS_<PHASE>=<path>` — an environment variable, not an edit (Phase 6) |
| indexing | sharded across GPUs and hosts; see Phase 3 |
| outputs | the campaign's own directory — **never `/tmp`** |

## R2. What healthy looks like

**There is no single number for "healthy".** Publishing one threshold produces false
alarms on the dense scans and silence on the broken ones. Every row carries its conditions.

| quantity | observed | condition |
|---|---|---|
| peaks/frame, few-grain | ≲ 50 | classical indexing works; this pipeline is overkill but fine |
| peaks/frame, design case | 100–500 | nulls matter here |
| peaks/frame, dense/streaky | ≳ 900 | expect the iterative peel and several s/frame |
| Zn/Zn sampleG | 96–167 peaks/frame at 99.8 | 201×201 at 1.000 µm (40,401 frames), wire parked |
| random-orientation null, sampleG | **max 9 hits in 30,000 draws** (`nhit`) | the analytic Poisson gate would have accepted down to 5 |
| random-orientation null, sampleH | `nhit` max **10**, `nhit_distinct` max **5**, full raster, 60,000 draws | each count has its own null; never gate one against the other's (`INVARIANTS.md` 15b) |
| re-gating at nhit > 9, sampleG | kept **83.2 %** (171,644 / 206,343) | of previously "validated" instances |
| stage-readback mislabels | 180 of 20,301 frames (0.89 %), sampleH scan 1 | present on **both** Zn scans — assume it until checked |
| plateau collapse in peak counts | 35–45 % of whole-frame counts | flat-top saturated reflections counted as dozens of peaks |
| map↔binary-mask correlation | tops out ~0.5 at perfect registration | a strong visual overlay is not a high pixel correlation |
| **transmission, 16-BM-D Si** | 16 sharp spots/frame median, max 93 | 40 % of raster positions off-sample and carrying zero spots is normal |
| **transmission, distinct observed** | median 45 per accepted orientation | ratio **1.02**, recorded as "stacking". If it is `NMatches` ÷ `unique_spots_per_orientation`, as on the 34-ID-E campaigns (`LAB_NOTEBOOK.md` §2c), it is a winner-take-all SHARING ratio, not harmonic stacking, which `NMatches` cannot do; a high value means orientations sharing peaks (glossary: `INVARIANTS.md` 15b) |
| **transmission, index rate** | **0.19 s/frame** on one GPU | after the 12.2 GB forward cache is built (~100 s, geometry-specific) |

## R3. Current pick-up point

> **Every session updates this before it ends.** A stale pick-up point is worse than none.

**Last updated: 2026-09-21.**

**State.** Three stations are covered: 34-ID-E and **TPS 21A** (reflection; TPS 21A is the
first XMAS-calibrated station, `LAB_NOTEBOOK_TPS21A.md`) and **16-BM-D** (transmission,
`LAB_NOTEBOOK_16BMD_Si.md`). Current release is **laue-index 0.7.1**; **0.7.2** (with
laue-torch 0.1.4) is prepared and not yet published. The beamline install is now a
pip environment with laue-index plus a separate editable-checkout environment; the old
`laue_rt` environment and the old checkout (archived 2026-08-30) are gone (Phase 3).
Invariants run to **38**; 33–38 (energy-window count, monotonic nulls, unique `ResultDir`,
CSL redundancy, fine-raster smoothness, transposed simulated frames) are the newest, and
**15b now carries the one glossary of the spot counts**. `ENVELOPE.md` gained strain,
orientation-floor and depth-projection rows.

**Open:**

1. **Absolute orientation is not recoverable from any Laue data this chain has.** The
   beam-azimuth gauge is exact (invariant 27, `ENVELOPE.md` §1 row 5). Breaking it needs
   external metrology, and the ask has gone to 16-BM-D. Until it comes back, quote relative
   quantities only.
2. `GenerateHKLs.py` θ_max fix (four-corner) is committed and released. It cannot drop
   reflections at 34-ID-E, but it has not been re-run there.
3. **Running 0.7.1 with several shards per host:** set `SCRIPTS`, `LAUE_PREPROCESS_WORKERS`
   (per shard) and `OPENCV_NUM_THREADS=1` by hand; `pipeline/launch_shard.sh` does not work
   as shipped. 0.7.2 fixes both (`pipeline/dispatch/`). Phase 3 item 2 and `DIAGNOSIS.md`.
   (`GaussSigmaMax`, listed here before as ignored by streaming, is parsed by the streaming
   config in 0.7.1.)
4. The transmission "search-safe gate ≥ 6 distinct observed" (which count: `INVARIANTS.md` 15b) rests on an extrapolation from
   three measured null bins. Treat as unverified.
5. The ~40 µm filament in the 16-BM-D wafer is unidentified; SAXS cannot separate crack from
   scribe from scratch.
6. `DIAGNOSIS.md` has fourteen entries, each keyed to a `symptom:` (local symptoms declared
   in its table) so `beamreport` can attach them.
7. Substrate/deposit direction flipped several times on a deposit-on-substrate campaign —
   **read `LAB_NOTEBOOK.md` §3b before re-arguing it.**
8. **Upgrading a streaming run from 0.7.1 to 0.7.2 can change its output.** Streaming
   post-processing now honours explicit `RobustFilter`, `MinGoodSpots`, `MinNrSpots` and
   `MaxAngle` as `RunImage.py` does. An ABSENT `RobustFilter` keeps the 0.7.1 legacy filter
   (one warning line). `MinGoodSpots` was ignored in 0.7.1 (`--min-unique 2` always): a config
   with `MinGoodSpots 4` raises the exclusive-label floor 2 → 4; `MinGoodSpots 2` reproduces
   0.7.1. A post-processing failure now exits non-zero instead of `Pipeline complete`. Phase 3
   §"Streaming post-processing".
9. **`nhit_distinct` is not yet validated as a gate.** The gates default to `nhit`; whether
   the extra grains `nhit_distinct` admits are real awaits the raw-image hit test on the newly
   admitted grains (Phase 4).
10. **A right-sized forward cache from a different geometry is still accepted** (open;
    `DIAGNOSIS.md`). Delete `ForwardFile` after any geometry or lattice change.
