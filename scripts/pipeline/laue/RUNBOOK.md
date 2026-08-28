# Laue runbook — operational state

> Part of the **Laue doc set**. The spine is [`README.md`](README.md).
>
> **This is the volatile document.** The handbook is procedure; the notebooks only grow.
> This file describes *right now*. **Update §R3 before you finish.**

---

## R1. Where it runs

The 34-ID-E operational detail — beamline access, the daemon, sharding across hosts, and
the a two-phase hcp/bcc alloy campaign state — is in
[`laue_torch/report/RUN_PROCESS_REPORT_HANDOFF.md`](../../../laue_torch/report/RUN_PROCESS_REPORT_HANDOFF.md).
That file predates this doc set and is the authority for host and access specifics.

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
| Zn/Zn sampleH | 96–167 peaks/frame at 99.8 | 201×201 at 1.000 µm, wire parked |
| random-orientation null, sampleH | **max 9 hits in 30,000 draws** | the analytic Poisson gate would have accepted down to 5 |
| re-gating at nhit > 9 | kept **83.2 %** (171,644 / 206,343) | of previously "validated" instances |
| stage-readback mislabels | 180 of 20,301 frames (0.89 %) | present on **both** Zn scans — assume it until checked |
| plateau collapse in peak counts | 35–45 % of whole-frame counts | flat-top saturated reflections counted as dozens of peaks |
| map↔binary-mask correlation | tops out ~0.5 at perfect registration | a strong visual overlay is not a high pixel correlation |
| **transmission, 16-BM-D Si** | 16 sharp spots/frame median, max 93 | 40 % of raster positions off-sample and carrying zero spots is normal |
| **transmission, distinct observed** | median 45 per accepted orientation | stacking ratio **1.02**; anything ≳ 2 is harmonic stacking, re-gate |
| **transmission, index rate** | **0.19 s/frame** on one GPU | after the 12.2 GB forward cache is built (~100 s, geometry-specific) |

## R3. Current pick-up point

> **Every session updates this before it ends.** A stale pick-up point is worse than none.

**Last updated: 2026-08-25.**

**State.** First campaign outside 34-ID-E reflection geometry is complete: **16-BM-D
white-beam transmission**, Si wafer, 15,300 frames over six ω settings, all indexed. The doc
set now covers both geometries — see the scope table in the spine. Invariants **23–32** and
`LAB_NOTEBOOK_16BMD_Si.md` are new; `ENVELOPE.md` §1 has a new row that applies to *both*
geometries and is the most consequential thing in this update.

**Open:**

1. **Absolute orientation is not recoverable from any Laue data this chain has.** The
   beam-azimuth gauge is exact (invariant 27, `ENVELOPE.md` §1 row 5). Breaking it needs
   external metrology, and the ask has gone to 16-BM-D. Until it comes back, quote relative
   quantities only.
2. `GenerateHKLs.py` θ_max fix (four-corner) is **applied but not committed** in the working
   tree. It cannot drop reflections at 34-ID-E, but it has not been re-run there.
3. `GaussSigmaMax` is silently ignored by the streaming config path — the detection blur is a
   default, not the params value.
4. The transmission "search-safe gate ≥ 6 distinct observed" rests on an extrapolation from
   three measured null bins. Treat as unverified.
5. The ~40 µm filament in the 16-BM-D wafer is unidentified; SAXS cannot separate crack from
   scribe from scratch.
6. `DIAGNOSIS.md` now has five entries.
7. Substrate/deposit direction on Zn/Zn flipped several times — **read
   `LAB_NOTEBOOK_ZnZn.md` §5 before re-arguing it.**
