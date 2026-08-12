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

## R3. Current pick-up point

> **Every session updates this before it ends.** A stale pick-up point is worse than none.

**Last updated: 2026-08-11.**

**State.** The handbook was split into this doc set today. No change to any procedure or
claim — the text was moved, and `DIAGNOSIS.md` and this file are new.

**Open:**

1. `DIAGNOSIS.md` has three entries. It grows the day someone works out what a strange
   plot meant, written the same day.
2. The material port is **done** (2026-07-24, on the Zn/Zn dataset); Phase 6 is a
   *verification* step, not a porting step.
3. Substrate/deposit direction on Zn/Zn flipped several times — **read
   `LAB_NOTEBOOK_ZnZn.md` §5 before re-arguing it.**
