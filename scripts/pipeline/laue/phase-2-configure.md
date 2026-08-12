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
