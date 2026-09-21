"""How much of the validated set clears the EMPIRICAL null, not just the analytic one?

The p<1e-4 gate in the validator is analytic Poisson, which assumes uniformly
scattered peaks. Real peak fields are clustered, so the measured null has heavier
tails: on the scan this was written for, 18,000 random draws reached 16 hits (alpha)
and 15 (beta), far beyond what Poisson(3.08)/Poisson(1.91) would allow. The analytic
gate is therefore optimistic.

This reports, per phase:
  - instances above the empirical null maximum (no random draw ever did this well)
  - instances above the empirical 99.9th percentile
  - the same, restricted to grains that also RECUR at >=5 positions, which is the
    tier the report stands behind (independent evidence, not a harsher single-frame cut)

The null is the one MEASURED ON THIS SCAN by null_model.py (read through
frame_peaks.load_null: $LAUE_WORK/peel_map/${LAUE_OUT_PREFIX}_null.json, with
LAUE_NULLMAX_<PHASE> overriding the max). There is no built-in fallback: this
script used to carry one scan's Ti null and apply it to every scan.

The statistic is LAUE_GATE_STAT = nhit (default) | nhit_distinct; the counts and
the null are both taken for that statistic.

usage: empirical_gate.py        (phases from LAUE_PHASES, default alpha,beta)
"""
import os
import sys

import numpy as np

from frame_peaks import gate_counts, gate_statistic, load_null, out_prefix

W = os.environ.get("LAUE_WORK") or sys.exit("LAUE_WORK is not set (peel_map/ is read under it)")
PREFIX = out_prefix()
PHASES = [p.strip() for p in os.environ.get("LAUE_PHASES", "alpha,beta").split(",") if p.strip()]
STAT = gate_statistic()
print(f"gate statistic: {STAT} (LAUE_GATE_STAT)")

for ph in PHASES:
    null = load_null(ph, W, PREFIX, STAT)
    src = f"{W}/peel_map/{PREFIX}_{ph}_validated.npz"
    z = np.load(src, allow_pickle=True)
    nhit = gate_counts(z, STAT, src); lab = z["labels"]
    n = len(nhit)
    mx, p999 = int(null["max"]), null.get("p999")
    counts = np.bincount(lab[lab >= 0])
    size_of = np.zeros(len(lab), int)
    size_of[lab >= 0] = counts[lab[lab >= 0]]
    rec5 = size_of >= 5

    print(f"\n=== {ph} ===   null: {STAT} max {mx}"
          + (f", p99.9 {p999:g}" if p999 is not None else "")
          + f"  [{null['source']}]")
    print(f"validated instances               {n:>8,}")
    if p999 is not None:
        print(f"  above empirical 99.9pct ({p999:>4g})   {int((nhit > p999).sum()):>8,}  "
              f"({100*(nhit > p999).mean():.1f}%)")
    else:
        print("  above empirical 99.9pct           (no p99.9: only the max was supplied)")
    print(f"  above empirical null MAX ({mx:>2})    {int((nhit > mx).sum()):>8,}  "
          f"({100*(nhit > mx).mean():.1f}%)")
    print(f"  in grains recurring >=5 positions {int(rec5.sum()):>8,}  "
          f"({100*rec5.mean():.1f}%)")
    print(f"  BOTH >null max AND recurring>=5   {int((rec5 & (nhit > mx)).sum()):>8,}")

    # grain-level: how many distinct grains have at least one instance above null max
    if lab.max() >= 0:
        g_above = len(np.unique(lab[(lab >= 0) & (nhit > mx)]))
        g_rec5 = int((counts >= 5).sum())
        both = np.unique(lab[(lab >= 0) & (nhit > mx) & rec5])
        print(f"  distinct grains, >=1 instance >null max: {g_above:,}")
        print(f"  distinct grains recurring >=5:           {g_rec5:,}")
        print(f"  distinct grains BOTH:                    {len(both):,}")
