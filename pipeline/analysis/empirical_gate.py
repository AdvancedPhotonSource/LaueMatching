"""How much of the validated set clears the EMPIRICAL null, not just the analytic one?

The p<1e-4 gate in the validator is analytic Poisson, which assumes uniformly
scattered peaks. Real peak fields are clustered, so the measured null has heavier
tails: 18,000 random draws reached 16 hits (alpha) and 15 (beta), far beyond what
Poisson(3.08)/Poisson(1.91) would allow. The analytic gate is therefore optimistic.

This reports, per phase:
  - instances above the empirical null maximum (no random draw ever did this well)
  - instances above the empirical 99.9th percentile
  - the same, restricted to grains that also RECUR at >=5 positions, which is the
    tier the report stands behind (independent evidence, not a harsher single-frame cut)
"""
import os
import numpy as np

W = os.environ.get("LAUE_WORK", "$LAUE_WORK")
PREFIX = os.environ.get("LAUE_OUT_PREFIX", "scan")
NULL = {"alpha": dict(mean=2.95, p999=12, mx=16), "beta": dict(mean=1.81, p999=10, mx=15)}

for ph in ("alpha", "beta"):
    z = np.load(f"{W}/peel_map/{PREFIX}_{ph}_validated.npz", allow_pickle=True)
    nhit = z["nhit"].astype(int); lab = z["labels"]
    n = len(nhit)
    mx, p999 = NULL[ph]["mx"], NULL[ph]["p999"]
    counts = np.bincount(lab[lab >= 0])
    size_of = np.zeros(len(lab), int)
    size_of[lab >= 0] = counts[lab[lab >= 0]]
    rec5 = size_of >= 5

    print(f"\n=== {ph} ===")
    print(f"validated instances               {n:>8,}")
    print(f"  above empirical 99.9pct ({p999:>2})     {int((nhit > p999).sum()):>8,}  "
          f"({100*(nhit > p999).mean():.1f}%)")
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
