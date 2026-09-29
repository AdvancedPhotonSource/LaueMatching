"""Substrate vs deposit, when both are the same phase.

Zn electroplated on Zn: no phase contrast, no orientation relationship, and no
depth resolution (the wire is parked). Three independent handles remain, and
they are only worth reporting where they agree:

  A. ORIENTATION PERSISTENCE. A substrate grain is large and continuous, so its
     orientation recurs over a wide contiguous area. Electrodeposit grains are
     small, so their orientations appear at few positions. If the specimen is
     part bare substrate and part covered, the distribution of cluster
     footprints should be two-population, not a single power law.

  B. PRESENCE/ABSENCE vs BACKGROUND. If the fluorescence pedestal really tracks
     how much Zn sits above the substrate, then where the pedestal is high the
     substrate orientation should be absent (buried) more often. This is a
     binary test and far more robust than the spectral-shift version.

  C. SPECTRAL HARDENING (see hardening.py) -- the subtlest of the three.

Every claim is checked against a null, because with thousands of candidate
clusters something always looks structured. B and C are map statistics, so their
null is SPATIAL (invariant 10): the pedestal map is toroidally shifted against the
fixed cluster maps (raster.toroidal_shift_null). Before 2026-09 they permuted
values/labels, which ignores autocorrelation; on a known-null synthetic that
permutation flagged 79% of independent smooth maps at p < 0.05 for B (the
toroidal null: 4.5%). LAUE_NULL_REPS sets the number of shifts (default 1000).

usage: substrate_deposit.py <clustered.npz> <pedestal.npz> <outdir>

Needs LAUE_NR (raster columns) and LAUE_NROWS (rows); connected components use the
shared neighbourhood of raster.structure() (LAUE_CONNECTIVITY, default 8 -- the
value this script always used).
"""
import os
import sys

import numpy as np
from scipy import ndimage as ndi

from raster import (nan_corr, positions_per_label, raster_positions, raster_shape, structure,
                    toroidal_shift_null)


def load(clustered, pedestal):
    z = np.load(clustered, allow_pickle=True)
    oms, lab, X, Z, nh, fr = (z["oms"], z["labels"], z["X"].astype(float),
                              z["Z"].astype(float), z["nhit"].astype(int), z["frames"])
    p = np.load(pedestal)
    return oms, lab, X, Z, nh, fr, p


NREPS = int(os.environ.get("LAUE_NULL_REPS", "1000"))


def main():
    clustered, pedestal, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    NROWS, NR = raster_shape()             # LAUE_NROWS, LAUE_NR -- required
    os.makedirs(outdir, exist_ok=True)
    oms, lab, X, Z, nh, fr, ped = load(clustered, pedestal)
    row, col = raster_positions(fr, Z=Z, shape=(NROWS, NR))
    flat = ped["flat"]
    print(f"{len(oms)} re-gated instances, {len(np.unique(lab))} clusters, "
          f"{len(set(zip(row.tolist(), col.tolist())))} distinct positions\n", flush=True)

    # ---- A. cluster footprints -------------------------------------------
    labs, counts = np.unique(lab, return_counts=True)     # INSTANCES per cluster
    # distinct POSITIONS per cluster: the footprint. `counts` (instances) was
    # printed as positions; two orientations at one position are one position.
    npos_all = positions_per_label(lab, row, col)
    npos = npos_all[labs] if len(npos_all) else np.zeros(0, int)
    order = np.argsort(-npos)
    print("=== A. ORIENTATION PERSISTENCE ===")
    print(f"  clusters: {len(labs)}")
    print(f"  singletons (one position only): {(npos==1).sum()} "
          f"({(npos==1).mean()*100:.1f}%)")
    for k in (2, 5, 10, 25, 50, 100, 500):
        print(f"  clusters spanning >= {k:4d} positions: {(npos>=k).sum()}")

    rows_out = []
    for li in labs[order][:40]:
        m = lab == li
        rr, cc = row[m], col[m]
        grid = np.zeros((NROWS, NR), bool)
        grid[rr, cc] = True
        ccl, ncc = ndi.label(grid, structure=structure())
        sizes = np.bincount(ccl.ravel())[1:] if ncc else np.array([0])
        rows_out.append((int(li), int(m.sum()), int(len(set(zip(rr.tolist(), cc.tolist())))),
                         int(ncc), int(sizes.max()),
                         float(np.median(nh[m]))))
    print(f"\n  {'cluster':>8} {'inst':>6} {'positions':>10} {'components':>11} "
          f"{'largest cc':>11} {'med nhit':>9}")
    for r in rows_out[:15]:
        print(f"  {r[0]:8d} {r[1]:6d} {r[2]:10d} {r[3]:11d} {r[4]:11d} {r[5]:9.1f}")

    # two-population test on the footprint distribution
    pos_per_cluster = np.array([r[2] for r in rows_out] +
                               [int(c) for c in npos[order][40:]])
    big = npos[order][0]
    print(f"\n  largest cluster covers {big} positions "
          f"({big/max(len(set(zip(row.tolist(),col.tolist()))),1)*100:.2f}% of positions)")

    # ---- B. presence/absence of the dominant cluster vs pedestal ----------
    print("\n=== B. IS THE DOMINANT ORIENTATION ABSENT WHERE THE PEDESTAL IS HIGH? ===")
    top = labs[order][0]
    m = lab == top
    present = np.zeros((NROWS, NR), bool)
    present[row[m], col[m]] = True
    # only positions that produced any validated orientation are informative;
    # a position with nothing indexed is not evidence of burial
    indexed = np.zeros((NROWS, NR), bool)
    indexed[row, col] = True
    ok = indexed & np.isfinite(flat)
    if ok.sum() < 100:
        print("  too few indexed positions for this test")
    else:
        pv = flat[ok]
        pr = present[ok]
        if pr.sum() == 0 or pr.sum() == pr.size:
            print("  dominant cluster is present everywhere or nowhere; test not informative")
        else:
            mu_p, mu_a = pv[pr].mean(), pv[~pr].mean()
            print(f"  pedestal where dominant orientation PRESENT : {mu_p:7.2f} ADU (n={pr.sum()})")
            print(f"  pedestal where it is ABSENT                 : {mu_a:7.2f} ADU (n={(~pr).sum()})")
            print(f"  difference                                  : {mu_a-mu_p:+7.2f} ADU")
            print("  PREDICTED if the pedestal is deposit thickness: ABSENT should be HIGHER")
            d0 = mu_a - mu_p

            def diff_b(f):
                okf = indexed & np.isfinite(f)
                prf = present[okf]
                if prf.sum() < 3 or (~prf).sum() < 3:
                    return np.nan
                return f[okf][~prf].mean() - f[okf][prf].mean()
            res = toroidal_shift_null(diff_b, flat, n=NREPS, rng=0)
            null = res["null"]
            print(f"  toroidal-shift p (spatial null, {res['n']} shifts of the pedestal map): "
                  f"{res['p']:.4g}")
            print(f"  null spread: sd {null.std():.2f} ADU -> effect is {abs(d0)/null.std():.1f} sigma")

    # ---- C. footprint vs pedestal, over all sizeable clusters -------------
    print("\n=== C. DO LARGE-FOOTPRINT CLUSTERS SIT AT LOW PEDESTAL? ===")
    sizeable = labs[npos >= 5]
    if len(sizeable) >= 10:
        fp = np.array([npos_all[li] for li in sizeable], float)
        members = [(row[lab == li], col[lab == li]) for li in sizeable]

        def corr_c(f):
            pm = np.array([np.nanmean(f[rr, cc]) if np.isfinite(f[rr, cc]).any() else np.nan
                           for rr, cc in members])
            return nan_corr(fp, pm)
        res = toroidal_shift_null(corr_c, flat, n=NREPS, rng=1)
        print(f"  clusters spanning >=5 positions: {len(fp)}")
        print(f"  corr(footprint, mean pedestal) = {res['obs']:+.3f}   "
              f"toroidal-shift p = {res['p']:.4g} ({res['n']} shifts)")
        print("  PREDICTED if big clusters are exposed substrate: NEGATIVE correlation")
    else:
        print(f"  only {len(sizeable)} clusters with >=5 instances; skipping")

    np.savez(f"{outdir}/substrate_deposit.npz",
             labels=lab, row=row, col=col, nhit=nh,
             cluster_sizes=counts, cluster_positions=npos, cluster_ids=labs)
    print(f"\nwrote {outdir}/substrate_deposit.npz")
    print("SUBSTRATE_DEPOSIT_DONE", flush=True)


if __name__ == "__main__":
    main()
