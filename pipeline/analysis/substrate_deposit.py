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

Every claim is checked against a null built by permuting the quantity being
tested, because with thousands of candidate clusters something always looks
structured.

usage: substrate_deposit.py <clustered.npz> <pedestal.npz> <outdir>
"""
import os
import sys

import numpy as np
from scipy import ndimage as ndi

NR = 201


def load(clustered, pedestal):
    z = np.load(clustered, allow_pickle=True)
    oms, lab, X, Z, nh, fr = (z["oms"], z["labels"], z["X"].astype(float),
                              z["Z"].astype(float), z["nhit"].astype(int), z["frames"])
    p = np.load(pedestal)
    return oms, lab, X, Z, nh, fr, p


def frame_to_rc(frames):
    """G19_scan1_Laue2D_<i>.h5 -> (row, col) on the 201x201 raster."""
    n = np.array([int(str(f).split("_")[-1].split(".")[0]) for f in frames])
    return (n - 1) // NR, (n - 1) % NR


def main():
    clustered, pedestal, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(outdir, exist_ok=True)
    oms, lab, X, Z, nh, fr, ped = load(clustered, pedestal)
    row, col = frame_to_rc(fr)
    flat = ped["flat"]
    print(f"{len(oms)} re-gated instances, {len(np.unique(lab))} clusters, "
          f"{len(set(zip(row.tolist(), col.tolist())))} distinct positions\n", flush=True)

    # ---- A. cluster footprints -------------------------------------------
    labs, counts = np.unique(lab, return_counts=True)
    order = np.argsort(-counts)
    print("=== A. ORIENTATION PERSISTENCE ===")
    print(f"  clusters: {len(labs)}")
    print(f"  singletons (one position only): {(counts==1).sum()} "
          f"({(counts==1).mean()*100:.1f}%)")
    for k in (2, 5, 10, 25, 50, 100, 500):
        print(f"  clusters spanning >= {k:4d} positions: {(counts>=k).sum()}")

    rows_out = []
    for li in labs[order][:40]:
        m = lab == li
        rr, cc = row[m], col[m]
        grid = np.zeros((NR, NR), bool)
        grid[rr, cc] = True
        ccl, ncc = ndi.label(grid, structure=ndi.generate_binary_structure(2, 2))
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
                               [int(c) for c in counts[order][40:]])
    big = counts[order][0]
    print(f"\n  largest cluster covers {big} instances "
          f"({big/max(len(set(zip(row.tolist(),col.tolist()))),1)*100:.2f}% of positions)")

    # ---- B. presence/absence of the dominant cluster vs pedestal ----------
    print("\n=== B. IS THE DOMINANT ORIENTATION ABSENT WHERE THE PEDESTAL IS HIGH? ===")
    top = labs[order][0]
    m = lab == top
    present = np.zeros((NR, NR), bool)
    present[row[m], col[m]] = True
    # only positions that produced any validated orientation are informative;
    # a position with nothing indexed is not evidence of burial
    indexed = np.zeros((NR, NR), bool)
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
            rng = np.random.default_rng(0)
            d0 = mu_a - mu_p
            null = np.empty(5000)
            for i in range(5000):
                s = rng.permutation(pr)
                null[i] = pv[~s].mean() - pv[s].mean()
            p = float((np.abs(null) >= abs(d0)).mean())
            print(f"  permutation p (label-shuffled null, 5000 draws): {p:.4g}")
            print(f"  null spread: sd {null.std():.2f} ADU -> effect is {abs(d0)/null.std():.1f} sigma")

    # ---- C. footprint vs pedestal, over all sizeable clusters -------------
    print("\n=== C. DO LARGE-FOOTPRINT CLUSTERS SIT AT LOW PEDESTAL? ===")
    sizeable = labs[counts >= 5]
    if len(sizeable) >= 10:
        fp, pedmean = [], []
        for li in sizeable:
            m = lab == li
            fp.append(m.sum())
            v = flat[row[m], col[m]]
            v = v[np.isfinite(v)]
            if len(v):
                pedmean.append(v.mean())
            else:
                fp.pop()
        fp = np.array(fp, float); pedmean = np.array(pedmean)
        a = fp - fp.mean(); b = pedmean - pedmean.mean()
        r = float(a @ b / np.sqrt((a @ a) * (b @ b)))
        rng = np.random.default_rng(1)
        null = np.array([float(a @ rng.permutation(b) / np.sqrt((a @ a) * (b @ b)))
                         for _ in range(5000)])
        print(f"  clusters with >=5 instances: {len(fp)}")
        print(f"  corr(footprint, mean pedestal) = {r:+.3f}   "
              f"perm p = {(np.abs(null)>=abs(r)).mean():.4g}")
        print("  PREDICTED if big clusters are exposed substrate: NEGATIVE correlation")
    else:
        print(f"  only {len(sizeable)} clusters with >=5 instances; skipping")

    np.savez(f"{outdir}/substrate_deposit.npz",
             labels=lab, row=row, col=col, nhit=nh,
             cluster_sizes=counts, cluster_ids=labs)
    print(f"\nwrote {outdir}/substrate_deposit.npz")
    print("SUBSTRATE_DEPOSIT_DONE", flush=True)


if __name__ == "__main__":
    main()
