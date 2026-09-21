"""Within-grain hardening test: substrate seen through deposit, controlled for
grain identity, using the fluorescence pedestal as the deposit proxy.

The naive between-grain test failed because it averaged each grain's spectrum
over its whole footprint -- mostly bare (thin, patchy deposit), so the buried
signal washed out. The fix is to compare each grain ONLY TO ITSELF across its
own footprint: positions with a higher pedestal have more overlying deposit, so
if the grain's reflections harden there, that is substrate seen through deposit.

Grain fixed effects: subtract each grain's own mean from both pedestal and spot
energy, then pool the residuals. This removes every between-grain confound
(orientation, spot count, structure factors) -- the estimate is the pure
within-grain relationship. Null: shuffle pedestal WITHIN each grain.

No optical registration needed: the pedestal is the deposit proxy in the scan's
own frame. (The optical image, with the known vertical flip, is an independent
cross-check for later.)

Positions (row, col) come from layer_separation.npz, written by separate_layers.py
through raster.raster_positions, so they index full_pedestal.npz on the same raster.

Environment: LAUE_WORK (required).
"""
import os
import sys

import numpy as np
from scipy import stats


def per_grain_split(li, pv, ev, keep_g, minsz):
    """Median energy, high-pedestal half minus low-pedestal half, for each kept grain.

    ``li`` holds each instance's grain as an index into the ``np.unique`` label
    array (the ``return_inverse`` of it), so the grains to visit are the INDICES
    where ``keep_g`` is true -- ``np.where(keep_g)[0]``. Until 2026-09 the loop ran
    over ``range(keep_g.sum())``, i.e. indices 0..n_kept-1, which are not the kept
    grains: the t-test ran on an arbitrary subset (the kept grains that happened to
    have a small index) and silently skipped the rest.

    Returns ``(deltas, grains)``: the per-grain dE and the grain index each came from.
    """
    deltas, grains = [], []
    for g in np.flatnonzero(keep_g):
        m = li == g
        if m.sum() < minsz:
            continue
        p_ = pv[m]; e_ = ev[m]
        if p_.std() < 1e-6:
            continue
        hi = p_ >= np.median(p_)
        if hi.sum() < 5 or (~hi).sum() < 5:
            continue
        deltas.append(np.median(e_[hi]) - np.median(e_[~hi]))
        grains.append(int(g))
    return np.array(deltas), np.array(grains, int)


def main():
    W = os.environ.get("LAUE_WORK")
    if not W:
        sys.exit("LAUE_WORK is not set (work directory holding analysis_out/)")

    sep = np.load(f"{W}/analysis_out/layer_separation.npz", allow_pickle=True)
    lab = sep["labels"]; row = sep["row"]; col = sep["col"]
    medE = sep["medE"]; foot = sep["footprint"]; nasg = sep["nassigned"]
    flat = np.load(f"{W}/analysis_out/full_pedestal.npz")["flat"]      # (LAUE_NROWS, LAUE_NR), same raster as row/col

    pedv = flat[row, col]
    ok = np.isfinite(medE) & np.isfinite(pedv) & (nasg >= 8)           # enough spots for a stable median
    lab, row, col, medE, foot, pedv = (a[ok] for a in (lab, row, col, medE, foot, pedv))
    print(f"{ok.sum()} instances with a stable spectrum (>=8 assigned spots)")

    # keep grains with enough positions AND enough pedestal spread to split
    labs, inv = np.unique(lab, return_inverse=True)
    counts = np.bincount(inv)
    MINSZ = 20
    keep_g = counts >= MINSZ
    print(f"grains with >= {MINSZ} instances: {keep_g.sum()} of {len(labs)}")

    # ---- grain fixed-effects: demean pedestal & energy within each grain --------
    gmask = keep_g[inv]
    li = inv[gmask]
    pv = pedv[gmask]; ev = medE[gmask]
    # per-grain means
    gp = np.bincount(li, pv) / np.bincount(li)
    ge = np.bincount(li, ev) / np.bincount(li)
    pr = pv - gp[li]        # pedestal residual (within grain)
    er = ev - ge[li]        # energy residual (within grain)

    def wcorr(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float(a @ b / np.sqrt((a @ a) * (b @ b)))

    r_fe = wcorr(pr, er)
    # slope: keV of spectral shift per ADU of pedestal, within grain
    slope = float(np.polyfit(pr, er, 1)[0])
    print("\n=== WITHIN-GRAIN (grain fixed-effects) ===")
    print(f"  instances used: {gmask.sum()} across {keep_g.sum()} grains")
    print(f"  corr(pedestal, spot energy) WITHIN grain = {r_fe:+.4f}")
    print(f"  slope = {slope*1000:+.3f} eV per ADU of pedestal")
    print(f"  pedestal within-grain spread (sd of residual): {pr.std():.1f} ADU")
    print(f"  => typical within-grain energy swing: {slope*pr.std()*1000:+.1f} eV over +-1 sd pedestal")
    print("  PREDICT > 0 if higher pedestal (more deposit) hardens the substrate reflections")

    # ---- null: shuffle pedestal residual within each grain ---------------------
    rng = np.random.default_rng(0)
    order = np.argsort(li, kind="stable")
    li_s = li[order]; pr_s = pr[order]; er_s = er[order]
    # group boundaries
    bnd = np.concatenate([[0], np.where(np.diff(li_s) != 0)[0] + 1, [len(li_s)]])
    null = np.empty(2000)
    for k in range(2000):
        prp = pr_s.copy()
        for a, b in zip(bnd[:-1], bnd[1:]):
            prp[a:b] = rng.permutation(prp[a:b])
        null[k] = wcorr(prp, er_s)
    p = float((np.abs(null) >= abs(r_fe)).mean())
    print(f"  within-grain-shuffle null: p = {p:.4g}  (null sd {null.std():.4f})")

    # ---- paired high/low split, per grain (a second, simpler view) -------------
    deltas, _ = per_grain_split(li, pv, ev, keep_g, MINSZ)
    t, pt = stats.ttest_1samp(deltas, 0.0)
    print("\n=== per-grain high-vs-low pedestal split ===")
    print(f"  grains: {len(deltas)}")
    print(f"  mean ΔE (high-pedestal − low-pedestal) = {deltas.mean()*1000:+.1f} eV")
    print(f"  fraction of grains with ΔE > 0: {(deltas > 0).mean():.3f}  (0.5 = no effect)")
    print(f"  one-sample t-test vs 0: t = {t:.2f}, p = {pt:.3g}")

    np.savez(f"{W}/analysis_out/within_grain.npz",
             r_fe=r_fe, slope=slope, p=p, deltas=deltas,
             ped_resid=pr, ene_resid=er)
    print("\nWITHIN_GRAIN_DONE")


if __name__ == "__main__":
    main()
