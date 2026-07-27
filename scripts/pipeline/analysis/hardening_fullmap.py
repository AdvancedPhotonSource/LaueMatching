"""Full-map test: does the fluorescence pedestal predict spectral hardening?

The hypothesis (Dina's, for Zn electroplated on Zn): a higher background marks
positions where the beam passes through more Zn. If so the DIFFRACTION must be
hardened there too, because the 1/e sampled depth in Zn is a strong function of
photon energy (3.4 um at 12 keV, 41 um at 30 keV at 45 deg incidence, from
midas_hkls.absorption) -- a thick overlayer removes low-energy reflections
preferentially. Two independent observables, one cause.

This is the full-map version. The pilot-row version was a weak test and returned
null: one row spans only part of the pedestal range, and within a row the scan
order and the X coordinate are the same variable, so a detector drift and a
spatial structure are indistinguishable. Over the whole raster they separate,
because the background ridge is not aligned with either scan axis.

Controls, all reported whether or not they are flattering:
  (a) peak count -- frames with more spots sample the energy distribution better;
      partialled out.
  (b) scan time -- the pedestal regressed on frame index, and the hardening test
      repeated on the residual.
  (c) I0.
  (d) a spatially-blocked permutation null: naive permutation is anticonservative
      when both fields are spatially autocorrelated, which these are. Blocks of
      whole rows are permuted instead.

usage: hardening_fullmap.py <spot_energy_merged.npz> <full_pedestal.npz> <outdir>
"""
import os
import sys

import numpy as np

NR = 201


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else np.nan


def partial(a, b, *ctrl):
    """corr(a,b) with the ctrl variables linearly removed from both."""
    A = np.column_stack([np.ones(len(a))] + [np.asarray(c, float) for c in ctrl])
    def resid(x):
        return x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    return pearson(resid(a), resid(b))


def blocked_perm_p(a, b, rows, n=5000, seed=0):
    """Permute whole rows, preserving within-row spatial structure.

    A naive element-wise permutation destroys the autocorrelation in both fields
    and therefore understates the null spread -- it will call almost any smooth
    map 'significant'.
    """
    r0 = abs(pearson(a, b))
    rng = np.random.default_rng(seed)
    uniq = np.unique(rows)
    cnt = 0
    for _ in range(n):
        perm = rng.permutation(uniq)
        mapping = dict(zip(uniq, perm))
        order = np.argsort([mapping[r] for r in rows], kind="stable")
        if abs(pearson(a, b[order])) >= r0:
            cnt += 1
    return (cnt + 1) / (n + 1)


def main():
    spe, pedf, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(outdir, exist_ok=True)

    d = np.load(spe, allow_pickle=True)
    spots, sources, counts = d["spots"], d["sources"], d["counts"]
    edges = np.concatenate([[0], np.cumsum(counts)])
    ped = np.load(pedf)
    flat, halo, i0 = ped["flat"], ped["halo"], ped["i0"]

    rows, cols, medE, fracLo, nsp, pedv, halov, i0v, fno = ([] for _ in range(9))
    for i, src in enumerate(sources):
        n = int(str(src).split("_")[-1].split(".")[0])
        r, c = (n - 1) // NR, (n - 1) % NR
        if not np.isfinite(flat[r, c]):
            continue
        a = spots[edges[i]:edges[i + 1]]
        good = np.isfinite(a[:, 1]) & (a[:, 5] < 5)
        E = a[good, 1]
        if len(E) < 10:
            continue
        rows.append(r); cols.append(c); fno.append(n)
        medE.append(np.median(E)); fracLo.append((E < 15).mean()); nsp.append(len(E))
        pedv.append(flat[r, c]); halov.append(halo[r, c]); i0v.append(i0[r, c])

    rows = np.array(rows); cols = np.array(cols); fno = np.array(fno, float)
    medE = np.array(medE); fracLo = np.array(fracLo); nsp = np.array(nsp, float)
    pedv = np.array(pedv); halov = np.array(halov); i0v = np.array(i0v, float)
    print(f"{len(rows)} positions with >=10 assigned spots and a valid pedestal\n")
    print(f"  pedestal  {pedv.min():7.1f} .. {pedv.max():7.1f} ADU  (pilot row spanned 126-214)")
    print(f"  median E  {medE.min():7.2f} .. {medE.max():7.2f} keV")
    print(f"  spots/pos {nsp.min():7.0f} .. {nsp.max():7.0f}\n")

    print("=== HARDENING TEST (full map) ===")
    res = {}
    for nm, y in (("median spot energy", medE),
                  ("fraction of spots < 15 keV", fracLo)):
        r = pearson(pedv, y)
        rp = partial(pedv, y, nsp)
        rpt = partial(pedv, y, nsp, fno)
        p = blocked_perm_p(pedv, y, rows)
        print(f"  corr(pedestal, {nm:26s}) = {r:+.4f}")
        print(f"      partialling out spots/position          : {rp:+.4f}")
        print(f"      partialling out spots/position AND time : {rpt:+.4f}")
        print(f"      row-blocked permutation p               : {p:.4g}")
        res[nm] = dict(r=r, r_partial=rp, r_partial_time=rpt, p_blocked=p)

    print("\n=== CONTROLS ===")
    print(f"  corr(pedestal, I0)            = {pearson(pedv, i0v):+.4f}")
    print(f"  corr(pedestal, frame index)   = {pearson(pedv, fno):+.4f}  (whole map; "
          f"within one row this is confounded with X)")
    print(f"  corr(pedestal, spots/pos)     = {pearson(pedv, nsp):+.4f}")
    print(f"  corr(pedestal, halo)          = {pearson(pedv, halov):+.4f}  "
          f"(flat vs forward-peaked components)")
    print(f"  corr(spots/pos, median E)     = {pearson(nsp, medE):+.4f}")

    print("\n=== DIRECTION: top vs bottom pedestal quintile ===")
    lo = pedv <= np.percentile(pedv, 20)
    hi = pedv >= np.percentile(pedv, 80)
    print(f"  low  pedestal (n={lo.sum():5d}, {pedv[lo].mean():6.1f} ADU): "
          f"median E {np.median(medE[lo]):6.2f} keV, frac<15 {fracLo[lo].mean():.4f}")
    print(f"  high pedestal (n={hi.sum():5d}, {pedv[hi].mean():6.1f} ADU): "
          f"median E {np.median(medE[hi]):6.2f} keV, frac<15 {fracLo[hi].mean():.4f}")
    dE = np.median(medE[hi]) - np.median(medE[lo])
    print(f"  difference: {dE:+.3f} keV")
    print("  PREDICTED if the pedestal is Zn thickness: HIGHER median E, LOWER frac<15")

    # how big an effect *should* a given overlayer produce? gives the null teeth.
    # midas_hkls is a pip dependency (pip install midas-hkls); absorption comes
    # from its NIST MAC tables -- do not hard-code a source path, which would
    # shadow the installed package.
    print("\n=== what thickness would this rule out? ===")
    try:
        from midas_hkls.absorption import linear_absorption_coefficient as mu_of
        HC = 12.398419739
        for t_um in (1, 2, 5, 10, 20):
            # extra attenuation of a reflection at E through t um of Zn at 45 deg
            w = []
            for E in (12.0, 25.0):
                mu = mu_of("Zn", HC / E)          # 1/cm
                w.append(np.exp(-mu * (2 * np.sqrt(2) * t_um * 1e-4)))
            print(f"    t = {t_um:2d} um: transmission 12 keV {w[0]:.3f}, "
                  f"25 keV {w[1]:.3f}  -> ratio {w[1]/max(w[0],1e-30):8.1f}")
    except Exception as exc:
        print(f"    (midas_hkls unavailable here: {exc})")

    np.savez(f"{outdir}/hardening_fullmap.npz",
             rows=rows, cols=cols, pedestal=pedv, halo=halov, i0=i0v,
             nspots=nsp, medE=medE, fracLo=fracLo, frame=fno)
    import json
    json.dump(res, open(f"{outdir}/hardening_fullmap.json", "w"), indent=1)
    print(f"\nwrote {outdir}/hardening_fullmap.json")
    print("HARDENING_FULLMAP_DONE", flush=True)


if __name__ == "__main__":
    main()
