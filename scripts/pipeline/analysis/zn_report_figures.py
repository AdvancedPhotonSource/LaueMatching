"""Report plates for the Zn/Zn same-phase analysis.

Kept in the repo rather than living in a scratch directory, so the figures in the
report can always be regenerated from the analysis outputs.

Every panel that reports a quantity also reports its null, because on this
dataset several quantities look structured and are not: the pilot-row hardening
correlation was r=+0.05 (p=0.48), and the background map survives a drift model
only because that was explicitly tested (96.8% of the structure remains).

Inputs are skipped individually if absent, so this can be run part-way through
the chain.

usage: zn_report_figures.py <analysis_out_dir> [outdir]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NR = 201
EXT = [-100, 100, -100, 100]          # um, both axes in the sample frame


def _save(fig, path):
    fig.tight_layout()
    # bbox_inches='tight' + padding so suptitles at y=1.02 are never clipped
    fig.savefig(path, dpi=130, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


def plate_background(A, out):
    """Flat pedestal vs halo, and what a scan-time drift can explain."""
    f = os.path.join(A, "full_pedestal.npz")
    if not os.path.exists(f):
        print("  (skip background plate: no full_pedestal.npz)")
        return
    z = np.load(f)
    flat, halo, i0 = z["flat"], z["halo"], z["i0"]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for a, (m, t) in zip(ax, ((flat, "flat pedestal (corners)\nisotropic — Zn K$\\alpha$ fluorescence"),
                              (halo, "halo excess (centre − corners)\nforward-peaked — air + thermal diffuse"),
                              (i0, "beam monitor I0\n(flat to ±1.5%)"))):
        im = a.imshow(m, origin="lower", extent=EXT, cmap="viridis")
        a.set_title(t, fontsize=10)
        a.set_xlabel("X (µm)"); a.set_ylabel("45° axis (µm)")
        plt.colorbar(im, ax=a)
    fig.suptitle("Zn/Zn sampleG scan1_Laue2D — background decomposition, 201×201 positions", y=1.02)
    _save(fig, os.path.join(out, "plate_background.png"))

    v = flat[np.isfinite(flat)]
    h = halo[np.isfinite(halo)]
    print(f"    flat pedestal {v.min():.0f}–{v.max():.0f} ADU ({v.max()/max(v.min(),1):.2f}×), "
          f"halo {h.min():.0f}–{h.max():.0f} ({h.max()/max(h.min(),1):.2f}×)")


def plate_hardening(A, out):
    """The test that ties the pedestal to deposit thickness — with its null."""
    f = os.path.join(A, "hardening_fullmap.npz")
    if not os.path.exists(f):
        print("  (skip hardening plate: no hardening_fullmap.npz)")
        return
    z = np.load(f)
    ped, medE, nsp, rows = z["pedestal"], z["medE"], z["nspots"], z["rows"]

    def pearson(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float(a @ b / np.sqrt((a @ a) * (b @ b)))

    r = pearson(ped, medE)
    rng = np.random.default_rng(0)
    uniq = np.unique(rows)
    null = []
    for _ in range(2000):
        mapping = dict(zip(uniq, rng.permutation(uniq)))
        order = np.argsort([mapping[q] for q in rows], kind="stable")
        null.append(pearson(ped, medE[order]))
    null = np.array(null)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].hexbin(ped, medE, gridsize=45, cmap="magma", mincnt=1)
    ax[0].set_xlabel("flat pedestal (ADU)  →  more Zn in the beam path?")
    ax[0].set_ylabel("median assigned-spot energy (keV)")
    ax[0].set_title(f"hardening test:  r = {r:+.3f}")
    lo, hi = np.percentile(ped, 20), np.percentile(ped, 80)
    for q, c in ((lo, "tab:blue"), (hi, "tab:red")):
        ax[0].axvline(q, color=c, ls="--", lw=1)

    ax[1].hist(null, bins=50, color="0.75", label="row-blocked permutation null")
    ax[1].axvline(r, color="crimson", lw=2, label=f"measured r = {r:+.3f}")
    ax[1].set_xlabel("correlation"); ax[1].set_ylabel("count")
    p = float((np.abs(null) >= abs(r)).mean())
    ax[1].set_title(f"null spread sd = {null.std():.3f},  p = {p:.3g}")
    ax[1].legend(fontsize=8)
    fig.suptitle("Does the fluorescence pedestal predict spectral hardening?  "
                 "(prediction: higher pedestal → higher median energy)", y=1.02, fontsize=11)
    _save(fig, os.path.join(out, "plate_hardening.png"))
    print(f"    r = {r:+.4f}, blocked-permutation p = {p:.4g}, null sd = {null.std():.4f}")


def plate_grainmap(A, out):
    """IPF-coloured orientation map + per-position grain footprint.

    Colours each point by the crystal orientation (Zn c-axis in the sample frame
    -> RGB), NOT by cluster id: connected-components labelling numbers clusters in
    raster order, so a label-coloured map shows a smooth gradient that is an
    artefact of numbering, not grain structure. Here a contiguous grain is a
    uniform colour and fine deposit is speckle.
    """
    f = os.path.join(A, "..", "peel_map", "full_zn_clustered.npz")
    f = os.path.normpath(f)
    if not os.path.exists(f):
        print("  (skip grain plate: no full_zn_clustered.npz)")
        return
    z = np.load(f, allow_pickle=True)
    oms, lab, fr, nh = z["oms"], z["labels"], z["frames"], z["nhit"].astype(int)
    n = np.array([int(str(x).split("_")[-1].split(".")[0]) for x in fr])
    row, col = (n - 1) // NR, (n - 1) % NR
    cnt = np.bincount(lab)

    best = {}
    for i in range(len(oms)):
        k = (row[i], col[i])
        if k not in best or nh[i] > best[k][1]:
            best[k] = (i, nh[i])

    rgb = np.zeros((NR, NR, 4)); size = np.zeros((NR, NR))
    for (r, c), (i, _) in best.items():
        v = np.abs(oms[i][:, 2]); v /= np.linalg.norm(v)
        rgb[r, c, :3] = v; rgb[r, c, 3] = 1.0
        size[r, c] = cnt[lab[i]]

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].imshow(rgb, origin="lower", extent=EXT, interpolation="nearest")
    ax[0].set_title("orientation map — RGB = c-axis in sample frame\n"
                    "(uniform colour = one grain; speckle = fine deposit)")
    lm = np.ma.masked_where(size == 0, size)
    im = ax[1].imshow(lm, origin="lower", extent=EXT, cmap="viridis",
                      norm=matplotlib.colors.LogNorm(vmin=1, vmax=max(cnt.max(), 2)),
                      interpolation="nearest")
    ax[1].set_title("grain footprint at each position (log)\n"
                    "bright = large contiguous grain; dark = fine deposit")
    for a in ax:
        a.set_xlabel("X (µm)"); a.set_ylabel("45° axis (µm)")
    plt.colorbar(im, ax=ax[1], label="instances in grain")
    fig.suptitle("Zn grain map — coarse (substrate) vs fine (deposit)", y=1.02)
    _save(fig, os.path.join(out, "plate_grainmap.png"))
    occ = size[size > 0]
    print(f"    {(size>0).sum()} indexed positions; "
          f"{(size>=100).mean()*100:.0f}% of grid in grains >=100 pos, "
          f"{((size>0)&(size<10)).sum()} positions in grains <10")


def plate_texture(A, out):
    f = os.path.join(A, "texture.npz")
    if not os.path.exists(f):
        print("  (skip texture plate: no texture.npz)")
        return
    z = np.load(f)
    oms, null_oms = z["oms_rep"], z["null_oms"]

    def cpole(o):
        d = o[:, :, 2]                       # crystal c-axis in sample frame
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        d = np.where(d[:, 2:3] < 0, -d, d)
        return d

    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2), subplot_kw={"aspect": "equal"})
    for a, (o, t) in zip(ax, ((oms, "measured (one per grain)"),
                              (null_oms, "indexability-matched null"))):
        d = cpole(o)
        # equal-area projection
        rr = np.sqrt(2 * (1 - d[:, 2]))
        a.hexbin(rr * d[:, 0] / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9),
                 rr * d[:, 1] / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9),
                 gridsize=30, cmap="inferno", mincnt=1)
        a.set_title(f"{t}  (n={len(o)})", fontsize=10)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Zn c-axis [0001] pole figure vs a null of random orientations that "
                 "WOULD have been indexable", y=1.0, fontsize=10)
    _save(fig, os.path.join(out, "plate_texture.png"))


def main():
    A = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(A, "figures")
    os.makedirs(out, exist_ok=True)
    print(f"reading {A}\nwriting {out}")
    plate_background(A, out)
    plate_hardening(A, out)
    plate_grainmap(A, out)
    plate_texture(A, out)
    print("ZN_FIGURES_DONE", flush=True)


if __name__ == "__main__":
    main()
