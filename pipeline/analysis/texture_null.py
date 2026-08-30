"""Texture of the Zn deposit, measured against a random-orientation null.

Electroplated Zn is usually strongly fibre-textured, so "there is texture" is
the expected answer and therefore the easy one to report without checking. Two
things make a texture claim non-trivial here:

  1. The indexer does not sample SO(3) uniformly. Detector coverage, the energy
     window and the reflection list all make some orientations easier to index
     than others, so an apparently peaked pole figure can be an artefact of what
     is *indexable* rather than what is present. The null must therefore be
     random orientations passed through the SAME indexability filter, not a flat
     sphere.

  2. Instances are not independent -- one grain contributes many positions.
     Statistics are computed on cluster REPRESENTATIVES (one orientation per
     grain), with the instance-weighted version reported alongside for contrast.

usage: texture_null.py <clustered.npz> <outdir> [phase]
"""
import os
import sys

import numpy as np

from laue_material import Phase


def rand_om(rng, n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1),
        np.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1),
        np.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)], -2)


def axis_directions(oms, hkl, B):
    """Sample-frame unit vectors of a crystal direction for each orientation."""
    v = B @ np.asarray(hkl, float)
    v = v / np.linalg.norm(v)
    d = np.einsum('nij,j->ni', oms, v)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def main():
    src, outdir = sys.argv[1], sys.argv[2]
    phase = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("LAUE_PHASE", "zn")
    os.makedirs(outdir, exist_ok=True)
    ph = Phase.load(phase)

    z = np.load(src, allow_pickle=True)
    oms, lab = z["oms"], z["labels"]

    # one representative per grain: instances of the same grain are not
    # independent samples of the texture
    labs = np.unique(lab)
    rep = np.array([np.where(lab == li)[0][0] for li in labs])
    oms_rep = oms[rep]
    print(f"{len(oms)} instances -> {len(oms_rep)} grain representatives", flush=True)

    # indexability-matched null: random orientations that WOULD have been
    # indexable, i.e. that put at least MinNrSpots reflections on the detector
    rng = np.random.default_rng(0)
    MINSPOTS = int(ph.raw.get("MinNrSpots", [8])[0])
    keep, tries, target = [], 0, 20000
    while len(keep) < target and tries < 400000:
        batch = rand_om(rng, 2000)
        tries += 2000
        for OM in batch:
            if len(ph.project(OM)) >= MINSPOTS:
                keep.append(OM)
                if len(keep) >= target:
                    break
    null_oms = np.array(keep)
    print(f"  indexability-matched null: {len(null_oms)} of {tries} random orientations "
          f"put >= {MINSPOTS} reflections on the detector ({len(null_oms)/tries*100:.1f}%)",
          flush=True)

    print("\n=== POLE DENSITY vs INDEXABILITY-MATCHED NULL ===")
    print(f"  {'direction':>12} {'measured max MRD':>18} {'null max MRD':>14} {'p':>8}")
    results = {}
    for name, hkl in (("c-axis [0001]", (0, 0, 1)),
                      ("a-axis [2-1-10]", (1, 0, 0)),
                      ("[10-10]", (0, 1, 0))):
        d_meas = axis_directions(oms_rep, hkl, ph.B)
        d_null = axis_directions(null_oms, hkl, ph.B)

        # density on an equal-area grid of the upper hemisphere
        def density(d, nb=24):
            dd = np.where(d[:, 2:3] < 0, -d, d)          # fold to upper hemisphere
            ct = dd[:, 2]                                 # equal-area in cos(theta)
            ph_ = np.arctan2(dd[:, 1], dd[:, 0])
            H, _, _ = np.histogram2d(ct, ph_, bins=[nb, 2 * nb],
                                     range=[[0, 1], [-np.pi, np.pi]])
            return H / H.mean()                           # multiples of random

        Hm, Hn = density(d_meas), density(d_null)
        mm, mn = Hm.max(), Hn.max()
        # null distribution of the max, by resampling the null to the measured size
        boot = []
        for _ in range(400):
            idx = rng.choice(len(d_null), size=len(d_meas), replace=True)
            boot.append(density(d_null[idx]).max())
        boot = np.array(boot)
        p = float((boot >= mm).mean())
        print(f"  {name:>12} {mm:18.2f} {np.median(boot):14.2f} {p:8.4g}")
        results[name] = dict(measured_max=float(mm), null_max_median=float(np.median(boot)),
                             p=p)

    # instance-weighted, for contrast: this is what you get if you forget that
    # one grain contributes many positions
    print("\n  instance-weighted (NOT independent -- shown for contrast only):")
    for name, hkl in (("c-axis [0001]", (0, 0, 1)),):
        d = axis_directions(oms, hkl, ph.B)
        dd = np.where(d[:, 2:3] < 0, -d, d)
        H, _, _ = np.histogram2d(dd[:, 2], np.arctan2(dd[:, 1], dd[:, 0]),
                                 bins=[24, 48], range=[[0, 1], [-np.pi, np.pi]])
        print(f"    {name}: max {H.max()/H.mean():.2f} MRD "
              f"(vs {results[name]['measured_max']:.2f} per-grain)")

    np.savez(f"{outdir}/texture.npz", oms_rep=oms_rep, null_oms=null_oms)
    import json
    json.dump(results, open(f"{outdir}/texture.json", "w"), indent=1)
    print(f"\nwrote {outdir}/texture.json")
    print("TEXTURE_DONE", flush=True)


if __name__ == "__main__":
    main()
