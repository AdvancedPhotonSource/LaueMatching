"""Validate all batch-peeled orientations per frame (Poisson spot-test), then
cluster validated instances map-wide (symmetry of LAUE_PHASE's space group) into
confidence-tiered grains.

Validator (analytic version of the test-frame SNR check): for each orientation,
h = predicted reflections landing within TOL px of a real peak (SNR>8 peaks of
that frame from frame_peaks.detect_peaks; nhit_distinct, the distinct peaks
explained, is saved alongside). Null: Poisson lambda = n_pred * N_peaks * pi*TOL^2/Npx^2.
VERIFIED if P(X >= h) < 1e-4.
"""
import os
import sys
import numpy as np, h5py, glob
from math import cos, sin, pi
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from scipy.stats import poisson
from concurrent.futures import ProcessPoolExecutor

# No default for the raw frames: the default here used to be another campaign's folder.
WORK = os.environ.get("LAUE_WORK") or sys.exit("LAUE_WORK is not set (peel_map/ is read and written under it)")
DATA = os.environ.get("LAUE_SCAN_DATA") or sys.exit("LAUE_SCAN_DATA is not set (folder of the raw frames)")
H5LOC = os.environ.get("LAUE_H5LOC", "/entry1/data/data")
from laue_material import phase_name
PHASE = phase_name()      # LAUE_PHASE, or the single phase in LAUE_PHASES
HC = 1.2398419739; TOL = 8.0

# Lattice, reflection list and detector geometry come from the parameter file
# the indexer used (see laue_material) -- not a second copy of the constants.
from laue_material import Phase
from frame_peaks import detect_peaks, count_matched_peaks, analytic_gate_note
_ph = Phase.load(PHASE)
B = _ph.B; HKLS = _ph.hkls; NPX = _ph.npx_x

def project(OM):
    return _ph.project(OM)

def validate_frame(item):
    fn, oms_flat = item
    oms = oms_flat.reshape(-1, 3, 3)
    with h5py.File(f"{DATA}/{fn}", "r") as f:
        raw = f[H5LOC][()].astype(float)
    # shared detector (frame_peaks). BEHAVIOUR CHANGE: replaces this script's
    # full-frame median_filter(25) background with the detector the null uses.
    xs, ys, _ = detect_peaks(raw)
    if len(xs) < 5:
        return fn, np.zeros(len(oms), bool), np.zeros(len(oms)), np.zeros(len(oms)), 0
    tree = cKDTree(np.c_[xs, ys]); npeaks = len(xs)
    ok = np.zeros(len(oms), bool); hits = np.zeros(len(oms)); hits_d = np.zeros(len(oms))
    for i, OM in enumerate(oms):
        pr = project(OM)
        if len(pr) == 0: continue
        # h = predicted reflections on a peak (the Poisson gate's statistic);
        # h_dist = distinct peaks explained (harmonics not stacked)
        h_dist, h = count_matched_peaks(tree, pr, TOL)
        lam = len(pr) * npeaks * pi * TOL * TOL / (NPX*NPX)
        ok[i] = poisson.sf(h - 1, lam) < 1e-4
        hits[i] = h; hits_d[i] = h_dist
    return fn, ok, hits, hits_d, npeaks

if __name__ == "__main__":
    analytic_gate_note("map_validate_cluster")
    with np.load(f"{WORK}/peel_map/accepted_per_frame.npz") as z:
        ACC = {fn: z[fn].copy() for fn in z.files}   # eager load: no shared zip handle across forks
    results = {}
    with ProcessPoolExecutor(max_workers=32) as ex:
        for fn, ok, hits, hits_d, npeaks in ex.map(validate_frame, list(ACC.items())):
            results[fn] = (ok, hits, hits_d, npeaks)
    tot = sum(len(v[0]) for v in results.values())
    ver = sum(int(v[0].sum()) for v in results.values())
    print(f"instances: {tot}, VERIFIED (Poisson p<1e-4): {ver} ({100*ver/tot:.0f}%)", flush=True)

    # map-wide clustering of VERIFIED instances; operators follow the phase's
    # space group (this used to be a hard-coded hex table whatever the phase)
    oms_v, fns_v, nh_v, nd_v = [], [], [], []
    for fn, (ok, hits, hits_d, npeaks) in results.items():
        oms = ACC[fn].reshape(-1, 3, 3)
        for OM, o, h, hd in zip(oms, ok, hits, hits_d):
            if o: oms_v.append(OM); fns_v.append(fn); nh_v.append(h); nd_v.append(hd)
    if nh_v:
        print(f"hits per verified instance: nhit median {int(np.median(nh_v))}, "
              f"nhit_distinct median {int(np.median(nd_v))}", flush=True)
    oms_v = np.array(oms_v)
    print(f"clustering {len(oms_v)} verified instances...", flush=True)
    labels = np.full(len(oms_v), -1); cid = 0
    for i in range(len(oms_v)):
        if labels[i] >= 0: continue
        un = np.where(labels < 0)[0]
        d = _ph.misorientation(oms_v[i], oms_v[un]); labels[un[d < 0.7]] = cid; cid += 1
    counts = np.bincount(labels)
    print(f"verified grains (clusters): {cid}")
    for k in (1, 2, 3, 5, 10):
        print(f"  in >= {k} frames: {(counts >= k).sum()}")
    per_frame = ver / max(1, len(results))
    print(f"mean VERIFIED grains per frame: {per_frame:.1f}")
    np.savez(f"{WORK}/peel_map/verified_clusters.npz",
             oms=oms_v, labels=labels, frames=np.array(fns_v),
             nhit=np.array(nh_v, int), nhit_distinct=np.array(nd_v, int))
    print("saved verified_clusters.npz")
