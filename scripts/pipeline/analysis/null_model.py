"""Measure the random-orientation null FOR THIS SCAN, rather than reusing the
number quoted for the SmallArea scan (peak counts and predicted-reflection counts
differ, so lambda differs).

For a sample of frames spread across the raster: detect SNR>8 peaks with the same
fast downsampled-median background the validator uses, then draw random orientations,
project their full Laue pattern, and count how many predicted reflections land within
TOL px of a real peak. Reports mean / 99th pct / max hits and the analytic Poisson
lambda, per phase.

usage: null_model.py [nframes] [ndraws] [nworkers]
"""
import os
import numpy as np, h5py, glob, json, sys
from math import pi, cos, sin
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from concurrent.futures import ProcessPoolExecutor

W = os.environ.get("LAUE_WORK", "$LAUE_WORK")
TESTSCANS = os.environ.get("LAUE_TESTSCANS", "$LAUE_DATA-2/Thompson_202607/Initial_Indexing_TestScans")
DATA = os.environ.get("LAUE_SCAN_DATA", f"{TESTSCANS}/ID26-10x10um_0p25umStepSize_TestingIndexing")
RESDIR = lambda ph: os.environ.get(f"LAUE_SCAN_{ph.upper()}", f"{TESTSCANS}/laue_Matching_Results/results/{ph}_20260717_161826")
H5LOC = "/entry1/data/data"; HC = 1.2398419739; TOL = 8.0
PREFIX = os.environ.get("LAUE_OUT_PREFIX", "scan")

NFR = int(sys.argv[1]) if len(sys.argv) > 1 else 150
NDR = int(sys.argv[2]) if len(sys.argv) > 2 else 200
NW  = int(sys.argv[3]) if len(sys.argv) > 3 else 4

# Lattice, reflection list, detector geometry and energy window all come from the
# parameter file the indexer itself used -- see laue_material. Set LAUE_PHASES to
# the phases present (single-phase materials: LAUE_PHASES=zn) and
# LAUE_PARAMS_<PHASE> to each params_*.txt.
from frame_peaks import detect_peaks
from laue_material import Phase
PHASES = [p.strip() for p in os.environ.get("LAUE_PHASES", "alpha,beta").split(",") if p.strip()]
BS = {ph: Phase.load(ph) for ph in PHASES}
NPX = next(iter(BS.values())).npx_x
for ph, obj in BS.items():
    print(f"  {ph}: {obj}", flush=True)

def rand_om(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

mapping = json.load(open(f"{RESDIR(PHASES[0])}/frame_mapping.json"))
img2file = {int(k): vv["file"] for k, vv in mapping.items() if isinstance(vv, dict) and "file" in vv}
allimg = sorted(img2file)
sel = allimg[::max(1, len(allimg)//NFR)][:NFR]

def job(inum):
    fn = img2file[inum]
    try:
        with h5py.File(f"{DATA}/{fn}", "r") as f:
            raw = f[H5LOC][()].astype(float)
    except Exception:
        return None
    # Shared with parentbeta_validate: the null gates the validator's output, so
    # the two must detect peaks with identical code. Includes the blooming-streak
    # filter -- an unfiltered detector stacks dozens of false peaks down the bloom
    # of a saturated reflection, which inflates the null.
    xs, ys, _ = detect_peaks(raw, NPX)
    if len(xs) < 5:
        return None
    tree = cKDTree(np.c_[xs, ys]); npeaks = len(xs)
    rng = np.random.default_rng(inum)
    out = {}
    for ph, phase in BS.items():
        hits = []; npred = []
        for _ in range(NDR):
            pr = phase.project(rand_om(rng))
            if not len(pr):
                hits.append(0); npred.append(0); continue
            d, _ = tree.query(pr); hits.append(int((d < TOL).sum())); npred.append(len(pr))
        out[ph] = (np.array(hits), float(np.mean(npred)), npeaks)
    return out

acc = {ph: [] for ph in BS}; lam = {ph: [] for ph in BS}; npk = []
with ProcessPoolExecutor(max_workers=NW) as ex:
    for r in ex.map(job, sel):
        if r is None:
            continue
        for ph, (h, mp, npeaks) in r.items():
            acc[ph].append(h)
            lam[ph].append(mp*npeaks*pi*TOL*TOL/(NPX*NPX))
        npk.append(list(r.values())[0][2])

print(f"frames sampled: {len(npk)}, draws/frame/phase: {NDR}")
print(f"median SNR>8 peaks per frame: {int(np.median(npk))}")
for ph in BS:
    h = np.concatenate(acc[ph])
    print(f"\n[{ph}] RANDOM-ORIENTATION NULL over {len(h):,} draws")
    print(f"   mean hits {h.mean():.2f}   median {int(np.median(h))}   "
          f"99th pct {np.percentile(h,99):.0f}   99.9th {np.percentile(h,99.9):.0f}   max {h.max()}")
    print(f"   analytic Poisson lambda (mean over frames): {np.mean(lam[ph]):.2f}")
