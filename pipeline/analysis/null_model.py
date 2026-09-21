"""Measure the random-orientation null FOR THIS SCAN, rather than reusing the
number quoted for the SmallArea scan (peak counts and predicted-reflection counts
differ, so lambda differs).

For a sample of frames spread across the raster: detect SNR>8 peaks with the same
fast downsampled-median background the validator uses, then draw random orientations,
project their full Laue pattern, and count how many predicted reflections land within
TOL px of a real peak. Reports mean / 99th pct / max hits and the analytic Poisson
lambda, per phase.

Both hit statistics are measured, with frame_peaks.count_matched_peaks -- the
same function the validator uses:
    nhit           predicted reflections that land on a peak (harmonics stack)
    nhit_distinct  distinct observed peaks explained
and written, per phase, to

    $LAUE_WORK/peel_map/${LAUE_OUT_PREFIX}_null.json
    {"schema": 1, "prefix": ..., "tol_px": 8.0, "n_frames": ..., "draws_per_frame": ...,
     "phases": {"<phase>": {"params": ..., "poisson_lambda_mean": ...,
                            "nhit":          {"statistic": "nhit", "n_draws": N,
                                              "mean", "median", "p99", "p999", "max"},
                            "nhit_distinct": {... same keys ...}}}}

which the gate scripts (regrain, empirical_gate, validated_figures,
collect_scan_metrics) read through frame_peaks.load_null. LAUE_NULLMAX_<PHASE>
overrides the max and must be the max of the statistic named by LAUE_GATE_STAT.

usage: null_model.py [nframes] [ndraws] [nworkers]
"""
import os
import numpy as np, h5py, glob, json, sys
from math import pi, cos, sin
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from concurrent.futures import ProcessPoolExecutor

# No defaults for data locations: a default here once pointed at another
# campaign's frames and measured THAT scan's null for this one.
W = os.environ.get("LAUE_WORK") or sys.exit("LAUE_WORK is not set (output root; peel_map/ is written under it)")
DATA = os.environ.get("LAUE_SCAN_DATA") or sys.exit("LAUE_SCAN_DATA is not set (folder of this scan's raw frames)")
def RESDIR(ph):
    v = f"LAUE_SCAN_{ph.upper()}"
    return os.environ.get(v) or sys.exit(f"{v} is not set (indexing-run directory holding frame_mapping.json)")
H5LOC = "/entry1/data/data"; HC = 1.2398419739; TOL = 8.0
from frame_peaks import out_prefix
PREFIX = out_prefix()

NFR = int(sys.argv[1]) if len(sys.argv) > 1 else 150
NDR = int(sys.argv[2]) if len(sys.argv) > 2 else 200
NW  = int(sys.argv[3]) if len(sys.argv) > 3 else 4

# Lattice, reflection list, detector geometry and energy window all come from the
# parameter file the indexer itself used -- see laue_material. Set LAUE_PHASES to
# the phases present (single-phase materials: LAUE_PHASES=zn) and
# LAUE_PARAMS_<PHASE> to each params_*.txt.
from frame_peaks import detect_peaks, count_matched_peaks, null_json_path
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
    xs, ys, _ = detect_peaks(raw)
    if len(xs) < 5:
        return None
    tree = cKDTree(np.c_[xs, ys]); npeaks = len(xs)
    rng = np.random.default_rng(inum)
    out = {}
    for ph, phase in BS.items():
        hits = []; hits_d = []; npred = []
        for _ in range(NDR):
            pr = phase.project(rand_om(rng))
            if not len(pr):
                hits.append(0); hits_d.append(0); npred.append(0); continue
            # identical counter to parentbeta_validate -- the null must be
            # measured with the same statistic it will gate (invariant 15b).
            hd, h = count_matched_peaks(tree, pr, TOL)
            hits.append(h); hits_d.append(hd); npred.append(len(pr))
        out[ph] = (np.array(hits), float(np.mean(npred)), npeaks, np.array(hits_d))
    return out

# Everything below drives the process pool. Guarded so the script also runs under
# the "spawn" start method (macOS default), where each worker re-imports this
# module: the workers need only the definitions above.
if __name__ == "__main__":
    acc = {ph: [] for ph in BS}; accd = {ph: [] for ph in BS}; lam = {ph: [] for ph in BS}; npk = []
    with ProcessPoolExecutor(max_workers=NW) as ex:
        for r in ex.map(job, sel):
            if r is None:
                continue
            for ph, (h, mp, npeaks, hd) in r.items():
                acc[ph].append(h)
                accd[ph].append(hd)
                lam[ph].append(mp*npeaks*pi*TOL*TOL/(NPX*NPX))
            npk.append(list(r.values())[0][2])

    if not npk:
        sys.exit(f"no frame yielded a null sample ({len(sel)} tried): every read failed or had "
                 f"< 5 peaks. Check LAUE_SCAN_DATA={DATA} and the frame_mapping.json file names.")
    print(f"frames sampled: {len(npk)}, draws/frame/phase: {NDR}")
    print(f"median SNR>8 peaks per frame: {int(np.median(npk))}")
    for ph in BS:
        h = np.concatenate(acc[ph])
        print(f"\n[{ph}] RANDOM-ORIENTATION NULL over {len(h):,} draws")
        print(f"   mean hits {h.mean():.2f}   median {int(np.median(h))}   "
              f"99th pct {np.percentile(h,99):.0f}   99.9th {np.percentile(h,99.9):.0f}   max {h.max()}")
        print(f"   analytic Poisson lambda (mean over frames): {np.mean(lam[ph]):.2f}")
        hd = np.concatenate(accd[ph])
        print(f"   [DISTINCT observed peaks -- invariant 15b] "
              f"mean {hd.mean():.2f}   median {int(np.median(hd))}   "
              f"99.9th {np.percentile(hd,99.9):.0f}   MAX {hd.max()}")
        print(f"   stacking in the null (predicted/distinct): {h.mean()/max(hd.mean(),1e-9):.3f}x")
        print(f"   -> pass LAUE_NULLMAX_{ph.upper()}={h.max()} to gate on nhit, "
              f"or {hd.max()} to gate on nhit_distinct. They are NOT interchangeable.")


    def _stats(arr, name):
        return {"statistic": name, "n_draws": int(len(arr)),
                "mean": round(float(arr.mean()), 4), "median": float(np.median(arr)),
                "p99": round(float(np.percentile(arr, 99)), 3),
                "p999": round(float(np.percentile(arr, 99.9)), 3),
                "max": int(arr.max())}


    out = {"schema": 1, "prefix": PREFIX, "tol_px": TOL, "n_frames": len(npk),
           "draws_per_frame": NDR, "phases": {}}
    for ph in BS:
        out["phases"][ph] = {
            "params": BS[ph].params_path,
            "poisson_lambda_mean": round(float(np.mean(lam[ph])), 4),
            "nhit": _stats(np.concatenate(acc[ph]), "nhit"),
            "nhit_distinct": _stats(np.concatenate(accd[ph]), "nhit_distinct"),
        }
    os.makedirs(f"{W}/peel_map", exist_ok=True)
    jpath = null_json_path(W, PREFIX)
    with open(jpath, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {jpath}")
