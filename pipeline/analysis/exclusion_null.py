"""Empirical null for the ALPHA-EXCLUSION test.

The census scores each beta grain by how many of its predicted reflections land on
peaks that no validated alpha grain of that frame explains, and assigns significance
with an analytic Poisson model. That model assumes uniformly scattered peaks; real
peak fields are clustered, and we already showed the analytic gate is optimistic
(measured null reached 15 hits where Poisson(1.91) forbids it).

So measure the same statistic under the null: on each sampled frame, build the
alpha-claimed mask exactly as the census does, then draw RANDOM beta orientations and
count how many of their reflections land on alpha-unclaimed peaks. Both counts are
reported, with the census's own detector and counter (frame_peaks.detect_peaks and
count_matched_peaks): nhit (predicted reflections; harmonics stack) and nhit_distinct
(distinct unclaimed peaks). Compare a census count only with the null of the same name.

usage: exclusion_null.py [nframes] [ndraws] [nworkers]
"""
import os
import zlib
import numpy as np, h5py, json, sys
from math import pi, cos, sin
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from concurrent.futures import ProcessPoolExecutor

# No defaults for data locations: the default here used to be another campaign's frames.
W = os.environ.get("LAUE_WORK") or sys.exit("LAUE_WORK is not set (peel_map/ is read under it)")
DATA = os.environ.get("LAUE_SCAN_DATA") or sys.exit("LAUE_SCAN_DATA is not set (folder of this scan's raw frames)")
H5LOC = "/entry1/data/data"; TOL = 8.0
from frame_peaks import out_prefix
PREFIX = out_prefix()
from laue_material import Phase
from frame_peaks import detect_peaks, count_matched_peaks
_PH_A = Phase.load("alpha")
_PH_B = Phase.load("beta")
NPX = _PH_A.npx_x
NFR = int(sys.argv[1]) if len(sys.argv) > 1 else 120
NDR = int(sys.argv[2]) if len(sys.argv) > 2 else 150
NW  = int(sys.argv[3]) if len(sys.argv) > 3 else 12

def rand_om(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

za = np.load(f"{W}/peel_map/{PREFIX}_alpha_validated.npz", allow_pickle=True)
a_oms = za["oms"]; a_fr = np.asarray([str(f) for f in za["frames"]])
alpha_by_frame = {}
for OM, fn in zip(a_oms, a_fr):
    alpha_by_frame.setdefault(fn, []).append(OM)

zb = np.load(f"{W}/peel_map/{PREFIX}_beta_validated.npz", allow_pickle=True)
b_fr = np.asarray([str(f) for f in zb["frames"]])
frames = sorted(set(b_fr) & set(alpha_by_frame))
sel = frames[::max(1, len(frames)//NFR)][:NFR]
print(f"frames with both phases: {len(frames)}, sampling {len(sel)}", flush=True)

def job(fn):
    try:
        with h5py.File(f"{DATA}/{fn}", "r") as f:
            raw = f[H5LOC][()].astype(float)
    except Exception:
        return None
    # identical detector and counter to beta_alpha_exclusion_census.py -- this is
    # its null, so the two must change together
    xs, ys, _ = detect_peaks(raw)
    if len(xs) < 5:
        return None
    tree = cKDTree(np.c_[xs, ys]); npeaks = len(xs)
    claimed = np.zeros(npeaks, bool)
    for OMa in alpha_by_frame.get(fn, []):
        pra = _PH_A.project(OMa)
        if len(pra):
            da, ja = tree.query(pra); claimed[ja[da < TOL]] = True
    # Stable per-frame seed. It was abs(hash(fn)), and Python salts str hashes per
    # process (PYTHONHASHSEED), so the null differed run to run and worker to worker.
    rng = np.random.default_rng(zlib.crc32(os.path.basename(fn).encode()))
    uniq = []; uniq_d = []
    for _ in range(NDR):
        pr = _PH_B.project(rand_om(rng))
        ud, u = count_matched_peaks(tree, pr, TOL, exclude=claimed)
        uniq.append(u); uniq_d.append(ud)
    return np.array(uniq), np.array(uniq_d), int(claimed.sum()), npeaks

# Everything below drives the process pool. Guarded so the script also runs under
# the "spawn" start method (macOS default), where each worker re-imports this
# module: the workers need only the definitions above.
if __name__ == "__main__":
    acc = []; accd = []; frac_claimed = []
    with ProcessPoolExecutor(max_workers=NW) as ex:
        for r in ex.map(job, sel):
            if r is None:
                continue
            u, ud, ncl, npk = r
            acc.append(u); accd.append(ud); frac_claimed.append(ncl/npk)

    if not acc:
        sys.exit(f"no frame yielded a null sample ({len(sel)} tried): every read failed or had "
                 f"< 5 peaks. Check LAUE_SCAN_DATA={DATA}.")
    print(f"\nalpha claims {100*np.mean(frac_claimed):.1f}% of peaks on average")
    for name, arr in (("nhit", np.concatenate(acc)), ("nhit_distinct", np.concatenate(accd))):
        print(f"\n[{name}] RANDOM-BETA NULL on ALPHA-UNCLAIMED peaks, {len(arr):,} draws")
        print(f"  mean {arr.mean():.2f}  median {int(np.median(arr))}  "
              f"99th {np.percentile(arr,99):.0f}  99.9th {np.percentile(arr,99.9):.0f}  max {arr.max()}")
    print("\nCompare with the census's per-grain alpha-unclaimed hit counts, SAME statistic "
          "(the census prints both).")
