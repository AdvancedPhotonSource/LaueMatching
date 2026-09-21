"""Full-map background decomposition at every raster position (40,401 on a 201x201 scan).

Reads only four 150x150 detector corners plus one central box per frame -- about
180 kB instead of 8.4 MB, so this adds ~2% to the I/O the indexer is already
doing rather than re-reading 357 GB.

Two components, because they mean different things:
  flat pedestal (corners)  -- isotropic emission; Zn K-alpha fluorescence sits here
  halo (centre - corners)  -- forward-peaked: air scatter + thermal diffuse
Only the flat component should track how much Zn is in the beam path.

Environment (required, no defaults):
  LAUE_SCAN_DATA   folder of the scan's raw .h5 frames, named <prefix>_<N>.h5
  LAUE_WORK        work directory; writes analysis_out/full_pedestal.npz
  LAUE_NR, LAUE_NROWS  raster columns and rows; frame N goes to
                   row (N-1)//LAUE_NR, col (N-1)%LAUE_NR (raster.raster_positions)
Optional:
  LAUE_FRAME_PREFIX  only frames whose name starts with this (if the folder holds
                   other .h5 files)

usage: fullped.py [n_threads]
"""
import numpy as np, h5py, time, os, sys, glob
from concurrent.futures import ThreadPoolExecutor

from raster import frame_number, raster_positions, raster_shape

S = os.environ.get("LAUE_SCAN_DATA")
if not S:
    sys.exit("LAUE_SCAN_DATA is not set (folder of the scan's raw .h5 frames)")
W = os.environ.get("LAUE_WORK")
if not W:
    sys.exit("LAUE_WORK is not set (work directory; output goes to analysis_out/)")
# created now, before the frame-read pass, so a bad LAUE_WORK fails in seconds
# rather than after reading every frame
os.makedirs(f"{W}/analysis_out", exist_ok=True)
NROWS, NR = raster_shape()             # LAUE_NROWS, LAUE_NR -- required
NTH = int(sys.argv[1]) if len(sys.argv) > 1 else 6

_pat = f"{S}/{os.environ.get('LAUE_FRAME_PREFIX', '')}*.h5"
FILES = {}
for _f in glob.glob(_pat):
    try:
        FILES[frame_number(_f)] = _f
    except ValueError:
        continue
if not FILES:
    sys.exit(f"no frames matched {_pat}")

def one(i):
    try:
        with h5py.File(FILES[i], "r") as h:
            ds = h["entry1/data/data"]
            tl = ds[:150, :150]; tr = ds[:150, -150:]
            bl = ds[-150:, :150]; br = ds[-150:, -150:]
            ctr = ds[949:1099, 949:1099]
            i0 = int(h["entry1/monitor/I0"][0])
            sx = float(h["entry1/sample/sampleX"][0])
            sy = float(h["entry1/sample/sampleY"][0])
            sz = float(h["entry1/sample/sampleZ"][0])
    except Exception:
        return i, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    corners = np.concatenate([tl.ravel(), tr.ravel(), bl.ravel(), br.ravel()])
    flat = float(np.median(corners))
    cen = float(np.median(ctr))
    # corner-to-corner spread: a check that "flat" really is flat
    spread = float(np.std([np.median(x) for x in (tl, tr, bl, br)]))
    return i, flat, cen, cen - flat, spread, float(i0), sx, sz

idx = list(range(1, NROWS * NR + 1))
extra = sorted(set(FILES) - set(idx))
if extra:
    sys.exit(f"{len(extra)} frames (e.g. {extra[:3]}) lie beyond a {NROWS} x {NR} raster: "
             f"LAUE_NROWS / LAUE_NR are wrong for this scan")
t0 = time.time()
out = np.full((len(idx), 8), np.nan)
done = 0
with ThreadPoolExecutor(NTH) as ex:
    for r in ex.map(one, idx, chunksize=64):
        out[r[0] - 1] = r
        done += 1
        if done % 5000 == 0:
            el = time.time() - t0
            print(f"  {done}/{len(idx)}  {el:.0f}s  ETA {el/done*(len(idx)-done):.0f}s", flush=True)
print(f"read {done} frames in {time.time()-t0:.0f}s", flush=True)

# place every frame by the shared convention (row-major: row = slow axis, col = X)
_r, _c = raster_positions([f"f_{i}.h5" for i in idx], shape=(NROWS, NR))
def grid(k):
    g = np.full((NROWS, NR), np.nan)
    g[_r, _c] = out[:, k]
    return g
flat, cen, halo, spread, i0, sx, sz = (grid(k) for k in range(1, 8))

# scan_label: what these maps are of, for figure titles (zn_report_figures.py)
np.savez_compressed(f"{W}/analysis_out/full_pedestal.npz",
                    flat=flat, halo=halo, centre=cen, corner_spread=spread,
                    i0=i0, sampleX=sx, sampleZ=sz,
                    scan_label=os.path.basename(os.path.normpath(S)))

def stat(nm, a):
    v = a[np.isfinite(a)]
    print(f"  {nm:16s} min {v.min():8.1f} med {np.median(v):8.1f} max {v.max():8.1f}  "
          f"ratio {v.max()/max(v.min(),1e-9):5.2f}x  sd {v.std():6.2f}")

print(f"\n=== full map ({NR}x{NROWS}) ===")
for nm, a in (("flat pedestal", flat), ("halo excess", halo), ("centre", cen),
              ("corner spread", spread), ("I0", i0)):
    stat(nm, a)

bad = ~np.isfinite(flat)
print(f"\n  unreadable frames: {bad.sum()}")
lowbeam = np.isfinite(i0) & (i0 < 0.5 * np.nanmedian(i0))
print(f"  beam-dropout frames (I0 < 50% median): {lowbeam.sum()}")
np.save(f"{W}/analysis_out/beam_dropout_mask.npy", lowbeam)
print("FULLPED_DONE", flush=True)
