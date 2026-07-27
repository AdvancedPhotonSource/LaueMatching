"""Full-map background decomposition at all 40,401 positions.

Reads only four 150x150 detector corners plus one central box per frame -- about
180 kB instead of 8.4 MB, so this adds ~2% to the I/O the indexer is already
doing rather than re-reading 357 GB.

Two components, because they mean different things:
  flat pedestal (corners)  -- isotropic emission; Zn K-alpha fluorescence sits here
  halo (centre - corners)  -- forward-peaked: air scatter + thermal diffuse
Only the flat component should track how much Zn is in the beam path.
"""
import numpy as np, h5py, time, os, sys
from concurrent.futures import ThreadPoolExecutor

S = "$LAUE_DATA-2/bt_34ide_jul26/sampleG/scan1_Laue2D"
W = "$LAUE_WORK"
NR = 201
NTH = int(sys.argv[1]) if len(sys.argv) > 1 else 6

def one(i):
    try:
        with h5py.File(f"{S}/G19_scan1_Laue2D_{i}.h5", "r") as h:
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

idx = list(range(1, NR * NR + 1))
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

flat = out[:, 1].reshape(NR, NR)      # row-major: row = slow axis, col = X
halo = out[:, 3].reshape(NR, NR)
cen  = out[:, 2].reshape(NR, NR)
spread = out[:, 4].reshape(NR, NR)
i0   = out[:, 5].reshape(NR, NR)
sx   = out[:, 6].reshape(NR, NR)
sz   = out[:, 7].reshape(NR, NR)

np.savez_compressed(f"{W}/analysis_out/full_pedestal.npz",
                    flat=flat, halo=halo, centre=cen, corner_spread=spread,
                    i0=i0, sampleX=sx, sampleZ=sz)

def stat(nm, a):
    v = a[np.isfinite(a)]
    print(f"  {nm:16s} min {v.min():8.1f} med {np.median(v):8.1f} max {v.max():8.1f}  "
          f"ratio {v.max()/max(v.min(),1e-9):5.2f}x  sd {v.std():6.2f}")

print("\n=== full map (201x201) ===")
for nm, a in (("flat pedestal", flat), ("halo excess", halo), ("centre", cen),
              ("corner spread", spread), ("I0", i0)):
    stat(nm, a)

bad = ~np.isfinite(flat)
print(f"\n  unreadable frames: {bad.sum()}")
lowbeam = np.isfinite(i0) & (i0 < 0.5 * np.nanmedian(i0))
print(f"  beam-dropout frames (I0 < 50% median): {lowbeam.sum()}")
np.save(f"{W}/analysis_out/beam_dropout_mask.npy", lowbeam)
print("FULLPED_DONE", flush=True)
