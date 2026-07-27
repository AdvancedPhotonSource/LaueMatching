"""Can we separate substrate from deposit signal within each frame?

Each frame superimposes many grains (the whole illuminated column). In a plated
region that is deposit near the surface PLUS substrate beneath it. Two
independent handles should tag each indexed orientation by layer:

  A. persistence  -- substrate is a continuous polycrystal, so a substrate grain
     recurs over a wide contiguous area (including under the deposit); deposit
     grains are fine/local. Footprint = how many positions a grain's cluster spans.
  B. spectral hardness -- in REFLECTION geometry a substrate reflection round-trips
     (in + out) through the overlying deposit and is absorption-hardened; a deposit
     reflection sits at the surface and stays soft. Per-orientation median energy
     of the ASSIGNED (detected) spots is the tag.

THE TEST: do A and B agree? If large-footprint grains also have harder spectra,
the two independent signatures of "substrate" coincide -> we have separated the
layers. This script computes per-orientation median assigned-spot energy, joins
it to cluster footprint, and reports the relationship (with a null).

Then it splits the map into a substrate layer and a deposit layer and writes both.
"""
import os, sys, glob, json
import numpy as np, h5py
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "$LAUE_WORK/analysis")
from laue_material import Phase

W = "$LAUE_WORK"
NR = 201
HC = 1.2398419739
PH = Phase(f"{W}/params/params_Zn_h_s1.txt", "zn")
B, ROTI, P, KI, dx, dy, NPX = PH.B, PH.roti, PH.P, PH.ki, PH.dx, PH.dy, PH.npx_x
Elo, Ehi = PH.Elo, PH.Ehi
# stream layout columns
OM0, GCOL = 23, 1
S_G, S_H, S_K, S_L = 1, 3, 4, 5

# ---- global frame -> (output.h5) map across all 6 shards --------------------
def build_frame_map():
    fmap = {}
    for d in sorted(glob.glob(f"{W}/results/alpha_20260724_13[5-9]*/")):
        prov = open(f"{d}/provenance.json").read()
        if "params_Zn_h_s" not in prov:
            continue
        mp = json.load(open(f"{d}/frame_mapping.json"))
        for k, v in mp.items():
            if isinstance(v, dict) and "file" in v:
                fmap[v["file"]] = f"{d}/results/image_{int(k):05d}.output.h5"
    return fmap


def spot_energies_for(OM, hkls):
    q = (OM @ B @ hkls.T).T
    ql = np.linalg.norm(q, axis=1); ok = ql > 1e-9
    E = np.full(len(hkls), np.nan)
    if not ok.any():
        return E
    qh = q[ok] / ql[ok, None]; st = -qh[:, 2]
    good = st > 1e-9
    Eo = np.where(good, HC * ql[ok] / np.where(good, st, np.nan) / (4 * np.pi), np.nan)
    E[ok] = Eo
    return E


def process_frame(args):
    """For one frame: match each requested OM to its output-grain, return the
    median energy of that grain's assigned spots."""
    fn, oms_req, h5path = args
    if not os.path.exists(h5path):
        return None
    try:
        with h5py.File(h5path, "r") as h:
            ori = h["entry/results/filtered_orientations"][()]
            sp = h["entry/results/filtered_spots"][()]
    except Exception:
        return None
    if not len(ori):
        return None
    out_oms = ori[:, OM0:OM0 + 9]                     # (M,9)
    out_g = ori[:, GCOL].astype(int)
    med = np.full(len(oms_req), np.nan)
    nasg = np.zeros(len(oms_req), int)
    for i, om in enumerate(oms_req):
        d = np.abs(out_oms - om.reshape(9)).max(axis=1)
        j = int(d.argmin())
        if d[j] > 1e-6:
            continue
        g = out_g[j]
        m = sp[:, S_G].astype(int) == g
        if not m.any():
            continue
        hkl = sp[m][:, [S_H, S_K, S_L]]
        E = spot_energies_for(om, hkl)
        E = E[np.isfinite(E)]
        if len(E):
            med[i] = np.median(E); nasg[i] = len(E)
    return fn, med, nasg


def main():
    z = np.load(f"{W}/peel_map/full_zn_clustered.npz", allow_pickle=True)
    oms, lab, fr, nh = z["oms"], z["labels"], z["frames"], z["nhit"].astype(int)
    n = np.array([int(str(f).split("_")[-1].split(".")[0]) for f in fr])
    row, col = (n - 1) // NR, (n - 1) % NR
    foot = np.bincount(lab)[lab]                       # footprint per instance
    print(f"{len(oms)} instances, {len(np.unique(lab))} clusters", flush=True)

    fmap = build_frame_map()
    print(f"frame map: {len(fmap)} frames", flush=True)

    # group instance indices by frame
    by_frame = {}
    for idx, f in enumerate(fr):
        by_frame.setdefault(str(f), []).append(idx)
    tasks = [(f, oms[ix], fmap.get(f, "")) for f, ix in by_frame.items() if f in fmap]
    print(f"{len(tasks)} frames to read", flush=True)

    medE = np.full(len(oms), np.nan); nasg = np.zeros(len(oms), int)
    idx_by_frame = by_frame
    done = 0
    with ProcessPoolExecutor(max_workers=int(os.environ.get("NW", "32"))) as ex:
        for res in ex.map(process_frame, tasks, chunksize=16):
            done += 1
            if res is None:
                continue
            f, med, na = res
            ix = idx_by_frame[f]
            for k, j in enumerate(ix):
                medE[j] = med[k]; nasg[j] = na[k]
            if done % 4000 == 0:
                print(f"  {done}/{len(tasks)} frames", flush=True)

    ok = np.isfinite(medE)
    print(f"\nper-orientation median energy computed for {ok.sum()} / {len(oms)} instances")

    # ---- THE TEST: footprint (persistence) vs spectral hardness -------------
    fo, me = np.log10(foot[ok]), medE[ok]
    r = float(np.corrcoef(fo, me)[0, 1])
    rng = np.random.default_rng(0)
    null = np.array([float(np.corrcoef(fo, rng.permutation(me))[0, 1]) for _ in range(2000)])
    p = float((np.abs(null) >= abs(r)).mean())
    print("\n=== persistence vs spectral hardness (per orientation) ===")
    print(f"  corr(log footprint, median energy) = {r:+.3f}   perm p = {p:.4g}")
    print("  PREDICTED if big grains = substrate seen through deposit: POSITIVE")

    for lo, hi, name in [(1, 1, "singletons (deposit?)"), (2, 9, "small 2-9"),
                         (10, 49, "medium 10-49"), (50, 199, "large 50-199"),
                         (200, 10 ** 9, "very large >=200 (substrate?)")]:
        m = ok & (foot >= lo) & (foot <= hi)
        if m.sum():
            print(f"  {name:32s} n={m.sum():6d}  median E = {np.median(medE[m]):6.2f} keV  "
                  f"frac<15keV = {(medE[m] < 15).mean():.3f}")

    # ---- split into substrate vs deposit layers ----------------------------
    # substrate: large footprint AND/OR hard; deposit: small AND soft.
    med_all = np.nanmedian(medE)
    big = foot >= 50
    hard = medE >= med_all
    substrate = ok & (big | (foot >= 10) & hard)
    deposit = ok & ~substrate
    print(f"\n=== layer split ===")
    print(f"  substrate-tagged instances: {substrate.sum()}  "
          f"({len(set(zip(row[substrate].tolist(), col[substrate].tolist())))} positions)")
    print(f"  deposit-tagged instances:   {deposit.sum()}  "
          f"({len(set(zip(row[deposit].tolist(), col[deposit].tolist())))} positions)")
    # how many positions carry BOTH a substrate and a deposit orientation?
    sub_pos = set(zip(row[substrate].tolist(), col[substrate].tolist()))
    dep_pos = set(zip(row[deposit].tolist(), col[deposit].tolist()))
    both = sub_pos & dep_pos
    print(f"  positions with BOTH layers indexed: {len(both)} "
          f"({len(both) / max(len(sub_pos | dep_pos), 1) * 100:.1f}% of occupied) "
          f"-- these are where we see substrate THROUGH the deposit")

    np.savez(f"{W}/analysis_out/layer_separation.npz",
             oms=oms, labels=lab, row=row, col=col, footprint=foot,
             medE=medE, nassigned=nasg, nhit=nh,
             substrate=substrate, deposit=deposit)
    print(f"\nwrote layer_separation.npz")
    print("SEPARATION_DONE", flush=True)


if __name__ == "__main__":
    main()
