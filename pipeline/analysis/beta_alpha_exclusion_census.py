"""Honest beta census under the ALPHA-EXCLUSION null.

For every verified beta cluster:
  - EACH INSTANCE's own BCC pattern (one per frame it was verified in; the text
    here used to say a cluster-mean pattern, which the code never computed);
  - on each frame the cluster was verified in, count predicted reflections that
    land on an SNR>8 peak NOT claimed by ANY verified alpha grain of that frame
    (TOL px), and the Poisson p on the alpha-UNCLAIMED peak subset only. Counted
    both ways by frame_peaks.count_matched_peaks: nhit (predicted reflections,
    harmonics stack) and nhit_distinct (distinct peaks). The analytic p-values
    are on nhit, the count their lambda models, whatever LAUE_GATE_STAT says
    (printed at run time); both counts are saved;
  - THE GATED p (since 2026-09) is the MATCHED EXCLUSION NULL of exclusion_stats:
    the instance's unclaimed count against the same count with the frame's
    alpha grains replaced by random alpha orientations, K times
    (LAUE_CENSUS_NULL_K, default 99), ties randomised; a cluster's p is the
    Fisher combination over its DISTINCT FRAMES (one instance per frame, the
    highest nhit). The analytic Poisson p and its Fisher combination over
    instances are still computed and saved, labelled NOT CALIBRATED: the
    instances were selected by the validator (p < 1e-4), so their p's are not
    uniform under the null, and on a known-null synthetic that combination put
    100% of null clusters below 0.01 (the matched null: 0.3%; 3.3% at 0.05).
    ``LAUE_CENSUS_P=analytic`` gates on the old combination to reproduce old counts.
  - nfr = number of DISTINCT FRAMES of a cluster (it counted instances).
Report how many beta grains survive gates 1e-2 / 1e-3 / 1e-4, split by
recurrence (>= 2 distinct frames), plus the alpha-unclaimed hit distribution. ("unique" in the saved key names tot_unq / best_punq means
alpha-UNCLAIMED, not winner-take-all and not distinct.)
Writes peel_map/beta_census.npz for the story.
"""
import os
import zlib
import numpy as np, h5py, sys
from math import pi
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from scipy.stats import poisson, combine_pvalues
from concurrent.futures import ProcessPoolExecutor

WORK = os.environ.get("LAUE_WORK") or sys.exit("LAUE_WORK is not set (output root; peel_map/ is read and written under it)")
# Root of the legacy test-scan folders; only the "id6_10x10" key uses it.
TESTSCANS = os.environ.get("LAUE_TESTSCANS", "")
# argv[1]=scan prefix, argv[2]=nworkers. "smallarea" reproduces the original hardcoded run;
# alpha/beta instances are read from <prefix>_{alpha,beta}_validated.npz (or the original
# verified_clusters/beta_verified pair for smallarea).
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "smallarea"
NW = int(sys.argv[2]) if len(sys.argv) > 2 else 32
from frame_peaks import out_prefix
SCANS = {
 # raw frames for the legacy smallarea run now come from LAUE_SCAN_DATA; the
 # hard-coded campaign path that used to sit here resolved on one machine only.
 "smallarea": dict(data=os.environ.get("LAUE_SCAN_DATA", ""),
                   alpha_npz="verified_clusters.npz", beta_npz="beta_verified.npz",
                   out="beta_census"),
 "env": dict(data=os.environ.get("LAUE_SCAN_DATA", ""),
                   alpha_npz=f'{out_prefix()}_alpha_validated.npz',
                   beta_npz=f'{out_prefix()}_beta_validated.npz',
                   out=f'{out_prefix()}_census'),
 # NB legacy key: this is specimen ID26 and the scan is 20x20 um in the sample
 # frame, not 10x10 -- the folder name is wrong. Kept so existing *_id6_10x10_*
 # outputs stay resolvable; prefer the "env" entry for new work.
 "id6_10x10": dict(data=f"{TESTSCANS}/ID26-10x10um_0p25umStepSize_TestingIndexing" if TESTSCANS else "",
                   alpha_npz="id6_10x10_alpha_validated.npz",
                   beta_npz="id6_10x10_beta_validated.npz",
                   out="id6_10x10_census"),
}
if PREFIX not in SCANS:
    sys.exit(f"unknown scan {PREFIX!r}; choose from {sorted(SCANS)}")
CFG = SCANS[PREFIX]; DATA = CFG["data"]
if not DATA or not os.path.isdir(DATA):
    sys.exit(f"raw-frame folder {DATA!r} does not exist -- set LAUE_SCAN_DATA"
             + (" (and LAUE_TESTSCANS for this legacy key)" if PREFIX == "id6_10x10" else ""))
H5LOC = "/entry1/data/data"; TOL = 8.0
from laue_material import Phase
from frame_peaks import detect_peaks, count_matched_peaks, analytic_gate_note, poisson_lambda, require_labels
from exclusion_stats import (claim_mask, cluster_combine, empirical_p, random_rotations,
                             unclaimed_hits)
NULL_K = int(os.environ.get("LAUE_CENSUS_NULL_K", "99"))
P_MODE = os.environ.get("LAUE_CENSUS_P", "matched").strip() or "matched"
if P_MODE not in ("matched", "analytic"):
    sys.exit(f"LAUE_CENSUS_P must be matched or analytic, got {P_MODE!r}")
_PH_A = Phase.load("alpha")
_PH_B = Phase.load("beta")
NPX = _PH_A.npx_x
# The p-values below are computed on this statistic (both counts are saved).
# The analytic Poisson p-values model PREDICTED reflections (nhit); there is no
# closed form for the distinct count, so they stay on nhit whatever LAUE_GATE_STAT
# says (announced at run time). Both counts are saved; compare each with the
# exclusion_null.py distribution of the SAME name.
STAT = "nhit"

# ---- load all verified instances ------------------------------------------
zb = np.load(f"{WORK}/peel_map/{CFG['beta_npz']}", allow_pickle=True)
b_oms, b_lab, b_fr = zb["oms"], zb["labels"], np.asarray([str(f) for f in zb["frames"]])
# refuse unclustered labels BEFORE the full frame pass, not after it
require_labels(b_lab, f"{WORK}/peel_map/{CFG['beta_npz']}")
za = np.load(f"{WORK}/peel_map/{CFG['alpha_npz']}", allow_pickle=True)
a_oms, a_fr = za["oms"], np.asarray([str(f) for f in za["frames"]])
print(f"[{PREFIX}] alpha instances {len(a_oms)}, beta instances {len(b_oms)}, "
      f"beta clusters {b_lab.max()+1}", flush=True)
alpha_by_frame = {}
for OM, fn in zip(a_oms, a_fr):
    alpha_by_frame.setdefault(fn, []).append(OM)

# per-frame job: for each beta instance ON this frame return
# (gidx, hit, unclaimed, hit_distinct, unclaimed_distinct, p_all, p_unq, p_matched)
inst_frame = list(b_fr)
frames = sorted(set(inst_frame))
def frame_job(fn):
    idx = np.where(b_fr == fn)[0]
    with h5py.File(f"{DATA}/{fn}", "r") as f: raw = f[H5LOC][()].astype(float)
    # Shared detector (frame_peaks): same downsampled-median background, 9 px
    # maximum filter and SNR 8 this census always used, plus the plateau / halo /
    # bloom handling -- and the SAME detector exclusion_null.py measures with.
    xs, ys, _ = detect_peaks(raw)
    npeaks = len(xs)
    out = []
    rng = np.random.default_rng(zlib.crc32(os.path.basename(fn).encode()))
    if npeaks < 5:
        for i in idx: out.append((int(i), 0, 0, 0, 0, 1.0, 1.0, float(rng.random())))
        return out
    tree = cKDTree(np.c_[xs, ys])
    a_here = alpha_by_frame.get(fn, [])
    claimed = claim_mask(tree, npeaks, [_PH_A.project(OMa) for OMa in a_here], TOL)
    # matched exclusion null: the same number of alpha grains, random orientations
    null_claims = [claim_mask(tree, npeaks,
                              [_PH_A.project(R) for R in random_rotations(rng, len(a_here))], TOL)
                   for _ in range(NULL_K)]
    n_uncl = int((~claimed).sum())
    for i in idx:
        pr = _PH_B.project(b_oms[i])
        if not len(pr): out.append((int(i), 0, 0, 0, 0, 1.0, 1.0, float(rng.random()))); continue
        # (distinct, predicted) on ALL peaks, and on alpha-UNCLAIMED peaks only
        hit_d, hit = count_matched_peaks(tree, pr, TOL)
        unq_d, unq = count_matched_peaks(tree, pr, TOL, exclude=claimed)
        h_all, h_unq = hit, unq
        lam = poisson_lambda(len(pr), npeaks, TOL, _PH_A.npx_x, _PH_A.npx_y)
        lam_u = poisson_lambda(len(pr), max(n_uncl, 1), TOL, _PH_A.npx_x, _PH_A.npx_y)
        p_all = poisson.sf(h_all-1, lam)
        p_unq = poisson.sf(h_unq-1, lam_u)
        u_null = [unclaimed_hits(tree, pr, TOL, nc) for nc in null_claims]
        p_m = empirical_p(unclaimed_hits(tree, pr, TOL, claimed), u_null, rng)
        out.append((int(i), hit, unq, hit_d, unq_d, float(p_all), float(p_unq), p_m))
    return out

# Everything below drives the process pool. Guarded so the script also runs under
# the "spawn" start method (macOS default), where each worker re-imports this
# module: the workers need only the definitions above.
if __name__ == "__main__":
    analytic_gate_note("beta_alpha_exclusion_census")
    inst_pall = np.ones(len(b_oms)); inst_punq = np.ones(len(b_oms)); inst_pm = np.ones(len(b_oms))
    inst_hit = np.zeros(len(b_oms), int); inst_unq = np.zeros(len(b_oms), int)
    inst_hit_d = np.zeros(len(b_oms), int); inst_unq_d = np.zeros(len(b_oms), int)
    with ProcessPoolExecutor(max_workers=NW) as ex:
        for res in ex.map(frame_job, frames):
            for i, h, u, hd, ud, pa, pu, pm in res:
                inst_pall[i] = pa; inst_punq[i] = pu; inst_hit[i] = h; inst_unq[i] = u
                inst_pm[i] = pm
                inst_hit_d[i] = hd; inst_unq_d[i] = ud

    # ---- aggregate per cluster -------------------------------------------------
    ncl = b_lab.max()+1
    counts = np.bincount(b_lab, minlength=ncl)
    rows = []
    for cid in range(ncl):
        ii = np.where(b_lab == cid)[0]
        best_punq = float(inst_punq[ii].min())
        # ANALYTIC (not calibrated): Fisher over the selected instances' Poisson p
        comb = combine_pvalues(inst_punq[ii], method="fisher")[1] if len(ii) >= 2 else best_punq
        # MATCHED: one instance per distinct frame (highest nhit), Fisher over frames
        nfr, comb_m = cluster_combine(b_fr[ii], inst_pm[ii], inst_hit[ii])
        rows.append((cid, nfr, int(inst_unq[ii].sum()), int(inst_unq[ii].max()),
                     best_punq, float(comb), int(inst_unq_d[ii].sum()), comb_m, len(ii)))
    rows = np.array(rows).reshape(-1, 9)
    nfr_c = rows[:, 1]; tot_unq = rows[:, 2]; best_p = rows[:, 4]; comb_p = rows[:, 5]
    tot_unq_d = rows[:, 6]; comb_m = rows[:, 7]; ninst_c = rows[:, 8]

    def report(gate, mode=P_MODE):
        single = (nfr_c == 1)
        multi = (nfr_c >= 2)
        if mode == "matched":
            s_pass = int(((comb_m < gate) & single).sum())
            m_pass = int(((comb_m < gate) & multi).sum())
        else:   # the old analytic combination (not calibrated)
            s_pass = int(((best_p < gate) & (ninst_c == 1)).sum())
            m_pass = int(((comb_p < gate) & (ninst_c >= 2)).sum())
            single, multi = ninst_c == 1, ninst_c >= 2
        return s_pass, m_pass, int(single.sum()), int(multi.sum())

    print(f"TOTAL verified beta clusters: {ncl}")
    print(f"  recurrent (>=2 distinct frames): {(nfr_c>=2).sum()}, single-frame: {(nfr_c==1).sum()}")
    print(f"  total beta instances: {len(b_oms)}")
    for mode in (P_MODE, "analytic" if P_MODE == "matched" else "matched"):
        what = ("MATCHED exclusion null (alpha replaced by random orientations, "
                f"K={NULL_K}; Fisher over distinct frames)" if mode == "matched" else
                "ANALYTIC Poisson, Fisher over selected instances -- NOT CALIBRATED")
        tag = "GATED" if mode == P_MODE else "for comparison only"
        print(f"\nALPHA-EXCLUSION SURVIVORS [{tag}] {what}, statistic {STAT}:")
        print(f"{'gate':>8} {'single':>10} {'multi(Fisher)':>14} {'TOTAL beta grains':>18}")
        for gate in (1e-2, 1e-3, 1e-4):
            s, m, ns, nm = report(gate, mode)
            print(f"{gate:>8.0e} {s:>4}/{ns:<5} {m:>4}/{nm:<8} {s+m:>10}")

    # grains with >=1 alpha-unclaimed hit at all. "unclaimed" = on a peak no validated
    # alpha grain of the frame explains; both counts reported (nhit stacks harmonics).
    for name, tot in (("nhit", tot_unq), ("nhit_distinct", tot_unq_d)):
        print(f"\n[{name}] clusters with >=1 alpha-unclaimed hit summed over frames: "
              f"{int((tot>=1).sum())}")
        print(f"[{name}] clusters with >=3 alpha-unclaimed hits (summed): {int((tot>=3).sum())}")
        print(f"[{name}] alpha-unclaimed hit total distribution (summed per grain): "
              f"median {int(np.median(tot))}, 90pct {int(np.percentile(tot,90))}, max {int(tot.max())}")

    np.savez(f"{WORK}/peel_map/{CFG['out']}.npz",
             cid=rows[:,0].astype(int), nfr=nfr_c.astype(int), tot_unq=tot_unq.astype(int),
             tot_unq_distinct=tot_unq_d.astype(int), p_statistic=STAT,
             best_punq=best_p, comb_punq=comb_p, counts=counts,
             comb_pmatched=comb_m, ninst=ninst_c.astype(int), null_k=NULL_K, p_mode=P_MODE)
    print(f"\nsaved {CFG['out']}.npz")
