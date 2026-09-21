"""Stage A of the parent-beta reconstruction: per-frame Poisson validation of
BOTH phases on the 10,201-frame 100x100um_TestScan_About1parentbeta scan, with
stage coordinates captured for the position-resolved analysis.

For each phase (alpha hex, beta bcc): read filtered_orientations (matrix columns by layout, frame_peaks.orientation_block)
from every image_*.output.h5, validate each against SNR>8 peaks of its frame
(full predicted pattern; Poisson p<1e-4 on the whole peak list), and save the
survivors with their (sampleX, sampleZ). Light greedy clustering afterwards.

Writes peel_map/parentbeta_{alpha,beta}_validated.npz  (oms, frames, X, Z,
nhit, nhit_distinct, labels). nhit counts predicted reflections on a peak
(harmonics stack); nhit_distinct counts distinct peaks explained -- see
frame_peaks.count_matched_peaks. The Poisson validation gate itself is on nhit,
WHATEVER LAUE_GATE_STAT says: its analytic lambda is the chance expectation of
PREDICTED reflections on a peak, and there is no closed form for the distinct count.
The script prints this at start-up; LAUE_GATE_STAT applies to the empirical-null
gates downstream (empirical_gate, regrain, validated_figures), which use nhit_distinct
from this file when asked to.

Exits non-zero, without writing, when an input is unset or when frames were
attempted but none validated (the first read error is printed).
"""
import os
import numpy as np, h5py, glob, json, sys
from math import pi, cos, sin
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from scipy.stats import poisson
from concurrent.futures import ProcessPoolExecutor

# No defaults for data locations: an unset variable used to resolve to another
# campaign's folders (or to "", which made every frame fail silently).
WORK = os.environ.get("LAUE_WORK") or sys.exit("LAUE_WORK is not set (output root; peel_map/ is written under it)")
# Root of the legacy test-scan folders; only the "parentbeta"/"id6_10x10" keys use it.
TESTSCANS = os.environ.get("LAUE_TESTSCANS", "")
# Scan registry: argv[3] selects. "parentbeta" reproduces the original hardcoded 100x100um run.
# resdir/{phase} is where the indexer's output.h5 + frame_mapping.json live; out/{phase} names the npz.
from frame_peaks import out_prefix
SCANS={
 "parentbeta": dict(data=f"{TESTSCANS}/ID6-100x100um_TestScan_About1parentbeta",
                    resdir=lambda ph: f"{WORK}/results/parentbeta_{ph}",
                    out=lambda ph: f"parentbeta_{ph}"),
 # ID6 10x10um 0.25um-step fine scan (6561 frames); raw data was renamed ID6 -> ID26
 "env": dict(data=os.environ.get("LAUE_SCAN_DATA", ""),
                    resdir=lambda ph: os.environ.get(f"LAUE_SCAN_{ph.upper()}", ""),
                    out=lambda ph: f'{out_prefix()}_{ph}'),
 # NB legacy key: this is specimen ID26 and the scan is 20x20 um in the sample
 # frame, not 10x10 -- the folder name is wrong. Kept so existing *_id6_10x10_*
 # outputs stay resolvable; prefer the "env" entry for new work.
 "id6_10x10":  dict(data=f"{TESTSCANS}/ID26-10x10um_0p25umStepSize_TestingIndexing",
                    resdir=lambda ph: f"{TESTSCANS}/laue_Matching_Results/results/{ph}_20260717_161826",
                    out=lambda ph: f"id6_10x10_{ph}"),
}
H5LOC="/entry1/data/data"; HC=1.2398419739; TOL=8.0
# Detector geometry, energy window and NPX now come from the phase's parameter
# file (see laue_material), resolved below once PHASE is known.

from laue_material import phase_name
PHASE=sys.argv[1] if len(sys.argv)>1 else phase_name()   # argv, else LAUE_PHASE / the single LAUE_PHASES entry
NW=int(sys.argv[2]) if len(sys.argv)>2 else 36
SCAN=sys.argv[3] if len(sys.argv)>3 else "parentbeta"
if SCAN not in SCANS:
    sys.exit(f"unknown scan {SCAN!r}; choose from {sorted(SCANS)}")
CFG=SCANS[SCAN]; DATA=CFG["data"]; RESDIR=CFG["resdir"](PHASE); RES=CFG["out"](PHASE)
if SCAN!="env" and not TESTSCANS:
    sys.exit(f"scan {SCAN!r} lives under LAUE_TESTSCANS, which is not set; "
             f"use scan 'env' with LAUE_SCAN_DATA + LAUE_SCAN_{PHASE.upper()} for new work")
if not DATA or not os.path.isdir(DATA):
    sys.exit(f"raw-frame folder {DATA!r} does not exist -- set LAUE_SCAN_DATA "
             f"(an unset value used to make every frame fail and write an empty npz)")
if not RESDIR or not os.path.isdir(RESDIR):
    sys.exit(f"indexing-run folder {RESDIR!r} does not exist -- set LAUE_SCAN_{PHASE.upper()}")
# Lattice / reflection list / geometry from the params file the indexer used.
from frame_peaks import detect_peaks, count_matched_peaks, analytic_gate_note, image_number, orientation_block
from laue_material import Phase
_ph=Phase.load(PHASE); B=_ph.B; HKL=_ph.hkls; NPX=_ph.npx_x
print(f"[{PHASE}] scan={SCAN} resdir={RESDIR}",flush=True)
print(f"[{PHASE}] {_ph}",flush=True)

def project(OM):
    return _ph.project(OM)

mapping=json.load(open(f"{RESDIR}/frame_mapping.json"))
img2file={int(k):vv["file"] for k,vv in mapping.items() if isinstance(vv,dict) and "file" in vv}
h5s=sorted(glob.glob(f"{RESDIR}/results/image_*.output.h5"))
print(f"[{PHASE}] {len(h5s)} output.h5 files",flush=True)

def validate(h5):
    """-> (instances or None, error string or None, attempted).

    ``attempted`` is True once the frame has orientations to test; a read failure
    after that is an ERROR, reported rather than swallowed -- swallowing it is how
    an unset LAUE_SCAN_DATA used to print "VALIDATED 0" and write an empty npz."""
    inum=image_number(h5); fn=img2file.get(inum)
    if fn is None: return None, None, False
    try:
        with h5py.File(h5,"r") as f:
            filt=f["entry/results/filtered_orientations"][()]
    except Exception as e: return None, f"{h5}: {type(e).__name__}: {e}", False
    if not len(filt): return None, None, False
    oms=orientation_block(filt,h5).reshape(-1,3,3)   # layout by column count
    try:
        with h5py.File(f"{DATA}/{fn}","r") as f:
            raw=f[H5LOC][()].astype(float)
            X=float(f["entry1/sample/sampleX"][()].ravel()[0]); Z=float(f["entry1/sample/sampleZ"][()].ravel()[0])
    except Exception as e: return None, f"{DATA}/{fn}: {type(e).__name__}: {e}", True
    # Shared with null_model so the count and the null it is gated against are
    # measured by identical code. Uses a downsampled-median background (~16x
    # faster than full median_filter(25), same peaks) and drops blooming streaks.
    xs, ys, _ = detect_peaks(raw)
    out=[]
    if len(xs)>=5:
        tree=cKDTree(np.c_[xs,ys]); npeaks=len(xs)
        for i,OM in enumerate(oms):
            pr=project(OM)
            if not len(pr): continue
            # h      = predicted reflections supported (what the measured null
            #          maxima in use were calibrated against -- do NOT redefine)
            # h_dist = DISTINCT detected peaks explained (invariant 15b). Both
            #          are carried so a gate on either has a null measured the
            #          same way. See frame_peaks.count_matched_peaks.
            h_dist, h = count_matched_peaks(tree, pr, TOL)
            lam=len(pr)*npeaks*pi*TOL*TOL/(NPX*NPX)
            if poisson.sf(h-1,lam)<1e-4:
                out.append((OM,fn,X,Z,h,h_dist))
    return out, None, True

# Everything below drives the process pool. Guarded so the script also runs under
# the "spawn" start method (macOS default), where each worker re-imports this
# module: the workers need only the definitions above.
if __name__ == "__main__":
    analytic_gate_note("parentbeta_validate")
    if not h5s:
        sys.exit(f"[{PHASE}] no image_*.output.h5 under {RESDIR}/results")
    oms_v=[]; fr_v=[]; X_v=[]; Z_v=[]; nh_v=[]; nd_v=[]; done=0; tot_inst=0
    attempted=0; errors=[]
    with ProcessPoolExecutor(max_workers=NW) as ex:
        for res,err,tried in ex.map(validate, h5s, chunksize=8):
            done+=1; attempted+=int(tried)
            if err: errors.append(err)
            if res:
                for OM,fn,X,Z,h,hd in res:
                    oms_v.append(OM); fr_v.append(fn); X_v.append(X); Z_v.append(Z); nh_v.append(h); nd_v.append(hd)
            if done%1000==0: print(f"[{PHASE}] {done}/{len(h5s)} frames, {len(oms_v)} validated so far",flush=True)
    oms_v=np.array(oms_v)
    print(f"[{PHASE}] VALIDATED instances: {len(oms_v)}  (frames attempted {attempted}, "
          f"read errors {len(errors)})",flush=True)
    if errors:
        print(f"[{PHASE}] first read error: {errors[0]}",flush=True)
    if attempted>0 and len(oms_v)==0:
        sys.exit(f"[{PHASE}] 0 instances validated out of {attempted} frames attempted -- "
                 f"refusing to write an empty npz. "
                 + (f"First error: {errors[0]}" if errors else
                    "No read errors: every orientation failed the Poisson test; check the "
                    "phase's params file, TOL and the frame<->result mapping."))
    # SAVE validated instances immediately (protect the expensive I/O before clustering)
    np.savez(f"{WORK}/peel_map/{RES}_validated.npz",
             oms=oms_v, frames=np.array(fr_v), X=np.array(X_v), Z=np.array(Z_v),
             nhit=np.array(nh_v), nhit_distinct=np.array(nd_v), labels=np.full(len(oms_v),-1))
    print(f"[{PHASE}] saved validated (pre-cluster)",flush=True)

    # Light greedy clustering; operators follow the space group, not the phase name.
    #
    # This loop is O(n_clusters x n_instances x n_sym_ops). That is fine for the
    # ~1e3-1e4 instances the test scans produced, but a full 201x201 raster yields
    # ~2e5 instances and tens of thousands of clusters, where it does not terminate
    # in useful time. Set LAUE_SKIP_CLUSTER=1 to stop after the (already saved)
    # pre-cluster file and cluster separately with cluster_orientations.py, which is
    # KD-tree based; the label column is then filled in there.
    if os.environ.get("LAUE_SKIP_CLUSTER") == "1":
        print(f"[{PHASE}] LAUE_SKIP_CLUSTER=1 -> leaving labels unset; "
              f"run cluster_orientations.py on {RES}_validated.npz",flush=True)
        sys.exit(0)
    if len(oms_v) > 50000:
        print(f"[{PHASE}] WARNING: {len(oms_v)} instances is beyond what this greedy "
              f"clustering handles in reasonable time; consider LAUE_SKIP_CLUSTER=1 "
              f"+ cluster_orientations.py",flush=True)
    OPS=_ph.sym_ops
    def miso_min(A,Bs):
        return _ph.misorientation(A,Bs)
    labels=np.full(len(oms_v),-1); cid=0
    for i in range(len(oms_v)):
        if labels[i]>=0: continue
        un=np.where(labels<0)[0]; d=miso_min(oms_v[i],oms_v[un]); labels[un[d<1.0]]=cid; cid+=1
    counts=np.bincount(labels) if len(oms_v) else np.array([])
    print(f"[{PHASE}] clusters (<1.0 deg): {cid}; top sizes {sorted(counts,reverse=True)[:12]}",flush=True)
    np.savez(f"{WORK}/peel_map/{RES}_validated.npz",
             oms=oms_v, frames=np.array(fr_v), X=np.array(X_v), Z=np.array(Z_v),
             nhit=np.array(nh_v), nhit_distinct=np.array(nd_v), labels=labels)
    print(f"[{PHASE}] saved {RES}_validated.npz",flush=True)
