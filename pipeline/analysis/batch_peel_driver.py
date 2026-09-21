"""Batch iterative peel over the full SmallAreaTest1 map.

Each PASS = one orchestrator run over ALL frames (daemon loads once), then
per-frame masking of every accepted grain's full projected pattern, writing
the residual frames for the next pass. Sigma-cap schedule widens per pass.
Outputs per-frame accepted orientation stacks + a status log.

Env: LAUE_WORK, LAUE_SCAN_DATA, LAUE_PARAMS_<PHASE> (or LAUE_PARAMS), LAUE_PHASE;
LAUE_LM (a LaueMatching checkout; default: the one this file is in). Each pass's
config is the base params file with only BackgroundFile and GaussSigmaMax rewritten.
An orchestrator failure, or a pass that writes no results, exits non-zero.
"""
import numpy as np, h5py, subprocess, os, sys, shutil, glob, json, time
from math import cos, sin, pi

WORK = os.environ.get("LAUE_WORK") or sys.exit("LAUE_WORK is not set (params/, results/, peel_map/ are written under it)")
# LaueMatching checkout whose scripts/laue_orchestrator.py runs each pass. Defaults to
# the checkout this file sits in (the old default was a deleted install path, and the
# resulting "can't open file" was hidden by capture_output and logged as "converged").
LM = os.environ.get("LAUE_LM") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ORCH = f"{LM}/scripts/laue_orchestrator.py"
if not os.path.isfile(ORCH):
    sys.exit(f"orchestrator not found at {ORCH}; set LAUE_LM to a LaueMatching checkout")
PY = sys.executable
DATA = os.environ.get("LAUE_SCAN_DATA") or sys.exit("LAUE_SCAN_DATA is not set (folder of the raw frames to peel)")
H5LOC = os.environ.get("LAUE_H5LOC", "/entry1/data/data")
from laue_material import phase_name
PHASE = phase_name()      # LAUE_PHASE, or the single phase in LAUE_PHASES
# Base parameter file to derive each pass's config from. Also the source of the
# lattice, reflection list and geometry -- so the peel cannot be run against one
# material's reflections while masking another's predicted spots.
BASE_PARAMS = os.environ.get(f"LAUE_PARAMS_{PHASE.upper()}") or os.environ.get("LAUE_PARAMS")
if not BASE_PARAMS:
    # Deliberately no default. A default of params_Ti_alpha.txt would resolve
    # successfully on the machine where that file happens to exist, and peel one
    # material's frames against another material's reflections without complaint.
    sys.exit(f"set LAUE_PARAMS_{PHASE.upper()} (or LAUE_PARAMS) to the params_*.txt used for indexing")
MAX_PASS = 6
MASK_R = 10
SIGCAPS = [2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
HC = 1.2398419739
ST = open(f"{WORK}/batch_peel_status.txt", "w", buffering=1)
def log(m): ST.write(m + "\n"); print(m, flush=True)

from laue_material import Phase
from frame_peaks import image_number, orientation_block
_ph = Phase.load(PHASE, BASE_PARAMS)
nPx = _ph.npx_x
log(f"[peel] {_ph}")

def project(OM):
    return _ph.project(OM)

OPS = _ph.sym_ops
def miso_min(A, Bs):
    return _ph.misorientation(A, Bs)

frames = sorted(os.path.basename(f) for f in glob.glob(f"{DATA}/*.h5"))
if not frames:
    sys.exit(f"no .h5 frames under LAUE_SCAN_DATA={DATA}")
log(f"BATCH PEEL: {len(frames)} frames, {MAX_PASS} passes max")
accepted = {fn: [] for fn in frames}
os.makedirs(f"{WORK}/peel_map", exist_ok=True)
cur_folder = DATA

for p_i in range(1, MAX_PASS+1):
    t0 = time.time()
    cfg = f"{WORK}/params/params_batchpeel_p{p_i}.txt"
    os.makedirs(f"{WORK}/params", exist_ok=True)
    base = open(BASE_PARAMS).read()
    # Only the two keys the peel schedule owns are rewritten: the per-pass
    # background and the widening sigma cap. Detection thresholds
    # (ThresholdPercentile, MinNrSpots, MinIntensity) come from the base params
    # file unchanged -- they used to be silently forced to 99.8 / 8 / 50, which
    # overrode whatever the indexing run had been tuned to.
    lines = []; have_sigma = False
    for ln in base.splitlines():
        k = ln.split()[0] if ln.split() else ""
        if k == "BackgroundFile": ln = f"BackgroundFile {WORK}/peel_map/bg_p{p_i}.bin"
        if k == "GaussSigmaMax":
            if have_sigma:
                continue                      # drop duplicates: one key, one value
            ln = f"GaussSigmaMax {SIGCAPS[p_i-1]}"; have_sigma = True
        lines.append(ln)
    if not have_sigma:
        lines.append(f"GaussSigmaMax {SIGCAPS[p_i-1]}")
    open(cfg, "w").write("\n".join(lines) + "\n")
    out = f"{WORK}/results/batchpeel_pass{p_i}"
    shutil.rmtree(out, ignore_errors=True)
    r = subprocess.run([PY, ORCH,
                        "--config", cfg, "--folder", cur_folder, "--h5-location", H5LOC,
                        "--ncpus", "32", "--port", "60517", "--flush-time", "120",
                        "--output-dir", out],
                       capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    # keep the orchestrator's output instead of discarding it
    with open(f"{WORK}/batch_peel_pass{p_i}.log", "w") as fh:
        fh.write(r.stdout or ""); fh.write(r.stderr or "")
    if r.returncode != 0:
        tail = "\n".join(((r.stderr or "") + (r.stdout or "")).strip().splitlines()[-20:])
        log(f"pass {p_i}: orchestrator FAILED (exit {r.returncode}); "
            f"log {WORK}/batch_peel_pass{p_i}.log\n{tail}")
        sys.exit(f"pass {p_i}: orchestrator exited {r.returncode}")
    mapping = {}
    try:
        mp = json.load(open(f"{out}/frame_mapping.json"))
        mapping = {int(k): vv["file"] for k, vv in mp.items() if isinstance(vv, dict) and "file" in vv}
    except Exception as e:
        log(f"pass {p_i}: mapping load failed: {e}")
    new_total = 0
    outs = glob.glob(f"{out}/results/image_*.output.h5")
    if not outs or not mapping:
        # an empty pass is a failure, not convergence
        log(f"pass {p_i}: {len(outs)} output files, {len(mapping)} mapped frames under {out} "
            f"-- the pass produced nothing; see {WORK}/batch_peel_pass{p_i}.log")
        sys.exit(f"pass {p_i}: no results")
    for h5 in outs:
        inum = image_number(h5)
        fn = mapping.get(inum)
        if fn is None: continue
        # residual frames are named identically to originals
        fn = os.path.basename(fn)
        try:
            with h5py.File(h5, "r") as f:
                filt = f["entry/results/filtered_orientations"][()]
        except Exception:
            continue
        for om9 in orientation_block(filt, h5):     # layout by column count
            OM = om9.reshape(3, 3)
            if accepted[fn] and miso_min(OM, np.array(accepted[fn])).min() < 0.7:
                continue
            accepted[fn].append(OM); new_total += 1
    tot = sum(len(x) for x in accepted.values())
    log(f"pass {p_i}: +{new_total} new (total {tot}) in {time.time()-t0:.0f}s")
    if new_total < max(5, 0.01*tot):
        log("converged."); break
    if p_i == MAX_PASS: break
    # build residuals
    t1 = time.time()
    nxt = f"{WORK}/peel_map/pass{p_i+1}"
    shutil.rmtree(nxt, ignore_errors=True); os.makedirs(nxt)
    yy, xx = np.mgrid[-MASK_R:MASK_R+1, -MASK_R:MASK_R+1]
    disk = (xx*xx + yy*yy) <= MASK_R*MASK_R
    for fn in frames:
        src = f"{DATA}/{fn}"
        with h5py.File(src, "r") as f:
            img = f[H5LOC][()]
        med = np.median(img)
        if accepted[fn]:
            for x, y in np.vstack([project(OM) for OM in accepted[fn]]):
                xi, yi = int(round(x)), int(round(y))
                x0, x1 = max(0, xi-MASK_R), min(nPx, xi+MASK_R+1)
                y0, y1 = max(0, yi-MASK_R), min(nPx, yi+MASK_R+1)
                img[y0:y1, x0:x1][disk[(y0-yi+MASK_R):(y1-yi+MASK_R), (x0-xi+MASK_R):(x1-xi+MASK_R)]] = med
        with h5py.File(f"{nxt}/{fn}", "w") as f:
            f.create_dataset(H5LOC, data=img)
    log(f"pass {p_i}: residuals written in {time.time()-t1:.0f}s")
    cur_folder = nxt

# save accepted stacks
np.savez(f"{WORK}/peel_map/accepted_per_frame.npz",
         **{fn: np.array(oms) for fn, oms in accepted.items() if oms})
log(f"BATCH PEEL DONE: {sum(len(x) for x in accepted.values())} orientation instances "
    f"across {sum(1 for x in accepted.values() if x)} frames")
