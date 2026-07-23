"""Collect every headline metric for every analysed scan into one JSON.

Pulls from each scan's analysis log (nulls, census, Burgers, coherence) and from its
saved npz (validated instances, geometry measured from the stage coordinates, grains
under the contiguity-aware definition). Folder names are never trusted for geometry.

usage: collect_scan_metrics.py > metrics.json
"""
import glob
import json
import os
import re
import sys
import numpy as np
from scipy import ndimage as ndi

W = "/net/hpcs34/data34c/for_Hemant/lauematching_ti"
ANA = f"{W}/analysis"
SQ2 = np.sqrt(2.0)

def grab(txt, pat, cast=float, default=None):
    m = re.search(pat, txt)
    return cast(m.group(1)) if m else default

def rmat(ax, deg):
    u = np.asarray(ax, float); u /= np.linalg.norm(u); t = np.radians(deg)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(t)*K + (1-np.cos(t))*(K@K)
HEXOPS = np.array([rmat([0, 0, 1], 60*k) for k in range(6)] +
                  [rmat([np.cos(np.radians(a)), np.sin(np.radians(a)), 0], 180)
                   for a in (0, 30, 60, 90, 120, 150)])
CUBOPS = np.array([np.eye(3)] + [rmat(a, d) for a, d in
    [([1,0,0],90),([1,0,0],180),([1,0,0],270),([0,1,0],90),([0,1,0],180),([0,1,0],270),
     ([0,0,1],90),([0,0,1],180),([0,0,1],270),([1,1,0],180),([1,-1,0],180),([1,0,1],180),
     ([-1,0,1],180),([0,1,1],180),([0,1,-1],180),([1,1,1],120),([1,1,1],240),([1,-1,1],120),
     ([1,-1,1],240),([-1,1,1],120),([-1,1,1],240),([1,1,-1],120),([1,1,-1],240)]])

def phase_metrics(d, pref, ph, nullmax):
    f = f"{d}/peel_map/{pref}_{ph}_validated.npz"
    if not os.path.isfile(f):
        return None
    z = np.load(f, allow_pickle=True)
    X, Z, lab, nhit = z["X"].astype(float), z["Z"].astype(float), z["labels"], z["nhit"].astype(int)
    out = {"instances": int(len(X))}

    # geometry MEASURED from the stage coordinates, never from the folder name
    Xu = np.unique(np.round(X, 4)); Zu = np.unique(np.round(Z, 4))
    dx = float(np.median(np.diff(Xu))) if len(Xu) > 1 else 0.0
    dz = float(np.median(np.diff(Zu))) if len(Zu) > 1 else 0.0
    out["geom"] = {
        "positions": int(len(set(zip(np.round(X, 4), np.round(Z, 4))))),
        "nx": int(len(Xu)), "nz": int(len(Zu)),
        "span_x_um": round(float(Xu.max()-Xu.min()), 3) if len(Xu) > 1 else 0.0,
        "span_z_lab_um": round(float(Zu.max()-Zu.min()), 3) if len(Zu) > 1 else 0.0,
        "step_x_um": round(dx, 4), "step_z_lab_um": round(dz, 4),
        # 45 deg mount: sample-frame Z is the lab Z de-projected
        "span_z_sample_um": round(float(Zu.max()-Zu.min())*SQ2, 3) if len(Zu) > 1 else 0.0,
        "step_z_sample_um": round(dz*SQ2, 4),
    }
    out["above_nullmax"] = int((nhit > nullmax).sum()) if nullmax else None
    out["above_nullmax_pct"] = round(100*float((nhit > nullmax).mean()), 1) if nullmax else None
    out["median_hits"] = int(np.median(nhit))

    # contiguity-aware grains
    Xi = {v: i for i, v in enumerate(Xu)}; Zi = {v: i for i, v in enumerate(Zu)}
    gi = np.array([Zi[round(v, 4)] for v in Z]); gj = np.array([Xi[round(v, 4)] for v in X])
    shape = (len(Zu), len(Xu)); st = ndi.generate_binary_structure(2, 2)
    grains, gold = 0, 0
    ge5 = 0
    for c in range(lab.max()+1):
        idx = np.where(lab == c)[0]
        if not len(idx):
            continue
        m = np.zeros(shape, bool); m[gi[idx], gj[idx]] = True
        cc, n = ndi.label(m, structure=st)
        for k in range(1, max(n, 1)+1):
            keep = cc[gi[idx], gj[idx]] == k if n >= 1 else np.ones(len(idx), bool)
            if not keep.any():
                continue
            g = idx[keep]
            grains += 1
            npos = len(set(zip(gi[g], gj[g])))
            if npos >= 5:
                ge5 += 1
                if nullmax and (nhit[g] > nullmax).any():
                    gold += 1
    out["clusters"] = int(lab.max()+1)
    out["grains"] = grains
    out["grains_ge5"] = ge5
    out["gold"] = gold
    return out

res = {}
for d in sorted(glob.glob(f"{ANA}/*/")):
    scan = os.path.basename(d.rstrip("/"))
    va = glob.glob(f"{d}/peel_map/*_alpha_validated.npz")
    if not va:
        continue
    pref = os.path.basename(va[0]).replace("_alpha_validated.npz", "")
    logf = sorted(glob.glob(f"{d}/*analysis*.log"))
    txt = open(logf[0]).read() if logf else ""

    nm = {}
    for ph in ("alpha", "beta"):
        blk = re.search(rf"\[{ph}\] RANDOM-ORIENTATION NULL.*?\n(.*?)\n", txt, re.S)
        line = blk.group(1) if blk else ""
        nm[ph] = {
            "mean": grab(line, r"mean hits ([0-9.]+)"),
            "p999": grab(line, r"99\.9th ([0-9]+)", int),
            "max": grab(line, r"max ([0-9]+)", int),
        }

    entry = {
        "prefix": pref,
        "null": nm,
        "alpha_claim_pct": grab(txt, r"alpha claims ([0-9.]+)% of peaks"),
        "burgers_parents": grab(txt, r"=> (\d+) significant prior-beta grain", int),
        "burgers_explained_pct": grab(txt, r"significant-alpha instances \((\d+)%\)", int),
        "coherence_obs": grab(txt, r"OBSERVED same-variant fraction: ([0-9.]+)"),
        "coherence_null": grab(txt, r"SHUFFLED null: mean ([0-9.]+)"),
        "coherence_z": grab(txt, r"\nz = ([0-9.]+)", float),
    }
    for ph in ("alpha", "beta"):
        entry[ph] = phase_metrics(d, pref, ph, nm[ph]["max"])
    res[scan] = entry
    print(f"collected {scan}", file=sys.stderr)

json.dump(res, sys.stdout, indent=1)
