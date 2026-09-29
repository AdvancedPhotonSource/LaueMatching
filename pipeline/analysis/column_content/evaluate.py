"""Frozen evaluator for column-content validation on synthetic Laue columns (sampleH column_eval_wide.py as a function).

Gates (PREREGISTER_column_content_wide.md): V1 false UB <= 0.10/frame; V2 recall (share >= 5%) >= 90%; V3 share error
<= 20%; V4 point orientation <= 0.02 deg; V6 median unexplained <= 0.15; V7 streak recall >= 85%; V8 extent Spearman
>= 0.6. Matching tolerance per truth crystal: 0.5 deg + half-width (streak) or + 2 sigma (clouds).
Plus recall by class / half-width / share, and extent true vs recovered.
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import torch
from scipy.stats import chi2, spearmanr

from .geom import DT, truth_summary


def evaluate(base: str, fd: str, outp: str, *, G, AH, mis):
    """base = pipeline outdir (summary.json, round<r>/reports.jsonl); fd = synthetic frame dir; mis(om, T) -> deg array
    (checked: the matching tolerances are in degrees)."""
    from .geom import require_degrees
    require_degrees(mis, batch=True, where="mis")
    ang = lambda a, B: np.asarray(mis(np.asarray(a), np.asarray(B)), float)  # noqa: E731
    summ = json.load(open(f"{base}/summary.json"))
    fin = summ.get("final_round", max(r["round"] for r in summ["rounds"]))
    reps = {r["frame"]: r for r in (json.loads(l) for l in open(f"{base}/round{fin}/reports.jsonl"))}
    n_false = nf = 0; rows = []; v3 = []; v4 = []; v8 = []; ue = []; inits = []
    for fname, rep in reps.items():
        if "error" in rep:
            continue
        with h5py.File(f"{fd}/{fname}", "r") as h:
            g = h["/entry1/truth"]
            T = g["centres"][()]; share = g["share"][()]
            cls = [c.decode() for c in g["spread_class"][()]]; par = g["spread_param"][()]
            comps = [g[f"comps_{k}"][()] for k in range(len(T))]
        nf += 1; ue.append(rep.get("unexplained_flux_frac", 1.0)); inits.append(rep.get("init_won"))   # no orientation found: nothing explained
        tol = np.array([0.5 + (par[i] if cls[i] == "streak" else 2 * par[i] if cls[i] in ("narrow", "wide3d") else 0.0)
                        for i in range(len(T))])
        rec = rep.get("orientations", [])
        A = np.array([ang(q["mean_om"], T) for q in rec]) if rec else np.zeros((0, len(T)))
        M = A < tol[None, :]
        n_false += int((~M.any(1)).sum()) if len(A) else 0
        for i in range(len(T)):
            found = bool(len(A) and M[:, i].any())
            rows.append(dict(frame=fname, i=i, share=float(share[i]), cls=cls[i], par=float(par[i]), found=found))
            if not found:
                continue
            k = int(np.argmin(np.where(M[:, i], A[:, i], np.inf)))
            one2one = int(np.argmin(A[k] - tol)) == i and int(M[k].sum()) == 1
            if not one2one:
                continue
            if share[i] >= 0.05:
                v3.append((cls[i], abs(rec[k]["share"] - share[i]) / share[i]))
            if cls[i] == "point":
                v4.append(float(A[k, i]))
            if cls[i] != "point" and share[i] >= 0.05 and rec[k].get("extent_perp_deg") is not None:
                om = torch.as_tensor(T[i], dtype=DT)
                px, py, _, ok = G.project(om[None], torch.as_tensor(AH, dtype=DT))
                blind = G.mean_q_axis(om, torch.as_tensor(AH[ok[0].numpy()], dtype=DT)).numpy()
                ts = truth_summary(comps[i], np.ones(len(comps[i])), T[i], blind)
                v8.append((cls[i], float(par[i]), float(ts["extent_perp_deg"]), float(rec[k]["extent_perp_deg"])))
    big = [r for r in rows if r["share"] >= 0.05]
    recall = float(np.mean([r["found"] for r in big])) if big else None
    sbig = [r for r in big if r["cls"] == "streak"]
    srec = float(np.mean([r["found"] for r in sbig])) if sbig else None
    ub = 0.5 * chi2.ppf(0.95, 2 * (n_false + 1)) / max(nf, 1)
    v3e = [e for _, e in v3]
    rho8 = float(spearmanr([x[2] for x in v8], [x[3] for x in v8]).correlation) if len(v8) > 3 else None
    # pipeline.run writes the gate as "g_disc"; this output keeps its "gdisc" key
    out = dict(frames=nf, final_round=fin, gdisc=summ.get("g_disc"), rounds=summ["rounds"],
               V1=dict(false_total=n_false, per_frame=n_false / max(nf, 1), ub95=ub, passed=ub <= 0.10),
               V2=dict(n=len(big), recall=recall, passed=recall is not None and recall >= 0.90),
               V3=dict(n=len(v3e), median_rel_err=float(np.median(v3e)) if v3e else None, passed=bool(v3e) and float(np.median(v3e)) <= 0.20),
               V4=dict(n=len(v4), median_deg=float(np.median(v4)) if v4 else None, passed=bool(v4) and float(np.median(v4)) <= 0.02),
               V6=dict(median_unexplained=float(np.median(ue)), q90=float(np.percentile(ue, 90)), passed=float(np.median(ue)) <= 0.15),
               V7=dict(n=len(sbig), recall=srec, passed=srec is not None and srec >= 0.85),
               V8=dict(n=len(v8), spearman=rho8, passed=rho8 is not None and rho8 >= 0.6))
    out["read"] = "VALIDATED" if all(out[k]["passed"] for k in ("V1", "V2", "V3", "V4", "V6", "V7", "V8")) else "NOT VALIDATED"
    out["by_class"] = {}
    for c in ("point", "narrow", "streak", "wide3d"):
        rr = [r for r in big if r["cls"] == c]
        e3 = [e for cc, e in v3 if cc == c]
        out["by_class"][c] = dict(n=len(rr), recall=float(np.mean([r["found"] for r in rr])) if rr else None,
                                  share_rel_err_median=float(np.median(e3)) if e3 else None)
    out["streak_by_halfwidth"] = {str(h): dict(n=len(rr), recall=float(np.mean([r["found"] for r in rr])) if rr else None,
                                               extent_true_med=float(np.median([x[2] for x in v8 if x[0] == "streak" and x[1] == h])) if any(x[0] == "streak" and x[1] == h for x in v8) else None,
                                               extent_rec_med=float(np.median([x[3] for x in v8 if x[0] == "streak" and x[1] == h])) if any(x[0] == "streak" and x[1] == h for x in v8) else None)
                                  for h in (0.1, 0.2, 0.4, 0.8) for rr in [[r for r in sbig if r["par"] == h]]}
    bins = [(0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 1.01)]
    out["recall_by_share"] = {f"{lo}-{hi}": dict(n=len(s), recall=float(np.mean([r["found"] for r in s])) if s else None)
                              for lo, hi in bins for s in [[r for r in rows if lo <= r["share"] < hi]]}
    out["init_won"] = {str(k): int(sum(1 for x in inits if x == k)) for k in set(inits)}
    conv = lambda o: bool(o) if isinstance(o, np.bool_) else float(o)  # noqa: E731
    json.dump(out, open(outp, "w"), indent=1, default=conv)
    return out


