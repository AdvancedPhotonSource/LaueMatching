"""Column-content pipeline for Laue frames (reference implementation; the sampleH column_pipeline.py generalised).

    round 0  : indexer solutions (nhit > gate0) for each frame
    round r  : joint fit (ColumnFit) of ALL current orientations on the ORIGINAL frame -> residual frames ->
               index the residuals -> NEW = nhit > g_disc and > new_min_deg from every current orientation -> refit
    g_disc   : MEASURED (see :func:`gate_from_scrambled`), never inherited from round 0.

The indexer is injected: ``index_fn(frame_dir, tag) -> {frame_name: output_h5}`` (at 34-ID-E this wraps the
LaueMatching C indexer dispatch, e.g. the sampleH ``index_set.sh``), and ``solutions(output_h5, gate) -> [(OM, nhit)]``.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List

import h5py
import numpy as np

_W: dict = {}


def _worker_init(params_path, phase, bg_path, kernel_npz, hkl_fn):
    import torch
    torch.set_num_threads(1)
    from .fit import ColumnFit
    from .geom import EmpKernel, Geom
    G = Geom(params_path, phase)
    _W.update(G=G, CF=ColumnFit, AH=np.asarray(hkl_fn(), float),
              BG=np.fromfile(bg_path, dtype=np.float64).reshape(G.ny, G.nx), KERN=EmpKernel(kernel_npz))


def fit_one(task):
    """task = (name, src_h5, oms, round_found, nhits, resdir, fit_kw)."""
    name, src, oms, found, nhits, resdir, fit_kw = task
    try:
        with h5py.File(src, "r") as h:
            raw = h["/entry1/data/data"][()]
        raw = raw[0] if raw.ndim == 3 else raw
        if not oms:
            return dict(frame=name, orientations=[], unexplained_flux_frac=1.0)
        fr = _W["CF"](_W["G"], raw, _W["BG"], [np.array(o) for o in oms], _W["AH"], _W["KERN"], K=fit_kw.get("K", 24))
        fr.fit(n_iter=fit_kw.get("n_iter", 300), lr=fit_kw.get("lr", 3e-4), inits=tuple(fit_kw.get("inits", (0.05, 0.4))))
        rep = fr.report()
        for q, rr, nh in zip(rep["orientations"], found, nhits):
            q["round_found"] = rr; q["nhit"] = nh
        if resdir:
            with h5py.File(f"{resdir}/{name}", "w") as h:
                h.create_dataset("/entry1/data/data", data=fr.residual_frame())
                for k in ("sampleX", "sampleZ", "sampleY"):
                    h.create_dataset(f"/entry1/sample/{k}", data=np.array([0.0]))
        rep["frame"] = name
        return rep
    except Exception as e:
        return dict(frame=name, error=repr(e))


def run(frames: Dict[str, str], round0: Dict[str, List], outdir: str, *, params_path: str, phase: str, bg_path: str,
        kernel_npz: str, hkl_fn: Callable, index_fn: Callable, solutions: Callable, misor_deg: Callable, g_disc: int,
        max_rounds: int = 3, new_min_deg: float = 1.0, workers: int = 60, fit_kw: dict | None = None) -> dict:
    """frames: {name: raw_h5}; round0: {name: [(OM, nhit), ...]}. Writes outdir/round<r>/{reports.jsonl, residual/}.
    misor_deg(OM_a, OM_b) must return DEGREES (checked): new_min_deg is compared against it."""
    from .geom import require_degrees
    require_degrees(misor_deg, where="misor_deg")
    fit_kw = fit_kw or {}
    state = {n: dict(oms=[o for o, _ in round0.get(n, [])], rnd=[0] * len(round0.get(n, [])), nh=[h for _, h in round0.get(n, [])])
             for n in frames}
    summary = dict(frames=len(state), rounds=[], g_disc=g_disc)
    for rnd in range(max_rounds + 1):
        rdir = f"{outdir}/round{rnd}"; resdir = f"{rdir}/residual"; os.makedirs(resdir, exist_ok=True)
        tasks = [(n, frames[n], st["oms"], st["rnd"], st["nh"], resdir, fit_kw) for n, st in state.items()]
        with ProcessPoolExecutor(workers, initializer=_worker_init, initargs=(params_path, phase, bg_path, kernel_npz, hkl_fn)) as ex:
            reps = list(ex.map(fit_one, tasks, chunksize=1))
        with open(f"{rdir}/reports.jsonl", "w") as fo:
            for r in reps:
                fo.write(json.dumps(r, default=float) + "\n")
        info = dict(round=rnd, n_errors=sum("error" in r for r in reps))
        if rnd == max_rounds:
            summary["rounds"].append(info); break
        resmap = index_fn(resdir, f"r{rnd + 1}")
        added = 0
        for n, st in state.items():
            p = resmap.get(n)
            if not p:
                continue
            for om, nh in solutions(p, g_disc):
                if all(misor_deg(om, o) > new_min_deg for o in st["oms"]):
                    st["oms"].append(om); st["rnd"].append(rnd + 1); st["nh"].append(nh); added += 1
        info["added"] = added; summary["rounds"].append(info)
        if added == 0:
            break
    summary["final_round"] = rnd
    json.dump(summary, open(f"{outdir}/summary.json", "w"), indent=1, default=float)
    return summary


def gate_from_scrambled(counts_by_gate: Dict[int, int], n_frames: int, ub_max: float = 0.10):
    """Smallest gate whose chance-solution count on SCRAMBLED residual frames has a Poisson 95% UB <= ub_max per frame.
    counts_by_gate = {gate: number of solutions with nhit > gate over n_frames scrambled residual frames}
    (sampleH: scramble_frames.py uniform mode, then the same indexer)."""
    from scipy.stats import chi2
    ub = {g: 0.5 * chi2.ppf(0.95, 2 * (c + 1)) / max(n_frames, 1) for g, c in counts_by_gate.items()}
    ok = [g for g in sorted(ub) if ub[g] <= ub_max]
    return (ok[0] if ok else None), ub
