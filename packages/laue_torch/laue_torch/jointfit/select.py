"""Grain-count model selection.

The greedy peel answers "how many grains?" by accretion: keep adding whatever
explains leftover peaks until a threshold stops firing.  That is what let a
fragmented cloud seed extra grains -- there was never a step that asked whether
a grain was WORTH its parameters.

Here the question is posed properly.  Adding a grain always reduces the residual
(weakly), so the comparison must charge for the parameters it costs:

    BIC = n ln(RSS/n) + k ln(n)

EFFECTIVE PARAMETER COUNT.  ``k`` counts only what the data actually
constrains.  A grain whose amplitudes all solve to zero contributes nothing to
the model, and its orientation and spread are then unidentifiable, so it costs
ZERO parameters -- not six.  Likewise only NON-ZERO amplitudes are counted, the
standard degrees-of-freedom treatment for a non-negative (sparse) linear model.
This matters: without it, a zero-amplitude grain would be penalized as though it
were doing work, and the selection would look decisive when it was only counting.

CAVEAT ON n.  Pixels are correlated through the PSF, so the effective sample
size is smaller than the pixel count and the ln(n) penalty is not calibrated in
an absolute sense.  Treat BIC differences as ORDINAL -- which model wins -- and
do not read them as evidence ratios.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor

from .design import ROI
from .model import JointGrainFit

__all__ = ["BicReport", "SelectionResult", "model_bic", "prune_grains"]


@dataclass
class BicReport:
    """BIC and its ingredients, kept separate so a surprising value is debuggable."""

    bic: float
    rss: float
    n_pixels: int
    n_effective_params: int
    n_active_grains: int
    active: Tensor          # (G,) bool -- which grains earned any amplitude


@dataclass
class SelectionResult:
    """Outcome of a pruning search.

    ``inactive`` are grains removed because they earned no amplitude at all --
    that is not a model-selection decision, just the removal of grains that
    explain nothing.  ``dropped`` are grains removed on BIC grounds.
    """

    keep: list[int]
    bic: float
    start_bic: float
    inactive: list[int] = field(default_factory=list)
    dropped: list[int] = field(default_factory=list)
    history: list[tuple[int, float]] = field(default_factory=list)


def _bic_value(rss: float, n: int, k: int) -> float:
    # Guard a numerically-zero residual: a perfect fit is not infinitely good
    # evidence, it just means the model is at the numerical floor.
    rss = max(rss, torch.finfo(torch.float64).tiny)
    return n * math.log(rss / n) + k * math.log(n)


def model_bic(
    model: JointGrainFit,
    rois: Sequence[ROI],
    images: Sequence[Tensor],
    amp_rtol: float = 1e-6,
    ridge: float = 0.0,
) -> BicReport:
    """BIC for ``model`` on the given ROIs.

    amp_rtol : an amplitude counts as non-zero above this fraction of the
               largest amplitude in the fit.  Solver output is exactly 0.0 for
               unused reflections, so this only guards near-degenerate cases.
    """
    with torch.no_grad():
        rss, _info = model.loss(rois, images, ridge=ridge)
        per_grain = model.grain_amplitudes(rois, images, ridge=ridge)

    n_pixels = int(sum(r.n_pixels for r in rois))
    scale = float(per_grain.max()) if per_grain.numel() else 0.0
    active = per_grain > (amp_rtol * scale if scale > 0 else 0.0)
    n_active = int(active.sum())

    # Non-zero amplitudes across all ROIs, counted once per (grain, reflection).
    n_amp = _count_active_amplitudes(model, rois, images, amp_rtol, ridge)
    per_grain_params = 6 if model.log_sigma_par.requires_grad else 3
    k = n_amp + per_grain_params * n_active

    return BicReport(
        bic=_bic_value(float(rss), n_pixels, k),
        rss=float(rss),
        n_pixels=n_pixels,
        n_effective_params=k,
        n_active_grains=n_active,
        active=active,
    )


@torch.no_grad()
def _count_active_amplitudes(
    model: JointGrainFit,
    rois: Sequence[ROI],
    images: Sequence[Tensor],
    amp_rtol: float,
    ridge: float,
) -> int:
    from .amplitudes import solve_amplitudes
    from .design import build_basis, suggested_window

    geom = model.spot_geometry()
    total = 0
    for roi, img in zip(rois, images):
        sel = model._select(geom, roi, 4.0)
        if sel.numel() == 0:
            continue
        win = suggested_window(geom.cov[sel])
        basis = build_basis(geom.px[sel], geom.py[sel], geom.cov[sel], roi, win)
        sol = solve_amplitudes(basis, img, ridge=ridge)
        if sol.amplitudes.numel() == 0:
            continue
        scale = float(sol.amplitudes.max())
        if scale <= 0:
            continue
        total += int((sol.amplitudes > amp_rtol * scale).sum())
    return total


def prune_grains(
    model: JointGrainFit,
    rois: Sequence[ROI],
    images: Sequence[Tensor],
    max_drops: int | None = None,
    ridge: float = 0.0,
) -> SelectionResult:
    """Remove grains that earn nothing, then greedily drop on BIC.

    Returns retained grain indices INTO THE ORIGINAL MODEL.  Sub-models are
    rebuilt with ``JointGrainFit.subset``, so retained grains keep their fitted
    parameters rather than being re-initialized.

    TWO STAGES, deliberately separate:

    1. INACTIVE removal.  A grain whose amplitudes all solve to zero explains
       none of the image.  BIC cannot express this -- such a grain adds zero
       residual AND zero effective parameters, so it ties exactly with the model
       that omits it, and a strict-improvement search would keep it forever.
       Dropping it is definitional, not statistical, so it happens first.
    2. BIC pruning on what remains, for grains that DO carry amplitude but may
       not be worth their parameters.  Ties are broken toward the SMALLER model
       (the ``<=`` below): if two models explain the data equally well, prefer
       the one claiming fewer grains.

    This is a greedy DROP search, the mirror of the peel's greedy ADD.  Greedy
    is defensible here in a way it is not there: removing a grain cannot reveal
    intensity that was hidden, because the amplitudes simply redistribute over
    what remains -- whereas adding a grain changes what every later step sees.
    """
    start_report = model_bic(model, rois, images, ridge=ridge)
    start = start_report.bic

    # -- stage 1: grains that earned no amplitude at all -------------------
    active_mask = start_report.active
    keep = [i for i in range(model.n_grains) if bool(active_mask[i])]
    inactive = [i for i in range(model.n_grains) if not bool(active_mask[i])]
    if not keep:                      # nothing earned anything; keep as-is
        return SelectionResult(keep=list(range(model.n_grains)), bic=start,
                               start_bic=start)
    current = model.subset(keep) if inactive else model
    best_bic = model_bic(current, rois, images, ridge=ridge).bic

    # -- stage 2: BIC pruning ----------------------------------------------
    dropped: list[int] = []
    history: list[tuple[int, float]] = []
    limit = len(keep) - 1 if max_drops is None else max_drops

    for _ in range(max(limit, 0)):
        if len(keep) <= 1:
            break
        candidates: list[tuple[float, int]] = []
        for pos in range(len(keep)):
            trial = current.subset([j for j in range(len(keep)) if j != pos])
            candidates.append((model_bic(trial, rois, images, ridge=ridge).bic, pos))
        candidates.sort()
        best_trial, pos = candidates[0]
        # `<=` breaks exact ties toward the smaller model.
        if best_trial > best_bic:
            break
        dropped.append(keep[pos])
        history.append((keep[pos], best_trial))
        current = current.subset([j for j in range(len(keep)) if j != pos])
        keep = [i for j, i in enumerate(keep) if j != pos]
        best_bic = best_trial

    return SelectionResult(keep=keep, bic=best_bic, start_bic=start,
                           inactive=inactive, dropped=dropped, history=history)
