"""Reference implementation of the Gürsoy *et al.* JAC 2022 two-stage solver.

Per-pixel reconstruction: given a vector ``d ∈ ℝ^M`` of M scan-frame
intensities at a single detector pixel, find ``(p, s)`` minimising

.. math::
   \\min_{p,\\,s\\ge 0}\\; \\| A_p\\, s - d \\|_2^2

where ``p`` is an integer offset into the binary coded sequence,
``A_p ∈ ℝ^{M×N}`` is the convolution-shifted coding matrix
(``A_p[i,j] = a_{p+i+j}``; see Eq. (2) of Gürsoy 2022), and ``s ∈ ℝ_+^N``
is the non-negative signal footprint along the beam path.

Algorithm:
1. Enumerate candidate offsets ``p ∈ [p_min, p_max]``.
2. For each ``p``, solve ``s = NNLS(A_p, d)`` (``scipy.optimize.nnls``).
3. Return the ``(p^*, s^*)`` with smallest residual.

Ray-tracing from ``(pixel, p^*)`` back to source depth follows the
standard Laue geometry; for the synthetic head-to-head comparison
script we report ``argmax(s^*)`` directly, since the recovered peak
index encodes depth via the mask's known per-bar geometry.

The implementation is **pure numpy + scipy**, with no torch
dependency.  This deliberately matches the published method exactly
and removes any "framework-effect" confound from the head-to-head
comparison with our differentiable refiner.

Reference: Gürsoy, Sheyfer, Wojcik, Liu & Tischler.
*J. Appl. Cryst.* **55**, 1104–1110 (2022), §2.2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import nnls


@dataclass
class TwoStagePixelResult:
    """Outcome of running the two-stage solver on one detector pixel."""

    p_best: int                          # integer offset minimising residual
    s_best: np.ndarray                   # (N,) non-negative signal footprint
    residual: float                      # ||A_{p*} s* − d||_2
    peak_index: int                      # argmax(s_best); position on beam path
    candidates_residuals: np.ndarray     # residual at every candidate p (for plotting)


def _build_A_p(p: int, M: int, N: int, coded_sequence: np.ndarray) -> np.ndarray:
    """Build the M × N coding matrix at offset ``p``.

    Entries that fall outside the coded sequence are zero (the rays are
    "off the mask" — full transmission, but with no modulation
    information so they contribute nothing to identifying ``p``).  In
    the published Si-calibration regime the scan range is sized so
    most entries are in-range; this is just a defensive boundary
    handling, not a published algorithmic choice.
    """
    L = coded_sequence.shape[0]
    A = np.zeros((M, N), dtype=np.float64)
    for i in range(M):
        for j in range(N):
            idx = p + i + j
            if 0 <= idx < L:
                A[i, j] = coded_sequence[idx]
    return A


def two_stage_pixel_reconstruct(
    d: np.ndarray,
    coded_sequence: np.ndarray,
    *,
    n_signal: int,
    p_candidates: Optional[Sequence[int]] = None,
    normalize: bool = True,
) -> TwoStagePixelResult:
    """Recover ``(p, s)`` for a single pixel via exhaustive search + NNLS.

    Parameters
    ----------
    d
        ``(M,)`` per-frame intensities at one detector pixel.
    coded_sequence
        ``(L,)`` array of coded-aperture transmissivities — typically
        ``[0, 1]`` for binary masks (``0`` = absorber, ``1`` = open).
        For multi-level masks pass the per-bar Beer-Lambert
        transmission at the relevant photon energy.
    n_signal
        Length of the signal vector ``s``.  Per Gürsoy 2022 should be
        ≥ the maximum expected signal extent; setting it to ``len(d) +
        a_few_bars`` is a safe default.
    p_candidates
        Candidate offsets to enumerate.  Defaults to all valid offsets
        ``0 ≤ p ≤ L − M − N``.
    normalize
        If True (default), bias-correct and rescale ``d`` into ``[0, 1]``
        before NNLS, matching Eq. (1) of Gürsoy 2023.  Avoids the
        absolute-flux ambiguity in the encoded measurements.
    """
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    coded_sequence = np.asarray(coded_sequence, dtype=np.float64).reshape(-1)
    M = d.shape[0]
    L = coded_sequence.shape[0]
    N = int(n_signal)

    if normalize and (d.max() - d.min()) > 1e-30:
        d_norm = (d - d.min()) / (d.max() - d.min())
    else:
        d_norm = d.copy()

    if p_candidates is None:
        p_candidates = list(range(0, max(L - M - N + 2, 1)))
    p_candidates = list(p_candidates)

    residuals = np.full(len(p_candidates), np.inf, dtype=np.float64)
    best = None
    for k, p in enumerate(p_candidates):
        A_p = _build_A_p(p, M, N, coded_sequence)
        if not np.any(A_p):
            continue
        s_hat, res = nnls(A_p, d_norm)
        residuals[k] = float(res)
        if best is None or res < best.residual:
            best = TwoStagePixelResult(
                p_best=int(p),
                s_best=s_hat,
                residual=float(res),
                peak_index=int(np.argmax(s_hat)),
                candidates_residuals=residuals,            # filled below
            )
    if best is None:
        raise RuntimeError("two-stage solver: no candidate produced a valid A_p")
    best.candidates_residuals = residuals
    return best


def two_stage_scan_reconstruct(
    intensity_matrix: np.ndarray,
    coded_sequence: np.ndarray,
    *,
    n_signal: int,
    p_candidates: Optional[Sequence[int]] = None,
    normalize: bool = True,
) -> list[TwoStagePixelResult]:
    """Per-pixel two-stage reconstruction over a list of pixels.

    Parameters
    ----------
    intensity_matrix
        ``(n_pixels, M)`` matrix — one row per pixel, M frames each.
    coded_sequence, n_signal, p_candidates, normalize
        Forwarded to :func:`two_stage_pixel_reconstruct`.
    """
    return [
        two_stage_pixel_reconstruct(
            row, coded_sequence,
            n_signal=n_signal, p_candidates=p_candidates,
            normalize=normalize,
        )
        for row in intensity_matrix
    ]
