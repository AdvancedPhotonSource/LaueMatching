"""Sanity tests for the Gürsoy 2022 two-stage baseline solver.

These don't aim to validate the *method* (it's the published method,
not ours); they confirm that our reference implementation correctly
recovers a synthetic delta-source pixel signal.
"""
from __future__ import annotations

import numpy as np
import pytest

from laue_torch.coded_aperture import (
    build_de_bruijn_sequence,
    two_stage_pixel_reconstruct,
)


def test_recovers_delta_source_position():
    """Inject a single-point source at known position; recover its
    absolute aperture-plane index ``p + j``.

    The two-stage method has a convolutional ambiguity: ``(p, j)`` and
    ``(p − k, j + k)`` reach identical columns of the coding matrix
    and are therefore indistinguishable on a single pixel.  Ray-tracing
    from the *aperture-plane intersection* — which is ``p + j`` — to
    source depth resolves the physical question.  We assert recovery
    of that combined index.
    """
    rng = np.random.default_rng(0)
    L = 64
    seq = build_de_bruijn_sequence(order=6, alphabet=2).numpy().astype(np.float64)
    assert seq.shape == (L,)

    M = 12         # number of scan frames
    N = 8          # signal length
    p_truth = 17   # truth offset
    j_truth = 3    # signal element with all the mass
    source_index_truth = p_truth + j_truth

    A_truth = np.zeros((M, N), dtype=np.float64)
    for i in range(M):
        for j in range(N):
            idx = p_truth + i + j
            if 0 <= idx < L:
                A_truth[i, j] = seq[idx]
    s_truth = np.zeros(N, dtype=np.float64)
    s_truth[j_truth] = 1.0
    d = A_truth @ s_truth + 0.01 * rng.standard_normal(M)

    result = two_stage_pixel_reconstruct(
        d, seq, n_signal=N,
        p_candidates=range(0, L - M - N + 1),
        normalize=False,
    )
    recovered_source = result.p_best + result.peak_index
    assert recovered_source == source_index_truth, (
        f"recovered source index = {recovered_source}, "
        f"truth {source_index_truth} "
        f"(p={result.p_best}, j={result.peak_index})"
    )


def test_residual_falls_at_truth():
    """Noise-free single-bar source: residual hits 0 across the
    ``(p, j)`` ambiguity plateau."""
    L = 32
    seq = build_de_bruijn_sequence(order=5, alphabet=2).numpy().astype(np.float64)
    M = 8
    N = 6
    p_truth = 11
    j_truth = 2
    source_index_truth = p_truth + j_truth

    A_truth = np.zeros((M, N), dtype=np.float64)
    for i in range(M):
        for j in range(N):
            idx = p_truth + i + j
            if 0 <= idx < L:
                A_truth[i, j] = seq[idx]
    d = A_truth[:, j_truth]

    result = two_stage_pixel_reconstruct(
        d, seq, n_signal=N,
        p_candidates=range(0, L - M - N + 1),
        normalize=False,
    )
    finite = result.candidates_residuals[np.isfinite(result.candidates_residuals)]
    assert finite.min() < 1e-6
    # Either branch of the ambiguity plateau is acceptable as long as
    # it recovers the absolute aperture-plane source index.
    assert result.p_best + result.peak_index == source_index_truth, (
        f"recovered source index = {result.p_best + result.peak_index}, "
        f"truth {source_index_truth}"
    )
