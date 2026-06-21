"""Characterization: lock the full image preprocessing pipeline.

Runs ``preprocess_image`` (background -> threshold -> connected components ->
small-component filter -> Gaussian blur) on a REAL Laue frame and pins summary
statistics of every intermediate.  This is the behaviour anchor for the planned
``Preprocessor`` stage (REFACTOR_PLAN §3 / §6.5).

The real frame (``Indent_KB_2D_scan2_1040.h5``) is a heavy, local-only fixture
(see ``local_fixture``); the test skips when it is unavailable (e.g. CI) so the
golden is only checked where the data exists.

Background subtraction is deliberately supplied explicitly (a zero background)
rather than computed: the in-pipeline ``compute_background`` uses diplib's
median filter, which is environment-dependent (and segfaults under the
duplicate-OpenMP runtime on macOS).  The refactor's Preprocessor will reuse the
same ``compute_background`` verbatim, so it is covered by reuse; here we pin the
deterministic post-background stages (threshold -> components -> filter -> blur).
"""
import numpy as np
import pytest

import laue_stream_utils as lsu
from _golden import check_golden, fixture, local_fixture

_FRAME = "Indent_KB_2D_scan2_1040.h5"


def _stats(arr):
    arr = np.asarray(arr)
    return {
        "shape": list(arr.shape),
        "nonzero": int((arr != 0).sum()),
        "sum": float(arr.sum()),
        "max": float(arr.max()) if arr.size else 0.0,
    }


@pytest.mark.skipif(local_fixture(_FRAME) is None,
                    reason=f"local fixture {_FRAME} unavailable")
def test_char_preprocess_real_frame():
    raw = lsu.load_h5_image(str(local_fixture(_FRAME)))
    cfg = lsu.parse_config(str(fixture("params_NiIndent.txt")))
    # Explicit zero background (see module docstring) -> deterministic, env-safe.
    background = np.zeros_like(raw)
    out = lsu.preprocess_image(raw, cfg, background=background,
                               return_intermediates=True)

    snap = {
        "raw": _stats(raw),
        "background": _stats(out["background"]),
        "thresholded": _stats(out["thresholded"]),
        "filt_img": _stats(out["filt_img"]),
        "blurred": _stats(out["blurred"]),
        "n_centers": int(len(out["centers"])),
    }
    check_golden("preprocess_real_frame", snap)


if __name__ == "__main__":
    if local_fixture(_FRAME) is None:
        print(f"SKIP  preprocess (no {_FRAME})")
    else:
        test_char_preprocess_real_frame()
        print("PASS  test_char_preprocess_real_frame")
