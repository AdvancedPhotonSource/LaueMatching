"""Characterization + §6.5/§6.2 unification anchor for the streaming postprocess.

laue_postprocess.process_single_image had no test.  Pin its filtered output on
the real frame101 fixture BEFORE refactoring its core onto the shared
PostProcessor stage, so the refactor is provably behaviour-preserving.
"""
import numpy as np
import pytest

import laue_stream_utils as lsu
import laue_postprocess as lp
from _golden import check_golden, fixture


def _run(tmp_path):
    d = np.load(fixture("frame101_filtertest.npz"))
    sol, spots, labels = d["solutions"], d["spots"], d["labels"]
    cfg = dict(lsu.DEFAULT_CONFIG)
    res = lp.process_single_image(
        image_nr=0, orientations=sol, spots=spots, cfg=cfg,
        output_dir=str(tmp_path), min_unique=2, labels=labels,
        write_indexfile=False)
    return res


def test_char_process_single_image(tmp_path):
    res = _run(tmp_path)
    snap = {
        "n_filtered": int(res["n_filtered"]),
        "kept_grains": sorted(int(r[0]) for r in res["filtered_orientations"]),
        "n_filtered_spots": int(len(res["filtered_spots"])),
    }
    check_golden("stream_process_single_image", snap)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        test_char_process_single_image(Path(d))
        print("PASS  test_char_process_single_image")
