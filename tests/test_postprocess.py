"""§6.5b — PostProcessor stage equivalence.

The PostProcessor composes calculate_unique_spots + sort + OrientationFilter +
spot-filter.  Pin that the composition reproduces the same kept grains as the
filtering golden / the bare functions on the real frame101 fixture.
"""
import numpy as np

import laue_stream_utils as lsu
from laue_index.postprocess import PostProcessor, sort_by_quality
from _golden import fixture


def _load():
    d = np.load(fixture("frame101_filtertest.npz"))
    return d["solutions"], d["spots"], d["labels"]


def test_postprocessor_robust_matches_pieces():
    sol, spots, labels = _load()
    pp = PostProcessor(robust=True, min_unique=2, min_total_spots=5,
                       max_angle_deg=5.0, space_group=225)
    res = pp(sol, spots, labels)

    # unique-spot info must equal the bare function
    usi = lsu.calculate_unique_spots(sol, spots, labels)
    assert {int(k) for k in res.unique_spot_info} == {int(k) for k in usi}

    # kept grains match the robust filter on the same data
    robust = lsu.filter_orientations_robust(
        sol, usi, min_unique=2, min_total_spots=5, max_angle_deg=5.0,
        csl_sigmas=(3,), csl_tol_deg=3.0, cubic=True)
    assert res.kept_grain_nrs == {int(r[0]) for r in robust}
    assert np.array_equal(res.filtered_orientations, robust)


def test_postprocessor_spot_filter_consistency():
    sol, spots, labels = _load()
    pp = PostProcessor(robust=True, space_group=225)
    res = pp(sol, spots, labels)
    # every filtered spot belongs to a kept grain
    if res.filtered_spots.size:
        kept = res.kept_grain_nrs
        assert set(res.filtered_spots[:, 0].astype(int)).issubset(kept)


def test_postprocessor_legacy_branch():
    sol, spots, labels = _load()
    usi = lsu.calculate_unique_spots(sol, spots, labels)
    legacy = lsu.filter_orientations(sol, usi, min_unique=2)
    pp = PostProcessor(robust=False, min_unique=2, space_group=225)
    res = pp(sol, spots, labels)
    assert np.array_equal(res.filtered_orientations, legacy)


def test_sort_by_quality_matches_lsu():
    sol, _, _ = _load()
    assert np.array_equal(sort_by_quality(sol, 4),
                          lsu.sort_orientations_by_quality(sol))


if __name__ == "__main__":
    for fn in [test_postprocessor_robust_matches_pieces,
               test_postprocessor_spot_filter_consistency,
               test_postprocessor_legacy_branch, test_sort_by_quality_matches_lsu]:
        fn(); print(f"PASS  {fn.__name__}")
