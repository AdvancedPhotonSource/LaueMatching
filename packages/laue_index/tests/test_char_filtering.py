"""Characterization: lock the current orientation-filtering pipeline.

Drives ``calculate_unique_spots`` + ``filter_orientations`` (legacy) +
``filter_orientations_robust`` (twin/CSL-aware) on a real captured frame
(``frame101_filtertest.npz``: solutions, spots, labels) and pins the outputs.

This is the behaviour anchor for REFACTOR_PLAN §4b / §6.2 — when filtering is
lifted into ``filtering.py`` behind the ``OrientationFilter`` interface and the
RunImage / laue_postprocess duplicates are deleted, these results must not move.
"""
import numpy as np

import laue_stream_utils as lsu
from _golden import check_golden, fixture


def _load():
    d = np.load(fixture("frame101_filtertest.npz"))
    return d["solutions"], d["spots"], d["labels"]


def test_char_unique_spots():
    sol, spots, labels = _load()
    usi = lsu.calculate_unique_spots(sol, spots, labels)
    # Pin per-grain claimed-spot counts (winner-take-all assignment).
    snap = {
        int(gn): {"count": int(v["count"]),
                  "unique_label_count": int(v["unique_label_count"])}
        for gn, v in usi.items()
    }
    check_golden("unique_spots", snap)


def test_char_filter_legacy():
    sol, spots, labels = _load()
    usi = lsu.calculate_unique_spots(sol, spots, labels)
    kept = lsu.filter_orientations(sol, usi, min_unique=2)
    snap = {
        "kept_grains": sorted(int(r[0]) for r in kept),
        "n_kept": int(len(kept)),
    }
    check_golden("filter_legacy", snap)


def test_char_filter_robust():
    sol, spots, labels = _load()
    usi = lsu.calculate_unique_spots(sol, spots, labels)
    # Same defaults RunImage wires in for the robust path.
    kept = lsu.filter_orientations_robust(
        sol, usi,
        min_unique=2, grain_col=0, quality_col=4, om_start_col=22,
        nmatches_col=5, max_angle_deg=5.0, min_total_spots=5, csl_sigmas=(3,),
    )
    snap = {
        "kept_grains": sorted(int(r[0]) for r in kept),
        "n_kept": int(len(kept)),
        # quality-ordered grain sequence (the robust filter re-sorts by quality)
        "kept_grains_by_quality": [int(r[0]) for r in kept],
    }
    check_golden("filter_robust", snap)


def test_filter_strategy_classes_match_functions():
    """The §6.2 OrientationFilter classes reproduce the bare functions exactly."""
    from laue_index.filtering import (LegacyUniqueSpotFilter, RobustCSLAwareFilter,
                                      filter_orientations, filter_orientations_robust)
    sol, spots, labels = _load()
    usi = lsu.calculate_unique_spots(sol, spots, labels)

    legacy_cls = LegacyUniqueSpotFilter(min_unique=2)(sol, usi)
    legacy_fn = filter_orientations(sol, usi, min_unique=2)
    assert np.array_equal(legacy_cls, legacy_fn)

    robust_cls = RobustCSLAwareFilter(
        min_unique=2, min_total_spots=5, max_angle_deg=5.0,
        csl_sigmas=(3,), csl_tol_deg=3.0, cubic=True)(sol, usi)
    robust_fn = filter_orientations_robust(
        sol, usi, min_unique=2, min_total_spots=5, max_angle_deg=5.0,
        csl_sigmas=(3,), csl_tol_deg=3.0, cubic=True)
    assert np.array_equal(robust_cls, robust_fn)


if __name__ == "__main__":
    for fn in [test_char_unique_spots, test_char_filter_legacy,
               test_char_filter_robust, test_filter_strategy_classes_match_functions]:
        fn(); print(f"PASS  {fn.__name__}")
