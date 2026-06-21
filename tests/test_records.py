"""§6.1 — typed Solution record + parse_solutions.

Guards the extraction with an *equivalence* check: parse_solutions must read the
exact same values the legacy positional-column code read via
``lsu.read_solutions``.  This is what makes switching the call sites
behaviour-preserving.
"""
import numpy as np

import laue_stream_utils as lsu
from laue_index import parse_solutions, SOLUTION_FORMATS
from laue_index.records import Solution
from _golden import check_golden, fixture


def test_parse_solutions_matches_legacy_columns():
    """Fields parsed by parse_solutions == legacy positional-column reads."""
    arr, _ = lsu.read_solutions(str(fixture("sample.solutions.txt")))
    sols = parse_solutions(arr, fmt="runimage")
    assert len(sols) == arr.shape[0]
    f = SOLUTION_FORMATS["runimage"]
    for i, s in enumerate(sols):
        row = arr[i]
        assert s.grain_nr == int(row[f.grain])
        assert s.quality == float(row[f.quality])
        assert s.n_matches == int(round(float(row[f.n_matches])))
        assert s.intensity == float(row[f.intensity])
        assert np.array_equal(s.orientation.ravel(), row[f.om_start:f.om_start + 9])
        assert np.array_equal(s.recip.ravel(), row[f.recip_start:f.recip_start + 9])
        assert np.array_equal(s.lattice, row[f.lattice_start:f.lattice_start + 6])
        assert s.misorientation_post_refine == float(row[f.misorientation])
        assert s.orientation_row_nr == int(round(float(row[f.row_nr])))
        assert s.image_nr is None


def test_parse_solutions_from_path_equals_from_array():
    arr, _ = lsu.read_solutions(str(fixture("sample.solutions.txt")))
    from_arr = parse_solutions(arr, fmt="runimage")
    from_path = parse_solutions(str(fixture("sample.solutions.txt")), fmt="runimage")
    assert len(from_arr) == len(from_path)
    for a, b in zip(from_arr, from_path):
        assert a.grain_nr == b.grain_nr
        assert np.allclose(a.orientation, b.orientation)


def test_stream_format_offsets_are_runimage_plus_one():
    """The stream layout is the runimage layout with ImageNr prepended."""
    ri, st = SOLUTION_FORMATS["runimage"], SOLUTION_FORMATS["stream"]
    for attr in ("grain", "intensity", "quality", "n_matches", "n_spots_calc",
                 "recip_start", "lattice_start", "om_start", "coarse_quality",
                 "misorientation", "row_nr", "spot_grain", "spot_x", "spot_y"):
        assert getattr(st, attr) == getattr(ri, attr) + 1, attr
    assert ri.image_nr == -1 and st.image_nr == 0
    assert st.n_cols == ri.n_cols + 1


def test_parse_solutions_empty():
    assert parse_solutions(np.empty((0, 34)), fmt="runimage") == []


def test_parse_solutions_snapshot():
    """Pin the parsed records (typed view of the golden in read_solutions)."""
    sols = parse_solutions(str(fixture("sample.solutions.txt")), fmt="runimage")
    snap = [
        {"grain_nr": s.grain_nr, "n_matches": s.n_matches,
         "quality": s.quality, "row_nr": s.orientation_row_nr,
         "om": s.orientation.tolist()}
        for s in sols
    ]
    check_golden("parse_solutions", snap)


if __name__ == "__main__":
    for fn in [test_parse_solutions_matches_legacy_columns,
               test_parse_solutions_from_path_equals_from_array,
               test_stream_format_offsets_are_runimage_plus_one,
               test_parse_solutions_empty, test_parse_solutions_snapshot]:
        fn(); print(f"PASS  {fn.__name__}")
