"""Characterization: lock the C-output text parsing (solutions / spots).

Pins ``read_solutions`` and ``read_spots`` on real, committed C-output files
(34-column RunImage solution layout; 11-column spots layout).  This is the
behaviour anchor for REFACTOR_PLAN §5 / §6.1 — when a typed ``Solution`` record
and ``parse_solutions`` replace the positional-column parsing, the values read
out of these files must be identical.
"""
import laue_stream_utils as lsu
from _golden import check_golden, fixture


def test_char_read_solutions():
    data, header = lsu.read_solutions(str(fixture("sample.solutions.txt")))
    snap = {
        "shape": list(data.shape),
        "header_tokens": header.split(),
        "data": data.tolist(),
    }
    check_golden("read_solutions", snap)


def test_char_read_spots():
    data, header = lsu.read_spots(str(fixture("sample.spots.txt")))
    snap = {
        "shape": list(data.shape),
        "header_tokens": header.split(),
        "data": data.tolist(),
    }
    check_golden("read_spots", snap)


if __name__ == "__main__":
    for fn in [test_char_read_solutions, test_char_read_spots]:
        fn(); print(f"PASS  {fn.__name__}")
