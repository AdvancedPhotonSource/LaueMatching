"""The duplicate merge must not depend on the order candidates arrive in.

GPU kernels append candidates in atomicAdd arrival order (varies run to run);
the CPU appends in database-row order. The merge is greedy, so before 0.8.0 a
chain of candidates just inside MaxAngle clustered differently depending on
which arrived first: GPU output was non-deterministic and differed from CPU.
Candidates are now sorted by (score descending, row ascending) before merging,
on every path, so each cluster is seeded by its strongest candidate.
See fixtures/merge_order.c for the chain.
"""
import os
import subprocess

import pytest

from _cbuild import FIXTURES, HEADERS, build


def test_merge_is_independent_of_arrival_order(tmp_path):
    if not os.path.isfile(HEADERS):
        pytest.skip("c_src not present (installed wheel, not a checkout)")
    exe = str(tmp_path / "merge_order")
    err = build(exe, [os.path.join(FIXTURES, "merge_order.c")])
    if err is not None and err.startswith("no toolchain"):
        pytest.skip(err)
    assert err is None, err
    out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout
    results = {l.split()[1]: l.split("|", 1)[1].strip()
               for l in out.splitlines() if l.startswith("PERM")}
    assert len(results) == 6
    assert len(set(results.values())) == 1, "\n".join(f"{k}: {v}" for k, v in results.items())
    # Seeded by the strongest candidate (row 1, 0.9 deg, the middle of the
    # chain), which takes both neighbours: one cluster of three.
    assert next(iter(results.values())) == "1:3:300.0"
