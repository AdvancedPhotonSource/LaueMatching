"""The orientation matrix is read from the indexer's OrientMatrix columns.

The solution table has two layouts (``laue_index.records.SOLUTION_FORMATS``,
duplicated in ``laue_torch.io``): 34 columns with the matrix at 22..30
(RunImage) and 35 columns with ImageNr prepended and the matrix at 23..31
(stream). ``LaueScanLoader`` used to read columns 1..9 (Intensity, scores,
NMatches, reciprocal-matrix entries -- never the orientation), and
``load_orientations`` detected a ``%`` header by calling ``str()`` on a float
array, which never matched, so it also returned columns 0..8.

Every fixture puts the known matrix in the layout's columns and JUNK
everywhere else, so reading any other column fails.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from laue_torch.io import load_orientations, solution_orientation_columns

U_TRUE = np.array([[0.867151, 0.494088, 0.062670],
                   [-0.052670, 0.216095, -0.974957],
                   [-0.495254, 0.842135, 0.213410]])
OM_START = {34: 22, 35: 23}


def _table(n_cols: int, n_rows: int = 2) -> np.ndarray:
    rng = np.random.default_rng(n_cols)
    arr = rng.uniform(100.0, 900.0, size=(n_rows, n_cols))      # junk
    for r in range(n_rows):
        arr[r, OM_START[n_cols]:OM_START[n_cols] + 9] = U_TRUE.reshape(9)
    return arr


def test_column_map_matches_laue_index_if_installed():
    records = pytest.importorskip("laue_index.records")
    for fmt in records.SOLUTION_FORMATS.values():
        assert solution_orientation_columns(fmt.n_cols) == (fmt.om_start, fmt.om_start + 9)


def test_unknown_column_count_raises():
    with pytest.raises(ValueError, match="34 .*35"):
        solution_orientation_columns(31)


@pytest.mark.parametrize("n_cols", [34, 35])
def test_scan_loader_reads_orientation_columns(tmp_path, n_cols):
    from laue_torch.realdata import LaueScanLoader
    with h5py.File(tmp_path / "image_00001.output.h5", "w") as hf:
        g = hf.create_group("/entry/results")
        g.create_dataset("orientations", data=_table(n_cols, 3))
        g.create_dataset("filtered_orientations", data=_table(n_cols, 2))
        hf.create_dataset("/entry/data/input_blurred", data=np.zeros((4, 6)))
    for use_filtered, n in ((True, 2), (False, 3)):
        (vox,) = list(LaueScanLoader(tmp_path, use_filtered=use_filtered))
        assert vox.U_seed_list.shape == (n, 3, 3)
        for U in vox.U_seed_list:
            assert torch.allclose(U, torch.tensor(U_TRUE))


def test_scan_loader_explicit_columns_and_bad_layout(tmp_path):
    from laue_torch.realdata import LaueScanLoader
    arr = np.full((1, 20), 5.0)
    arr[0, 4:13] = U_TRUE.reshape(9)
    with h5py.File(tmp_path / "x.h5", "w") as hf:
        hf.create_dataset("/entry/results/filtered_orientations", data=arr)
        hf.create_dataset("/entry/data/input_blurred", data=np.zeros((4, 6)))
    with pytest.raises(ValueError, match="20 columns"):
        list(LaueScanLoader(tmp_path))
    (vox,) = list(LaueScanLoader(tmp_path, orientation_columns=(4, 13)))
    assert torch.allclose(vox.U_seed_list[0], torch.tensor(U_TRUE))


def test_scan_loader_empty_solutions_gives_no_seeds(tmp_path):
    from laue_torch.realdata import LaueScanLoader
    with h5py.File(tmp_path / "x.h5", "w") as hf:
        hf.create_dataset("/entry/results/filtered_orientations", data=np.empty((0,)))
        hf.create_dataset("/entry/data/input_blurred", data=np.zeros((4, 6)))
    (vox,) = list(LaueScanLoader(tmp_path))
    assert vox.U_seed_list.shape == (0, 3, 3)


@pytest.mark.parametrize("n_cols", [34, 35])
@pytest.mark.parametrize("header", ["%GrainNr\tNumberOfSolutions\tIntensity\n", "# comment\n", ""])
def test_load_orientations_solutions_file(tmp_path, n_cols, header):
    path = tmp_path / "solutions.txt"
    body = "\n".join("\t".join(f"{v:.6f}" for v in row) for row in _table(n_cols))
    path.write_text(header + body + "\n")
    U = load_orientations(path)
    assert U.shape == (2, 3, 3)
    assert torch.allclose(U, torch.tensor(U_TRUE).expand(2, 3, 3))


def test_load_orientations_plain_nine_column_file(tmp_path):
    path = tmp_path / "orient.csv"
    np.savetxt(path, U_TRUE.reshape(1, 9), delimiter="\t")
    assert torch.allclose(load_orientations(path)[0], torch.tensor(U_TRUE))
