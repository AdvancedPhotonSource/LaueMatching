"""Shared fixtures for laue_torch tests.

The parity tests validate the differentiable forward against the repository's
reference NumPy/C simulator, so they need the LaueMatching REPO checkout
(``simulation/`` and ``scripts/``) -- not just an installed ``laue_torch``.

Rather than counting ``parents[N]`` (which silently breaks whenever the package
moves in the tree, as it did when laue_torch moved to ``packages/laue_torch/``),
the repo root is discovered by walking up for a marker file. If no checkout is
found -- the normal case when testing an installed wheel or sdist -- the
repo-dependent fixtures skip instead of erroring.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

#: Marker identifying a real LaueMatching checkout.
_MARKER = Path("simulation") / "params_sim.txt"


def _find_repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / _MARKER).is_file():
            return candidate
    return None


ROOT = _find_repo_root()

# Legacy flat modules (laue_stream_utils, laue_config) live in scripts/.
if ROOT is not None:
    _scripts = str(ROOT / "scripts")
    if _scripts not in sys.path:
        sys.path.insert(0, _scripts)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    if ROOT is None:
        pytest.skip("needs a LaueMatching checkout (simulation/params_sim.txt not found)")
    return ROOT


def _fixture_file(root: Path, *parts: str) -> Path:
    path = root.joinpath(*parts)
    if not path.is_file():
        pytest.skip(f"missing reference fixture: {path}")
    return path


@pytest.fixture(scope="session")
def params_path(repo_root: Path) -> Path:
    return _fixture_file(repo_root, "simulation", "params_sim.txt")


@pytest.fixture(scope="session")
def hkl_csv(repo_root: Path) -> Path:
    # NOTE: simulation/valid_hkls.csv is gitignored -- it is generated, not
    # committed -- so a fresh clone will not have it until the sim has run.
    return _fixture_file(repo_root, "simulation", "valid_hkls.csv")


@pytest.fixture(scope="session")
def four_orients(repo_root: Path) -> Path:
    return _fixture_file(repo_root, "simulation", "fourOrientations.csv")
