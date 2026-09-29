"""Shared setup for the laue_jax parity suite.

The only test here is a parity gate against ``laue_torch`` on the golden
LaueMatching config, so it needs a real repo checkout (``simulation/``), not
just an installed ``laue_jax``.

The repo root is discovered by walking up for a marker file rather than by
counting ``parents[N]`` -- the hardcoded depth silently broke when the packages
moved to ``packages/laue_jax/``, and it pointed at a directory that does not
exist instead of skipping. If no checkout is found (installed-wheel testing),
the fixtures skip.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Run from ``packages/`` (``python -m pytest laue_jax/tests``), the cwd is on
# sys.path and holds the PROJECT dirs ``laue_torch/`` and ``laue_jax/``. Those
# import as empty PEP 420 namespace packages ahead of the editable-install
# finders, so ``from laue_torch import LaueForwardModel`` failed at collection.
# Drop that entry; the real packages come from the install.
_PACKAGES_DIR = Path(__file__).resolve().parents[2]


def _is_packages_dir(entry: str) -> bool:
    try:
        return Path(entry or os.getcwd()).resolve() == _PACKAGES_DIR
    except OSError:
        return False


sys.path[:] = [p for p in sys.path if not _is_packages_dir(p)]
for _name in ("laue_torch", "laue_jax"):
    _mod = sys.modules.get(_name)
    if _mod is not None and getattr(_mod, "__file__", None) is None:
        del sys.modules[_name]      # a namespace-package shadow already imported

_MARKER = Path("simulation") / "params_sim.txt"


def _find_repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / _MARKER).is_file():
            return candidate
    return None


ROOT = _find_repo_root()


@pytest.fixture(scope="session")
def repo_root() -> Path:
    if ROOT is None:
        pytest.skip("needs a LaueMatching checkout (simulation/params_sim.txt not found)")
    return ROOT


@pytest.fixture(scope="session")
def sim_dir(repo_root: Path) -> Path:
    """simulation/, with each required fixture checked.

    NOTE: valid_hkls.csv is gitignored -- generated, not committed -- so a fresh
    clone will not have it until the simulation has been run once.
    """
    sim = repo_root / "simulation"
    for name in ("params_sim.txt", "valid_hkls.csv", "fourOrientations.csv"):
        if not (sim / name).is_file():
            pytest.skip(f"missing reference fixture: {sim / name}")
    return sim
