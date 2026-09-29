"""P1 (code read 2026-09-28): pipeline/dispatch fixes.

* preflight.sh checked a RELATIVE BackgroundFile against its own cwd; the image
  server resolves it in each run's own (timestamped) output directory, where it
  does not exist, and silently computes a background from frame 1.
* mkrun.py accepted a shard of more than 65535 frames; the wire protocol's
  image number is a uint16, and the image server stops at 65535, dropping the
  shard's tail.
"""
import subprocess
import sys

import pytest

from conftest import _REPO_ROOT
from test_pipeline_launch_scripts import _bash, _mkrun, _one_line_plan

pytestmark = pytest.mark.skipif(_REPO_ROOT is None, reason="pipeline/ needs a checkout")
DISPATCH = _REPO_ROOT / "pipeline" / "dispatch" if _REPO_ROOT else None


def test_preflight_refuses_relative_background(tmp_path):
    import numpy as np
    shard = tmp_path / "s"
    shard.mkdir()
    (shard / "f_000001.h5").write_bytes(b"")
    np.full(8, 100.0).tofile(tmp_path / "bg.bin")
    (tmp_path / "p").write_text(
        f"SpaceGroup 194\nResultDir {tmp_path}/rd\nBackgroundFile bg.bin\nNrPxX 4\nNrPxY 2\n")
    plan = _one_line_plan(tmp_path, 0, 61200)
    r = _bash(DISPATCH / "preflight.sh", plan, cwd=tmp_path,
              env={"PY": sys.executable, "LOCAL_PY": sys.executable})
    assert r.returncode != 0
    assert "BackgroundFile must be an absolute path" in r.stderr


def test_mkrun_refuses_a_shard_over_65535_frames(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    tpl = tmp_path / "tpl.txt"
    tpl.write_text("SpaceGroup 194\nBackgroundFile /bg.bin\n")
    # 300 x 300 raster in one shard = 90000 frames. Refused before any source
    # frame is looked at (none exist here).
    r = _mkrun("--work", tmp_path / "work", "--run", "sA", "--raw-dir", raw,
               "--prefix", "scan_", "--ncol", 300, "--nrow", 300, "--slots", "hA:0",
               "--port-base", 61200, "--phase", f"alpha={tpl}")
    assert r.returncode != 0
    assert "65535" in r.stderr
    assert "MISSING SOURCE FRAME" not in r.stderr


@pytest.mark.parametrize("template", ["params_alpha.template.txt", "params_beta.template.txt"])
def test_templates_set_robust_filter_1(template):
    """D4: the streaming path treats an ABSENT RobustFilter as the legacy filter,
    which deletes real Sigma3 twins; the templates now say RobustFilter 1."""
    import re
    text = (_REPO_ROOT / "pipeline" / template).read_text()
    assert re.search(r"^RobustFilter\s+1\b", text, re.M), template


@pytest.mark.parametrize("template", ["params_alpha.template.txt", "params_beta.template.txt"])
def test_templates_carry_every_required_key(template):
    """P3/D5: a params file without one of config_schema.REQUIRED_KEYS is refused,
    so the templates must set every one (placeholders count as set)."""
    from laue_index import config_schema as S
    text = (_REPO_ROOT / "pipeline" / template).read_text()
    keys = {ln.split()[0] for ln in text.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")}
    assert not (S.REQUIRED_KEYS - keys), sorted(S.REQUIRED_KEYS - keys)
