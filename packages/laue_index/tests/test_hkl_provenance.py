"""GenerateHKLs writes an artifact record carrying its full configuration.

An HKL list generated for another lattice or detector is silently wrong for
the run that uses it. The list now carries `<file>.meta.json` (kind hkl_list):
every generator input under the params-file key names, the column layout, the
row count and the full SHA-256, so a reader can compare the record with the
params file it is about to run (artifacts.check(expect=...)).
"""
import json
import subprocess
import sys

from laue_index import artifacts as A

LAT = ["0.26649", "0.26649", "0.49468", "90", "90", "120"]


def _generate(tmp_path, lat=LAT):
    out = tmp_path / "valid_hkls.csv"
    cmd = [sys.executable, "-c",
           "import sys; from laue_index.pipeline import add_to_path; add_to_path();"
           "import GenerateHKLs; GenerateHKLs.main()",
           "-resultFileName", str(out), "-sym", "P", "-sgnum", "194",
           "-latticeParameter", *lat, "-RArray", "-1.2", "-1.21", "-1.22",
           "-PArray", "0.0288", "0.0027", "0.513", "-NumPxX", "256", "-NumPxY", "256",
           "-dx", "0.0016", "-dy", "0.0016", "-Ehi", "20", "-Elo", "5"]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=tmp_path)
    return out


def test_hkl_list_record_has_the_full_configuration(tmp_path):
    out = _generate(tmp_path)
    rec = A.read_record(out)
    assert rec and rec["schema"] == A.SCHEMA and rec["kind"] == "hkl_list"
    cfg = rec["config"]
    assert cfg["SpaceGroup"] == 194 and cfg["Symmetry"] == "P"
    assert cfg["LatticeParameter"] == [float(v) for v in LAT]
    assert cfg["P_Array"] == [0.0288, 0.0027, 0.513] and cfg["R_Array"] == [-1.2, -1.21, -1.22]
    assert cfg["NrPxX"] == 256 and cfg["PxX"] == 0.0016 and cfg["Ehi"] == 20 and cfg["Elo"] == 5
    lay = rec["artifact"]["layout"]
    assert lay["columns"][:3] == ["h", "k", "l"]
    assert lay["rows"] == sum(1 for _ in open(out))
    assert A.check(out, "hkl_list", full=True).status == "ok"


def test_hkl_list_for_another_lattice_is_refused(tmp_path):
    out = _generate(tmp_path)
    c = A.check(out, "hkl_list", expect={"SpaceGroup": 194,
                                         "LatticeParameter": [0.2921, 0.2921, 0.4665, 90, 90, 120]})
    assert c.status == "mismatch" and "LatticeParameter" in c.reasons[0]


def test_no_legacy_sidecar_is_written(tmp_path):
    out = _generate(tmp_path)
    assert not (tmp_path / "valid_hkls.csv.provenance.json").exists()
