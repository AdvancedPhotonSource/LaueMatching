"""P1 (code read 2026-09-28): orchestrator fixes, with the daemon, image server
and post-processor stubbed (pattern of test_orchestrator_postprocess_failure).

* ResultDir: the daemon (LaueMatchingGPUStream.cu) writes to ``results_stream``
  when the params file has no ResultDir line; the orchestrator looked in
  ``results`` (the RunImage default), waited out its flush timeout and exited 1.
"""
from __future__ import annotations

import subprocess

import pytest

import laue_orchestrator as lo

REQUIRED = """SpaceGroup 225
Symmetry F
LatticeParameter 0.35238 0.35238 0.35238 90 90 90
P_Array 0.0 0.0 0.5
R_Array 0.0 0.0 0.0
Elo 5
Ehi 30
NrPxX 64
NrPxY 64
PxX 0.0002
PxY 0.0002
MaxNrLaueSpots 30
MinIntensity 0
NMeadianPasses 1
MinGoodSpots 2
RobustFilter 1
BackgroundFile bg.bin
"""


class _FakeProc:
    pid = 4242
    returncode = 0

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def send_signal(self, sig):
        pass

    def kill(self):
        pass


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    daemon = tmp_path / "LaueMatchingGPUStream"
    daemon.write_bytes(b"stub")
    monkeypatch.setattr(lo, "_find_daemon_binary", lambda: str(daemon))
    monkeypatch.setattr(lo, "_wait_for_daemon_port", lambda *a, **k: True)
    monkeypatch.setattr(lo, "_monitor", lambda *a, **k: None)
    real_popen, real_run = subprocess.Popen, subprocess.run
    state = {"pp_cmd": None, "popen_cwd": []}

    def popen(cmd, *a, **k):
        if cmd and (cmd[0] == str(daemon) or
                    any(str(c).endswith("laue_image_server.py") for c in cmd)):
            state["popen_cwd"].append(k.get("cwd"))
            return _FakeProc()
        return real_popen(cmd, *a, **k)

    def run(cmd, *a, **k):
        if any(str(c).endswith("laue_postprocess.py") for c in cmd):
            state["pp_cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(lo.subprocess, "Popen", popen)
    monkeypatch.setattr(lo.subprocess, "run", run)
    frames = tmp_path / "frames"
    frames.mkdir()
    out = tmp_path / "run"

    def go(params_text, **kw):
        params = tmp_path / "params.txt"
        params.write_text(params_text)
        lo.run_pipeline(config_file=str(params), folder=str(frames),
                        orient_file=str(tmp_path / "o.bin"),
                        hkl_file=str(tmp_path / "h.bin"), output_dir=str(out), **kw)
    return state, go, out


def _daemon_outputs(out, result_dir):
    d = out / result_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / "solutions.txt").write_text("hdr\n")
    (d / "spots.txt").write_text("hdr\n")
    return d


def test_no_resultdir_line_reads_results_stream(stubbed):
    state, go, out = stubbed
    d = _daemon_outputs(out, "results_stream")
    go(REQUIRED)
    assert state["pp_cmd"] is not None
    assert state["pp_cmd"][state["pp_cmd"].index("--solutions") + 1] == str(d / "solutions.txt")


def test_explicit_resultdir_is_used(stubbed):
    state, go, out = stubbed
    d = _daemon_outputs(out, "mine")
    go(REQUIRED + "ResultDir mine   # comment\n")
    assert state["pp_cmd"][state["pp_cmd"].index("--solutions") + 1] == str(d / "solutions.txt")


@pytest.mark.parametrize("text,expect", [
    ("", "results_stream"),
    ("ResultDir a\nResultDir b\n", "b"),           # last line wins, as in the C loop
    ("  ResultDir x\n", "results_stream"),          # the C matches at column 0
    ("ResultDir\n", "results_stream"),              # sscanf leaves the default
    ("ResultDir /abs/dir\n", "/abs/dir"),
    ("ResultDirX y\n", "results_stream"),          # whole-token match, as the C (0.8.0)
])
def test_result_dir_parsed_as_the_c_does(tmp_path, text, expect):
    p = tmp_path / "p.txt"
    p.write_text(text)
    assert lo._daemon_result_dir(str(p)) == expect


# --------------------------------------------------------------------------- #
# Relative ForwardFile / BackgroundFile: the daemon and the image server run   #
# with cwd = output_dir, so that is where a relative path is looked up.        #
# --------------------------------------------------------------------------- #

def test_relative_paths_checked_where_the_daemon_looks(stubbed, tmp_path, monkeypatch, caplog):
    state, go, out = stubbed
    _daemon_outputs(out, "results_stream")
    # present in the orchestrator's cwd, absent from output_dir
    launch = tmp_path / "launch"
    launch.mkdir()
    (launch / "fwd.bin").write_bytes(b"x")
    (launch / "bg.bin").write_bytes(b"x")
    monkeypatch.chdir(launch)
    with caplog.at_level("WARNING", logger="laue_orchestrator"):
        go(REQUIRED + "ForwardFile fwd.bin\nBackgroundFile bg.bin\n")
    assert f"Forward simulation file not found: {out / 'fwd.bin'}" in caplog.text
    assert f"BackgroundFile not found: {out / 'bg.bin'}" in caplog.text


def test_relative_paths_present_in_output_dir_do_not_warn(stubbed, tmp_path, monkeypatch, caplog):
    state, go, out = stubbed
    _daemon_outputs(out, "results_stream")
    (out / "fwd.bin").write_bytes(b"x")
    (out / "bg.bin").write_bytes(b"x")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    with caplog.at_level("WARNING", logger="laue_orchestrator"):
        go(REQUIRED + "ForwardFile fwd.bin\nBackgroundFile bg.bin\n")
    assert "not found" not in caplog.text


# --------------------------------------------------------------------------- #
# Drain: wait for the daemon to REPORT every frame the server sent, not for   #
# solutions.txt to stop growing. No-solution frames never grow that file, so   #
# a tail of them (or one slow frame) read as "drained" and was SIGTERMed away. #
# --------------------------------------------------------------------------- #

_FAKE_DAEMON = r'''
import os, signal, sys, time
sol, marker = sys.argv[1], sys.argv[2]
printed = []
def on_term(sig, frm):
    with open(marker, "w") as f:
        f.write(" ".join(printed))
    sys.exit(0)
signal.signal(signal.SIGTERM, on_term)
os.makedirs(os.path.dirname(sol), exist_ok=True)
with open(sol, "w") as f:
    f.write("hdr\n1 0 0\n")
def say(n, what):
    print(f"[Image {n}] {what}", flush=True)
    printed.append(str(n))
say(1, "Total: 0.100 s (GPU: 1 ms, CPU: 1 ms)")
time.sleep(12)            # > the old 10 s solutions.txt quiescence window
say(2, "No matches, skipping fitting.")
say(3, "Total: 0.100 s (GPU: 1 ms, CPU: 1 ms)")   # a frame with no solution row
while True:
    time.sleep(0.2)
'''


def test_drain_waits_for_the_daemon_to_report_every_sent_frame(stubbed, tmp_path, monkeypatch):
    import json
    import sys
    state, go, out = stubbed
    out.mkdir(parents=True, exist_ok=True)
    (out / "results_stream").mkdir()
    (out / "results_stream" / "spots.txt").write_text("hdr\n")
    (out / "frame_mapping.json").write_text(json.dumps({
        "1": {"file": "a.h5", "frame": 0, "skipped": False},
        "2": {"file": "a.h5", "frame": 1, "skipped": False},
        "3": {"file": "a.h5", "frame": 2, "skipped": False},
        "4": {"file": "a.h5", "frame": 3, "skipped": True, "reason": "no_spots"}}))
    script = tmp_path / "fake_daemon.py"
    script.write_text(_FAKE_DAEMON)
    marker = tmp_path / "terminated_after.txt"
    popen = lo.subprocess.Popen
    real_popen = subprocess.Popen

    def popen_real_daemon(cmd, *a, **k):
        if cmd and str(cmd[0]).endswith("LaueMatchingGPUStream"):
            return real_popen([sys.executable, str(script),
                               str(out / "results_stream" / "solutions.txt"), str(marker)],
                              stdout=k["stdout"], stderr=k.get("stderr"), cwd=k.get("cwd"))
        return popen(cmd, *a, **k)
    monkeypatch.setattr(lo.subprocess, "Popen", popen_real_daemon)
    go(REQUIRED)
    assert marker.read_text().split() == ["1", "2", "3"], \
        "daemon was terminated before it reported every sent frame"


def test_drain_progress_parses_the_daemons_terminal_lines(tmp_path):
    log = tmp_path / "daemon.log"
    log.write_text("[Image 5] Submitting on stream 0...\n"
                   "[Image 5] GPU: 3 ms (H2D:1 Kern:1 D2H:1), 0 matches\n"
                   "[Image 5] No matches, skipping fitting.\n"
                   "[Image 6] 2 unique orientations found\n"
                   "[Image 7] Total: 0.5 s (GPU: 1 ms, CPU: 1 ms)\n[Image 8] Tot")
    prog = lo._DaemonProgress(str(log))
    assert prog.update() == {5, 7}
    with open(log, "a") as f:
        f.write("al: 0.5 s (GPU: 1 ms, CPU: 1 ms)\n")
    assert prog.update() == {5, 7, 8}          # a line split across two reads
