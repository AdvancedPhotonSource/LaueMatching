"""A failed post-processing step fails the pipeline.

``run_pipeline`` used to log "Post-processing failed" and then "Pipeline
complete" and return normally (exit 0). ``pipeline/dispatch/wait_static.sh``
keys on the "Pipeline complete" line, so a shard with no per-image results
counted as finished. Now: non-zero exit and no completion line; the line is
kept, unchanged, for success.

Daemon, image server and post-processor are all stubbed; only run_pipeline's
own control flow runs.
"""
import logging
import subprocess

import pytest

import laue_orchestrator as lo


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
    state = {"pp_rc": 0}

    def popen(cmd, *a, **k):
        if cmd and (cmd[0] == str(daemon) or
                    any(str(c).endswith("laue_image_server.py") for c in cmd)):
            return _FakeProc()
        return real_popen(cmd, *a, **k)

    def run(cmd, *a, **k):
        if any(str(c).endswith("laue_postprocess.py") for c in cmd):
            return subprocess.CompletedProcess(cmd, state["pp_rc"], "",
                                               "Traceback: boom")
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(lo.subprocess, "Popen", popen)
    monkeypatch.setattr(lo.subprocess, "run", run)

    params = tmp_path / "params.txt"
    params.write_text("SpaceGroup 225\nResultDir results_stream\n")
    frames = tmp_path / "frames"
    frames.mkdir()
    out = tmp_path / "run"
    (out / "results_stream").mkdir(parents=True)
    (out / "results_stream" / "solutions.txt").write_text("hdr\n")
    (out / "results_stream" / "spots.txt").write_text("hdr\n")

    def go():
        lo.run_pipeline(config_file=str(params), folder=str(frames),
                        orient_file=str(tmp_path / "o.bin"),
                        hkl_file=str(tmp_path / "h.bin"), output_dir=str(out))
    return state, go


def test_postprocess_failure_exits_nonzero_without_completion_line(stubbed, caplog):
    state, go = stubbed
    state["pp_rc"] = 3
    with caplog.at_level(logging.INFO, logger="laue_orchestrator"), \
            pytest.raises(SystemExit) as exc:
        go()
    assert exc.value.code == 3
    assert "Pipeline complete" not in caplog.text
    assert "Post-processing failed" in caplog.text


def test_signal_killed_postprocess_still_exits_nonzero(stubbed):
    state, go = stubbed
    state["pp_rc"] = -9
    with pytest.raises(SystemExit) as exc:
        go()
    assert exc.value.code == 1


def test_success_keeps_the_completion_line(stubbed, caplog):
    state, go = stubbed
    state["pp_rc"] = 0
    with caplog.at_level(logging.INFO, logger="laue_orchestrator"):
        go()
    assert "Pipeline complete" in caplog.text
