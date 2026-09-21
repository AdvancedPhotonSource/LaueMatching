"""The shell launch path in the repo's pipeline/ directory.

Pins the behaviours that failed silently before 0.7.2:

* ``run_laue.sh`` resolved ``SCRIPTS`` to the repo root, launched detached, and
  printed a pid while the orchestrator died with "can't open file" in a log.
  It must now refuse a bad ``SCRIPTS`` before launching anything, resolve an
  unset one to a directory holding ``laue_orchestrator.py``, and report a launch
  that dies within its liveness wait.
* The parameter templates carry ``__SET_ME__`` placeholders that the launchers
  refuse, and the config parser rejects too (a malformed SpaceGroup / Symmetry /
  LatticeParameter / P_Array / R_Array is fatal since 0.7.2).
* ``pipeline/dispatch/mkrun.py`` makes row-aligned shards with zero-padded links
  and unique ports / ResultDirs, and refuses a port that another plan holds.
* ``run_analysis_chain.sh`` stops at the first failing step, and hands the null
  maximum that ``null_model.py`` measured to the gate as ``LAUE_NULLMAX_<PHASE>``.
* ``dispatch.sh`` sizes the preprocessing pool per shard, not per host.

Everything here needs the checkout (pipeline/ is not in the wheel) and is
skipped against an installed package.
"""
import os
import shutil
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from conftest import _REPO_ROOT

pytestmark = pytest.mark.skipif(_REPO_ROOT is None, reason="pipeline/ needs a checkout")

if _REPO_ROOT is not None:
    PIPE = _REPO_ROOT / "pipeline"
    DISPATCH = PIPE / "dispatch"
    RUN_LAUE = PIPE / "run_laue.sh"
    CHAIN = PIPE / "analysis" / "run_analysis_chain.sh"
    MKRUN = DISPATCH / "mkrun.py"
    SHELL_SCRIPTS = [RUN_LAUE, PIPE / "launch_shard.sh", CHAIN,
                     *sorted(DISPATCH.glob("*.sh"))] if DISPATCH.is_dir() else []
else:  # pragma: no cover - module is skipped
    PIPE = DISPATCH = RUN_LAUE = CHAIN = MKRUN = None
    SHELL_SCRIPTS = []

BASH = shutil.which("bash")


def _bash(script, *args, env=None, cwd=None, timeout=60):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run([BASH, str(script), *map(str, args)], capture_output=True,
                          text=True, env=e, cwd=cwd, timeout=timeout, stdin=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# syntax
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_shell_scripts_parse(script):
    r = subprocess.run([BASH, "-n", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_shell_scripts_pass_shellcheck():
    r = subprocess.run(["shellcheck", "-x", *map(str, SHELL_SCRIPTS)], capture_output=True,
                       text=True, cwd=DISPATCH)
    assert r.returncode == 0, r.stdout + r.stderr


def test_launch_shard_is_a_loud_stub():
    """The retired launcher must fail and point at the replacement, not run."""
    r = _bash(PIPE / "launch_shard.sh", "1", "0", "61200", "16")
    assert r.returncode != 0
    assert "pipeline/dispatch" in r.stderr


# ---------------------------------------------------------------------------
# run_laue.sh
# ---------------------------------------------------------------------------

@pytest.fixture
def laue_work(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    work = tmp_path / "work"
    (work / "params").mkdir(parents=True)
    cfg = work / "params" / "params_alpha.txt"
    cfg.write_text("SpaceGroup 194\n")
    return data, work, cfg


def test_run_laue_bogus_scripts_exits_before_launch(laue_work, tmp_path):
    data, work, cfg = laue_work
    r = _bash(RUN_LAUE, data, env={"WORK": str(work), "SCRIPTS": str(tmp_path / "nope"),
                                   "ALPHA_CONFIG": str(cfg), "BETA_CONFIG": ""})
    assert r.returncode != 0
    assert "laue_orchestrator.py not found" in r.stderr
    assert not (work / "results").exists(), "nothing may be created or launched"


def test_run_laue_unset_scripts_resolves_to_orchestrator(laue_work):
    data, work, cfg = laue_work
    env = {"WORK": str(work), "ALPHA_CONFIG": str(cfg), "BETA_CONFIG": "",
           "PY": sys.executable, "DRY_RUN": "1"}
    env_run = {k: v for k, v in os.environ.items() if k != "SCRIPTS"}
    env_run.update(env)
    r = subprocess.run([BASH, str(RUN_LAUE), str(data)], capture_output=True, text=True,
                       env=env_run, stdin=subprocess.DEVNULL, timeout=60)
    assert r.returncode == 0, r.stderr
    line = [ln for ln in r.stdout.splitlines() if "SCRIPTS=" in ln][0]
    scripts = Path(line.split("SCRIPTS=")[1].split()[0])
    assert (scripts / "laue_orchestrator.py").is_file()
    assert not (work / "results").exists()


@pytest.mark.parametrize("template", ["params_alpha.template.txt", "params_beta.template.txt"])
def test_run_laue_refuses_unfilled_template(laue_work, template):
    data, work, _ = laue_work
    r = _bash(RUN_LAUE, data, env={"WORK": str(work), "PY": sys.executable,
                                   "ALPHA_CONFIG": str(PIPE / template), "BETA_CONFIG": ""})
    assert r.returncode != 0
    assert "__SET_ME__" in r.stderr
    assert not (work / "results").exists()


def _fake_install(tmp_path, body):
    """A SCRIPTS dir whose orchestrator runs ``body``, and a PATH with a setsid
    that just execs (macOS has none; on Linux the real one execs in place too)."""
    scripts = tmp_path / "fake_scripts"
    scripts.mkdir()
    (scripts / "laue_orchestrator.py").write_text(body)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    setsid = bindir / "setsid"
    setsid.write_text('#!/bin/bash\nexec "$@"\n')
    setsid.chmod(0o755)
    return scripts, f"{bindir}{os.pathsep}{os.environ['PATH']}"


def test_run_laue_reports_launch_that_dies(laue_work, tmp_path):
    data, work, cfg = laue_work
    scripts, path = _fake_install(
        tmp_path, "import sys; print('boom: cannot start'); sys.exit(3)\n")
    r = _bash(RUN_LAUE, data, env={"WORK": str(work), "SCRIPTS": str(scripts), "PATH": path,
                                   "PY": sys.executable, "ALPHA_CONFIG": str(cfg),
                                   "BETA_CONFIG": "", "LIVENESS_WAIT": "1"})
    assert r.returncode != 0
    assert "exited with status 3" in r.stderr
    assert "boom: cannot start" in r.stderr, "the launch log tail must be shown"


def test_run_laue_reports_live_launch(laue_work, tmp_path):
    data, work, cfg = laue_work
    scripts, path = _fake_install(tmp_path, "import time; time.sleep(60)\n")
    r = _bash(RUN_LAUE, data, env={"WORK": str(work), "SCRIPTS": str(scripts), "PATH": path,
                                   "PY": sys.executable, "ALPHA_CONFIG": str(cfg),
                                   "BETA_CONFIG": "", "LIVENESS_WAIT": "1"})
    try:
        assert r.returncode == 0, r.stderr
        assert "alive after 1s" in r.stdout
    finally:
        for ln in r.stdout.splitlines():
            if ": pid " in ln:
                try:
                    os.kill(int(ln.split(": pid ")[1].split()[0]), signal.SIGTERM)
                except (ProcessLookupError, ValueError):
                    pass


# ---------------------------------------------------------------------------
# templates vs the config parser
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("template", ["params_alpha.template.txt", "params_beta.template.txt"])
def test_config_parser_rejects_template_placeholders(template):
    import laue_config
    with pytest.raises((SystemExit, ValueError)):
        laue_config.ConfigurationManager(str(PIPE / template))


@pytest.mark.parametrize("template", ["params_alpha.template.txt", "params_beta.template.txt"])
def test_templates_carry_no_experiment_geometry(template):
    """Every experiment-specific value is a placeholder, and the units are stated right."""
    text = (PIPE / template).read_text()
    live = {ln.split()[0]: ln.split("#")[0].split()[1:] for ln in text.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")}
    for key in ("SpaceGroup", "Symmetry", "LatticeParameter", "P_Array", "R_Array",
                "Elo", "Ehi", "OrientationFile", "ForwardFile", "HKLFile", "BackgroundFile", "ResultDir"):
        assert live[key] and all(v == "__SET_ME__" for v in live[key]), key
    assert "RADIANS" in text and "FRACTIONS" in text
    assert "(Rodrigues)" not in text


# ---------------------------------------------------------------------------
# mkrun.py
# ---------------------------------------------------------------------------

def _raster(tmp_path, n):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(1, n + 1):
        (raw / f"scan_{i}.h5").write_bytes(b"")
    tpl = tmp_path / "tpl.txt"
    tpl.write_text("SpaceGroup 194\nResultDir /somewhere/else\nBackgroundFile /bg.bin\n")
    return raw, tpl


def _mkrun(*args):
    return subprocess.run([sys.executable, str(MKRUN), *map(str, args)],
                          capture_output=True, text=True, timeout=60)


def test_mkrun_row_aligned_padded_unique(tmp_path):
    # 5 columns x 4 rows, scan stopped after 18 frames; 2 shards, 2 phases
    raw, tpl = _raster(tmp_path, 18)
    work = tmp_path / "work"
    r = _mkrun("--work", work, "--run", "sA", "--raw-dir", raw, "--prefix", "scan_",
               "--ncol", 5, "--nrow", 4, "--nframes", 18, "--slots", "hA:0,hA:1",
               "--port-base", 61200, "--phase", f"alpha={tpl}", "--phase", f"beta={tpl}")
    assert r.returncode == 0, r.stderr

    s1 = sorted(os.listdir(work / "shards" / "sA_alpha_sh1"))
    s2 = sorted(os.listdir(work / "shards" / "sA_alpha_sh2"))
    # rows 0-1 -> frames 1-10, rows 2-3 -> frames 11-18: whole rows per shard
    assert s1 == [f"scan_{i:06d}.h5" for i in range(1, 11)]
    assert s2 == [f"scan_{i:06d}.h5" for i in range(11, 19)]
    assert os.path.realpath(work / "shards" / "sA_alpha_sh2" / "scan_000011.h5") == \
        os.path.realpath(raw / "scan_11.h5")

    rows = []
    for ph in ("alpha", "beta"):
        for ln in (work / f"plan_sA_{ph}.txt").read_text().splitlines():
            if ln.strip() and not ln.startswith("#"):
                rows.append(ln.split())
    assert len(rows) == 4 and all(len(r_) == 7 for r_ in rows)
    ports = [r_[2] for r_ in rows]
    assert len(set(ports)) == 4
    rds = []
    for r_ in rows:
        text = Path(r_[5]).read_text()
        rd = [ln.split()[1] for ln in text.splitlines() if ln.startswith("ResultDir")]
        assert len(rd) == 1, "the template's ResultDir must be replaced, not duplicated"
        rds.append(rd[0])
        assert "/somewhere/else" not in text
    assert len(set(rds)) == 4
    assert all(Path(rd).is_dir() for rd in rds)


def test_mkrun_refuses_port_held_by_another_plan(tmp_path):
    raw, tpl = _raster(tmp_path, 20)
    work = tmp_path / "work"
    common = ["--work", work, "--raw-dir", raw, "--prefix", "scan_", "--ncol", 5, "--nrow", 4,
              "--phase", f"alpha={tpl}"]
    r = _mkrun(*common, "--run", "sA", "--slots", "hA:0,hA:1", "--port-base", 61200)
    assert r.returncode == 0, r.stderr
    r = _mkrun(*common, "--run", "sB", "--slots", "hB:0", "--port-base", 61201)
    assert r.returncode != 0
    assert "PORT COLLISION" in r.stderr
    assert not (work / "plan_sB_alpha.txt").exists()
    # regenerating the SAME run is not a collision with itself
    r = _mkrun(*common, "--run", "sA", "--slots", "hA:0,hA:1", "--port-base", 61200)
    assert r.returncode == 0, r.stderr


def test_mkrun_refuses_unfilled_template(tmp_path):
    raw, _ = _raster(tmp_path, 20)
    r = _mkrun("--work", tmp_path / "work", "--run", "sA", "--raw-dir", raw,
               "--prefix", "scan_", "--ncol", 5, "--nrow", 4, "--slots", "hA:0",
               "--port-base", 61200, "--phase", f"alpha={PIPE / 'params_alpha.template.txt'}")
    assert r.returncode != 0
    assert "__SET_ME__" in r.stderr
    assert not (tmp_path / "work").exists()


def test_mkrun_refuses_missing_frame(tmp_path):
    raw, tpl = _raster(tmp_path, 19)          # raster says 20
    r = _mkrun("--work", tmp_path / "work", "--run", "sA", "--raw-dir", raw,
               "--prefix", "scan_", "--ncol", 5, "--nrow", 4, "--slots", "hA:0",
               "--port-base", 61200, "--phase", f"alpha={tpl}")
    assert r.returncode != 0
    assert "MISSING SOURCE FRAME" in r.stderr


# ---------------------------------------------------------------------------
# dispatch / watch / wait (no ssh: DRY_RUN and fake run trees)
# ---------------------------------------------------------------------------

def test_dispatch_sizes_pool_per_shard(tmp_path):
    plan = tmp_path / "plan.txt"
    plan.write_text(textwrap.dedent(f"""\
        # comment
        hostA 0 61200 {tmp_path} t_sh1 {tmp_path}/p1 {tmp_path}/s1
        hostA 1 61201 {tmp_path} t_sh2 {tmp_path}/p2 {tmp_path}/s2   # trailing comment

        hostB 0 61202 {tmp_path} t_sh3 {tmp_path}/p3 {tmp_path}/s3
        """))
    r = _bash(DISPATCH / "dispatch.sh", plan, 0,
              env={"DRY_RUN": "1", "DISPATCH_NCPU": "112", "PY": sys.executable})
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "3 runs to launch" in out
    # 112 * 3 / (4 * 2) = 42 per shard on the host carrying two; 84 on the other
    assert "t_sh1" in out and "workers42 shards_on_host2" in out
    assert "workers84 shards_on_host1" in out
    assert "nothing launched" in out


_FAKE_TOOLS = {
    "nproc": "echo 112\n",
    # this user's processes: two orchestrators already running; -p PID: owner + command
    "ps": textwrap.dedent("""\
        case " $* " in
          *" -p 111 "*) echo "otheruser /opt/x/LaueMatchingGPUStream p.txt o.bin h.bin 8";;
          *" -p 222 "*) echo "someone python train.py";;
          *" -u "*) printf '%s\\n' "python /x/scripts/laue_orchestrator.py --config a" \\
                                   "python /x/scripts/laue_orchestrator.py --config b" "bash";;
        esac
        """),
    # GPU 1 holds another user's daemon, GPU 2 an unrelated job, GPU 0 is free
    "nvidia-smi": textwrap.dedent("""\
        case "$*" in
          *query-gpu*) printf '0, GPU-a\\n1, GPU-b\\n2, GPU-c\\n';;
          *query-compute-apps*) printf '111, GPU-b\\n222, GPU-c\\n';;
        esac
        """),
    "ss": 'echo "LISTEN 0 128 0.0.0.0:61250 0.0.0.0:*"\n',
}


def _fake_host_tools(tmp_path, skip=()):
    """A PATH whose nproc / ps / nvidia-smi / ss describe a busy GPU host. The plan
    names THIS host, so dispatch.sh runs its probe locally through the same
    `bash -s` script it sends over ssh."""
    b = tmp_path / "fakebin"
    b.mkdir()
    for name, body in _FAKE_TOOLS.items():
        if name in skip:
            continue
        f = b / name
        f.write_text("#!/bin/bash\n" + body)
        f.chmod(0o755)
    return f"{b}{os.pathsep}{os.environ['PATH']}"


def _this_host():
    return subprocess.run(["hostname", "-s"], capture_output=True, text=True).stdout.strip()


def _one_line_plan(tmp_path, gpu, port, name="plan.txt"):
    p = tmp_path / name
    p.write_text(f"{_this_host()} {gpu} {port} {tmp_path} t_sh{gpu} {tmp_path}/p {tmp_path}/s\n")
    return p


def _dispatch_probe(tmp_path, gpu, port, skip=(), **env):
    path = _fake_host_tools(tmp_path, skip)
    plan = _one_line_plan(tmp_path, gpu, port)
    return _bash(DISPATCH / "dispatch.sh", plan, 0,
                 env={"PATH": path, "DRY_RUN": "1", "PY": sys.executable, **env})


def test_dispatch_free_gpu_counts_running_orchestrators(tmp_path):
    """GPU 0 is free: go ahead, and size the pool over this plan's shard PLUS the two
    orchestrators of this user already on the host: 112 * 3 / (4 * 3) = 28."""
    r = _dispatch_probe(tmp_path, 0, 61200)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 shard(s) in this plan + 2 orchestrator(s) of yours already running = 3" in r.stdout
    assert "workers28 shards_on_host3" in r.stdout


def test_dispatch_refuses_gpu_held_by_any_users_daemon(tmp_path):
    r = _dispatch_probe(tmp_path, 1, 61201)
    assert r.returncode != 0
    assert "GPU 1 already runs a daemon (otheruser" in r.stderr


def test_dispatch_shared_gpu_can_be_allowed(tmp_path):
    r = _dispatch_probe(tmp_path, 1, 61201, ALLOW_SHARED_GPU="1")
    assert r.returncode == 0, r.stderr
    assert "SHARED with a running daemon" in r.stdout


def test_dispatch_other_job_on_gpu_warns_only(tmp_path):
    r = _dispatch_probe(tmp_path, 2, 61202)
    assert r.returncode == 0, r.stderr
    assert "GPU 2 also runs another compute process" in r.stderr


def test_dispatch_refuses_bound_port_even_when_gpu_sharing_allowed(tmp_path):
    r = _dispatch_probe(tmp_path, 0, 61250, ALLOW_SHARED_GPU="1")
    assert r.returncode != 0
    assert "port(s) already bound: 61250" in r.stderr


def test_dispatch_refuses_host_it_cannot_check(tmp_path):
    # The fake PATH is PREPENDED to the real one, so leaving the fake out cannot
    # hide a real nvidia-smi: on a GPU host the probe finds it and the host looks
    # checkable. Measured on a 4-GPU Linux host, where this test failed for
    # exactly that reason. It runs wherever nvidia-smi is absent (CI, macOS).
    import shutil
    if shutil.which("nvidia-smi"):
        pytest.skip("a real nvidia-smi is on PATH here and cannot be hidden")
    r = _dispatch_probe(tmp_path, 0, 61200, skip=("nvidia-smi",))
    assert r.returncode != 0
    assert "no nvidia-smi" in r.stderr


def test_dispatch_missing_scripts_dir_says_so(tmp_path):
    """A dispatch/ copied out of the checkout has no ../../scripts: a clear ERROR,
    not cd's bare message."""
    d = tmp_path / "a" / "b" / "dispatch"
    shutil.copytree(DISPATCH, d, ignore=shutil.ignore_patterns("__pycache__"))
    env = {k: v for k, v in os.environ.items() if k != "SCRIPTS"}
    env.update(PY=sys.executable, DRY_RUN="1")
    r = subprocess.run([BASH, str(d / "dispatch.sh"), str(_one_line_plan(tmp_path, 0, 61200))],
                       capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL, timeout=60)
    assert r.returncode != 0
    assert "ERROR: SCRIPTS is not set" in r.stderr


def _preflight_setup(tmp_path, nx, ny, nbytes):
    import numpy as np
    shard = tmp_path / "s"
    shard.mkdir()
    (shard / "f_000001.h5").write_bytes(b"")
    bg = tmp_path / "bg.bin"
    np.full(nbytes // 8, 100.0).tofile(bg)
    px = "".join(f"{k} {v}\n" for k, v in (("NrPxX", nx), ("NrPxY", ny)) if v is not None)
    (tmp_path / "p").write_text(f"SpaceGroup 194\nResultDir {tmp_path}/rd\nBackgroundFile {bg}\n{px}")
    return _one_line_plan(tmp_path, 0, 61200)


def test_preflight_background_size_follows_params_detector(tmp_path):
    """A 4 x 2 detector needs a 64-byte background, not 2048 x 2048's 33554432."""
    plan = _preflight_setup(tmp_path, 4, 2, 64)
    r = _bash(DISPATCH / "preflight.sh", plan, env={"PY": sys.executable, "LOCAL_PY": sys.executable})
    assert "background is" not in r.stderr, r.stderr
    assert "background bg.bin median 100.0  OK" in r.stdout


def test_preflight_refuses_params_without_detector_size(tmp_path):
    plan = _preflight_setup(tmp_path, None, 2048, 2048 * 2048 * 8)
    r = _bash(DISPATCH / "preflight.sh", plan, env={"PY": sys.executable, "LOCAL_PY": sys.executable})
    assert r.returncode != 0
    assert "NrPxX missing" in r.stderr


def _fake_runs(tmp_path, tags, n_out, clean=True, finished=True, extra_daemon=""):
    for t in tags:
        res = tmp_path / "run" / t / "results"
        (res / "alpha_1" / "results").mkdir(parents=True)
        (res / "alpha_1" / "daemon.log").write_text(
            ("LaueMatchingGPUStream exited cleanly.\n" if clean else "working\n") + extra_daemon)
        if finished:
            (res / "alpha_1.launch.log").write_text("Pipeline complete in 3.0s\n")
        for i in range(n_out):
            (res / "alpha_1" / "results" / f"image_{i}.output.h5").write_bytes(b"")
    plan = tmp_path / "plan.txt"
    plan.write_text("".join(f"h 0 {61200 + k} {tmp_path} {t} p s\n" for k, t in enumerate(tags)))
    return plan


def test_watch_arm_clean_and_failure(tmp_path):
    plan = _fake_runs(tmp_path, ["a_sh1", "a_sh2"], 3)
    r = _bash(DISPATCH / "watch_arm.sh", plan, env={"WATCH_INTERVAL": "0"})
    assert r.returncode == 0 and "ALL 2 RUNS EXITED CLEANLY" in r.stdout
    with open(tmp_path / "run" / "a_sh2" / "results" / "alpha_1" / "daemon.log", "a") as f:
        f.write("Can't spawn new thread: res = 11\nGPUassert: busy or unavailable\n")
    r = _bash(DISPATCH / "watch_arm.sh", plan, env={"WATCH_INTERVAL": "0"})
    assert r.returncode == 1
    assert "a_sh2[GPUassert,res = 11]" in r.stdout


def test_wait_static_complete_short_and_unfinished(tmp_path):
    plan = _fake_runs(tmp_path, ["a_sh1", "a_sh2"], 3)
    r = _bash(DISPATCH / "wait_static.sh", 6, plan, env={"WAIT_INTERVAL": "0"})
    assert r.returncode == 0 and "COMPLETE AND STATIC: 6/6" in r.stdout
    r = _bash(DISPATCH / "wait_static.sh", 8, plan, env={"WAIT_INTERVAL": "0"})
    assert r.returncode == 1 and "STATIC BUT SHORT: 6/8" in r.stdout
    # an orchestrator still post-processing: a static count is NOT yet "short"
    (tmp_path / "run" / "a_sh1" / "results" / "alpha_1.launch.log").unlink()
    r = _bash(DISPATCH / "wait_static.sh", 8, plan,
              env={"WAIT_INTERVAL": "0", "WAIT_MAX_POLLS": "3"})
    assert r.returncode == 2 and "WAIT_TIMEOUT" in r.stdout


# ---------------------------------------------------------------------------
# run_analysis_chain.sh
# ---------------------------------------------------------------------------

_STUB = "import sys; sys.exit(0)\n"


def _chain_scripts(tmp_path, overrides):
    """Stub analysis scripts, plus the REAL frame_peaks.py: the chain reads the
    null json through its null_json_path / gate_statistic, so the stub dir has to
    carry the module the real scripts share."""
    d = tmp_path / "scripts"
    d.mkdir()
    shutil.copy(PIPE / "analysis" / "frame_peaks.py", d / "frame_peaks.py")
    names = ["parentbeta_validate", "null_model", "empirical_gate",
             "beta_alpha_exclusion_census", "exclusion_null", "parentbeta_reconstruct",
             "anchor_null", "variant_coherence", "validated_figures"]
    for n in names:
        (d / f"{n}.py").write_text(overrides.get(n, _STUB))
    return d


def _null_writer(phases):
    """A null_model.py stub writing the json schema null_model.py documents."""
    return textwrap.dedent(f"""\
        import json, os
        w, p = os.environ["LAUE_WORK"], os.environ["LAUE_OUT_PREFIX"]
        os.makedirs(os.path.join(w, "peel_map"), exist_ok=True)
        ph = {{k: {{"nhit": {{"statistic": "nhit", "max": a}},
                   "nhit_distinct": {{"statistic": "nhit_distinct", "max": b}}}}
              for k, (a, b) in {phases!r}.items()}}
        json.dump({{"schema": 1, "phases": ph}},
                  open(os.path.join(w, "peel_map", p + "_null.json"), "w"))
        """)


_NULL_BOTH = _null_writer({"alpha": (13, 9), "beta": (11, 8)})
_GATE_PRINTS_ENV = textwrap.dedent("""\
    import os
    print("GATE SAW", os.environ.get("LAUE_NULLMAX_ALPHA"), os.environ.get("LAUE_NULLMAX_BETA"))
    """)


def _chain_env(tmp_path, scripts, **extra):
    """A complete two-phase environment; ``extra`` overrides, ``None`` removes."""
    params = tmp_path / "params"
    params.mkdir(exist_ok=True)
    for ph in ("alpha", "beta", "zn"):
        (params / f"params_{ph}.txt").write_text("SpaceGroup 194\n")
    env = {"LAUE_PHASES": "alpha,beta",
           "LAUE_WORK": str(tmp_path / "work"), "LAUE_SCAN_DATA": str(tmp_path),
           "LAUE_SCAN_ALPHA": str(tmp_path), "LAUE_SCAN_BETA": str(tmp_path),
           "LAUE_PARAMS_ALPHA": str(params / "params_alpha.txt"),
           "LAUE_PARAMS_BETA": str(params / "params_beta.txt"),
           "LAUE_MOUNT_DEG": "45",
           "LAUE_OUT_PREFIX": "t", "SCRIPTDIR": str(scripts), "PY": sys.executable,
           "LAUE_NULLMAX_ALPHA": "16",   # a stale inherited value must be replaced
           "LAUE_GATE_STAT": "nhit"}
    env.update(extra)
    return {k: v for k, v in env.items() if v is not None}


def _chain(env):
    """Run the chain with ONLY the given LAUE_* variables (none leak in from the
    shell running the tests)."""
    full = {k: v for k, v in os.environ.items() if not k.startswith("LAUE_")}
    full.update(env)
    return subprocess.run([BASH, str(CHAIN)], capture_output=True, text=True, env=full,
                          stdin=subprocess.DEVNULL, timeout=120)


def _needs_chain_imports():
    try:
        import h5py, matplotlib, numpy, scipy  # noqa: F401
    except ImportError:
        pytest.skip("run_analysis_chain.sh checks PY can import numpy/scipy/h5py/matplotlib")


_TWO_PHASE_STEPS = ("beta_alpha_exclusion_census", "exclusion_null", "parentbeta_reconstruct",
                    "anchor_null", "variant_coherence")


def _marking_stubs(names):
    """Stubs that print RAN <name> [<argv>] so the test can see what ran."""
    return {n: f"import sys; print('RAN {n}', *sys.argv[1:])\n" for n in names}


@pytest.mark.parametrize("stat, expect", [("nhit", "13 11"), ("nhit_distinct", "9 8")])
def test_chain_exports_measured_null_and_finishes(tmp_path, stat, expect):
    """The gate sees the maximum null_model.py measured, for the gate statistic in
    force -- not the stale LAUE_NULLMAX_ALPHA=16 the caller had exported."""
    _needs_chain_imports()
    s = _chain_scripts(tmp_path, {**_marking_stubs(_TWO_PHASE_STEPS + ("validated_figures",)),
                                  "null_model": _NULL_BOTH, "empirical_gate": _GATE_PRINTS_ENV})
    r = _chain(_chain_env(tmp_path, s, LAUE_GATE_STAT=stat))
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"GATE SAW {expect}" in r.stdout
    for n in _TWO_PHASE_STEPS + ("validated_figures",):
        assert f"RAN {n}" in r.stdout, n
    assert "analysis chain finished" in r.stdout


def test_chain_single_phase_skips_two_phase_steps(tmp_path):
    """LAUE_PHASES=zn: validate + null + gate for zn only; every alpha+beta step and
    the alpha/beta figure plate are skipped and announced. LAUE_PARAMS may stand in
    for LAUE_PARAMS_ZN, and neither LAUE_SCAN_BETA nor LAUE_MOUNT_DEG is needed."""
    _needs_chain_imports()
    s = _chain_scripts(tmp_path, {
        **_marking_stubs(_TWO_PHASE_STEPS + ("validated_figures", "parentbeta_validate")),
        "null_model": _null_writer({"zn": (12, 7)}),
        "empirical_gate": "import os; print('GATE SAW', os.environ['LAUE_NULLMAX_ZN'], "
                          "os.environ['LAUE_PHASES'])\n"})
    env = _chain_env(tmp_path, s, LAUE_PHASES="zn", LAUE_SCAN_ZN=str(tmp_path),
                     LAUE_PARAMS=str(tmp_path / "params" / "params_zn.txt"),
                     LAUE_SCAN_ALPHA=None, LAUE_SCAN_BETA=None, LAUE_PARAMS_ALPHA=None,
                     LAUE_PARAMS_BETA=None, LAUE_MOUNT_DEG=None, LAUE_NULLMAX_ALPHA=None)
    r = _chain(env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RAN parentbeta_validate zn" in r.stdout
    assert "RAN parentbeta_validate alpha" not in r.stdout
    assert "GATE SAW 12 zn" in r.stdout
    for n in _TWO_PHASE_STEPS + ("validated_figures",):
        assert f"RAN {n}" not in r.stdout, n
    assert "SKIPPED (need both alpha and beta" in r.stdout
    assert "SKIPPED: validated_figures.py" in r.stdout
    assert "analysis chain finished" in r.stdout


def test_chain_space_separated_phases_exported_comma_separated(tmp_path):
    """laue_material / null_model / empirical_gate split LAUE_PHASES on commas only,
    so the chain accepts spaces but hands the scripts the comma form."""
    _needs_chain_imports()
    s = _chain_scripts(tmp_path, {"null_model": _NULL_BOTH,
                                  "empirical_gate": "import os; print('PHASES', os.environ['LAUE_PHASES'])\n"})
    r = _chain(_chain_env(tmp_path, s, LAUE_PHASES="alpha beta"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PHASES alpha,beta" in r.stdout


def test_chain_reports_missing_mount_deg_before_any_step(tmp_path):
    _needs_chain_imports()
    s = _chain_scripts(tmp_path, _marking_stubs(("parentbeta_validate", "null_model")))
    r = _chain(_chain_env(tmp_path, s, LAUE_MOUNT_DEG=None))
    assert r.returncode != 0
    assert "LAUE_MOUNT_DEG" in r.stderr
    assert "RAN" not in r.stdout, "no step may run before the environment is complete"
    assert not (tmp_path / "work").exists()


def test_chain_reports_all_missing_variables_at_once(tmp_path):
    _needs_chain_imports()
    s = _chain_scripts(tmp_path, _marking_stubs(("parentbeta_validate",)))
    r = _chain(_chain_env(tmp_path, s, LAUE_MOUNT_DEG=None, LAUE_SCAN_BETA=None,
                          LAUE_PARAMS_ALPHA=None, LAUE_OUT_PREFIX=None))
    assert r.returncode != 0
    for var in ("LAUE_MOUNT_DEG", "LAUE_SCAN_BETA", "LAUE_PARAMS_ALPHA", "LAUE_OUT_PREFIX"):
        assert var in r.stderr, var
    assert "RAN" not in r.stdout


def test_chain_requires_laue_phases(tmp_path):
    s = _chain_scripts(tmp_path, _marking_stubs(("parentbeta_validate",)))
    r = _chain(_chain_env(tmp_path, s, LAUE_PHASES=None))
    assert r.returncode != 0
    assert "LAUE_PHASES is not set" in r.stderr
    assert "RAN" not in r.stdout


def test_chain_refuses_generic_params_for_two_phases(tmp_path):
    """LAUE_PARAMS cannot describe two phases (laue_material refuses it too)."""
    _needs_chain_imports()
    s = _chain_scripts(tmp_path, _marking_stubs(("parentbeta_validate",)))
    r = _chain(_chain_env(tmp_path, s, LAUE_PARAMS_ALPHA=None, LAUE_PARAMS_BETA=None,
                          LAUE_PARAMS=str(tmp_path / "params" / "params_alpha.txt")))
    assert r.returncode != 0
    assert "LAUE_PARAMS_ALPHA" in r.stderr and "LAUE_PARAMS_BETA" in r.stderr
    assert "RAN" not in r.stdout


def test_chain_stops_at_first_failing_step(tmp_path):
    _needs_chain_imports()
    s = _chain_scripts(tmp_path, {"null_model": _NULL_BOTH,
                                  "beta_alpha_exclusion_census": "raise ImportError('boom')\n",
                                  "exclusion_null": "print('SHOULD NOT RUN')\n"})
    r = _chain(_chain_env(tmp_path, s))
    assert r.returncode != 0
    assert "STEP FAILED" in r.stdout
    assert "SHOULD NOT RUN" not in r.stdout
    assert "analysis chain finished" not in r.stdout


def test_chain_refuses_when_null_not_measured(tmp_path):
    _needs_chain_imports()
    s = _chain_scripts(tmp_path, {"null_model": _null_writer({"alpha": (13, 9)}),
                                  "empirical_gate": "print('GATE RAN')\n"})
    r = _chain(_chain_env(tmp_path, s))
    assert r.returncode != 0
    assert "no measured null for phase beta" in r.stdout
    assert "GATE RAN" not in r.stdout
