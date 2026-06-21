"""§6.5 — Indexer stage (C-binary wrapper) unit checks.

The execution path itself is covered end-to-end by test_char_e2e (which calls
RunImage -> run_indexer -> the real binary).  Here we pin the cheap, pure bits:
executable selection and the missing-binary guard.
"""
from laue_index.indexer import resolve_executable, run_indexer


def test_resolve_executable_cpu_default():
    assert resolve_executable("/repo", "CPU", do_forward=False).endswith("bin/LaueMatchingCPU")


def test_resolve_executable_gpu():
    assert resolve_executable("/repo", "GPU", do_forward=False).endswith("bin/LaueMatchingGPU")


def test_resolve_executable_gpu_with_forward_falls_back_to_cpu():
    # DoFwd has no GPU path -> CPU binary.
    assert resolve_executable("/repo", "GPU", do_forward=True).endswith("bin/LaueMatchingCPU")


def test_run_indexer_missing_binary():
    res = run_indexer(
        repo_root="/nonexistent_repo_xyz", config_file="c.txt",
        orient_db_file="o.bin", hkl_file="h.csv", image_bin="i.bin",
        ncpus=1, output_path="/tmp/_laue_test_out")
    assert res.success is False
    assert "not found" in (res.error or "")


if __name__ == "__main__":
    for fn in [test_resolve_executable_cpu_default, test_resolve_executable_gpu,
               test_resolve_executable_gpu_with_forward_falls_back_to_cpu,
               test_run_indexer_missing_binary]:
        fn(); print(f"PASS  {fn.__name__}")
