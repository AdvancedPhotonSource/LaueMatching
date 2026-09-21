"""Every analysis script must get past its own module body before it touches data.

The 2026-09 audit found scripts in ``pipeline/analysis/`` that raised ``NameError``
at import -- a name used on a line above the one that defines it (``NPX =
_PH.npx_x`` before ``_PH = Phase.load(...)``), or never defined at all (``B``,
``B_alp``). They could not have run in their committed form, and a shell chain
that does not check exit codes reported "finished" after them.

Two checks, both static, so no data, parameter file or environment is needed:

* ``py_compile`` -- the file parses.
* an ``ast`` walk for undefined names: at MODULE level, a name loaded before any
  statement above it binds it; inside a function, a name that is neither local,
  enclosing, module-level nor a builtin. pyflakes does the second part better when
  it is installed; this is the fallback that needs nothing, and it is exercised on
  known-bad snippets below so a silent checker cannot pass the suite.
"""
from __future__ import annotations

import ast
import builtins
import py_compile
import re
from pathlib import Path

import pytest

try:
    from conftest import _REPO_ROOT
except ImportError:  # pragma: no cover - conftest always importable under pytest
    _REPO_ROOT = None

ANALYSIS_FILES = [
    "frame_peaks.py", "laue_material.py", "null_model.py", "parentbeta_validate.py",
    "distinct_peak_gate.py", "beta_alpha_exclusion_census.py", "exclusion_null.py",
    "beta_map_validate.py", "map_validate_cluster.py", "grain_extent_backfill.py",
    "parentbeta_backfill.py", "exposure_signal_check.py", "scan_map.py", "anchor_null.py",
    "big_grain_diagnostic.py", "big_grain_split_test.py", "parentbeta_reconstruct.py",
    "empirical_gate.py", "validated_figures.py", "regrain.py", "batch_peel_driver.py",
    "collect_scan_metrics.py", "texture_null.py", "variant_coherence.py", "spot_energy.py",
    "tolerance_sensitivity.py", "cluster_orientations.py", "catalog_figures.py",
    "fix_positions.py",
]

# Every .py in pipeline/analysis (map scripts included), for the name and spawn checks.
ALL_FILES = (sorted(p.name for p in (Path(_REPO_ROOT) / "pipeline" / "analysis").glob("*.py"))
             if _REPO_ROOT is not None else ANALYSIS_FILES)

_BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__spec__",
                                  "__builtins__", "__loader__", "__package__"}


def _analysis_dir() -> Path:
    if _REPO_ROOT is None:
        pytest.skip("not running from a repo checkout (pipeline/analysis absent)")
    d = Path(_REPO_ROOT) / "pipeline" / "analysis"
    if not d.is_dir():
        pytest.skip(f"{d} not present")
    return d


# ---------------------------------------------------------------------------
# the checker
# ---------------------------------------------------------------------------
def _stored(node) -> set:
    """Names a statement/target binds (not descending into nested scopes)."""
    out = set()
    for n in _walk_scope(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                if a.name == "*":
                    continue
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            out.update(n.names)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
    return out


def _walk_scope(node):
    """ast.walk that yields nested function/class/lambda nodes but not their bodies."""
    stack = [node]
    first = True
    while stack:
        n = stack.pop()
        yield n
        if not first and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                          ast.ClassDef, ast.Lambda)):
            continue
        first = False
        stack.extend(ast.iter_child_nodes(n))


def _loads_now(node) -> list:
    """Name loads executed immediately by a module-level statement.

    Function and lambda bodies are deferred (checked separately); decorators,
    default values and class bodies run now. Comprehension targets bind locally.
    """
    out = []

    def visit(n, local):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in n.decorator_list:
                visit(d, local)
            for d in n.args.defaults + [x for x in n.args.kw_defaults if x is not None]:
                visit(d, local)
            return
        if isinstance(n, ast.Lambda):
            for d in n.args.defaults:
                visit(d, local)
            return
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            loc = set(local)
            for g in n.generators:
                visit(g.iter, loc)
                loc |= _stored(g.target)
                for c in g.ifs:
                    visit(c, loc)
            for part in ([n.key, n.value] if isinstance(n, ast.DictComp) else [n.elt]):
                visit(part, loc)
            return
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if n.id not in local:
                out.append(n)
            return
        for c in ast.iter_child_nodes(n):
            visit(c, local)

    visit(node, set())
    return out


def _module_level(stmts, bound, problems):
    """Walk statements in execution order; report loads of not-yet-bound names."""
    for s in stmts:
        if isinstance(s, ast.If):
            for n in _loads_now(s.test):
                if n.id not in bound and n.id not in _BUILTINS:
                    problems.append((n.lineno, n.id))
            b1, b2 = set(bound), set(bound)
            _module_level(s.body, b1, problems)
            _module_level(s.orelse, b2, problems)
            bound |= b1 | b2
            continue
        if isinstance(s, (ast.For, ast.While)):
            head = s.iter if isinstance(s, ast.For) else s.test
            for n in _loads_now(head):
                if n.id not in bound and n.id not in _BUILTINS:
                    problems.append((n.lineno, n.id))
            if isinstance(s, ast.For):
                bound |= _stored(s.target)
            _module_level(s.body, bound, problems)
            _module_level(s.orelse, bound, problems)
            continue
        if isinstance(s, ast.With):
            for it in s.items:
                for n in _loads_now(it.context_expr):
                    if n.id not in bound and n.id not in _BUILTINS:
                        problems.append((n.lineno, n.id))
                if it.optional_vars is not None:
                    bound |= _stored(it.optional_vars)
            _module_level(s.body, bound, problems)
            continue
        if isinstance(s, ast.Try):
            _module_level(s.body, bound, problems)
            for h in s.handlers:
                hb = set(bound)
                if h.name:
                    hb.add(h.name)
                _module_level(h.body, hb, problems)
                bound |= hb
            _module_level(s.orelse, bound, problems)
            _module_level(s.finalbody, bound, problems)
            continue
        for n in _loads_now(s):
            if n.id not in bound and n.id not in _BUILTINS:
                problems.append((n.lineno, n.id))
        bound |= _stored(s)


def _is_main_guard(n) -> bool:
    return (isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
            and isinstance(n.test.left, ast.Name) and n.test.left.id == "__name__")


def _function_scopes(tree, module_names, main_names=frozenset()):
    """Loads inside functions that resolve nowhere (local, enclosing, module, builtin).

    ``module_names`` are the globals bound OUTSIDE any ``if __name__ == "__main__":``
    block; ``main_names`` those bound only inside it. A function defined outside the
    guard sees only ``module_names``: under the 'spawn' start method a worker
    re-imports the script as ``__mp_main__``, the guard does not run, and a name bound
    only there raises NameError in the worker. A function defined inside the guard
    runs in the parent and may use both.
    """
    problems = []

    def visit(n, enclosing, in_main):
        for c in ast.iter_child_nodes(n):
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                local = set()
                for a in (c.args.posonlyargs + c.args.args + c.args.kwonlyargs):
                    local.add(a.arg)
                if c.args.vararg:
                    local.add(c.args.vararg.arg)
                if c.args.kwarg:
                    local.add(c.args.kwarg.arg)
                body = c.body if isinstance(c.body, list) else [c.body]
                for b in body:
                    local |= _stored(b)
                    for m in ast.walk(b):     # comprehension targets anywhere inside
                        if isinstance(m, ast.comprehension):
                            local |= _stored(m.target)
                scope = enclosing | local
                visible = module_names | (main_names if in_main else set())
                for b in body:
                    for m in _walk_scope(b):
                        if (isinstance(m, ast.Name) and isinstance(m.ctx, ast.Load)
                                and m.id not in scope and m.id not in visible
                                and m.id not in _BUILTINS):
                            problems.append((m.lineno, m.id))
                    visit(b, scope, in_main)
            else:
                visit(c, enclosing, in_main or (n is tree and _is_main_guard(c)))

    visit(tree, set(), False)
    return problems


def undefined_names(src: str) -> list:
    """[(lineno, name)] for names that would raise NameError."""
    tree = ast.parse(src)
    problems = []
    _module_level(tree.body, set(), problems)
    module_names, guarded = set(), set()
    for s in tree.body:
        target = guarded if _is_main_guard(s) else module_names
        target |= _stored(s)
        for m in _walk_scope(s):      # names bound inside module-level if/for/try
            if isinstance(m, ast.Name) and isinstance(m.ctx, ast.Store):
                target.add(m.id)
    problems += _function_scopes(tree, module_names, guarded - module_names)
    return sorted(set(problems))


# ---------------------------------------------------------------------------
# the checker must be able to FAIL
# ---------------------------------------------------------------------------
def test_checker_flags_use_before_definition():
    """The exact shape of the audited bug: NPX from _PH on the line above _PH."""
    bad = "import os\nNPX = _PH.npx_x\n_PH = object()\n"
    assert undefined_names(bad) == [(2, "_PH")]


def test_checker_flags_name_never_defined_inside_function():
    bad = "def project(OM):\n    return OM @ B\n"
    assert undefined_names(bad) == [(2, "B")]


def test_checker_accepts_ordinary_code():
    ok = ("import numpy as np\nX = 1\n"
          "def f(a, *b, **c):\n    y = [i for i in range(a)]\n    return np, X, y, b, c\n"
          "for k in range(2):\n    Z = k\nprint(Z, f)\n")
    assert undefined_names(ok) == []


def test_checker_flags_worker_using_a_name_bound_only_under_main():
    """The spawn failure: under 'spawn' a worker re-imports the script without
    running the guard, so TABLE does not exist there."""
    bad = ("def job(i):\n    return TABLE[i]\n"
           "if __name__ == '__main__':\n    TABLE = [1, 2]\n    print(job(0))\n")
    assert undefined_names(bad) == [(2, "TABLE")]


def test_checker_allows_main_only_names_in_main_only_functions():
    ok = ("if __name__ == '__main__':\n    TABLE = [1, 2]\n"
          "    def report():\n        return TABLE\n    print(report())\n")
    assert undefined_names(ok) == []


def test_checker_allows_worker_globals_bound_above_the_guard():
    ok = ("TABLE = [1, 2]\ndef job(i):\n    return TABLE[i]\n"
          "if __name__ == '__main__':\n    print(job(0))\n")
    assert undefined_names(ok) == []


# ---------------------------------------------------------------------------
# the scripts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ANALYSIS_FILES)
def test_analysis_script_compiles(name, tmp_path):
    p = _analysis_dir() / name
    assert p.is_file(), f"{p} missing"
    py_compile.compile(str(p), cfile=str(tmp_path / (name + "c")), doraise=True)


@pytest.mark.parametrize("name", ALL_FILES)
def test_analysis_script_has_no_undefined_names(name):
    p = _analysis_dir() / name
    bad = undefined_names(p.read_text())
    assert not bad, f"{name}: names used before/without definition: {bad}"


# ---------------------------------------------------------------------------
# process pools only under a __main__ guard (spawn-safe), one prefix source
# ---------------------------------------------------------------------------
def _module_level_pool_calls(tree):
    """ProcessPoolExecutor(...) calls NOT inside a function or an
    ``if __name__ == "__main__":`` block."""
    bad = []

    def guarded(node):
        return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__")

    def visit(n):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) or guarded(n):
            return
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "ProcessPoolExecutor"):
            bad.append(n.lineno)
        for c in ast.iter_child_nodes(n):
            visit(c)

    visit(tree)
    return bad


@pytest.mark.parametrize("name", ALL_FILES)
def test_process_pool_only_under_main_guard(name):
    """Under 'spawn' (macOS) each worker re-imports the script; a pool created at
    module level then recurses. Every pool must sit in a function or the guard."""
    tree = ast.parse((_analysis_dir() / name).read_text())
    assert _module_level_pool_calls(tree) == [], f"{name}: unguarded ProcessPoolExecutor"


@pytest.mark.parametrize("name", ANALYSIS_FILES)
def test_out_prefix_read_through_one_helper(name):
    """LAUE_OUT_PREFIX has one default, in frame_peaks.out_prefix(); the census,
    its null and the gates used to default to env / scan / parentbeta."""
    if name == "frame_peaks.py":
        return
    src = (_analysis_dir() / name).read_text()
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    assert not re.search(r"environ(\.get)?\s*[\(\[]\s*[\"']LAUE_OUT_PREFIX", code), \
        f"{name} reads LAUE_OUT_PREFIX itself"
