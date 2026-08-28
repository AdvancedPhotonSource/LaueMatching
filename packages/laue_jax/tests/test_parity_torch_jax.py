"""Parity gate: laue_jax.laue_forward must match laue_torch.LaueForwardModel.

Uses the golden LaueMatching config (simulation/params_sim.txt +
valid_hkls.csv + fourOrientations.csv) so real spots land on the detector.
Images must agree to ~1e-6 (relative) across rotation/strain parameterizations.
This is the Phase-0 gate of Plan A — nothing downstream (JAX-CPFEM coupling) is
trusted until this passes.

Run (local CPU is fine for parity; GPU smoke test is separate)::

    KMP_DUPLICATE_LIB_OK=TRUE pytest laue_jax/tests/test_parity_torch_jax.py -q
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

# float64 everywhere so JAX matches torch's default double precision.
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import torch

from laue_torch import LaueForwardModel, parse_params
import laue_jax


@pytest.fixture(scope="module")
def golden(sim_dir):
    """The golden config. ``sim_dir`` (conftest) locates the repo checkout and
    skips if it or any fixture file is absent."""
    p = parse_params(str(sim_dir / "params_sim.txt"))
    t = p.to_tensors()
    hkl_data = np.genfromtxt(sim_dir / "valid_hkls.csv")
    hkls = hkl_data[:, :3].astype(np.int64)
    orients = np.genfromtxt(sim_dir / "fourOrientations.csv").reshape(-1, 3, 3)
    return t, hkls, orients


def _torch_image(t, hkls, orients, *, strain_mode="none", strain=None, psf_eta=0.0):
    model = LaueForwardModel(
        hkls=torch.tensor(hkls, dtype=torch.long),
        n_pix=t["n_pix"], px_size=t["px_size"],
        psf_sigma=t["psf_sigma"], psf_eta=psf_eta, render_window=13,
        rotation="matrix", detector_rotation="rodrigues",
        strain_mode=strain_mode, hard=False, reduce="sum",
    )
    st = None if strain is None else torch.tensor(strain, dtype=torch.float64)
    img = model(
        U=torch.tensor(orients, dtype=torch.float64),
        lattice=t["lattice"], P=t["P"], R=t["R"],
        strain=st, E_range=t["E_range"],
    )
    return img.detach().numpy()


def _jax_image(t, hkls, orients, *, strain_mode="none", strain=None, psf_eta=0.0):
    st = None if strain is None else jnp.asarray(strain)
    img = laue_jax.laue_forward(
        U=jnp.asarray(orients),
        lattice=jnp.asarray(t["lattice"].numpy()),
        P=jnp.asarray(t["P"].numpy()), R=jnp.asarray(t["R"].numpy()),
        hkls=jnp.asarray(hkls), n_pix=t["n_pix"], px_size=t["px_size"],
        strain=st, strain_mode=strain_mode,
        rotation="matrix", detector_rotation="rodrigues",
        psf_sigma=t["psf_sigma"], psf_eta=psf_eta, render_window=13,
        hard=False, reduce="sum", E_range=t["E_range"],
    )
    return np.asarray(img)


def _assert_close(a, b, tag):
    amax = max(a.max(), b.max(), 1e-12)
    diff = np.abs(a - b).max()
    rel = diff / amax
    assert rel < 1e-6, (
        f"[{tag}] max abs diff {diff:.3e} (rel {rel:.3e}); "
        f"torch sum={a.sum():.6f} jax sum={b.sum():.6f}")


def test_parity_no_strain(golden):
    t, hkls, orients = golden
    a = _torch_image(t, hkls, orients)
    b = _jax_image(t, hkls, orients)
    assert a.sum() > 0, "degenerate: no spots landed on detector"
    _assert_close(a, b, "no_strain")


def test_parity_pseudovoigt(golden):
    t, hkls, orients = golden
    a = _torch_image(t, hkls, orients, psf_eta=0.4)
    b = _jax_image(t, hkls, orients, psf_eta=0.4)
    assert a.sum() > 0
    _assert_close(a, b, "pseudo_voigt")


def test_parity_voigt_strain(golden):
    t, hkls, orients = golden
    G = orients.shape[0]
    rng = np.random.default_rng(11)
    strain = rng.normal(scale=1e-3, size=(G, 6))
    a = _torch_image(t, hkls, orients, strain_mode="voigt", strain=strain)
    b = _jax_image(t, hkls, orients, strain_mode="voigt", strain=strain)
    assert a.sum() > 0
    _assert_close(a, b, "voigt_strain")


def test_jax_grad_flows(golden):
    """Scalar image loss has a finite, nonzero gradient wrt orientation."""
    t, hkls, orients = golden
    lattice = jnp.asarray(t["lattice"].numpy())
    P = jnp.asarray(t["P"].numpy()); R = jnp.asarray(t["R"].numpy())
    hk = jnp.asarray(hkls)

    def loss(U):
        img = laue_jax.laue_forward(
            U=U, lattice=lattice, P=P, R=R, hkls=hk,
            n_pix=t["n_pix"], px_size=t["px_size"],
            rotation="matrix", detector_rotation="rodrigues",
            psf_sigma=t["psf_sigma"], render_window=13,
            hard=False, reduce="sum", E_range=t["E_range"],
        )
        return (img ** 2).sum()

    g = np.asarray(jax.grad(loss)(jnp.asarray(orients)))
    assert np.all(np.isfinite(g)), "non-finite gradient"
    assert np.abs(g).max() > 0, "gradient is identically zero"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
