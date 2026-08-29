"""Spot intensities come from |F(hkl)|^2, not a random draw.

The simulator used to give every spot ``np.random.randint(500, 16000)`` -- an
intensity unrelated to the crystal. That is not merely unphysical: it is a
systematic that any forward model must fail to reproduce, so it puts a floor
under every fit made against these images (measured: 0.0050 deg of orientation
error, against 0.00015 for an image whose intensities the model can represent).

Structure factors come from ``midas_hkls`` -- the form-factor tables, the
Debye-Waller term and the symmetry expansion all live there and are not
reimplemented here. These tests check the WIRING: that the right |F|^2 reaches
the right spot, which is the part this file can get wrong.

The factorisation is I = |F|^2 * I0(E), matching ``laue_torch.forward``'s
``per_spot_intensity * spectrum(E)``. Lorentz, polarisation, absorption,
detector response and extinction are deliberately absent: a simulator that
applies a factor the model cannot represent manufactures a mismatch.
"""
from __future__ import annotations

import numpy as np
import pytest

from laue_index.pipeline import add_to_path

add_to_path()
import GenerateSimulation as GS  # noqa: E402

midas_hkls = pytest.importorskip("midas_hkls", reason="structure factors need midas-hkls")
from midas_hkls.crystal import Atom  # noqa: E402

# FCC nickel: one atom at the origin, F m -3 m.
_FCC = [Atom("Ni", (0.0, 0.0, 0.0))]
_HKLS = np.array([[1, 1, 1], [2, 0, 0], [1, 0, 0], [2, 1, 0],
                  [2, 2, 0], [3, 1, 1], [4, 0, 0]], dtype=np.int64)
_MIXED = np.array([len({h % 2, k % 2, l % 2}) > 1 for h, k, l in _HKLS])


def _sim(model="structure", atoms=_FCC):
    params = {
        'sgNum': 225,
        'latC': "0.35238 0.35238 0.35238 90 90 90",   # nm, as ConfigParser stores it
        'gaussWidth': 2,
    }
    s = GS.DiffractionSimulator.__new__(GS.DiffractionSimulator)
    s.params = params
    s.intensity_model = model
    s.phase_atoms = atoms
    s.spectrum = None
    s.PEAK_TARGET = 16000.0
    s._crystal_t = None
    s._flat_warned = False
    return s


def test_forbidden_reflections_get_zero_intensity():
    """Mixed-parity hkl are absent in FCC. A spot there is a wrong crystal."""
    I = _sim().spot_intensities(_HKLS, np.full(len(_HKLS), 12.0))
    assert np.all(I[_MIXED] < 1e-9), f"forbidden reflections lit: {I[_MIXED]}"
    assert np.all(I[~_MIXED] > 1e-9)


def test_intensities_are_proportional_to_midas_hkls_F_squared():
    """Catches a permutation: hkl is threaded through four filter masks, and
    mismatching it against the spot list would put every intensity on the
    wrong reflection while still looking plausible."""
    from midas_hkls.crystal import Crystal, Lattice
    from midas_hkls.space_group import SpaceGroup
    from midas_hkls.structure_factor import (structure_factor_intensity,
                                             structure_factors)

    I = _sim().spot_intensities(_HKLS, np.full(len(_HKLS), 12.0))
    lat = Lattice(3.5238, 3.5238, 3.5238, 90.0, 90.0, 90.0)
    crystal = Crystal(lattice=lat, space_group=SpaceGroup.from_number(225), atoms=_FCC)
    ref = structure_factor_intensity(structure_factors(crystal.to_torch(), _HKLS)).numpy()

    ok = ref > 1e-9
    ratio = I[ok] / ref[ok]
    assert np.allclose(ratio, ratio[0], rtol=1e-9), (
        f"intensities not proportional to |F|^2 -- ordering broken: {ratio}")


def test_lattice_parameters_are_converted_nm_to_angstrom():
    """The config is in nm and midas_hkls is in Angstrom. A factor of 10 in the
    cell rescales s = sin(theta)/lambda and hence every form factor, tilting
    the whole pattern without ever looking obviously wrong."""
    got = _sim()._crystal_tensor()
    a = float(np.asarray(got.lattice_params)[0])
    assert 3.4 < a < 3.6, f"lattice a = {a}; expected ~3.52 A, not 0.352 or 35.2"


def test_a_spectrum_multiplies_each_spot_by_I0_at_its_own_energy():
    class _Linear:
        def __call__(self, E):
            import torch
            return torch.as_tensor(np.asarray(E, dtype=float) / 10.0)

    s = _sim()
    flat_E = np.full(len(_HKLS), 10.0)          # I0 == 1 everywhere
    base = s.spot_intensities(_HKLS, flat_E)
    s.spectrum = _Linear()
    varied = np.array([6.0, 12.0, 18.0, 24.0, 6.0, 12.0, 18.0])
    got = s.spot_intensities(_HKLS, varied)

    ok = base > 1e-9
    pred = base[ok] * (varied[ok] / 10.0)
    assert np.allclose(got[ok] / got[ok].max(), pred / pred.max(), atol=1e-12)


def test_no_phase_basis_falls_back_to_equal_intensities():
    """`auto` must not require a phase: every existing config lacks one, and
    silently switching them to structure factors would change their images."""
    I = _sim(model="auto", atoms=[]).spot_intensities(_HKLS, np.full(len(_HKLS), 12.0))
    assert np.allclose(I, I[0])
    assert I[0] > 0


def test_random_model_is_still_available_for_reproducing_old_images():
    I = _sim(model="random", atoms=[]).spot_intensities(_HKLS, np.full(len(_HKLS), 12.0))
    assert I.shape == (len(_HKLS),)
    assert np.all((I >= 500) & (I < 16000))
