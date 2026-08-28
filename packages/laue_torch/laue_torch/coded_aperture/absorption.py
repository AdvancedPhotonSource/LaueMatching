"""Linear absorption coefficients for coded-aperture materials.

Thin wrapper over ``midas_hkls.absorption.linear_absorption_coefficient``
that returns μ in **1/µm** (the natural unit for the few-µm Au + Si\\ :sub:`3`\\ N\\ :sub:`4`
mask thicknesses of the Gürsoy *et al.* coded-aperture setup) and adds a
mass-weighted compound combiner for Si\\ :sub:`3`\\ N\\ :sub:`4`.

Auto-differentiable through ``wavelength_A`` when it is a torch tensor.
"""
from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor

from midas_hkls.absorption import (
    linear_absorption_coefficient,
    mass_attenuation_coefficient,
)


_CM_PER_UM = 1.0e-4
# Si3N4 mass fractions (M_Si = 28.086, M_N = 14.007).
_M_SI = 28.086
_M_N = 14.007
_W_SI = 3.0 * _M_SI / (3.0 * _M_SI + 4.0 * _M_N)
_W_N = 4.0 * _M_N / (3.0 * _M_SI + 4.0 * _M_N)
_RHO_SI3N4_G_CM3 = 3.17


def mu_au(wavelength_A: Tensor) -> Tensor:
    """μ_Au(λ) in units of **1/µm**.

    Differentiable in ``wavelength_A``.  ``wavelength_A`` may be a scalar
    or any tensor shape; the result is broadcast to match.
    """
    mu_per_cm = linear_absorption_coefficient("Au", wavelength_A)
    return mu_per_cm * _CM_PER_UM


def mu_si3n4(wavelength_A: Tensor) -> Tensor:
    """μ_{Si₃N₄}(λ) in units of **1/µm**.

    Computed as the mass-weighted compound attenuation:

    .. math::
        \\mu/\\rho_{Si_3N_4}(\\lambda) = w_{Si}\\,\\mu/\\rho_{Si}(\\lambda)
                                       + w_N\\,\\mu/\\rho_N(\\lambda)

    multiplied by the Si\\ :sub:`3`\\ N\\ :sub:`4` density (3.17 g/cm³).

    Differentiable in ``wavelength_A``.
    """
    mac_si = mass_attenuation_coefficient("Si", wavelength_A)
    mac_n = mass_attenuation_coefficient("N", wavelength_A)
    mac_compound = _W_SI * mac_si + _W_N * mac_n
    mu_per_cm = mac_compound * _RHO_SI3N4_G_CM3
    return mu_per_cm * _CM_PER_UM
