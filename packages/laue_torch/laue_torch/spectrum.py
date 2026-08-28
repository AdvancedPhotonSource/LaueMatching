"""Beam-time incident-spectrum model :math:`I_0(\\lambda)` for the
``laue_torch`` white-beam Laue forward model.

The APS 34-ID-E undulator produces a strongly harmonically-peaked spectrum,
not a flat band.  :class:`UndulatorSpectrum` holds a Gaussian-mixture-on-
harmonics parameterisation fitted from a single-crystal calibrant exposure
(see ``experiments``/the Si-calibrant spectrum-fit workflow) and evaluates it
in a fully torch-differentiable, device-agnostic way so it can be used as the
per-spot spectral weight in :mod:`laue_torch.forward` (which otherwise assumes
a flat ``per_spot_intensity``).

Artefact format (JSON), written by the calibrant fit::

    {
      "parameterization": "gaussian_mixture_on_harmonics",
      "energy_range_keV": [5.0, 30.0],
      "fit_parameters": {
        "amplitudes":   [A1, A2, ...],   # per Gaussian
        "centres_keV":  [E1, E2, ...],   # Gaussian centres (keV)
        "widths_keV":   [s1, s2, ...],   # Gaussian sigmas  (keV)
        "baseline_poly":[c0, c1, ...]    # baseline polynomial in E (keV)
      }
    }

Notes
-----
The recovered :math:`I_0` is the *effective* spectral weight in the
convention ``I_pred = I0(E) * |F|^2 * L * P * A * Omega`` used by the fit;
callers must apply the same geometric factors.  It is defined up to an overall
scale (only ratios across energy are meaningful).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

_HC_KEV_A = 12.398419739  # keV·Å  (E[keV] = _HC_KEV_A / λ[Å])

__all__ = ["UndulatorSpectrum", "load_instrument_psf_sigma"]


def load_instrument_psf_sigma(path: str | Path) -> float:
    """Return the *measured* instrument PSF sigma (pixels) from a beam-time
    calibration artefact's ``instrument_resolution`` block.

    This is the detector+geometry resolution measured from a pristine
    single-crystal standard (zero intrinsic orientation spread), and is the
    value that should be passed to the ODF forward model as a *fixed* input so
    that recovered mosaic widths reflect the material, not the instrument.  It
    is distinct from the LaueMatching ``SimulationSmoothingWidth`` (a rendering
    smoothing width used during indexing).
    """
    with open(path) as fh:
        c = json.load(fh)
    ir = c.get("instrument_resolution")
    if not ir or "psf_sigma_px" not in ir:
        raise KeyError(
            f"{path}: no 'instrument_resolution.psf_sigma_px' — run the PSF/"
            "resolution extraction on a pristine-crystal standard first."
        )
    return float(ir["psf_sigma_px"])


class UndulatorSpectrum:
    """Differentiable Gaussian-mixture-on-harmonics incident spectrum.

    Parameters
    ----------
    amplitudes, centres_keV, widths_keV:
        Per-Gaussian amplitude, centre and sigma (keV).
    baseline_poly:
        Polynomial coefficients ``[c0, c1, ...]`` for a slowly-varying floor
        ``sum_k c_k E^k`` (E in keV).  Default: constant 0.
    energy_range_keV:
        ``(Elo, Ehi)`` bandpass; energies outside are clamped to 0.
    meta:
        Optional provenance dict (beamtime id, undulator, twin scale, ...).

    All parameters are stored as buffers on ``device``/``dtype`` and moved to
    match the query tensor at call time, so the object works transparently on
    CPU / CUDA / MPS and participates in autograd.
    """

    def __init__(
        self,
        amplitudes: Sequence[float] | Tensor,
        centres_keV: Sequence[float] | Tensor,
        widths_keV: Sequence[float] | Tensor,
        baseline_poly: Sequence[float] | Tensor = (0.0,),
        energy_range_keV: tuple[float, float] = (5.0, 30.0),
        meta: dict | None = None,
    ) -> None:
        self.A = torch.as_tensor(amplitudes, dtype=torch.float64)
        self.E0 = torch.as_tensor(centres_keV, dtype=torch.float64)
        self.sigma = torch.as_tensor(widths_keV, dtype=torch.float64)
        self.baseline = torch.as_tensor(baseline_poly, dtype=torch.float64)
        if not (self.A.shape == self.E0.shape == self.sigma.shape):
            raise ValueError(
                "amplitudes, centres_keV, widths_keV must have equal length; got "
                f"{tuple(self.A.shape)}, {tuple(self.E0.shape)}, {tuple(self.sigma.shape)}"
            )
        self.Elo, self.Ehi = float(energy_range_keV[0]), float(energy_range_keV[1])
        self.meta = dict(meta or {})

    # ── construction ────────────────────────────────────────────────────────
    @classmethod
    def from_json(cls, path: str | Path) -> "UndulatorSpectrum":
        """Load an ``undulator_spectrum_*.json`` calibrant artefact."""
        with open(path) as fh:
            c = json.load(fh)
        p = c["fit_parameters"]
        return cls(
            amplitudes=p["amplitudes"],
            centres_keV=p["centres_keV"],
            widths_keV=p["widths_keV"],
            baseline_poly=p.get("baseline_poly", [0.0]),
            energy_range_keV=tuple(c.get("energy_range_keV", (5.0, 30.0))),
            meta={k: v for k, v in c.items() if k != "fit_parameters"},
        )

    def to_json(self, path: str | Path) -> Path:
        """Write this spectrum to a JSON artefact (class-compatible format)."""
        out = dict(self.meta)
        out.update(
            parameterization="gaussian_mixture_on_harmonics",
            energy_range_keV=[self.Elo, self.Ehi],
            fit_parameters=dict(
                amplitudes=self.A.tolist(),
                centres_keV=self.E0.tolist(),
                widths_keV=self.sigma.tolist(),
                baseline_poly=self.baseline.tolist(),
            ),
        )
        path = Path(path)
        path.write_text(json.dumps(out, indent=2))
        return path

    # ── evaluation ──────────────────────────────────────────────────────────
    def __call__(self, energy_keV: Tensor) -> Tensor:
        """Evaluate :math:`I_0(E)` at ``energy_keV`` (keV). Differentiable.

        Returns a tensor of the same shape/device/dtype as ``energy_keV``,
        clamped to 0 outside the bandpass and at 0-floored negatives.
        """
        E = torch.as_tensor(energy_keV)
        dev, dt = E.device, (E.dtype if E.is_floating_point() else torch.float64)
        A = self.A.to(dev, dt)
        E0 = self.E0.to(dev, dt)
        sig = self.sigma.to(dev, dt)
        base = self.baseline.to(dev, dt)
        Ef = E.to(dt)
        # Gaussian mixture: sum_k A_k exp(-(E-E0_k)^2 / (2 sigma_k^2))
        z = (Ef.unsqueeze(-1) - E0) / sig
        gauss = (A * torch.exp(-0.5 * z * z)).sum(dim=-1)
        # baseline polynomial sum_k c_k E^k
        powers = torch.stack([Ef ** k for k in range(base.shape[0])], dim=-1)
        floor = (base * powers).sum(dim=-1)
        val = gauss + floor
        in_band = (Ef >= self.Elo) & (Ef <= self.Ehi)
        return torch.where(in_band, val.clamp_min(0.0), torch.zeros_like(val))

    def at_wavelength(self, lambda_A: Tensor) -> Tensor:
        """Evaluate :math:`I_0` at wavelength λ (Å)."""
        lam = torch.as_tensor(lambda_A)
        return self(_HC_KEV_A / lam.clamp_min(1e-9))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        bt = self.meta.get("beamtime_id", "?")
        return (f"UndulatorSpectrum(beamtime={bt!r}, n_gauss={self.A.numel()}, "
                f"E=[{self.Elo},{self.Ehi}]keV)")
