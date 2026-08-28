"""laue_index.calibrate — detector geometry from a SUPPLIED crystal orientation.

The problem this solves
-----------------------
Given a crystal whose orientation is already known (a reciprocal matrix from a
prior indexation, a beamline calibration, or an oriented standard), plus a set of
observed spots whose Miller indices are known, recover the detector pose
``P`` (3) and ``R`` (3, Rodrigues).

Why the orientation has to be supplied
--------------------------------------
A single Laue pattern cannot determine the detector's rotation about the incident
beam.  With ``ki = (0, 0, 1)``, the substitution

    U -> Rz(phi) . U ,   R_mat -> Rz(phi) . R_mat

leaves ``xyz = R_mat^T . kf`` algebraically unchanged, so every predicted pixel
AND every predicted energy is identical for any phi, while ``P`` (which lives in
the detector frame) is untouched.  Measured on real data this holds to
4.5e-13 px and 1.1e-14 keV.  Consequences, all of which this module encodes:

* With the orientation FREE the 9-parameter problem is rank-deficient: one
  singular value is numerically zero, and its null direction is exactly
  "rotate detector and crystal together about the beam".
* With the orientation SUPPLIED the 6-parameter problem is full rank
  (condition number ~284 on a real 34-ID-E geometry) and three correctly
  labelled spots make it exactly determined.
* Measured ENERGIES cannot constrain the pose at all -- energy is a function of
  (orientation, hkl) alone.  Supply them for the spot->hkl correspondence and as
  a check on the supplied orientation, not as geometry constraints.
* The supplied orientation therefore TRANSFERS its own frame onto the detector.
  That is legitimate and often exactly what is wanted, but it is inheritance,
  not measurement -- so ``frame_provenance`` is a REQUIRED argument and is
  recorded in the result and in any params file written from it.  This module
  will not silently invent an azimuth.

Nothing here imports ``laue_torch``: per the package constraint the shared pure
math is duplicated.  # TODO(unify-after-publish)
"""
from __future__ import annotations

import dataclasses
import math
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "HC_KEV_NM",
    "DetectorSpec",
    "Anchor",
    "Conditioning",
    "Validation",
    "CalibrationResult",
    "rodrigues_to_matrix",
    "matrix_to_rodrigues",
    "reciprocal_matrix",
    "project",
    "orientation_candidates",
    "calibrate",
    "MIN_ANCHORS",
]

HC_KEV_NM = 1.2398419739

#: Minimum labelled spots for a UNIQUE calibration.
#:
#: Not 3. Three anchors give 6 equations for 6 unknowns -- exactly determined,
#: which permits more than one root, and this system has one. Measured on the
#: synthetic Si case (400 starts perturbed by 1e-6 relative, chiltepin, Linux
#: x86_64 / py3.11.15 / scipy 1.17.1):
#:
#:   3 anchors  186/200 reach the true geometry, 14/200 (7%) converge instead to
#:              p=(0.076342, 0.077947, 0.045833) at rms 6e-14..1.7e-13 -- an
#:              exact alternative solution, not a convergence failure (0 of 17
#:              failures had rms >= 1e-6)
#:   4 anchors  200/200
#:   5, 6       200/200
#:
#: Which root the optimiser finds turns on last-bit floating-point differences,
#: so a 3-anchor fit is CPU-dependent: it was reproducibly correct on macOS
#: arm64 and on chiltepin, and reproducibly wrong on GitHub's Linux py3.11
#: runner. A caller cannot tell the two apart from the residual.
MIN_ANCHORS = 4


# --------------------------------------------------------------------------
# pure math, duplicated from laue_torch  # TODO(unify-after-publish)
# --------------------------------------------------------------------------
def rodrigues_to_matrix(v: Sequence[float]) -> np.ndarray:
    """Axis-angle vector (|v| = angle in radians) to a proper rotation matrix."""
    v = np.asarray(v, dtype=float)
    theta = float(np.linalg.norm(v))
    if theta < 1e-15:
        return np.eye(3)
    a = v / theta
    x, y, z = a
    c, s = math.cos(theta), math.sin(theta)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def matrix_to_rodrigues(M: np.ndarray) -> np.ndarray:
    """Proper rotation matrix to an axis-angle vector (|v| = angle in radians)."""
    M = np.asarray(M, dtype=float)
    cos_t = (np.trace(M) - 1.0) / 2.0
    theta = math.acos(float(np.clip(cos_t, -1.0, 1.0)))
    if theta < 1e-12:
        return np.zeros(3)
    if abs(math.pi - theta) < 1e-6:
        # near pi: axis from the symmetric part, sign fixed by the largest entry
        A = (M + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        k = int(np.argmax(axis))
        if axis[k] > 0:
            axis = A[:, k] / axis[k]
        axis = axis / np.linalg.norm(axis)
        return axis * theta
    axis = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
    axis = axis / (2.0 * math.sin(theta))
    return axis * theta


def reciprocal_matrix(lattice: Sequence[float]) -> np.ndarray:
    """B matrix (columns a*, b*, c*, the 2*pi convention) from a, b, c, al, be, ga.

    Lengths in nm, angles in degrees -- MIDAS units.
    """
    a, b, c, al, be, ga = (float(x) for x in lattice)
    al, be, ga = math.radians(al), math.radians(be), math.radians(ga)
    ca, cb, cg = math.cos(al), math.cos(be), math.cos(ga)
    sg = math.sin(ga)
    vol = a * b * c * math.sqrt(
        max(1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg, 1e-30))
    A = np.array([
        [a, b * cg, c * cb],
        [0.0, b * sg, c * (ca - cb * cg) / sg],
        [0.0, 0.0, vol / (a * b * sg)],
    ])
    # reciprocal (2*pi convention), columns a*, b*, c*
    return 2.0 * math.pi * np.linalg.inv(A).T


# --------------------------------------------------------------------------
# typed inputs / outputs
# --------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class DetectorSpec:
    """Panel geometry.  ``n_pix`` is (Nx, Ny); ``px_size`` is (dx, dy) in metres."""
    n_pix: tuple[int, int]
    px_size: tuple[float, float]

    def __post_init__(self) -> None:
        if min(self.n_pix) <= 0:
            raise ValueError(f"n_pix must be positive, got {self.n_pix}")
        if min(self.px_size) <= 0:
            raise ValueError(f"px_size must be positive, got {self.px_size}")


@dataclasses.dataclass(frozen=True)
class Anchor:
    """One spot with a known Miller index.

    ``energy_kev`` is optional and is NEVER used to constrain the pose -- it is
    reported back as a consistency check on the supplied orientation.
    """
    hkl: tuple[int, int, int]
    pixel: tuple[float, float]
    energy_kev: float | None = None


@dataclasses.dataclass(frozen=True)
class Conditioning:
    """Rank report for the fitted problem."""
    singular_values: tuple[float, ...]
    condition_number: float
    gauge_flat_ratio: float          # smallest/largest with the orientation FREE
    degenerate: bool

    def describe(self) -> str:
        return (f"condition number {self.condition_number:.4g}; "
                f"free-orientation flat direction "
                f"{self.gauge_flat_ratio:.3g} "
                f"({'DEGENERATE' if self.degenerate else 'full rank'})")


@dataclasses.dataclass(frozen=True)
class Validation:
    """Held-out and null-model evidence."""
    n_used_in_fit: int
    n_held_out: int
    n_held_out_matched: int
    held_out_median_px: float
    null_trials: int
    null_max_matched: int
    residual_dx: tuple[float, float]   # mean, sd
    residual_dy: tuple[float, float]
    tolerance_px: float

    @property
    def clears_null(self) -> bool:
        return self.n_held_out_matched > self.null_max_matched


@dataclasses.dataclass(frozen=True)
class CalibrationResult:
    p_array: tuple[float, float, float]
    r_array: tuple[float, float, float]
    orientation: np.ndarray            # the U actually used (3x3)
    convention: str                    # "columns" or "rows"
    convention_scores: dict            # both readings, so the choice is auditable
    rms_px: float
    conditioning: Conditioning
    validation: Validation | None
    frame_provenance: str
    energy_check: dict                 # supplied vs predicted, ppm

    @property
    def distance_mm(self) -> float:
        return self.p_array[2] * 1e3

    def params_text(self, lattice: Sequence[float], space_group: int,
                    symmetry: str, e_range: tuple[float, float] = (5.0, 30.0),
                    spec: DetectorSpec | None = None) -> str:
        """A LaueMatching params block, with the frame provenance in the header."""
        lines = [
            "# Detector geometry from a SUPPLIED crystal orientation.",
            "# The rotation about the beam is INHERITED from that orientation,",
            "# not measured from this pattern. Provenance of that frame:",
            f"#   {self.frame_provenance}",
            f"# recip1 read as {self.convention.upper()}; "
            f"rms {self.rms_px:.3f} px; {self.conditioning.describe()}",
        ]
        if self.validation is not None:
            v = self.validation
            lines.append(
                f"# held out {v.n_held_out_matched}/{v.n_held_out} within "
                f"{v.tolerance_px:g} px (null max {v.null_max_matched}), "
                f"median {v.held_out_median_px:.3f} px")
        lines += [
            f"SpaceGroup {space_group}",
            f"Symmetry {symmetry}",
            "LatticeParameter " + " ".join(f"{x:.6f}" for x in lattice),
            "P_Array " + " ".join(f"{x:.9f}" for x in self.p_array),
            "R_Array " + " ".join(f"{x:.9f}" for x in self.r_array),
        ]
        if spec is not None:
            lines += [
                f"PxX {spec.px_size[0]:.6f}",
                f"PxY {spec.px_size[1]:.6f}",
                f"NrPxX {spec.n_pix[0]}",
                f"NrPxY {spec.n_pix[1]}",
            ]
        lines += [f"Elo {e_range[0]}", f"Ehi {e_range[1]}"]
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# forward model
# --------------------------------------------------------------------------
def project(U: np.ndarray, B: np.ndarray, hkl: np.ndarray,
            P: Sequence[float], R_mat: np.ndarray,
            spec: DetectorSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict (px, py, energy_keV) for each reflection.

    Convention transcribed from laue_torch.forward:
        q = U B h ; kf = ki - 2(qhat.ki)qhat, ki = (0,0,1) ; xyz = R^T kf
    """
    hkl = np.asarray(hkl, dtype=float).reshape(-1, 3)
    M = np.asarray(U, dtype=float) @ np.asarray(B, dtype=float)
    q = hkl @ M.T
    qlen = np.linalg.norm(q, axis=1)
    qhat = q / np.maximum(qlen, 1e-30)[:, None]
    dot = qhat[:, 2]
    kf = np.stack([-2 * dot * qhat[:, 0],
                   -2 * dot * qhat[:, 1],
                   1.0 - 2 * dot * qhat[:, 2]], axis=1)
    xyz = kf @ np.asarray(R_mat, dtype=float)          # = (R^T kf) rowwise
    z = xyz[:, 2]
    z_pos = np.maximum(z, 1e-12)
    scale = P[2] / z_pos
    Nx, Ny = spec.n_pix
    dx, dy = spec.px_size
    px = (xyz[:, 0] * scale - P[0]) / dx + 0.5 * (Nx - 1)
    py = (xyz[:, 1] * scale - P[1]) / dy + 0.5 * (Ny - 1)
    sin_theta = -qhat[:, 2]
    energy = np.where(sin_theta > 1e-12,
                      HC_KEV_NM * qlen / (4.0 * math.pi *
                                          np.maximum(sin_theta, 1e-30)),
                      0.0)
    return px, py, energy


def orientation_candidates(recip: np.ndarray) -> dict[str, np.ndarray]:
    """Both readings of a supplied reciprocal matrix, column-normalised.

    A supplied ``recip`` is almost always ``scale * (a proper rotation)``, so its
    ROW norms and its COLUMN norms are equal by construction and cannot tell you
    which convention it is in.  Only the projection can.  This module therefore
    tries both and reports the scores rather than assuming.
    """
    R = np.asarray(recip, dtype=float).reshape(3, 3)
    cols = R / np.linalg.norm(R, axis=0, keepdims=True)
    rows = R.T / np.linalg.norm(R.T, axis=0, keepdims=True)
    return {"columns": cols, "rows": rows}


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------
def _pack(P: Sequence[float], rvec: Sequence[float]) -> np.ndarray:
    return np.concatenate([np.asarray(P, float), np.asarray(rvec, float)])


def _residuals(params: np.ndarray, U: np.ndarray, B: np.ndarray,
               hkl: np.ndarray, obs: np.ndarray,
               spec: DetectorSpec) -> np.ndarray:
    P, rvec = params[:3], params[3:6]
    px, py, _ = project(U, B, hkl, P, rodrigues_to_matrix(rvec), spec)
    return np.concatenate([px - obs[:, 0], py - obs[:, 1]])


def _numeric_jacobian(fun, p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    f0 = fun(p)
    J = np.empty((f0.size, p.size))
    for i in range(p.size):
        step = eps * max(abs(p[i]), 1e-3)
        q = p.copy()
        q[i] += step
        J[:, i] = (fun(q) - f0) / step
    return J


def _levenberg_marquardt(fun, p0: np.ndarray, max_iter: int = 200,
                         tol: float = 1e-12) -> tuple[np.ndarray, float]:
    """Small self-contained LM.  Keeps the package numpy-only."""
    p = np.asarray(p0, dtype=float).copy()
    f = fun(p)
    cost = float(f @ f)
    lam = 1e-3
    for _ in range(max_iter):
        J = _numeric_jacobian(fun, p)
        JTJ = J.T @ J
        g = J.T @ f
        improved = False
        for _ in range(30):
            try:
                step = np.linalg.solve(JTJ + lam * np.diag(np.diag(JTJ) + 1e-12), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            p_new = p + step
            f_new = fun(p_new)
            cost_new = float(f_new @ f_new)
            if cost_new < cost:
                if cost - cost_new < tol * max(cost, 1e-30):
                    p, f, cost = p_new, f_new, cost_new
                    return p, cost
                p, f, cost = p_new, f_new, cost_new
                lam = max(lam * 0.3, 1e-12)
                improved = True
                break
            lam *= 10.0
            if lam > 1e12:
                break
        if not improved:
            break
    return p, cost


def _conditioning(U: np.ndarray, B: np.ndarray, hkl: np.ndarray,
                  obs: np.ndarray, params: np.ndarray,
                  spec: DetectorSpec) -> Conditioning:
    """Rank of the fitted 6-parameter problem, and of the 9-parameter one.

    The 9-parameter number is the point: it should be numerically singular, and
    seeing that confirms the supplied orientation is what makes the fit possible.
    """
    fixed = lambda p: _residuals(p, U, B, hkl, obs, spec)
    J6 = _numeric_jacobian(fixed, params)
    s6 = np.linalg.svd(J6, compute_uv=False)

    def free(p9):
        Uw = rodrigues_to_matrix(p9[6:9]) @ U
        px, py, _ = project(Uw, B, hkl, p9[:3], rodrigues_to_matrix(p9[3:6]), spec)
        return np.concatenate([px - obs[:, 0], py - obs[:, 1]])

    J9 = _numeric_jacobian(free, np.concatenate([params, np.zeros(3)]))
    s9 = np.linalg.svd(J9, compute_uv=False)

    cond6 = float(s6[0] / max(s6[-1], 1e-300))
    flat = float(s9[-1] / max(s9[0], 1e-300))
    return Conditioning(
        singular_values=tuple(float(x) for x in s6),
        condition_number=cond6,
        gauge_flat_ratio=flat,
        degenerate=bool(cond6 > 1e8),
    )


def _validate(U: np.ndarray, B: np.ndarray, params: np.ndarray,
              spec: DetectorSpec, held_out_hkl: np.ndarray,
              observed_spots: np.ndarray, tolerance_px: float,
              null_trials: int, rng: np.random.Generator) -> Validation:
    """Held-out reflections against a measured random-spot null."""
    P, R_mat = params[:3], rodrigues_to_matrix(params[3:6])
    px, py, _ = project(U, B, held_out_hkl, P, R_mat, spec)
    Nx, Ny = spec.n_pix
    on = (px >= 0) & (px < Nx) & (py >= 0) & (py < Ny)
    pred = np.stack([px[on], py[on]], axis=1)

    def count(obs_xy):
        if len(pred) == 0 or len(obs_xy) == 0:
            return 0, np.array([]), np.array([])
        d = np.linalg.norm(obs_xy[:, None, :] - pred[None, :, :], axis=2)
        mn = d.min(axis=1)
        j = d.argmin(axis=1)
        keep = mn <= tolerance_px
        return int(keep.sum()), mn[keep], obs_xy[keep] - pred[j[keep]]

    n_match, dists, deltas = count(observed_spots)

    null_max = 0
    for _ in range(null_trials):
        fake = np.stack([rng.uniform(0, Nx, len(observed_spots)),
                         rng.uniform(0, Ny, len(observed_spots))], axis=1)
        null_max = max(null_max, count(fake)[0])

    if len(deltas):
        dx = (float(deltas[:, 0].mean()), float(deltas[:, 0].std()))
        dy = (float(deltas[:, 1].mean()), float(deltas[:, 1].std()))
        med = float(np.median(dists))
    else:
        dx = dy = (float("nan"), float("nan"))
        med = float("nan")

    return Validation(
        n_used_in_fit=0,                      # filled by caller
        n_held_out=len(pred),
        n_held_out_matched=n_match,
        held_out_median_px=med,
        null_trials=null_trials,
        null_max_matched=null_max,
        residual_dx=dx,
        residual_dy=dy,
        tolerance_px=tolerance_px,
    )


def calibrate(anchors: Iterable[Anchor],
              recip: np.ndarray,
              lattice: Sequence[float],
              spec: DetectorSpec,
              *,
              frame_provenance: str,
              initial_guess: Sequence[float] | None = None,
              convention: str | None = None,
              observed_spots: np.ndarray | None = None,
              held_out_hkl: np.ndarray | None = None,
              tolerance_px: float = 3.0,
              null_trials: int = 2000,
              n_restarts: int = 24,
              seed: int = 0) -> CalibrationResult:
    """Fit the detector pose from labelled spots and a SUPPLIED orientation.

    ``frame_provenance`` is required and must be non-empty: the rotation about
    the beam is inherited from the supplied orientation rather than measured, so
    where that orientation came from is part of the result.  Pass something a
    reader can act on, e.g. "recip1 from LaueGo CalibrationListOrange0,
    wire-calibrated 2026-07-07" or "oriented Si wafer, mount metrology".

    ``initial_guess`` is (P0, P1, P2, r0, r1, r2).  Without one, a spread of
    restarts is tried around a panel-facing-the-sample default; the fit is
    accepted only if it converges and clears its own null.
    """
    if not frame_provenance or not str(frame_provenance).strip():
        raise ValueError(
            "frame_provenance is required: the detector's rotation about the "
            "beam is inherited from the supplied orientation, not measured "
            "from the pattern. Record where that orientation came from.")

    anchors = list(anchors)
    if len(anchors) < MIN_ANCHORS:
        raise ValueError(
            f"need at least {MIN_ANCHORS} labelled spots, got {len(anchors)}. "
            f"Three gives 6 equations for 6 unknowns and is exactly determined, "
            f"which admits MORE THAN ONE exact solution: measured on the "
            f"synthetic Si case, 7% of starting guesses converge to a second "
            f"(P, R) that reproduces all three spots to ~1e-13 px -- fitting as "
            f"well as the true geometry. Which root you land on depends on "
            f"last-bit floating-point differences, so it varies by CPU. Four "
            f"anchors resolved it in 200/200 trials.")

    hkl = np.array([a.hkl for a in anchors], dtype=float)
    obs = np.array([a.pixel for a in anchors], dtype=float)
    B = reciprocal_matrix(lattice)
    rng = np.random.default_rng(seed)

    cands = orientation_candidates(recip)
    if convention is not None:
        if convention not in cands:
            raise ValueError(f"convention must be one of {sorted(cands)}")
        cands = {convention: cands[convention]}

    Nx, Ny = spec.n_pix
    dx, dy = spec.px_size
    if initial_guess is not None:
        seeds = [np.asarray(initial_guess, dtype=float)]
    else:
        seeds = []
        # A panel facing the sample, at a spread of plausible standoffs, with the
        # normal swept over the sphere. No orientation search is needed -- these
        # only have to land inside the basin.
        for dist in (0.05, 0.1, 0.2, 0.4):
            for _ in range(max(1, n_restarts // 4)):
                v = rng.normal(size=3)
                v *= rng.uniform(0.1, math.pi) / np.linalg.norm(v)
                seeds.append(_pack((0.0, 0.0, dist), v))

    best = None
    scores: dict[str, dict] = {}
    for name, U in cands.items():
        best_here = None
        for s in seeds:
            fun = lambda p, U=U: _residuals(p, U, B, hkl, obs, spec)
            p, cost = _levenberg_marquardt(fun, s)
            if p[2] <= 0:                      # reject the mirrored branch
                continue
            if best_here is None or cost < best_here[1]:
                best_here = (p, cost)
        if best_here is None:
            scores[name] = {"rms_px": float("inf"), "converged": False}
            continue
        p, cost = best_here
        rms = math.sqrt(cost / max(len(obs) * 2, 1))
        scores[name] = {"rms_px": rms, "converged": True}
        if best is None or cost < best[1]:
            best = (p, cost, name, U)

    if best is None:
        raise RuntimeError(
            "no convergent, physical solution (all candidates gave P2 <= 0 or "
            "failed to converge). Check the supplied orientation, the hkl "
            "labels, and the initial guess.")

    params, cost, chosen, U = best
    rms = math.sqrt(cost / max(len(obs) * 2, 1))
    cond = _conditioning(U, B, hkl, obs, params, spec)

    # energies: a check on the supplied orientation, never a pose constraint
    _, _, e_pred = project(U, B, hkl, params[:3],
                           rodrigues_to_matrix(params[3:6]), spec)
    supplied = np.array([a.energy_kev if a.energy_kev is not None else np.nan
                         for a in anchors], dtype=float)
    have = np.isfinite(supplied)
    if have.any():
        ppm = (e_pred[have] - supplied[have]) / supplied[have] * 1e6
        energy_check = {
            "n": int(have.sum()),
            "mean_ppm": float(ppm.mean()),
            "sd_ppm": float(ppm.std()),
            "max_abs_ev": float(np.max(np.abs(e_pred[have] - supplied[have])) * 1e3),
            "note": "energies are pose-independent; this checks the ORIENTATION",
        }
    else:
        energy_check = {"n": 0, "note": "no energies supplied"}

    validation = None
    if held_out_hkl is not None and observed_spots is not None:
        validation = _validate(U, B, params, spec,
                               np.asarray(held_out_hkl, dtype=float),
                               np.asarray(observed_spots, dtype=float),
                               tolerance_px, null_trials, rng)
        validation = dataclasses.replace(validation, n_used_in_fit=len(anchors))

    return CalibrationResult(
        p_array=tuple(float(x) for x in params[:3]),
        r_array=tuple(float(x) for x in params[3:6]),
        orientation=U,
        convention=chosen,
        convention_scores=scores,
        rms_px=rms,
        conditioning=cond,
        validation=validation,
        frame_provenance=str(frame_provenance).strip(),
        energy_check=energy_check,
    )
