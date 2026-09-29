"""Arc diagnostics for intensity a column fit leaves unexplained (reference implementation; sampleH arc_axes.py + arc_v2.py).

Validated on sampleH (PREREGISTER_arc_v2.md, PREREGISTER_arc_crowding.md; /verify PROVISIONAL, 3 of 4 lenses survive):
* arcs = components of (frame - bg) > 5 MAD, >= 15 px, intensity-weighted L >= 20 px and L/W >= 3;
* BEADS: contrast B = std(profile / 21-sample running mean) along the principal axis (0.5 px bilinear samples, +/-3 px
  perpendicular sum); beaded if B > 0.189 (controls: beaded synthetic 95%, continuous 4%); spacing = first
  autocorrelation peak, in plane-normal degrees (a LOWER bound on the lattice step);
* COMMON AXIS (pooled over all arcs): each arc's tangent t in plane-normal space must be perpendicular to the rotation
  axis u (t ~ u x n); statistic = max over a 5000-point hemisphere of the fraction of arcs within 5 deg, against a
  null that rotates each tangent randomly in its tangent plane. Detection works; LOCATION is only a band (rotation about
  the mean plane normal is unobservable) -- report the band, never a point.
* Per-frame axis recovery is NOT feasible on a 34-ID-E-like panel (fresh controls: 12.6 deg error) -- not provided.
* Chance chaining of compact spots produces no >= 20 px arcs at sampleH crowding (crowded null: 0 in 200 frames);
  re-measure that null for a new sample/detector.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))       # pipeline/analysis
from frame_peaks import SAT_LEVEL  # noqa: E402
from .fit import peak_labels  # noqa: E402

BEAD_THETA = 0.189
AXIS_TOL_DEG = 5.0


def pix2n(G, px, py):
    """pixel (x = col, y = row) -> unit plane normal (lab), energy-independent inverse of Geom.project."""
    P = G.P.numpy(); rinv = np.linalg.inv(G.roti.numpy()); ki = G.ki.numpy()
    xs = np.stack([(np.asarray(px) - 0.5 * (G.nx - 1)) * G.dx + P[0], (np.asarray(py) - 0.5 * (G.ny - 1)) * G.dy + P[1],
                   np.full(np.shape(px), P[2])], -1)
    xd = xs / np.linalg.norm(xs, axis=-1, keepdims=True)
    d = xd @ rinv.T - ki
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def extract_arcs(G, frame: np.ndarray, bg: np.ndarray, *, lmin=20.0, ar=3.0, min_px=15, sat_level=SAT_LEVEL):
    """Arcs of one frame (raw or residual) with normal-space mean/tangent/length and the bead profile statistics.
    Labels via fit.peak_labels: axis-aligned arcs >= 81 px are kept (frame_peaks' de-streak would erase them); only
    components touching a pixel >= sat_level are de-streaked (blooming)."""
    sub = frame.astype(float) - bg; sub -= np.median(sub)
    lab, _, _, _ = peak_labels(frame, sat_level)
    out = []
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        m = lab[sl] == k
        if m.sum() < min_px:
            continue
        yy, xx = np.nonzero(m); yy = yy + sl[0].start; xx = xx + sl[1].start
        w = np.clip(sub[yy, xx], 0, None); F = w.sum()
        if F <= 0:
            continue
        c = np.cov(np.vstack([xx, yy]), aweights=w); ev, evec = np.linalg.eigh(c); ev = np.clip(ev, 1e-9, None)
        L, W = 2 * math.sqrt(ev[1]), 2 * math.sqrt(ev[0])
        if not (L >= lmin and L / W >= ar):
            continue
        n = pix2n(G, xx.astype(float), yy.astype(float))
        nb = (w[:, None] * n).sum(0); nb /= np.linalg.norm(nb)
        d = n - nb; C = (w[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0) / F
        e2, v2 = np.linalg.eigh(C)
        t = v2[:, 2] - (v2[:, 2] @ nb) * nb; t /= np.linalg.norm(t)
        cx, cy = (w * xx).sum() / F, (w * yy).sum() / F; ax = evec[:, 1]; pe = np.array([-ax[1], ax[0]])
        s = (xx - cx) * ax[0] + (yy - cy) * ax[1]; ss = np.arange(s.min(), s.max() + 1e-9, 0.5)
        prof = np.zeros(len(ss))
        for o in np.arange(-3, 3.01, 1.0):
            prof += ndi.map_coordinates(sub, [cy + ss * ax[1] + o * pe[1], cx + ss * ax[0] + o * pe[0]], order=1, mode="constant")
        B = per = None
        if len(prof) >= 30:
            rm = np.convolve(prof, np.ones(21) / 21, mode="same"); good = rm > 0.1 * rm.max(); r = prof[good] / rm[good]
            if len(r) >= 20:
                B = float(np.std(r)); z = r - r.mean(); acf = np.correlate(z, z, "full")[len(z) - 1:]; acf /= max(acf[0], 1e-12)
                pk = [i for i in range(2, min(len(acf) - 1, 40)) if acf[i] > acf[i - 1] and acf[i] >= acf[i + 1] and acf[i] > 0.1]
                per = pk[0] * 0.5 if pk else None
        n1 = pix2n(G, np.array([cx, cx + ax[0]]), np.array([cy, cy + ax[1]]))
        dpx = math.degrees(math.acos(min(1.0, abs(float(n1[0] @ n1[1])))))
        out.append(dict(n=nb.tolist(), t=t.tolist(), lam_deg=math.degrees(4 * math.sqrt(max(e2[2], 0))), L=L, F=float(F), B=B,
                        beaded=None if B is None else bool(B > BEAD_THETA), period_px=per,
                        period_deg=None if per is None else per * dpx, deg_per_px=dpx))
    return out


def fib(nn=5000):
    i = np.arange(nn) + 0.5; z = i / nn; phi = math.pi * (1 + 5 ** 0.5) * i; r = np.sqrt(1 - z ** 2)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def _frac(N, T, G_):
    P = np.cross(G_[:, None, :], N[None]); pn = np.linalg.norm(P, axis=2)
    ok = (pn > 0.2) & (np.abs((P * T[None]).sum(2)) / np.clip(pn, 1e-12, None) > math.cos(math.radians(AXIS_TOL_DEG)))
    return ok.mean(1)


def common_axis_test(arcs, *, n_draws=200, seed=0, band_drop=0.02):
    """Pooled common-rotation-axis DETECTION with the axis reported as a BAND (lab frame). Needs many arcs."""
    rng = np.random.default_rng(seed)
    N = np.array([a["n"] for a in arcs]); T = np.array([a["t"] for a in arcs]); G_ = fib()
    f = _frac(N, T, G_); C2 = float(f.max())
    null = []
    for _ in range(n_draws):
        phi = rng.uniform(0, 2 * math.pi, len(N)); Tr = np.cos(phi)[:, None] * T + np.sin(phi)[:, None] * np.cross(N, T)
        null.append(float(_frac(N, Tr, G_).max()))
    band = G_[f >= C2 - band_drop]
    S = (band[:, :, None] * band[:, None, :]).sum(0); ev, evec = np.linalg.eigh(S)
    exc = C2 - float(np.percentile(null, 95))
    return dict(n_arcs=len(N), C2=C2, null_p95=float(np.percentile(null, 95)), excess=exc,
                read="DETECTED" if exc >= 0.10 else ("NOT DETECTED" if exc < 0.05 else "WEAK"),
                band_fraction_of_hemisphere=float((f >= C2 - band_drop).mean()), band_centroid=evec[:, 2].tolist(),
                band_elongation=evec[:, 1].tolist())
