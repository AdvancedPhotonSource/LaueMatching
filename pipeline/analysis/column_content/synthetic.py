"""Synthetic Laue COLUMNS of known orientation content (reference implementation; sampleH make_columns_wide.py generalised).

Per frame: N crystals. Orientation 40% random, 40% on a fibre (optional axis), 20% a NEAR partner of an earlier crystal
(Delta in {0.05, 0.1, 0.3, 1, 3} deg). Brightness log-uniform over 1.5 decades; per-reflection brightness x log-normal(0,
1.5), shared by a crystal's sub-orientations. Spread classes (default mix = the sampleH wide validation): point, narrow cloud
(sigma 0.02/0.05 deg), 1-D streak (half-width 0.1-0.8 deg), wide 3-D cloud (sigma 0.2 deg). Spots are the MEASURED
calibrant kernel scaled by s_true ~ U[1.00, 1.10], stamped from laue_torch.LaueForwardModel(return_aux=True) positions,
which a guard checks against Geom.project (< 1e-3 px) before anything is written.
Truth per frame (/entry1/truth): centres, share (flux / total), brightness, sigma_t, spread_class, spread_param, kind,
partner, delta_deg, kernel_scale, peak_adu, comps_<k>.
"""
from __future__ import annotations

import math
import os

import h5py
import numpy as np
import torch

DEFAULT_MIX = (("point", 0.30, (0.0,)), ("narrow", 0.20, (0.02, 0.05)), ("streak", 0.35, (0.1, 0.2, 0.4, 0.8)),
               ("wide3d", 0.15, (0.2,)))


def _expmap(w):
    th = float(np.linalg.norm(w))
    if th < 1e-15:
        return np.eye(3)
    k = w / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def render_columns(outdir: str, n_crystals: int, n_frames: int, seed: int, *, params_path: str, phase: str,
                   bg_path: str, kernel_npz: str, fibre_axis_lab=None, mix=DEFAULT_MIX, peak_range=(6000.0, 1e5),
                   sat=65535, device=None, prefix=None):
    from laue_torch import LaueForwardModel, generate_hkls, parse_params
    from scipy.ndimage import map_coordinates
    from .geom import Geom
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(outdir, exist_ok=True)
    p = parse_params(params_path)
    lat = torch.tensor(np.asarray(p.lattice, float), dtype=torch.float64, device=device)
    hkls = generate_hkls(p.sg_num, np.asarray(p.lattice, float), p.E_hi); NH = len(hkls)
    t = p.to_tensors(dtype=torch.float64)
    model = LaueForwardModel(hkls=hkls, n_pix=t["n_pix"], px_size=t["px_size"], psf_sigma=torch.tensor(1.45, dtype=torch.float64),
                             rotation="matrix", detector_rotation="rodrigues", strain_mode="none", hard=True).to(device)
    PP, RR = t["P"].to(device), t["R"].to(device)
    G = Geom(params_path, phase)
    ny, nx = G.ny, G.nx
    bg = np.fromfile(bg_path, dtype=np.float64).reshape(ny, nx)
    ker = np.load(kernel_npz); KF, KHW, KHF = ker["Kf"], float(ker["hw"]), float(ker["hf"])
    KH = 8; PH = (np.arange(10) + 0.5) / 10.0; SUBO = np.array([-1.0, 0.0, 1.0]) / 3.0
    rng = np.random.default_rng(seed)

    def lut_for(scale):
        k = np.arange(-KH, KH + 1, dtype=float); lut = np.zeros((10, 10, len(k), len(k)))
        for by, fy in enumerate(PH):
            for bx, fx in enumerate(PH):
                acc = np.zeros((len(k), len(k)))
                for oy in SUBO:
                    for ox in SUBO:
                        gx = ((k[None, :] + ox - fx) / scale + KHW) / KHF; gy = ((k[:, None] + oy - fy) / scale + KHW) / KHF
                        acc += map_coordinates(KF, [np.broadcast_to(gy, acc.shape), np.broadcast_to(gx, acc.shape)], order=1, mode="constant")
                lut[by, bx] = acc / 9.0
        return lut / lut.sum(axis=(2, 3), keepdims=True)

    def rrot():
        M = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        if np.linalg.det(M) < 0:
            M[:, 0] *= -1
        return M

    def fibre_rot():
        e1 = np.asarray(fibre_axis_lab, float); e1 /= np.linalg.norm(e1)
        a = rng.normal(size=3); a -= (a @ e1) * e1; e2 = a / np.linalg.norm(a); e3 = np.cross(e1, e2)
        return _expmap(e1 * rng.uniform(0, 2 * math.pi)) @ np.stack([e1, e2, e3], 1)

    def spots(M):
        with torch.no_grad():
            _, aux = model(torch.tensor(M, dtype=torch.float64, device=device)[None], lat, PP, RR, E_range=t["E_range"], return_aux=True)
        keep = aux.intensity > 0
        return aux.px[keep].cpu().numpy(), aux.py[keep].cpu().numpy(), aux.intensity[keep].cpu().numpy(), aux.hkl_idx[keep].cpu().numpy()

    # guard: laue_torch positions == Geom.project positions
    M = rrot(); px, py, _, hi = spots(M)
    gx, gy, _, ok = G.project(torch.as_tensor(M, dtype=torch.float64)[None], torch.as_tensor(np.asarray(hkls)[hi], dtype=torch.float64))
    d = np.hypot(gx[0].numpy() - px, gy[0].numpy() - py)[ok[0].numpy()]
    if not (len(d) > 10 and float(np.max(d)) < 1e-3):
        raise SystemExit(f"GUARD FAILED: laue_torch vs Geom.project {np.max(d) if len(d) else None} px on {len(d)} spots")

    classes = [m[0] for m in mix]; probs = np.cumsum([m[1] for m in mix]); probs /= probs[-1]
    for i in range(1, n_frames + 1):
        s_true = float(rng.uniform(1.00, 1.10)); lut = lut_for(s_true); crystals = []
        for c in range(n_crystals):
            u = rng.uniform()
            if c > 0 and u < 0.2:
                j = int(rng.integers(0, c)); dl = float(rng.choice([0.05, 0.1, 0.3, 1.0, 3.0]))
                ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
                Mc, kind, partner, delta = _expmap(ax * math.radians(dl)) @ crystals[j]["M"], "near", j, dl
            elif u < 0.6 and fibre_axis_lab is not None:
                Mc, kind, partner, delta = fibre_rot(), "fibre", -1, 0.0
            else:
                Mc, kind, partner, delta = rrot(), "random", -1, 0.0
            ci = int(np.searchsorted(probs, rng.uniform())); cls, pars = mix[ci][0], mix[ci][2]
            par = float(rng.choice(pars))
            if cls == "point":
                comps = [Mc]
            elif cls == "streak":
                ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
                comps = [_expmap(ax * math.radians(a)) @ Mc for a in np.linspace(-par, par, 41)]
            else:
                comps = [_expmap(w) @ Mc for w in rng.normal(0, math.radians(par), size=(60 if cls == "wide3d" else 30, 3))]
            crystals.append(dict(M=Mc, comps=comps, kind=kind, partner=partner, delta=delta, cls=cls, par=par,
                                 sigma_t=par if cls in ("narrow", "wide3d") else 0.0,
                                 B=float(np.exp(rng.uniform(0, math.log(31.6)))), fac=np.exp(rng.normal(0, 1.5, size=NH))))
        img = np.zeros((ny, nx)); flux = []
        for cr in crystals:
            one = np.zeros((ny, nx))
            for C in cr["comps"]:
                sx, sy, inten, hi = spots(C)
                for x, y, a, h in zip(sx, sy, inten, hi):
                    x0, y0 = int(math.floor(x)), int(math.floor(y)); bx, by = min(int((x - x0) * 10), 9), min(int((y - y0) * 10), 9)
                    ya, yb, xa, xb = y0 - KH, y0 + KH + 1, x0 - KH, x0 + KH + 1
                    if ya < 0 or xa < 0 or yb > ny or xb > nx:
                        continue
                    one[ya:yb, xa:xb] += a * cr["fac"][h] * lut[by, bx]
            one *= cr["B"] / len(cr["comps"]); flux.append(float(one.sum())); img += one
        peak = float(np.exp(rng.uniform(np.log(peak_range[0]), np.log(peak_range[1]))))
        frame = np.clip(rng.poisson(np.clip(bg + img * peak / max(img.max(), 1e-300), 0, None)), 0, sat).astype(np.uint16)
        tot = sum(flux)
        with h5py.File(f"{outdir}/{prefix or f'C{n_crystals}'}_{i:06d}.h5", "w") as h:
            h.create_dataset("/entry1/data/data", data=frame)
            for k in ("sampleX", "sampleZ", "sampleY"):
                h.create_dataset(f"/entry1/sample/{k}", data=np.array([0.0]))
            g = h.create_group("/entry1/truth")
            g.create_dataset("centres", data=np.stack([c["M"] for c in crystals])); g.create_dataset("share", data=np.array(flux) / tot)
            for key in ("B", "sigma_t", "partner", "delta", "par"):
                g.create_dataset({"B": "brightness", "delta": "delta_deg", "par": "spread_param"}.get(key, key), data=np.array([c[key] for c in crystals]))
            g.create_dataset("kind", data=np.array([c["kind"].encode() for c in crystals]))
            g.create_dataset("spread_class", data=np.array([c["cls"].encode() for c in crystals]))
            g.create_dataset("kernel_scale", data=np.array([s_true])); g.create_dataset("peak_adu", data=np.array([peak]))
            for k, c in enumerate(crystals):
                g.create_dataset(f"comps_{k}", data=np.stack(c["comps"]))
