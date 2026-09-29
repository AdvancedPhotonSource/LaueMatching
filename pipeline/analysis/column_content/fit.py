"""JOINT mixture fit of ALL orientations in one Laue frame (column-content reference implementation).

Ported from the sampleH Zn analysis (column_fit_wide.py; validated in PREREGISTER_column_content_wide.md: for crystals that
are FOUND, share error 16%, extent Spearman 0.93, unexplained 0.138, no false orientations; discovery of spread crystals
limits recall). Changes from that code: no private dependency (exact Cholesky NNLS, local Rodrigues), and the window
limits are arguments.

Model on the union pixel set P:  model(p) = c + sum_j sum_h a[j,h] sum_k w[j,k] Kern(p - pos(j,k,h))
  * windows: bounding box of the DETECTED component(s) touching each predicted spot (never the model's own output),
    dilated DIL px, clipped to [WMIN, WMAX]; stored ragged;
  * K components per orientation; Adam lr 3e-4 (2e-3 collapsed the fit), two inits, the lower loss wins;
  * alternating NNLS for a (per reflection) and w (per component); the final solve restarts from the BEST iterate;
    (a, w) are reset if either collapses to zero;
  * one global kernel width scale in (0.85, 1.25), fitted.
Share_j = flux_j / (modelled + UNEXPLAINED flux over ALL detected-peak pixels) (an undetected crystal lowers the others).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))       # pipeline/analysis
from frame_peaks import SAT_LEVEL, detect_peaks, directional_streaks  # noqa: E402
from .geom import DT, cloud_summary, tangent_rotation  # noqa: E402


def solve_nnls_exact(D: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """min ||D^T x - y||, x >= 0 (D is (m, NP)), solved exactly in m dimensions via Cholesky of D D^T."""
    from scipy.optimize import nnls
    G = (D @ D.T).detach().numpy(); b = (D @ y).detach().numpy()
    G = G + 1e-12 * max(float(np.trace(G)) / max(len(G), 1), 1e-300) * np.eye(len(G))
    L = np.linalg.cholesky(G)
    x, _ = nnls(L.T, np.linalg.solve(L, b), maxiter=50 * len(G))
    return torch.as_tensor(x, dtype=D.dtype)


HW = 12                                                          # edge margin (px) for an observed reflection


def observed_reflections(G, OM, AH, onpeak):
    om = torch.as_tensor(np.asarray(OM, float), dtype=DT)[None]
    px, py, _, ok = G.project(om, torch.as_tensor(AH, dtype=DT))
    px, py, ok = px[0].numpy(), py[0].numpy(), ok[0].numpy()
    H, W = onpeak.shape
    sel, pos = [], []
    for h in np.where(ok)[0]:
        x, y = int(round(px[h])), int(round(py[h]))
        if not (HW <= x < W - HW and HW <= y < H - HW) or not onpeak[y, x]:
            continue
        if any(math.hypot(px[h] - a, py[h] - b) < 1.5 for a, b in pos):
            continue                                            # harmonic: one reflection per spot
        sel.append(h); pos.append((px[h], py[h]))
    return np.array(sel, int), np.array(pos, float).reshape(-1, 2)



WMIN, WMAX, DIL, TOUCH = 25, 121, 6, 2          # defaults validated on sampleH (wide mode)


def peak_labels(raw, sat_level=SAT_LEVEL, k=5.0):
    """Labelled components of frame_peaks' cleaned frame above k MAD, for windows, the detected-pixel set and arcs.

    frame_peaks' de-streak (thin < 15 px, long >= 81 px, axis aligned) targets detector BLOOMING, but it also erases a
    real axis-aligned arc >= 81 px, which then gets no window and drops out of the unexplained total. Blooming is charge
    overflow from a saturated source, so the de-streak is applied only inside components that touch a saturated pixel;
    everything else is labelled as detected. detect_peaks keeps its own SAT_LEVEL (plateau collapse, halo sources), as
    before; sat_level here only selects the components to de-streak. Returns (lab, xs, ys, info) from detect_peaks."""
    xs, ys, info = detect_peaks(raw, drop_streaks=False)
    sub, mad = info["sub"], info["mad"]
    keep = sub > k * mad
    lab, _ = ndi.label(keep)
    # within 3 px of a saturated pixel: halo subtraction hollows the clipped core out of `sub`, so the bloom's
    # component usually starts just outside it
    hit = np.unique(lab[ndi.binary_dilation(raw >= sat_level, iterations=3) & (lab > 0)])
    if len(hit):
        near_sat = np.isin(lab, hit)
        keep = np.where(near_sat, (sub - directional_streaks(sub, mad)) > k * mad, keep)
        lab, _ = ndi.label(keep)
    return lab, xs, ys, info


def adaptive_box(lab, objs, x, y, H, W, wmin=WMIN, wmax=WMAX):
    """Window for a reflection predicted at (x, y): components of `lab` within TOUCH px, bbox dilated DIL px,
    clipped to [WMIN, WMAX] per side around the predicted spot."""
    xi, yi = int(round(x)), int(round(y))
    ids = set(np.unique(lab[max(yi - TOUCH, 0):yi + TOUCH + 1, max(xi - TOUCH, 0):xi + TOUCH + 1])) - {0}
    if ids:
        ys, xs = [], []
        for i in ids:
            sl = objs[i - 1]
            ys += [sl[0].start, sl[0].stop - 1]; xs += [sl[1].start, sl[1].stop - 1]
        y0, y1, x0, x1 = min(ys) - DIL, max(ys) + DIL, min(xs) - DIL, max(xs) + DIL
    else:
        y0, y1, x0, x1 = yi, yi, xi, xi
    h = wmax // 2; m = wmin // 2
    y0, y1 = max(y0, yi - h), min(y1, yi + h); x0, x1 = max(x0, xi - h), min(x1, xi + h)
    y0, y1 = min(y0, yi - m), max(y1, yi + m); x0, x1 = min(x0, xi - m), max(x1, xi + m)
    return max(y0, 0), min(y1, H - 1), max(x0, 0), min(x1, W - 1)




class ColumnFit:
    def __init__(self, G, raw, bg, oms, AH, kern, K=24, sat_level=65000.0, wmin=WMIN, wmax=WMAX):
        self.G, self.kern, self.K = G, kern, K
        self.u = torch.zeros((), dtype=DT, requires_grad=True)
        raw = raw.astype(np.float64)
        sub = raw - bg; sub -= np.median(sub)
        self.raw, self.bg, self.sat_level = raw, bg, float(sat_level)
        self.pedestal = bg + float(np.median(raw - bg))                    # what `sub` treats as zero signal
        sat = ndi.binary_dilation(raw >= sat_level, iterations=1)
        self.sat = sat                                                     # excluded from the fit and the residual
        lab0, xs, ys, _ = peak_labels(raw, sat_level)                      # undilated components (windows)
        self.peaks = np.c_[xs, ys].astype(float)
        objs = ndi.find_objects(lab0)
        onpeak = ndi.binary_dilation(lab0 > 0, iterations=1)
        self.onpeak = onpeak
        H, W = raw.shape
        self.oms, self.refl, self.pos, self.boxes = [], [], [], []
        for OM in oms:
            sel, pos = observed_reflections(G, OM, AH, onpeak)
            self.oms.append(np.asarray(OM, float)); self.refl.append(AH[sel]); self.pos.append(pos)
            self.boxes.append([adaptive_box(lab0, objs, x, y, H, W, wmin, wmax) for x, y in pos])
        mask = np.zeros((H, W), bool)
        for bx in self.boxes:
            for y0, y1, x0, x1 in bx:
                mask[y0:y1 + 1, x0:x1 + 1] = True
        mask &= ~sat
        self.mask = mask
        self.pix = np.flatnonzero(mask.ravel()); self.NP = len(self.pix)
        idx = np.full(H * W, -1, np.int64); idx[self.pix] = np.arange(self.NP)
        # ragged per-crystal pixel lists: pixel index into P, reflection index, pixel x, y
        self.pidx, self.pref, self.px, self.py = [], [], [], []
        for bx in self.boxes:
            pi, pr, qx, qy = [], [], [], []
            for h, (y0, y1, x0, x1) in enumerate(bx):
                yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
                ii = idx[(yy * W + xx).ravel()]
                keep = ii >= 0
                pi.append(ii[keep]); pr.append(np.full(int(keep.sum()), h)); qx.append(xx.ravel()[keep]); qy.append(yy.ravel()[keep])
            cat = lambda v, dt: torch.as_tensor(np.concatenate(v) if v else np.zeros(0), dtype=dt)  # noqa: E731
            self.pidx.append(cat(pi, torch.long)); self.pref.append(cat(pr, torch.long))
            self.px.append(cat(qx, DT)); self.py.append(cat(qy, DT))
        self.y = torch.as_tensor(sub.ravel()[self.pix], dtype=DT)
        self.om_t = [torch.as_tensor(o, dtype=DT) for o in self.oms]
        self.hk_t = [torch.as_tensor(np.asarray(r, float), dtype=DT) for r in self.refl]
        self.nref = [len(r) for r in self.refl]

    def kscale(self):
        return 1.05 + 0.20 * torch.tanh(self.u)

    def blocks(self, omegas):
        """Per crystal j: (K, P_j) unit-flux kernel values of each component at each of its window pixels."""
        out = []
        for j, om in enumerate(omegas):
            if self.nref[j] == 0 or len(self.pidx[j]) == 0:
                out.append(None); continue
            oms = tangent_rotation(om) @ self.om_t[j]
            px, py, _, ok = self.G.project(oms, self.hk_t[j])                          # (K, n_j)
            r = self.pref[j]
            dx = self.px[j][None] - px[:, r]; dy = self.py[j][None] - py[:, r]          # (K, P_j)
            out.append(self.kern(dx, dy, scale=self.kscale()) * ok.to(DT)[:, r])
        return out

    def _scatter_rows(self, j, V):
        """V (m, P_j) -> (m, NP)."""
        out = torch.zeros(V.shape[0], self.NP, dtype=DT)
        out.index_add_(1, self.pidx[j], V)
        return out

    def _per_refl(self, j, v):
        """v (P_j,) -> (n_j, NP): each reflection's pixels scattered into its own row."""
        out = torch.zeros(self.nref[j] * self.NP, dtype=DT)
        out.index_add_(0, self.pref[j] * self.NP + self.pidx[j], v)
        return out.reshape(self.nref[j], self.NP)

    def model(self, B, a, w, c=0.0, zero_c=False):
        m = torch.zeros(self.NP, dtype=DT)
        for j in range(len(B)):
            if B[j] is None:
                continue
            v = (w[j][:, None] * B[j]).sum(0) * a[j][self.pref[j]]
            m = m.index_add(0, self.pidx[j], v)
        return m if zero_c else m + c

    def linear(self, B, a, w, n_alt=4):
        J = len(B); c = torch.zeros((), dtype=DT)
        for _ in range(n_alt):
            cols = []
            for j in range(J):
                if B[j] is not None:
                    cols.append(self._scatter_rows(j, B[j] * a[j][self.pref[j]][None]))
            D = torch.cat(cols, 0)
            wv = solve_nnls_exact(D, self.y - c)
            p = 0
            for j in range(J):
                if B[j] is not None:
                    w[j] = wv[p:p + self.K]; p += self.K
            cols = []
            for j in range(J):
                if B[j] is not None:
                    cols.append(self._per_refl(j, (w[j][:, None] * B[j]).sum(0)))
            D = torch.cat(cols, 0)
            av = solve_nnls_exact(D, self.y - c)
            p = 0
            for j in range(J):
                if B[j] is not None:
                    a[j] = av[p:p + self.nref[j]]; p += self.nref[j]
            c = (self.y - self.model(B, a, w, zero_c=True)).mean()
        return a, w, c

    def _fit_once(self, n_iter, lr, init_deg, seed):
        g = torch.Generator().manual_seed(seed)
        J = len(self.oms)
        with torch.no_grad():
            self.u.zero_()
        omegas = [(torch.randn(self.K, 3, generator=g, dtype=DT) * math.radians(init_deg)).requires_grad_(True) for _ in range(J)]
        a = [torch.ones(n, dtype=DT) for n in self.nref]
        w = [torch.full((self.K,), 1.0 / self.K, dtype=DT) for _ in range(J)]
        opt = torch.optim.Adam([{"params": omegas, "lr": lr}, {"params": [self.u], "lr": 0.02}])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter, eta_min=lr * 0.05)
        den = float((self.y ** 2).sum()); best = (float("inf"), None, 0.0, None, None)
        for _ in range(n_iter):
            opt.zero_grad()
            B = self.blocks(omegas)
            with torch.no_grad():
                a, w, c = self.linear([b.detach() if b is not None else None for b in B], a, w)
            loss = ((self.model(B, a, w, c) - self.y) ** 2).sum() / den
            loss.backward(); opt.step(); sched.step()
            v = float(loss.detach())
            if v < best[0]:
                best = (v, [o.detach().clone() for o in omegas], float(self.u.detach()),
                        [x.clone() for x in a], [x.clone() for x in w])
            if any(float(x.sum()) <= 0 for x in a + w):                     # bilinear NNLS absorbing zero: restart
                a = [torch.ones(n, dtype=DT) for n in self.nref]
                w = [torch.full((self.K,), 1.0 / self.K, dtype=DT) for _ in range(J)]
        with torch.no_grad():
            self.u.fill_(best[2])
            a, w = best[3], best[4]                                         # re-solve from the BEST iterate
            B = self.blocks(best[1]); a, w, c = self.linear(B, a, w)
            loss = float((((self.model(B, a, w, c) - self.y) ** 2).sum() / den))
        return loss, best[1], a, w, c, B, best[2]

    def fit(self, n_iter=300, lr=3e-4, inits=(0.05, 0.4), seed=0):
        runs = [(d,) + self._fit_once(n_iter, lr, d, seed) for d in inits]
        d, loss, om, a, w, c, B, u = min(runs, key=lambda r: r[1])
        with torch.no_grad():
            self.u.fill_(u)
        self.omegas, self.a, self.w, self.c, self.B, self.loss, self.init_won = om, a, w, c, B, loss, d
        self.init_losses = {str(r[0]): r[1] for r in runs}
        return loss

    def report(self):
        from scipy.spatial.transform import Rotation
        J = len(self.oms)
        fl = [float(self.a[j].sum() * self.w[j].sum()) if self.B[j] is not None else 0.0 for j in range(J)]
        full = np.zeros(self.raw.size)
        full[self.pix] = self.model(self.B, self.a, self.w, zero_c=True).detach().numpy()
        sub = (self.raw - self.bg).ravel(); sub = sub - np.median(sub) - float(self.c)
        onp = self.onpeak.ravel() & (self.raw.ravel() < self.sat_level)
        yy = np.clip(sub[onp], 0, None); mm = np.clip(full[onp], 0, None)
        unexpl = float(np.clip(yy - mm, 0, None).sum()); tot = sum(fl) + unexpl
        C_int = float(np.minimum(mm, yy).sum() / max(yy.sum(), 1e-9))
        per = []
        for j in range(J):
            sizes = [((y1 - y0 + 1), (x1 - x0 + 1)) for y0, y1, x0, x1 in self.boxes[j]]
            if self.B[j] is None or float(self.w[j].sum()) <= 0:
                per.append(dict(n_refl=self.nref[j], share=0.0, share_of_modelled=0.0, mean_om=self.oms[j].tolist(),
                                extent_perp_deg=None, extent_blind_deg=None, window_sizes=sizes)); continue
            blind = self.G.mean_q_axis(self.om_t[j], self.hk_t[j])
            cs = cloud_summary(self.omegas[j], self.w[j], blind)
            wn = (self.w[j] / self.w[j].sum()).numpy()
            mu = (wn[:, None] * self.omegas[j].numpy()).sum(0)
            mean_om = Rotation.from_rotvec(mu).as_matrix() @ self.oms[j]
            per.append(dict(n_refl=self.nref[j], share=fl[j] / max(tot, 1e-9), share_of_modelled=fl[j] / max(sum(fl), 1e-9),
                            mean_om=mean_om.tolist(), window_sizes=sizes, **cs))
        return dict(orientations=per, C_int=C_int, unexplained_flux_frac=unexpl / max(tot, 1e-9),
                    kernel_scale=float(self.kscale().detach()), init_won=self.init_won, init_losses=self.init_losses,
                    n_detected=int(len(self.peaks)), loss=self.loss, n_pixels=int(self.NP))

    def residual_frame(self):
        """raw - model; the saturated pixels (and the dilation ring the fit excluded) are set to the background, since
        they were never fitted and a clipped core left at raw would seed a false crystal in discovery."""
        img = np.zeros(self.raw.size)
        img[self.pix] = self.model(self.B, self.a, self.w, zero_c=True).detach().numpy()
        res = self.raw - img.reshape(self.raw.shape)
        res[self.sat] = self.pedestal[self.sat]
        return np.clip(np.rint(res), 0, 65535).astype(np.uint16)
