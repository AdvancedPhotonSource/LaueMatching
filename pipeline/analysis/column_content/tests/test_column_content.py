"""Unit tests for the Laue column-content reference (run: pytest pipeline/analysis/column_content/tests).
The end-to-end fit is checked against the original sampleH code on real geometry (see the package README)."""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # pipeline/analysis
from column_content.arcs import BEAD_THETA, common_axis_test, fib  # noqa: E402
from column_content.fit import adaptive_box, solve_nnls_exact  # noqa: E402
from column_content.geom import cloud_summary, tangent_rotation  # noqa: E402


def test_nnls_exact_matches_scipy_on_the_full_design():
    from scipy.optimize import nnls
    rng = np.random.default_rng(0)
    D = np.abs(rng.normal(size=(30, 5000))); D[3] = D[2]          # coincident components
    y = D.T @ np.abs(rng.normal(size=30)) + rng.normal(0, 0.1, 5000)
    x_ref, _ = nnls(D.T, y)
    x = solve_nnls_exact(torch.as_tensor(D), torch.as_tensor(y)).numpy()
    assert abs(np.linalg.norm(D.T @ x - y) - np.linalg.norm(D.T @ x_ref - y)) < 1e-9 * np.linalg.norm(y)
    assert np.all(x >= 0)


def test_tangent_rotation_matches_scipy():
    from scipy.spatial.transform import Rotation
    w = np.array([[0.1, -0.2, 0.05], [0.0, 0.0, 0.0], [1.0, 2.0, -0.5]])
    ours = tangent_rotation(torch.as_tensor(w)).numpy()
    for k in range(3):
        assert np.allclose(ours[k], Rotation.from_rotvec(w[k]).as_matrix(), atol=1e-12)


def test_adaptive_box_follows_the_detected_component_and_clips():
    from scipy import ndimage as ndi
    lab = np.zeros((200, 200), int)
    lab[100:103, 60:140] = 1                                          # an 80 px streak
    objs = ndi.find_objects(lab)
    y0, y1, x0, x1 = adaptive_box(lab, objs, 100, 101, 200, 200)
    assert x0 <= 60 - 6 and x1 >= 139 + 6                              # the whole streak + dilation
    y0, y1, x0, x1 = adaptive_box(lab, objs, 40, 40, 200, 200)       # nothing touches (interior) -> the 25 px minimum
    assert (x1 - x0 + 1) == 25 and (y1 - y0 + 1) == 25
    lab2 = np.zeros((400, 400), int); lab2[200:202, 0:400] = 1
    y0, y1, x0, x1 = adaptive_box(lab2, ndi.find_objects(lab2), 200, 200, 400, 400)
    assert (x1 - x0 + 1) <= 121                                        # clipped at the maximum


def test_cloud_summary_splits_perp_and_blind():
    om = torch.zeros(4, 3, dtype=torch.float64)
    om[:, 0] = torch.tensor([-1.0, 1.0, -1.0, 1.0]) * math.radians(0.1)   # spread only along x
    s = cloud_summary(om, torch.ones(4, dtype=torch.float64), torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    assert s["extent_blind_deg"] == pytest.approx(0.1, rel=1e-6) and s["extent_perp_deg"] == pytest.approx(0.0, abs=1e-12)


def _arcs_with_axis(u, n_arcs, rng):
    arcs = []
    for _ in range(n_arcs):
        n = rng.normal(size=3); n[2] = abs(n[2]) + 1.5; n /= np.linalg.norm(n)   # normals in a limited cone
        t = np.cross(u, n); t /= np.linalg.norm(t)
        arcs.append(dict(n=n.tolist(), t=t.tolist()))
    return arcs


def test_common_axis_detected_for_shared_axis_and_not_for_random():
    rng = np.random.default_rng(1)
    u = np.array([0.0, 1.0, 1.0]) / math.sqrt(2)
    shared = common_axis_test(_arcs_with_axis(u, 150, rng), n_draws=40)
    rand = common_axis_test(sum((_arcs_with_axis(v / np.linalg.norm(v), 1, rng) for v in rng.normal(size=(150, 3))), []), n_draws=40)
    assert shared["read"] == "DETECTED" and shared["excess"] > 0.5
    assert rand["read"] != "DETECTED"
    c = np.array(shared["band_centroid"])
    assert abs(float(c @ u)) > math.cos(math.radians(25))               # u lies in / near the reported band


def test_bead_threshold_is_the_registered_value():
    assert BEAD_THETA == 0.189


# ---------------------------------------------------------------------------------------------------------------------
# Small synthetic scene: NiIndent-style params on a 256 px panel (same angular coverage, 8x larger pixels), a primitive
# hkl list (gcd 1, so no harmonics), a measured-format Gaussian kernel and a flat background.
N_PX, PX_M, BG_ADU = 256, 1.6e-3, 100.0


def _scene(tmp_path, elo=5.0, ehi=30.0, sigma=1.3):
    from column_content.geom import EmpKernel, Geom
    hk = [(h, k, l) for h in range(-5, 6) for k in range(-5, 6) for l in range(-5, 6)
          if (h, k, l) != (0, 0, 0) and math.gcd(math.gcd(abs(h), abs(k)), abs(l)) == 1]
    hkl_path = str(tmp_path / "hkl.csv"); np.savetxt(hkl_path, np.array(hk, float))
    params = str(tmp_path / "params.txt")
    with open(params, "w") as f:
        f.write("SpaceGroup 225\nSymmetry F\nLatticeParameter 0.352380 0.352380 0.352380 90 90 90\n"
                "P_Array 0.028828 0.002715 0.512993\nR_Array -1.20161887 -1.21404493 -1.21852276\n"
                f"PxX {PX_M}\nPxY {PX_M}\nNrPxX {N_PX}\nNrPxY {N_PX}\nElo {elo}\nEhi {ehi}\nHKLFile {hkl_path}\n")
    hw, hf = 8.0, 0.25
    g = np.arange(-hw, hw + 1e-9, hf)
    Kf = np.exp(-(g[None, :] ** 2 + g[:, None] ** 2) / (2 * sigma ** 2)) / (2 * math.pi * sigma ** 2)
    kern_path = str(tmp_path / "kern.npz"); np.savez(kern_path, Kf=Kf, hw=hw, hf=hf)
    bg = np.full((N_PX, N_PX), BG_ADU); bg_path = str(tmp_path / "bg.bin"); bg.tofile(bg_path)
    return dict(params=params, hkl=hkl_path, kern_path=kern_path, bg_path=bg_path, bg=bg,
                G=Geom(params, "ni"), AH=np.array(hk, float), K=EmpKernel(kern_path))


def _spots(S, M, margin=14):
    """(x, y) of the reflections of M that land >= margin px inside the panel."""
    x, y, _, ok = S["G"].project(torch.as_tensor(np.asarray(M, float))[None], torch.as_tensor(S["AH"]))
    x, y, ok = x[0].numpy(), y[0].numpy(), ok[0].numpy()
    sel = ok & (x >= margin) & (x < N_PX - margin) & (y >= margin) & (y < N_PX - margin)
    return np.c_[x[sel], y[sel]]


def _stamp(img, S, x, y, amp, h=8):
    xi, yi = int(round(x)), int(round(y))
    yy, xx = np.mgrid[yi - h:yi + h + 1, xi - h:xi + h + 1]
    v = S["K"](torch.as_tensor(xx - x, dtype=torch.float64), torch.as_tensor(yy - y, dtype=torch.float64)).numpy()
    img[yi - h:yi + h + 1, xi - h:xi + h + 1] += amp * v


def _rot(seed):
    from scipy.spatial.transform import Rotation
    return Rotation.random(random_state=seed).as_matrix()


def _rz(deg):
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("z", deg, degrees=True).as_matrix()


def test_geom_band_defaults_to_the_params_energy_range(tmp_path):
    S = _scene(tmp_path, elo=8.0, ehi=20.0)
    assert (S["G"].elo, S["G"].ehi) == (8.0, 20.0)


def test_evaluate_and_run_refuse_a_radian_misorientation(tmp_path):
    from column_content import evaluate as ev
    from column_content import pipeline as pl
    from scipy.spatial.transform import Rotation
    rad = lambda a, b: np.asarray([Rotation.from_matrix(np.asarray(a) @ np.asarray(B).T).magnitude()  # noqa: E731
                                   for B in np.asarray(b).reshape(-1, 3, 3)])
    with pytest.raises(ValueError, match="DEGREES"):
        pl.run({}, {}, str(tmp_path), params_path="", phase="", bg_path="", kernel_npz="", hkl_fn=None,
               index_fn=None, solutions=None, misor_deg=lambda a, b: float(rad(a, b)[0]), g_disc=5)
    with pytest.raises(ValueError, match="DEGREES"):
        ev.evaluate(str(tmp_path), str(tmp_path), str(tmp_path / "e.json"), G=None, AH=None, mis=rad)


def _sat_scene(tmp_path, sat_level):
    """One crystal; its brightest reflection clipped at sat_level; plus a flat unpredicted blob at 35000 ADU."""
    S = _scene(tmp_path); M = _rot(0); xy = _spots(S, M)
    img = S["bg"].copy()
    for k, (x, y) in enumerate(xy):
        _stamp(img, S, x, y, 3e4 if k else 3e6)                               # reflection 0 saturates
    d = np.hypot(*(np.mgrid[0:N_PX, 0:N_PX][::-1][:, :, :, None] - xy.T[:, None, None, :])).min(-1)
    cand = np.argwhere((d > 30) & (np.mgrid[0:N_PX, 0:N_PX][0] > 20) & (np.mgrid[0:N_PX, 0:N_PX][0] < N_PX - 20)
                       & (np.mgrid[0:N_PX, 0:N_PX][1] > 20) & (np.mgrid[0:N_PX, 0:N_PX][1] < N_PX - 20))
    by, bx = cand[len(cand) // 2]
    img[by - 2:by + 3, bx - 2:bx + 3] = 35000.0
    raw = np.clip(np.random.default_rng(0).poisson(img), 0, sat_level).astype(float)
    return S, M, raw, (by, bx)


def test_residual_is_background_on_saturated_pixels(tmp_path):
    from column_content.fit import ColumnFit
    S, M, raw, _ = _sat_scene(tmp_path, 30000)
    fit = ColumnFit(S["G"], raw, S["bg"], [M], S["AH"], S["K"], K=4, sat_level=30000)
    fit.fit(n_iter=3, inits=(0.05,))
    res = fit.residual_frame().astype(float)
    sat = fit.sat
    assert sat.sum() > 25 and (raw >= 30000).sum() > 0
    ped = np.rint(S["bg"] + np.median(raw - S["bg"]))
    assert np.array_equal(res[sat], ped[sat])


def test_report_uses_the_fit_saturation_level(tmp_path):
    from column_content.fit import ColumnFit
    S, M, raw, (by, bx) = _sat_scene(tmp_path, 30000)
    fit = ColumnFit(S["G"], raw, S["bg"], [M], S["AH"], S["K"], K=4, sat_level=30000)
    assert fit.sat_level == 30000
    fit.fit(n_iter=3, inits=(0.05,))
    rep = fit.report()
    # the 25 px blob at 35000 ADU is clipped at 30000: it must not count as unexplained flux (it would be ~7e5 ADU,
    # comparable to the whole crystal)
    blob = 25 * (30000 - BG_ADU)
    tot_mod = sum(o["share"] for o in rep["orientations"])
    assert rep["unexplained_flux_frac"] < 0.1, (rep["unexplained_flux_frac"], blob, tot_mod)


def _streak_frame(rng, amp_mad=20.0, sigma=10.0):
    img = 100.0 + rng.normal(0, sigma, size=(N_PX, N_PX))
    img[50:200, 120:123] += amp_mad * sigma                                  # vertical 3 x 150 px arc
    return img


def test_extract_arcs_keeps_an_axis_aligned_arc(tmp_path):
    from column_content.arcs import extract_arcs
    S = _scene(tmp_path)
    arcs = extract_arcs(S["G"], _streak_frame(np.random.default_rng(3)), S["bg"])
    assert len(arcs) == 1
    # L is 2 sigma of the intensity-weighted pixel spread: a uniform 150 px line gives 150 / sqrt(3) = 86.6 px
    assert arcs[0]["L"] >= 80.0


def test_fit_window_keeps_an_axis_aligned_arc_but_not_a_saturated_bloom(tmp_path):
    from column_content.fit import ColumnFit
    S = _scene(tmp_path); M = _rot(0); xy = _spots(S, M)
    pick = [k for k, (x, y) in enumerate(xy) if 85 <= y <= N_PX - 85 and 20 <= x <= N_PX - 20
            and all(abs(x - a) > 15 for j, (a, b) in enumerate(xy) if j != k)]
    assert pick
    ks = pick[0]; xs, ys = xy[ks]
    img = S["bg"].copy()
    for k, (x, y) in enumerate(xy):
        if k == ks:
            xi, yi = int(round(xs)), int(round(ys))
            img[yi - 75:yi + 75, xi - 1:xi + 2] += 400.0                      # the reflection is a 3 x 150 px arc
        else:
            _stamp(img, S, x, y, 3e4)
    raw = np.random.default_rng(1).poisson(img).astype(float)
    fit = ColumnFit(S["G"], raw, S["bg"], [M], S["AH"], S["K"], K=4)
    assert any(math.hypot(px - xs, py - ys) < 1.5 for px, py in fit.pos[0])  # the arc reflection has a window
    xi, yi = int(round(xs)), int(round(ys))
    assert fit.onpeak[yi - 70:yi + 70, xi].mean() > 0.95                     # and its pixels count as detected
    # a saturated core blooming down its column is treated exactly as before (frame_peaks de-streaks it; a noise
    # remnant survives the opening in both versions)
    img2 = S["bg"].copy()
    for x, y in xy:
        _stamp(img2, S, x, y, 3e4)
    kb = [k for k, (x, y) in enumerate(xy) if 10 < y < N_PX - 115 and not any(
        abs(x - a) <= 12 and y - 12 <= b <= y + 115 for j, (a, b) in enumerate(xy) if j != k)]
    assert kb
    xb, yb = int(round(xy[kb[0]][0])), int(round(xy[kb[0]][1]))
    img2[yb - 2:yb + 3, xb - 2:xb + 3] = 70000.0
    img2[yb:yb + 110, xb - 1:xb + 2] += 3000.0                                 # bloom running down the column
    raw2 = np.clip(np.random.default_rng(2).poisson(img2), 0, 65535).astype(float)
    from column_content.fit import peak_labels
    from frame_peaks import detect_peaks
    from scipy import ndimage as ndi
    _, _, info = detect_peaks(raw2)                                          # the original, de-streaked labelling
    old = ndi.label(info["sub"] > 5 * info["mad"])[0] > 0
    new = peak_labels(raw2)[0] > 0
    bl = (slice(yb + 30, yb + 110), slice(xb - 1, xb + 2))
    assert new[bl].mean() < 0.9 and np.array_equal(new[bl], old[bl])


def test_end_to_end_two_point_crystals(tmp_path):
    """Two point crystals, shares 0.3 / 0.7: ColumnFit from truth + 0.05 deg, then pipeline.run + evaluate."""
    import functools
    import json

    import h5py
    from column_content import evaluate as ev
    from column_content import pipeline as pl
    from column_content.fit import ColumnFit
    from scipy.spatial.transform import Rotation
    S = _scene(tmp_path)
    truth = [_rot(0), _rot(2)]; share = np.array([0.3, 0.7]); F = 2e6
    rng = np.random.default_rng(7)
    img = S["bg"].copy()
    for M, s in zip(truth, share):
        xy = _spots(S, M); a = rng.uniform(0.5, 1.5, len(xy)); a *= s * F / a.sum()
        for (x, y), amp in zip(xy, a):
            _stamp(img, S, x, y, amp)
    raw = np.random.default_rng(8).poisson(img).astype(np.uint16)
    assert raw.max() < 65000
    start = [Rotation.from_rotvec(ax / np.linalg.norm(ax) * math.radians(0.05)).as_matrix() @ M
             for ax, M in zip(rng.normal(size=(2, 3)), truth)]

    fit = ColumnFit(S["G"], raw.astype(float), S["bg"], start, S["AH"], S["K"], K=8)
    fit.fit(n_iter=40, inits=(0.05,))
    rep = fit.report()
    for q, M, s in zip(rep["orientations"], truth, share):
        assert abs(q["share"] - s) / s < 0.20, q["share"]
        assert Rotation.from_matrix(np.asarray(q["mean_om"]) @ M.T).magnitude() < math.radians(0.02)
    assert rep["unexplained_flux_frac"] < 0.15

    fd = tmp_path / "frames"; fd.mkdir()
    with h5py.File(fd / "F_000001.h5", "w") as h:
        h.create_dataset("/entry1/data/data", data=raw)
        g = h.create_group("/entry1/truth")
        g.create_dataset("centres", data=np.stack(truth)); g.create_dataset("share", data=share)
        g.create_dataset("spread_class", data=np.array([b"point", b"point"]))
        g.create_dataset("spread_param", data=np.zeros(2))
        for k, M in enumerate(truth):
            g.create_dataset(f"comps_{k}", data=M[None])
    from laue_material import Phase
    ph = Phase(S["params"], name="ni")
    out = str(tmp_path / "run")
    summ = pl.run({"F_000001.h5": str(fd / "F_000001.h5")}, {"F_000001.h5": [(start[0], 30), (start[1], 30)]}, out,
                  params_path=S["params"], phase="ni", bg_path=S["bg_path"], kernel_npz=S["kern_path"],
                  hkl_fn=functools.partial(np.loadtxt, S["hkl"]), index_fn=lambda d, t: {}, solutions=lambda p, g: [],
                  misor_deg=lambda a, b: float(ph.misorientation(a, b)[0]), g_disc=7, max_rounds=1, workers=1,
                  fit_kw=dict(K=8, n_iter=40, inits=(0.05,)))
    assert summ["final_round"] == 0 and summ["rounds"][0]["n_errors"] == 0
    res = ev.evaluate(out, str(fd), str(tmp_path / "eval.json"), G=S["G"], AH=S["AH"], mis=ph.misorientation)
    assert res["gdisc"] == 7 and json.load(open(tmp_path / "eval.json"))["gdisc"] == 7   # round trip of the gate
    assert res["V1"]["false_total"] == 0 and res["V2"]["recall"] == 1.0
