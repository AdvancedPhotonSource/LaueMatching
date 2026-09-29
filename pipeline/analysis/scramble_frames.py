"""Spot-scrambled frames: the input of the SEARCH null (invariant 29).

A gate safe against one random orientation is not safe against the best of the
indexer's whole search. The cheap, measured way to get the search's bar is to run
the SAME search on a frame whose spots have been scrambled: every connected
component the indexer would see keeps its exact pixel pattern and intensities,
only its POSITION is randomised. So, per frame:

* the connected-component count is unchanged (components are placed so they never
  touch, 8-connectivity, the labelling the indexer's preprocessing uses);
* every component's pixel intensities, and so the lit-pixel total and the total
  intensity, are unchanged (pixels are moved, never resampled);
* positions are drawn inside the frame's own SUPPORT (invariant 28: a null must
  have the same spatial support as the data) -- by default the band of scattering
  angle 2theta that the frame's lit pixels span, computed from the detector
  geometry -- and never onto a masked pixel or a detector gap.

The validator's peaks (``frame_peaks.detect_peaks`` on the raw frame) are
scrambled WITH the components that contain them, so the re-scoring statistic
(``count_matched_peaks``) sees the same scrambled field the indexer searched;
peaks outside every component are moved individually, inside the same support.

``keep_positions=True`` returns the frame unchanged. That is the NEGATIVE CONTROL
for the null itself: a "scramble" that keeps positions must reproduce the real
frame's solutions, so a null measured with it must FLAG (fail to clear) a real
crystal. A search-null block written that way is refused by every gate
(``frame_peaks.select_null``).

usage (selftest): scramble_frames.py --selftest
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

# 8-connectivity: cv2.connectedComponentsWithStats(..., 8) in
# laue_index.preprocess.find_connected_components, which segments the indexer input.
STRUCT8 = np.ones((3, 3), bool)
MAX_TRIES = 4000


class ScrambleError(RuntimeError):
    """A component could not be placed inside the allowed support."""


def twotheta_map(phase):
    """Scattering angle 2theta (degrees) of every detector pixel, shape (NrPxY, NrPxX).

    The inverse of ``laue_material.Phase.project``: pixel (x=col, y=row) sits at
    detector-frame point ``((x - (Nx-1)/2) dx + P0, (y - (Ny-1)/2) dy + P1, P2)``,
    the outgoing ray is ``rot @ that`` and 2theta is its angle to the beam (+z).
    """
    ny, nx = phase.npx_y, phase.npx_x
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    xs = np.stack([(xx - 0.5 * (nx - 1)) * phase.dx + phase.P[0],
                   (yy - 0.5 * (ny - 1)) * phase.dy + phase.P[1],
                   np.full_like(xx, phase.P[2])], axis=-1)
    kf = xs @ phase.rot.T
    kf /= np.linalg.norm(kf, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(kf[..., 2], -1.0, 1.0)))


def support_mask(lit, mask=None, twotheta=None, pad_deg=0.0):
    """Pixels a scrambled spot may occupy.

    ``lit``      bool image of the frame's lit (indexer-input) pixels
    ``mask``     bool image, True = masked / gap / excluded (never occupied)
    ``twotheta`` optional 2theta map; if given, the support is the band
                 [min, max] of 2theta over the lit pixels (+- ``pad_deg``) --
                 the frame's own radial support. Without it, the whole unmasked
                 detector.
    """
    allowed = np.ones(lit.shape, bool) if mask is None else ~np.asarray(mask, bool)
    if twotheta is not None and lit.any():
        t = twotheta[lit]
        allowed &= (twotheta >= t.min() - pad_deg) & (twotheta <= t.max() + pad_deg)
    return allowed


def _components(img):
    lab, n = ndi.label(img > 0, structure=STRUCT8)
    return lab, n, ndi.find_objects(lab)


def scramble_frame(img, rng, allowed=None, peaks=None, keep_positions=False,
                   max_tries=MAX_TRIES):
    """Scramble the components of one indexer-input frame.

    Parameters
    ----------
    img : 2-D array, the indexer's segmented input (non-zero = lit), e.g.
          ``/entry/data/cleaned_data_threshold_filtered``.
    rng : numpy Generator.
    allowed : bool image of pixels a component may occupy (``support_mask``);
          default everything. Lit pixels outside it are placed anyway only if
          their component can be put fully inside it; otherwise ScrambleError.
    peaks : optional (n, 2) array of (x, y) validator peaks on this frame.

    Returns ``(scrambled_img, scrambled_peaks, info)``; ``info`` carries the
    component count, lit-pixel total and intensity total before and after (they
    are equal by construction; the caller may assert it).
    """
    img = np.asarray(img)
    H, W = img.shape
    allowed = np.ones((H, W), bool) if allowed is None else np.asarray(allowed, bool)
    lab, n, objs = _components(img)
    pk = np.zeros((0, 2)) if peaks is None else np.asarray(peaks, float).reshape(-1, 2)
    info = {"n_components": int(n), "lit_px": int((img > 0).sum()),
            "total_intensity": float(img[img > 0].sum()), "keep_positions": bool(keep_positions)}
    if keep_positions:
        out = img.copy()
        info.update(n_components_out=info["n_components"], lit_px_out=info["lit_px"],
                    total_intensity_out=info["total_intensity"])
        return out, pk.copy(), info

    # which component each peak rides with (nearest lit pixel within 2 px)
    near = ndi.maximum_filter(lab, size=5) if n else lab
    pki = np.rint(pk).astype(int)
    inside = ((pki[:, 0] >= 0) & (pki[:, 0] < W) & (pki[:, 1] >= 0) & (pki[:, 1] < H))
    pk_comp = np.zeros(len(pk), int)
    pk_comp[inside] = near[pki[inside, 1], pki[inside, 0]]

    out = np.zeros_like(img)
    blocked = np.zeros((H, W), bool)          # occupied, dilated by one pixel
    free_y, free_x = np.nonzero(allowed)
    if not len(free_y):
        raise ScrambleError("the allowed support is empty")
    shift = np.zeros((n + 1, 2), int)
    # largest first: they are the hardest to place
    order = sorted(range(1, n + 1), key=lambda k: -int((lab[objs[k - 1]] == k).sum()))
    for k in order:
        sl = objs[k - 1]
        foot = lab[sl] == k
        fy, fx = np.nonzero(foot)
        vals = img[sl][foot]
        oy, ox = sl[0].start, sl[1].start
        ok = False
        for _ in range(max_tries):
            j = rng.integers(len(free_y))
            # anchor a random pixel of the footprint on a random allowed pixel
            a = rng.integers(len(fy))
            ty, tx = free_y[j] - fy[a], free_x[j] - fx[a]
            yy, xx = fy + ty, fx + tx
            if yy.min() < 0 or xx.min() < 0 or yy.max() >= H or xx.max() >= W:
                continue
            if not allowed[yy, xx].all() or blocked[yy, xx].any():
                continue
            ok = True
            break
        if not ok:
            raise ScrambleError(f"could not place component {k} ({len(fy)} px) inside "
                                f"the allowed support after {max_tries} tries")
        out[yy, xx] = vals
        y0, y1 = max(yy.min() - 1, 0), min(yy.max() + 2, H)
        x0, x1 = max(xx.min() - 1, 0), min(xx.max() + 2, W)
        tmp = np.zeros((y1 - y0, x1 - x0), bool)
        tmp[yy - y0, xx - x0] = True
        blocked[y0:y1, x0:x1] |= ndi.binary_dilation(tmp, structure=STRUCT8)
        shift[k] = (tx - ox, ty - oy)                # (dx, dy) of the whole component

    new_pk = pk.copy()
    if len(pk):
        rides = pk_comp > 0
        new_pk[rides, 0] += shift[pk_comp[rides], 0]
        new_pk[rides, 1] += shift[pk_comp[rides], 1]
        lone = np.flatnonzero(~rides)
        if len(lone):
            j = rng.integers(len(free_y), size=len(lone))
            new_pk[lone, 0] = free_x[j]
            new_pk[lone, 1] = free_y[j]

    lab2, n2 = ndi.label(out > 0, structure=STRUCT8)
    info.update(n_components_out=int(n2), lit_px_out=int((out > 0).sum()),
                total_intensity_out=float(out[out > 0].sum()))
    return out, new_pk, info


def selftest():
    rng = np.random.default_rng(0)
    img = np.zeros((200, 300), np.uint16)
    for _ in range(25):
        y, x = rng.integers(10, 190), rng.integers(10, 290)
        img[y:y + rng.integers(2, 6), x:x + rng.integers(2, 6)] = rng.integers(100, 5000)
    mask = np.zeros(img.shape, bool)
    mask[:, 140:150] = True                               # a detector gap
    img[mask] = 0
    peaks = np.argwhere(img > 0)[::7][:, ::-1].astype(float)
    out, pk, info = scramble_frame(img, rng, allowed=~mask, peaks=peaks)
    assert info["n_components_out"] == info["n_components"], info
    assert info["lit_px_out"] == info["lit_px"], info
    assert abs(info["total_intensity_out"] - info["total_intensity"]) < 1e-6, info
    assert not (out[mask] > 0).any(), "a scrambled spot landed in the gap"
    assert sorted(out[out > 0].tolist()) == sorted(img[img > 0].tolist())
    # every carried peak still sits on a lit pixel
    pki = np.rint(pk).astype(int)
    assert (out[pki[:, 1], pki[:, 0]] > 0).mean() > 0.9
    same, _, _ = scramble_frame(img, rng, allowed=~mask, peaks=peaks, keep_positions=True)
    assert np.array_equal(same, img)
    print(f"scramble_frames selftest OK ({info['n_components']} components, "
          f"{info['lit_px']} lit px preserved; gap never occupied)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        selftest()
    else:
        sys.exit(__doc__)
