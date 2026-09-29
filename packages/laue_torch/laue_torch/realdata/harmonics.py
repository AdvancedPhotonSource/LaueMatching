"""One reflection per predicted pixel: the harmonic grouping shared by the
real-data refiners.

Harmonics ((111), (222), (333), ...) share a q-hat, so they land on the same
detector pixel. A forward model that gives every reflection intensity 1 (or
the same observed amplitude) renders such a spot n times too bright. The
refiners keep only the LOWEST-ORDER member (smallest |hkl|^2) of each group of
in-band reflections that round to the same seed pixel; the group key is the
rounded seed pixel, so distinct reflections that coincide on a pixel are
merged too (the observed spot holds both, one amplitude is right).
Reflections that are near but not on the same pixel are not merged.
"""
from __future__ import annotations

import torch
from torch import Tensor


def lowest_order_per_pixel(cx: Tensor, cy: Tensor, keep_h: Tensor,
                           hkls: Tensor) -> list[int]:
    """Indices (into ``hkls``) of the lowest-order reflection per pixel.

    ``cx``, ``cy``: (H,) integer (rounded) seed pixel of every reflection.
    ``keep_h``: 1-D indices of the reflections that are in band and on the
    detector at the seed. Ties in |hkl|^2 keep the first index (stable sort).
    """
    if keep_h.numel() == 0:
        return []
    order_key = (hkls.to(dtype=torch.float64, device=keep_h.device) ** 2).sum(-1)
    order = torch.argsort(order_key[keep_h], stable=True)
    seen: set = set()
    out: list[int] = []
    for h in keep_h[order].tolist():
        key = (int(cx[h]), int(cy[h]))
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out
