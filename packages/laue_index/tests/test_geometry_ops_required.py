"""``disorientation_deg_axis`` has no default symmetry.

It used to default ``ops`` to the cubic group, so a caller that forgot to pass
the crystal's operators got a cubic disorientation for a hexagonal (or any
other) crystal without any sign of it. ``ops`` is now a required keyword.
"""
import numpy as np
import pytest

from laue_index.geometry import CUBIC_OPS, HEX_OPS, disorientation_deg_axis


def test_calling_without_ops_raises():
    with pytest.raises(TypeError):
        disorientation_deg_axis(np.eye(3), np.eye(3))


def test_ops_is_keyword_only():
    with pytest.raises(TypeError):
        disorientation_deg_axis(np.eye(3), np.eye(3), CUBIC_OPS)


def test_explicit_ops_still_work():
    # 90 deg about z: identity under cubic symmetry, 30 deg under hexagonal (6/mmm)
    rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert disorientation_deg_axis(np.eye(3), rz, ops=CUBIC_OPS)[0] < 1e-6
    assert abs(disorientation_deg_axis(np.eye(3), rz, ops=HEX_OPS)[0] - 30.0) < 1e-6
