"""Tests for RayleighDamping."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.damping import RayleighDamping
from femtoolkit.exceptions import ValidationError


def test_default_damping_is_zero() -> None:
    damping = RayleighDamping()
    assert damping.alpha == 0.0
    assert damping.beta == 0.0


def test_damping_matrix_formula() -> None:
    mass = np.eye(3) * 2.0
    stiffness = np.eye(3) * 100.0
    damping = RayleighDamping(alpha=0.5, beta=0.01)

    c = damping.damping_matrix(mass, stiffness)

    assert_allclose(c, 0.5 * mass + 0.01 * stiffness)


def test_damping_matrix_is_symmetric_for_symmetric_inputs() -> None:
    mass = np.array([[2.0, 0.1], [0.1, 2.0]])
    stiffness = np.array([[100.0, -10.0], [-10.0, 100.0]])
    damping = RayleighDamping(alpha=0.1, beta=0.001)

    c = damping.damping_matrix(mass, stiffness)

    assert_allclose(c, c.T)


def test_rejects_negative_alpha() -> None:
    with pytest.raises(ValidationError):
        RayleighDamping(alpha=-0.1, beta=0.0)


def test_rejects_negative_beta() -> None:
    with pytest.raises(ValidationError):
        RayleighDamping(alpha=0.0, beta=-0.1)


def test_accepts_zero_coefficients() -> None:
    damping = RayleighDamping(alpha=0.0, beta=0.0)
    mass = np.eye(2)
    stiffness = np.eye(2) * 10.0
    assert_allclose(damping.damping_matrix(mass, stiffness), np.zeros((2, 2)))


def test_rejects_mismatched_shapes() -> None:
    damping = RayleighDamping(alpha=0.1, beta=0.001)
    mass = np.eye(2)
    stiffness = np.eye(3)
    with pytest.raises(ValidationError):
        damping.damping_matrix(mass, stiffness)


def test_rejects_non_finite_coefficients() -> None:
    with pytest.raises(ValidationError):
        RayleighDamping(alpha=float("nan"), beta=0.0)
