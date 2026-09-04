"""Tests for femtoolkit.continuum.invariants: principal and isochoric invariants of C."""

import numpy as np
import pytest

from femtoolkit.continuum.invariants import (
    first_invariant,
    isochoric_first_invariant,
    isochoric_second_invariant,
    jacobian_from_right_cauchy_green,
    second_invariant,
    third_invariant,
    validate_right_cauchy_green,
)
from femtoolkit.exceptions import InvalidDeformationGradientError


def test_invariants_at_reference_configuration() -> None:
    c = np.eye(3)
    assert first_invariant(c) == pytest.approx(3.0)
    assert second_invariant(c) == pytest.approx(3.0)
    assert third_invariant(c) == pytest.approx(1.0)
    assert jacobian_from_right_cauchy_green(c) == pytest.approx(1.0)


def test_third_invariant_equals_jacobian_squared() -> None:
    f = np.array([[1.2, 0.1, 0.0], [0.0, 0.9, 0.05], [0.0, 0.0, 1.05]])
    c = f.T @ f
    jacobian = np.linalg.det(f)
    assert third_invariant(c) == pytest.approx(jacobian**2)
    assert jacobian_from_right_cauchy_green(c) == pytest.approx(abs(jacobian))


def test_isochoric_invariants_are_three_under_pure_volumetric_scaling() -> None:
    for alpha in (0.5, 1.0, 1.5, 2.0):
        c = (alpha**2) * np.eye(3)
        assert isochoric_first_invariant(c) == pytest.approx(3.0)
        assert isochoric_second_invariant(c) == pytest.approx(3.0)


def test_isochoric_invariants_depend_only_on_shape_not_volume() -> None:
    """Scaling C by any positive factor must not change the isochoric invariants."""
    base = np.array([[1.3, 0.05, 0.0], [0.05, 0.9, 0.0], [0.0, 0.0, 1.1]])
    scaled = 4.0 * base

    assert isochoric_first_invariant(scaled) == pytest.approx(isochoric_first_invariant(base))
    assert isochoric_second_invariant(scaled) == pytest.approx(isochoric_second_invariant(base))


def test_second_invariant_formula() -> None:
    c = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    # For a diagonal C, I2 = sum of principal 2x2 minors.
    expected = 2.0 * 3.0 + 3.0 * 4.0 + 2.0 * 4.0
    assert second_invariant(c) == pytest.approx(expected)


def test_validate_right_cauchy_green_accepts_valid_tensor() -> None:
    c = np.eye(3)
    assert validate_right_cauchy_green(c) == pytest.approx(1.0)


def test_validate_right_cauchy_green_rejects_non_finite() -> None:
    c = np.eye(3)
    c[0, 0] = np.nan
    with pytest.raises(InvalidDeformationGradientError):
        validate_right_cauchy_green(c)


def test_validate_right_cauchy_green_rejects_collapsed_tensor() -> None:
    c = np.zeros((3, 3))
    with pytest.raises(InvalidDeformationGradientError):
        validate_right_cauchy_green(c)


def test_jacobian_raises_before_log_would_be_computed_on_invalid_c() -> None:
    """Guards the 'never compute log(J) when J<=0' requirement at the invariants layer."""
    c = -np.eye(3)
    with pytest.raises(InvalidDeformationGradientError):
        jacobian_from_right_cauchy_green(c)
