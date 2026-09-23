"""Tests for femtoolkit.verification.metrics (Version 29)."""

from __future__ import annotations

import numpy as np
import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.verification.metrics import (
    absolute_error,
    energy_norm_error,
    l2_error,
    relative_error,
    relative_l2_error,
)


def test_absolute_error_basic() -> None:
    assert absolute_error(1.05, 1.0) == pytest.approx(0.05)


def test_absolute_error_is_non_negative_regardless_of_sign() -> None:
    assert absolute_error(0.95, 1.0) == pytest.approx(0.05)


def test_relative_error_basic() -> None:
    assert relative_error(1.1, 1.0) == pytest.approx(0.1)


def test_relative_error_uses_epsilon_floor_at_zero_reference() -> None:
    error = relative_error(1e-13, 0.0, epsilon=1e-12)
    assert error == pytest.approx(1e-13 / 1e-12)


def test_relative_error_rejects_non_positive_epsilon() -> None:
    with pytest.raises(ValidationError):
        relative_error(1.0, 1.0, epsilon=0.0)
    with pytest.raises(ValidationError):
        relative_error(1.0, 1.0, epsilon=-1e-9)


def test_l2_error_basic() -> None:
    numerical = np.array([1.0, 2.0, 3.0])
    reference = np.array([1.0, 2.0, 4.0])
    assert l2_error(numerical, reference) == pytest.approx(1.0)


def test_l2_error_zero_for_identical_vectors() -> None:
    vector = np.array([1.0, -2.0, 3.5])
    assert l2_error(vector, vector) == pytest.approx(0.0)


def test_l2_error_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValidationError):
        l2_error(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_relative_l2_error_basic() -> None:
    numerical = np.array([1.0, 2.0])
    reference = np.array([1.0, 2.0])
    assert relative_l2_error(numerical, reference) == pytest.approx(0.0)

    numerical = np.array([1.1, 2.2])
    reference = np.array([1.0, 2.0])
    expected = l2_error(numerical, reference) / np.linalg.norm(reference)
    assert relative_l2_error(numerical, reference) == pytest.approx(expected)


def test_relative_l2_error_uses_epsilon_at_zero_reference() -> None:
    numerical = np.array([1e-13, 0.0])
    reference = np.array([0.0, 0.0])
    error = relative_l2_error(numerical, reference, epsilon=1e-12)
    assert error == pytest.approx(1e-13 / 1e-12)


def test_energy_norm_error_zero_for_identical_displacements() -> None:
    stiffness = np.array([[2.0, -1.0], [-1.0, 2.0]])
    displacements = np.array([0.5, -0.3])
    assert energy_norm_error(displacements, displacements, stiffness) == pytest.approx(0.0)


def test_energy_norm_error_matches_hand_computation() -> None:
    stiffness = np.eye(2) * 4.0
    numerical = np.array([1.0, 0.0])
    reference = np.array([0.0, 0.0])
    # error = numerical - reference = [1, 0]; e^T K e = 4; sqrt(4) = 2.
    # reference energy = 0 -> denominator floored at epsilon.
    result = energy_norm_error(numerical, reference, stiffness, epsilon=1e-6)
    assert result == pytest.approx(2.0 / 1e-6)


def test_energy_norm_error_rejects_mismatched_stiffness_shape() -> None:
    stiffness = np.eye(3)
    with pytest.raises(ValidationError):
        energy_norm_error(np.array([1.0, 2.0]), np.array([1.0, 2.0]), stiffness)


def test_energy_norm_error_rejects_mismatched_displacement_shapes() -> None:
    stiffness = np.eye(2)
    with pytest.raises(ValidationError):
        energy_norm_error(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]), stiffness)
