"""Tests for Newton-Raphson convergence criteria."""

import numpy as np
import pytest

from femtoolkit.analysis.convergence import (
    NORM_FLOOR,
    displacement_correction_ratio,
    has_converged,
    residual_norm_ratio,
)
from femtoolkit.exceptions import ValidationError


def test_residual_norm_ratio_basic() -> None:
    residual = np.array([1.0, 0.0])
    external_force = np.array([100.0, 0.0])

    assert residual_norm_ratio(residual, external_force) == pytest.approx(0.01)


def test_residual_norm_ratio_floors_near_zero_external_force() -> None:
    residual = np.array([1e-13, 0.0])
    external_force = np.array([0.0, 0.0])

    # denominator floors to 1.0, so the ratio is just the residual's own norm.
    assert residual_norm_ratio(residual, external_force) == pytest.approx(1e-13)


def test_residual_norm_ratio_never_produces_nan_or_inf() -> None:
    residual = np.zeros(3)
    external_force = np.zeros(3)

    ratio = residual_norm_ratio(residual, external_force)
    assert np.isfinite(ratio)


def test_displacement_correction_ratio_basic() -> None:
    delta_u = np.array([0.001, 0.0])
    u = np.array([0.1, 0.0])

    assert displacement_correction_ratio(delta_u, u) == pytest.approx(0.01)


def test_displacement_correction_ratio_floors_near_zero_displacement() -> None:
    delta_u = np.array([1e-13])
    u = np.array([0.0])

    assert displacement_correction_ratio(delta_u, u) == pytest.approx(1e-13)


def test_norm_floor_is_small() -> None:
    assert 0.0 < NORM_FLOOR < 1e-6


def test_has_converged_residual_criterion() -> None:
    residual = np.array([1.0])
    external_force = np.array([1e9])

    assert has_converged(
        "residual",
        residual=residual,
        external_force=external_force,
        delta_u=np.array([0.0]),
        u=np.array([1.0]),
        tolerance=1e-6,
    )


def test_has_converged_residual_criterion_not_converged() -> None:
    residual = np.array([1.0])
    external_force = np.array([1.0])

    assert not has_converged(
        "residual",
        residual=residual,
        external_force=external_force,
        delta_u=np.array([0.0]),
        u=np.array([1.0]),
        tolerance=1e-6,
    )


def test_has_converged_displacement_criterion() -> None:
    assert has_converged(
        "displacement",
        residual=np.array([1e9]),
        external_force=np.array([1e9]),
        delta_u=np.array([1e-10]),
        u=np.array([1.0]),
        tolerance=1e-6,
    )


def test_has_converged_rejects_unknown_criterion() -> None:
    with pytest.raises(ValidationError):
        has_converged(
            "bogus",
            residual=np.array([0.0]),
            external_force=np.array([1.0]),
            delta_u=np.array([0.0]),
            u=np.array([1.0]),
            tolerance=1e-6,
        )
