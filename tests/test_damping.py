"""Tests for RayleighDamping and ModalDamping."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.damping import (
    ModalDamping,
    RayleighDamping,
    modal_damping_ratios_from_matrix,
)
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


# --- ModalDamping ---


def test_constant_ratio_broadcasts_to_every_mode() -> None:
    damping = ModalDamping(damping_ratios=0.02)
    assert_allclose(damping.ratios_for(4), [0.02, 0.02, 0.02, 0.02])


def test_per_mode_ratios_returned_as_is() -> None:
    damping = ModalDamping(damping_ratios=(0.01, 0.02, 0.05))
    assert_allclose(damping.ratios_for(3), [0.01, 0.02, 0.05])


def test_per_mode_ratios_wrong_count_raises() -> None:
    damping = ModalDamping(damping_ratios=(0.01, 0.02))
    with pytest.raises(ValidationError):
        damping.ratios_for(3)


def test_modal_damping_coefficients_formula() -> None:
    damping = ModalDamping(damping_ratios=0.05)
    omega = np.array([10.0, 20.0, 30.0])
    coefficients = damping.modal_damping_coefficients(omega)
    assert_allclose(coefficients, 2.0 * 0.05 * omega)


def test_modal_damping_rejects_negative_constant_ratio() -> None:
    with pytest.raises(ValidationError):
        ModalDamping(damping_ratios=-0.01)


def test_modal_damping_rejects_negative_per_mode_ratio() -> None:
    with pytest.raises(ValidationError):
        ModalDamping(damping_ratios=(0.01, -0.02, 0.03))


def test_modal_damping_rejects_non_finite_ratio() -> None:
    with pytest.raises(ValidationError):
        ModalDamping(damping_ratios=float("nan"))


def test_modal_damping_accepts_zero_ratio() -> None:
    damping = ModalDamping(damping_ratios=0.0)
    assert_allclose(damping.ratios_for(2), [0.0, 0.0])


# --- modal_damping_ratios_from_matrix ---


def test_derived_ratios_match_rayleigh_formula() -> None:
    """For mass-normalized modes (M = I here) and Rayleigh damping,
    zeta_i = (alpha + beta*omega_i^2) / (2*omega_i) exactly.
    """
    rayleigh = RayleighDamping(alpha=0.5, beta=0.001)
    omega = np.array([10.0, 50.0, 100.0])
    mass = np.eye(3)
    stiffness = np.diag(omega**2)
    damping_matrix = rayleigh.damping_matrix(mass, stiffness)
    mode_shapes = np.eye(3)  # already mass-normalized since mass = I

    ratios = modal_damping_ratios_from_matrix(mode_shapes, damping_matrix, omega)

    expected = (0.5 + 0.001 * omega**2) / (2.0 * omega)
    assert_allclose(ratios, expected, rtol=1e-9)


def test_derived_ratios_reject_non_positive_frequency() -> None:
    mode_shapes = np.eye(2)
    damping_matrix = np.eye(2)
    omega = np.array([10.0, 0.0])
    with pytest.raises(ValidationError):
        modal_damping_ratios_from_matrix(mode_shapes, damping_matrix, omega)
