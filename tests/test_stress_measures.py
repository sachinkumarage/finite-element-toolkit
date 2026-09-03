"""Tests for the finite-strain stress-measure conversions in femtoolkit.continuum.stress."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.continuum.stress import (
    cauchy_stress_from_second_piola_kirchhoff,
    first_piola_kirchhoff_from_second,
)


def test_identity_deformation_gives_equal_stress_measures() -> None:
    """At F=I (undeformed), Cauchy = P1 = P2 exactly -- all three measures coincide."""
    f = np.eye(3)
    s = np.diag([1e6, 2e6, 3e6])
    p1 = first_piola_kirchhoff_from_second(f, s)
    cauchy = cauchy_stress_from_second_piola_kirchhoff(f, s)
    assert_allclose(p1, s)
    assert_allclose(cauchy, s)


def test_first_piola_kirchhoff_formula() -> None:
    f = np.array([[1.2, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 1.0]])
    s = np.diag([1e6, 2e6, 3e6])
    p1 = first_piola_kirchhoff_from_second(f, s)
    assert_allclose(p1, f @ s)


def test_cauchy_stress_formula_and_symmetry() -> None:
    f = np.array([[1.2, 0.1, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 1.05]])
    s = np.array([[1e6, 5e4, 0.0], [5e4, 2e6, 0.0], [0.0, 0.0, 3e6]])
    cauchy = cauchy_stress_from_second_piola_kirchhoff(f, s)

    jacobian = np.linalg.det(f)
    expected = (f @ s @ f.T) / jacobian
    assert_allclose(cauchy, expected)
    assert_allclose(cauchy, cauchy.T)  # Cauchy stress is always symmetric


def test_p1_and_cauchy_consistency() -> None:
    """Cauchy = P1 @ F^T / det(F), the standard relationship between the two."""
    f = np.array([[1.1, 0.0, 0.0], [0.0, 1.0, 0.05], [0.0, 0.0, 0.95]])
    s = np.array([[2e6, 0.0, 1e5], [0.0, 1e6, 0.0], [1e5, 0.0, 0.5e6]])

    p1 = first_piola_kirchhoff_from_second(f, s)
    cauchy = cauchy_stress_from_second_piola_kirchhoff(f, s)
    jacobian = np.linalg.det(f)

    assert_allclose(cauchy, (p1 @ f.T) / jacobian)


def test_uniaxial_case_matches_hand_calculation() -> None:
    """A pure uniaxial stretch with a known S gives an analytically checkable Cauchy stress."""
    stretch = 1.5
    f = np.diag([stretch, 1.0, 1.0])
    s_xx = 1.0e6
    s = np.diag([s_xx, 0.0, 0.0])

    cauchy = cauchy_stress_from_second_piola_kirchhoff(f, s)
    # sigma_xx = (1/det(F)) * F_xx * S_xx * F_xx = stretch^2 * S_xx / stretch = stretch * S_xx
    expected_sigma_xx = stretch * s_xx
    assert cauchy[0, 0] == pytest.approx(expected_sigma_xx)
