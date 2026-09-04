"""Tests for femtoolkit.materials.hyperelastic: the HyperelasticMaterial base class machinery.

Uses a minimal concrete subclass (a hand-checkable quadratic energy in the
first invariant) to test the base class's generic numerical
stress/tangent/stress-measure machinery in isolation from any specific
published material model (those get their own test files).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.continuum.invariants import first_invariant
from femtoolkit.materials.hyperelastic import HyperelasticMaterial


class _QuadraticFirstInvariantMaterial(HyperelasticMaterial):
    """``W = k/2 * (I1 - 3)^2``, a simple hand-differentiable toy energy.

    ``dW/dI1 = k*(I1-3)``, ``dI1/dC = I`` (identity), so
    ``S = 2*dW/dC = 2*k*(I1-3)*I``. Used only to check the base class's
    numerical differentiation machinery against a hand-derived closed
    form, independent of any specific published material.
    """

    def __init__(self, k: float) -> None:
        self.k = k

    def _energy_from_right_cauchy_green(self, right_cauchy_green_tensor: np.ndarray) -> float:
        i1 = first_invariant(right_cauchy_green_tensor)
        return 0.5 * self.k * (i1 - 3.0) ** 2

    def analytical_second_piola_kirchhoff(
        self, deformation_gradient_tensor: np.ndarray
    ) -> np.ndarray:
        from femtoolkit.continuum.deformation import right_cauchy_green

        c = right_cauchy_green(deformation_gradient_tensor)
        i1 = first_invariant(c)
        s_tensor = 2.0 * self.k * (i1 - 3.0) * np.eye(3)
        from femtoolkit.continuum.tensor import tensor_to_voigt_stress

        return tensor_to_voigt_stress(s_tensor)


@pytest.fixture
def toy_material() -> _QuadraticFirstInvariantMaterial:
    return _QuadraticFirstInvariantMaterial(k=1.0e6)


def test_strain_energy_density_zero_at_reference(
    toy_material: _QuadraticFirstInvariantMaterial,
) -> None:
    assert toy_material.strain_energy_density(np.eye(3)) == pytest.approx(0.0, abs=1e-9)


def test_numerical_stress_matches_analytical_stress(
    toy_material: _QuadraticFirstInvariantMaterial,
) -> None:
    f = np.array([[1.2, 0.05, 0.0], [0.0, 0.95, 0.03], [0.0, 0.0, 1.1]])
    numerical = toy_material.second_piola_kirchhoff_stress(f)
    analytical = toy_material.analytical_second_piola_kirchhoff(f)
    assert_allclose(numerical, analytical, rtol=1e-4, atol=1.0)


def test_stress_is_zero_at_reference_configuration(
    toy_material: _QuadraticFirstInvariantMaterial,
) -> None:
    stress = toy_material.second_piola_kirchhoff_stress(np.eye(3))
    assert_allclose(stress, np.zeros(6), atol=1e-4)


def test_first_piola_kirchhoff_equals_f_times_second(
    toy_material: _QuadraticFirstInvariantMaterial,
) -> None:
    from femtoolkit.continuum.tensor import voigt_stress_to_tensor

    f = np.array([[1.15, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 1.0]])
    s_voigt = toy_material.second_piola_kirchhoff_stress(f)
    s_tensor = voigt_stress_to_tensor(s_voigt)
    expected_p = f @ s_tensor
    assert_allclose(toy_material.first_piola_kirchhoff_stress(f), expected_p, rtol=1e-4, atol=1.0)


def test_cauchy_stress_matches_conversion_formula(
    toy_material: _QuadraticFirstInvariantMaterial,
) -> None:
    from femtoolkit.continuum.tensor import voigt_stress_to_tensor

    f = np.array([[1.15, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 1.0]])
    jacobian = np.linalg.det(f)
    s_voigt = toy_material.second_piola_kirchhoff_stress(f)
    s_tensor = voigt_stress_to_tensor(s_voigt)
    expected_sigma = (1.0 / jacobian) * f @ s_tensor @ f.T
    assert_allclose(toy_material.cauchy_stress(f), expected_sigma, rtol=1e-4, atol=1.0)


def test_material_tangent_is_symmetric(toy_material: _QuadraticFirstInvariantMaterial) -> None:
    f = np.array([[1.2, 0.05, 0.0], [0.0, 0.95, 0.03], [0.0, 0.0, 1.1]])
    tangent = toy_material.material_tangent(f)
    assert_allclose(tangent, tangent.T, atol=1e-3)


def test_material_tangent_shape(toy_material: _QuadraticFirstInvariantMaterial) -> None:
    tangent = toy_material.material_tangent(np.eye(3))
    assert tangent.shape == (6, 6)
