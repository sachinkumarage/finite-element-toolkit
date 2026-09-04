"""Tests for the PK1/PK2/Cauchy stress-measure conversions on hyperelastic materials.

These conversions themselves (``first_piola_kirchhoff_from_second``,
``cauchy_stress_from_second_piola_kirchhoff``) are Version 16 code, already
tested in ``tests/test_stress_measures.py``; this file checks that
:class:`~femtoolkit.materials.hyperelastic.HyperelasticMaterial`'s
``first_piola_kirchhoff_stress``/``cauchy_stress`` wrappers apply them
correctly for the two new Version 17 materials specifically.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.materials.mooney_rivlin import MooneyRivlin3D
from femtoolkit.materials.neo_hookean import NeoHookean3D

_DEFORMATIONS = [
    np.array([[1.2, 0.1, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 1.05]]),
    np.array([[1.0, 0.0, 0.0], [0.15, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    np.array([[0.85, 0.02, 0.01], [0.0, 1.1, -0.03], [0.0, 0.0, 0.95]]),
]

_MATERIALS = [
    NeoHookean3D(youngs_modulus=5.0e6, poisson_ratio=0.45),
    MooneyRivlin3D(c10=0.4e6, c01=0.1e6, bulk_modulus=200e6),
]


@pytest.mark.parametrize("material", _MATERIALS, ids=["neo_hookean", "mooney_rivlin"])
@pytest.mark.parametrize("f", _DEFORMATIONS)
def test_first_piola_kirchhoff_equals_f_dot_s(material, f: np.ndarray) -> None:
    from femtoolkit.continuum.tensor import voigt_stress_to_tensor

    s_voigt = material.second_piola_kirchhoff_stress(f)
    s_tensor = voigt_stress_to_tensor(s_voigt)
    expected_p = f @ s_tensor
    assert_allclose(material.first_piola_kirchhoff_stress(f), expected_p, rtol=1e-6, atol=1e-3)


@pytest.mark.parametrize("material", _MATERIALS, ids=["neo_hookean", "mooney_rivlin"])
@pytest.mark.parametrize("f", _DEFORMATIONS)
def test_cauchy_stress_equals_one_over_j_times_f_s_ft(material, f: np.ndarray) -> None:
    from femtoolkit.continuum.tensor import voigt_stress_to_tensor

    jacobian = np.linalg.det(f)
    s_voigt = material.second_piola_kirchhoff_stress(f)
    s_tensor = voigt_stress_to_tensor(s_voigt)
    expected_sigma = (1.0 / jacobian) * f @ s_tensor @ f.T
    assert_allclose(material.cauchy_stress(f), expected_sigma, rtol=1e-6, atol=1e-3)


@pytest.mark.parametrize("material", _MATERIALS, ids=["neo_hookean", "mooney_rivlin"])
def test_cauchy_stress_is_symmetric(material) -> None:
    for f in _DEFORMATIONS:
        sigma = material.cauchy_stress(f)
        assert_allclose(sigma, sigma.T, atol=1e-6)


@pytest.mark.parametrize("material", _MATERIALS, ids=["neo_hookean", "mooney_rivlin"])
def test_first_piola_kirchhoff_is_nonsymmetric_for_a_general_triaxial_shear_state(material) -> None:
    """P = F @ S is generally non-symmetric (a two-point tensor); demonstrated with
    a combined shear + non-proportional stretch state where no special cancellation occurs."""
    f = np.array([[1.05, 0.3, 0.0], [0.1, 0.92, 0.0], [0.0, 0.05, 1.02]])
    p = material.first_piola_kirchhoff_stress(f)
    assert not np.allclose(p, p.T, atol=1e-3)
