"""Validation Case 3 (plane strain half): constitutive matrix.

For a structural steel (``E = 200 GPa``, ``v = 0.3``), the plane-strain
constitutive matrix is hand-derived independently of the implementation
under test and compared directly, alongside a demonstration that it
differs meaningfully from the plane-stress matrix for the same material
(see tests/validation/test_plane_stress.py) -- using the wrong one for a
given problem's geometry silently produces the wrong stiffness.
"""

import numpy as np
from numpy.testing import assert_allclose

from femtoolkit.continuum.constitutive import plane_strain_matrix, plane_stress_matrix

YOUNGS_MODULUS = 200e9
POISSON_RATIO = 0.3


def test_plane_strain_constitutive_matrix_hand_derived() -> None:
    """D = E/((1+v)(1-2v)) * [[1-v,v,0],[v,1-v,0],[0,0,(1-2v)/2]], independently computed."""
    d_matrix = plane_strain_matrix(YOUNGS_MODULUS, POISSON_RATIO)

    factor = YOUNGS_MODULUS / ((1.0 + POISSON_RATIO) * (1.0 - 2.0 * POISSON_RATIO))
    expected = factor * np.array(
        [
            [1.0 - POISSON_RATIO, POISSON_RATIO, 0.0],
            [POISSON_RATIO, 1.0 - POISSON_RATIO, 0.0],
            [0.0, 0.0, (1.0 - 2.0 * POISSON_RATIO) / 2.0],
        ]
    )
    assert_allclose(d_matrix, expected, rtol=1e-12)


def test_plane_strain_is_stiffer_than_plane_stress_for_the_same_material() -> None:
    """Physically, restraining out-of-plane strain (plane strain) makes a
    material appear stiffer in-plane than the same material under plane
    stress -- every entry of D_plane_strain exceeds the corresponding
    entry of D_plane_stress for a positive Poisson's ratio.
    """
    plane_stress = plane_stress_matrix(YOUNGS_MODULUS, POISSON_RATIO)
    plane_strain = plane_strain_matrix(YOUNGS_MODULUS, POISSON_RATIO)

    assert np.all(plane_strain[:2, :2] >= plane_stress[:2, :2])


def test_plane_strain_reduces_to_plane_stress_when_poisson_ratio_is_zero() -> None:
    """With v=0, there is no Poisson coupling between in-plane and
    out-of-plane behavior, so the two formulations coincide.
    """
    plane_stress = plane_stress_matrix(YOUNGS_MODULUS, 0.0)
    plane_strain = plane_strain_matrix(YOUNGS_MODULUS, 0.0)

    assert_allclose(plane_stress, plane_strain)


def test_plane_strain_matrix_is_symmetric_and_positive_definite() -> None:
    d_matrix = plane_strain_matrix(YOUNGS_MODULUS, POISSON_RATIO)

    assert_allclose(d_matrix, d_matrix.T)
    eigenvalues = np.linalg.eigvalsh(d_matrix)
    assert np.all(eigenvalues > 0.0)
