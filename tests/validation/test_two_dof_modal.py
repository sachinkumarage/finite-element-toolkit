"""Validation Case: a 2-DOF spring-mass system, solved independently of any FEM model.

Two unit masses in series: mass 1 tied to a fixed wall by spring
``k1``, and to mass 2 by spring ``k2``. Standard Lagrangian mechanics
gives the mass and stiffness matrices directly (no finite element
discretization involved -- this test validates the eigenvalue
*mathematics* in :mod:`femtoolkit.analysis.modal`, not the FEM
pipeline; :mod:`test_fem_dynamic_validation` covers that separately):

.. code-block:: text

    M = [[m1, 0], [0, m2]]
    K = [[k1+k2, -k2], [-k2, k2]]

For ``m1 = m2 = 1``, ``k1 = k2 = 1``, the characteristic equation
``det(K - lambda*M) = 0`` reduces to ``lambda^2 - 3*lambda + 1 = 0``,
with the exact golden-ratio-related roots:

.. code-block:: text

    lambda = (3 -+ sqrt(5)) / 2
"""

import math

import numpy as np
from numpy.testing import assert_allclose

from femtoolkit.analysis.modal import natural_frequencies

MASS_1 = 1.0
MASS_2 = 1.0
STIFFNESS_1 = 1.0
STIFFNESS_2 = 1.0


def _system():
    mass = np.array([[MASS_1, 0.0], [0.0, MASS_2]])
    stiffness = np.array(
        [
            [STIFFNESS_1 + STIFFNESS_2, -STIFFNESS_2],
            [-STIFFNESS_2, STIFFNESS_2],
        ]
    )
    return mass, stiffness


def test_eigenvalues_match_analytical_characteristic_equation() -> None:
    mass, stiffness = _system()
    result = natural_frequencies(stiffness, mass)

    expected = np.array([(3 - math.sqrt(5)) / 2, (3 + math.sqrt(5)) / 2])
    assert_allclose(result.eigenvalues, expected, rtol=1e-12)


def test_natural_frequencies_match_analytical() -> None:
    mass, stiffness = _system()
    result = natural_frequencies(mass=mass, stiffness=stiffness)

    expected_omega = np.sqrt(np.array([(3 - math.sqrt(5)) / 2, (3 + math.sqrt(5)) / 2]))
    assert_allclose(result.angular_frequencies, expected_omega, rtol=1e-12)
    assert_allclose(result.frequencies, expected_omega / (2 * math.pi), rtol=1e-12)


def test_mode_shapes_satisfy_generalized_eigenvalue_equation() -> None:
    """Direct algebraic check: K @ phi == lambda * M @ phi for each mode,
    independent of any normalization convention.
    """
    mass, stiffness = _system()
    result = natural_frequencies(stiffness, mass)

    for mode_index in range(2):
        phi = result.mode_shapes[:, mode_index]
        lam = result.eigenvalues[mode_index]
        assert_allclose(stiffness @ phi, lam * (mass @ phi), atol=1e-10)


def test_mode_shape_ratios_match_hand_derivation() -> None:
    """For mode 1 (lambda1), the eigenvector ratio phi2/phi1 solves
    ``(k1+k2-lambda*m1)*phi1 = k2*phi2``, i.e. ``phi2/phi1 = (k1+k2-lambda*m1)/k2``.
    Verified independently for both modes, sign/scale-agnostic (ratios only).
    """
    mass, stiffness = _system()
    result = natural_frequencies(stiffness, mass)

    for mode_index in range(2):
        lam = result.eigenvalues[mode_index]
        phi1, phi2 = result.mode_shapes[:, mode_index]
        expected_ratio = (STIFFNESS_1 + STIFFNESS_2 - lam * MASS_1) / STIFFNESS_2
        assert_allclose(phi2 / phi1, expected_ratio, rtol=1e-9)


def test_first_mode_is_in_phase_second_is_out_of_phase() -> None:
    """Physically: the lowest mode moves both masses together (in
    phase); the higher mode moves them in opposition (out of phase) --
    a standard qualitative check for a 2-DOF chain.
    """
    mass, stiffness = _system()
    result = natural_frequencies(stiffness, mass)

    mode_1 = result.mode_shapes[:, 0]
    mode_2 = result.mode_shapes[:, 1]
    assert mode_1[0] * mode_1[1] > 0  # same sign
    assert mode_2[0] * mode_2[1] < 0  # opposite sign
