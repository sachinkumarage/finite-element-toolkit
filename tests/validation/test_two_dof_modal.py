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

from femtoolkit.analysis.modal import modal_analysis, natural_frequencies

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


# --- Modal response: participation factors and effective modal mass ---


def test_participation_factors_match_hand_derived_mass_normalized_ratio() -> None:
    """For M = I, the mass-normalized mode shape is just the
    unit-norm eigenvector, ``phi_mn = phi / sqrt(phi[0]^2 + phi[1]^2)``,
    and (since M = I, r = [1, 0]) the participation factor is exactly
    its first component, ``Gamma_i = phi_mn[0]``. Using the closed-form
    eigenvector ratio ``phi2/phi1 = (k1+k2-lambda_i*m1)/k2`` already
    validated in :func:`test_mode_shape_ratios_match_hand_derivation`,
    this gives a fully independent, hand-derivable expected value.
    """
    mass, stiffness = _system()
    r = np.array([1.0, 0.0])
    result = modal_analysis(stiffness, mass, direction=r)

    for mode_index in range(2):
        lam = result.eigenvalues[mode_index]
        ratio = (STIFFNESS_1 + STIFFNESS_2 - lam * MASS_1) / STIFFNESS_2
        expected_magnitude = 1.0 / math.sqrt(1.0 + ratio**2)
        assert_allclose(
            abs(result.participation_factors[mode_index]), expected_magnitude, rtol=1e-9
        )


def test_effective_modal_mass_sums_to_total_participating_mass() -> None:
    """Sum of effective modal mass over ALL modes must exactly equal
    the total mass participating in the requested direction,
    ``r^T * M * r`` -- an exact consequence of eigenbasis completeness,
    independent of this 2-DOF system's specific numbers.
    """
    mass, stiffness = _system()
    r = np.array([1.0, 0.0])
    result = modal_analysis(stiffness, mass, direction=r)

    total_participating_mass = r @ mass @ r
    assert_allclose(result.effective_modal_mass.sum(), total_participating_mass, rtol=1e-9)
    assert_allclose(result.cumulative_mass_ratio[-1], 1.0, rtol=1e-9)


def test_modal_response_reconstructs_static_displacement() -> None:
    """Static-limit validation via modal superposition: for a constant
    force pattern F = r * F0, the exact static displacement
    ``u = K^-1 @ F`` must equal the modal reconstruction
    ``u = sum_i(phi_i * Gamma_i * F0 / omega_i^2)`` using mass-normalized
    mode shapes -- the static limit of the general modal-superposition
    equation ``q_i = Gamma_i * F0 / omega_i^2`` for a constant load.
    """
    mass, stiffness = _system()
    force_magnitude = 5.0
    r = np.array([1.0, 0.0])
    result = modal_analysis(stiffness, mass, direction=r)

    static_displacement = np.linalg.solve(stiffness, r * force_magnitude)

    modal_displacement = np.zeros(2)
    for mode_index in range(2):
        phi = result.mass_normalized_mode_shapes[:, mode_index]
        gamma = result.participation_factors[mode_index]
        omega = result.angular_frequencies[mode_index]
        modal_displacement += phi * gamma * force_magnitude / omega**2

    assert_allclose(modal_displacement, static_displacement, rtol=1e-9)
