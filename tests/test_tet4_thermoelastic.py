"""Tests confirming ThermoelasticMaterial3D (bound via at_temperature) plugs into the
TET4 small-strain dispatch (femtoolkit.analysis.nonlinear_elements) with zero changes
to that module -- the same NonlinearMaterial interface every material since Version 13
implements.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.nonlinear_elements import (
    initial_element_state,
    tet4_internal_force_and_tangent,
)
from femtoolkit.materials import LinearElastic3D, ThermoelasticMaterial3D
from femtoolkit.mesh import Node, Tet4Element3D


@pytest.fixture
def tet() -> Tet4Element3D:
    nodes = (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=0.0, y=1.0, z=0.0),
        Node(id=4, x=0.0, y=0.0, z=1.0),
    )
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    return Tet4Element3D(id=1, nodes=nodes, material=placeholder)


@pytest.fixture
def base_material() -> ThermoelasticMaterial3D:
    return ThermoelasticMaterial3D(
        youngs_modulus=200e9,
        poisson_ratio=0.3,
        thermal_expansion_coefficient=12e-6,
        reference_temperature=293.15,
        density=7850.0,
    )


def test_zero_displacement_at_reference_temperature_gives_zero_force(
    tet: Tet4Element3D, base_material: ThermoelasticMaterial3D
) -> None:
    material = base_material.at_temperature(293.15)
    committed = initial_element_state(tet, material)
    f_int, k_t, trial_state = tet4_internal_force_and_tangent(
        tet, material, np.zeros(12), committed
    )
    assert_allclose(f_int, np.zeros(12), atol=1e-6)
    assert_allclose(trial_state.states[0].stress, np.zeros(6), atol=1e-6)


def test_zero_displacement_at_elevated_temperature_gives_nonzero_thermal_force(
    tet: Tet4Element3D, base_material: ThermoelasticMaterial3D
) -> None:
    """Fully restrained (u=0) at T != T_ref: nonzero internal force from thermal stress."""
    material = base_material.at_temperature(393.15)
    committed = initial_element_state(tet, material)
    f_int, k_t, trial_state = tet4_internal_force_and_tangent(
        tet, material, np.zeros(12), committed
    )
    assert not np.allclose(f_int, np.zeros(12), atol=1.0)
    assert trial_state.states[0].stress[0] < 0.0  # compressive under restrained heating


def test_stiffness_uses_elastic_constants_at_the_material_temperature(
    tet: Tet4Element3D, base_material: ThermoelasticMaterial3D
) -> None:
    """K_t must equal the standard V*B^T@D@B, D evaluated at the material's temperature."""
    material = base_material.at_temperature(293.15)
    committed = initial_element_state(tet, material)
    displacements = np.array([0, 0, 0, 1e-3, 0, 0, 0, 1e-3, 0, 0, 0, 1e-3])
    _, k_t, _ = tet4_internal_force_and_tangent(tet, material, displacements, committed)

    d_matrix = base_material.constitutive_matrix_at(293.15)
    expected = tet.volume * tet.b_matrix.T @ d_matrix @ tet.b_matrix
    assert_allclose(k_t, expected, rtol=1e-10)


def test_thermal_strain_alone_gives_near_zero_stress(
    tet: Tet4Element3D, base_material: ThermoelasticMaterial3D
) -> None:
    """A displacement field matching pure free thermal expansion gives ~zero stress."""
    temperature = 393.15
    material = base_material.at_temperature(temperature)
    committed = initial_element_state(tet, material)

    alpha, delta_temperature = 12e-6, temperature - 293.15
    thermal_strain_normal = alpha * delta_temperature
    ref_coords = np.array([[n.x, n.y, n.z] for n in tet.nodes])
    displacements = (ref_coords * thermal_strain_normal).flatten()

    _, _, trial_state = tet4_internal_force_and_tangent(tet, material, displacements, committed)
    assert_allclose(trial_state.states[0].stress, np.zeros(6), atol=1.0)


def test_combined_mechanical_and_thermal_displacement(
    tet: Tet4Element3D, base_material: ThermoelasticMaterial3D
) -> None:
    temperature = 393.15
    material = base_material.at_temperature(temperature)
    committed = initial_element_state(tet, material)

    alpha, delta_temperature = 12e-6, temperature - 293.15
    thermal_strain_normal = alpha * delta_temperature
    ref_coords = np.array([[n.x, n.y, n.z] for n in tet.nodes])
    thermal_displacements = (ref_coords * thermal_strain_normal).flatten()
    extra_mechanical = np.array([0, 0, 0, 1e-4, 0, 0, 0, 0, 0, 0, 0, 0])
    displacements = thermal_displacements + extra_mechanical

    _, _, trial_state = tet4_internal_force_and_tangent(tet, material, displacements, committed)
    expected_mechanical_strain = np.array([1e-4, 0, 0, 0, 0, 0])
    expected_stress = base_material.constitutive_matrix_at(temperature) @ expected_mechanical_strain
    assert_allclose(trial_state.states[0].stress, expected_stress, atol=1.0)
