"""Tests confirming ThermoelasticMaterial3D plugs into the HEX8 small-strain dispatch
(femtoolkit.analysis.nonlinear_elements) with zero changes to that module -- mirrors
tests/test_tet4_thermoelastic.py for the 8-node solid element.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.nonlinear_elements import (
    hex8_internal_force_and_tangent,
    initial_element_state,
)
from femtoolkit.materials import LinearElastic3D, ThermoelasticMaterial3D
from femtoolkit.mesh import Hex8Element3D, Node

_UNIT_CUBE_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


@pytest.fixture
def hexa() -> Hex8Element3D:
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_UNIT_CUBE_COORDS))
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    return Hex8Element3D(id=1, nodes=nodes, material=placeholder)


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
    hexa: Hex8Element3D, base_material: ThermoelasticMaterial3D
) -> None:
    material = base_material.at_temperature(293.15)
    committed = initial_element_state(hexa, material)
    f_int, _, trial_state = hex8_internal_force_and_tangent(
        hexa, material, np.zeros(24), committed
    )
    assert_allclose(f_int, np.zeros(24), atol=1e-6)
    for state in trial_state.states:
        assert_allclose(state.stress, np.zeros(6), atol=1e-6)


def test_zero_displacement_at_elevated_temperature_gives_nonzero_thermal_force(
    hexa: Hex8Element3D, base_material: ThermoelasticMaterial3D
) -> None:
    material = base_material.at_temperature(393.15)
    committed = initial_element_state(hexa, material)
    f_int, _, trial_state = hex8_internal_force_and_tangent(
        hexa, material, np.zeros(24), committed
    )
    assert not np.allclose(f_int, np.zeros(24), atol=1.0)
    for state in trial_state.states:
        assert state.stress[0] < 0.0


def test_thermal_strain_alone_gives_near_zero_stress_at_every_gauss_point(
    hexa: Hex8Element3D, base_material: ThermoelasticMaterial3D
) -> None:
    temperature = 393.15
    material = base_material.at_temperature(temperature)
    committed = initial_element_state(hexa, material)

    alpha, delta_temperature = 12e-6, temperature - 293.15
    thermal_strain_normal = alpha * delta_temperature
    ref_coords = np.array(_UNIT_CUBE_COORDS)
    displacements = (ref_coords * thermal_strain_normal).flatten()

    _, _, trial_state = hex8_internal_force_and_tangent(hexa, material, displacements, committed)
    for state in trial_state.states:
        assert_allclose(state.stress, np.zeros(6), atol=1.0)


def test_stiffness_uses_elastic_constants_at_the_material_temperature(
    hexa: Hex8Element3D, base_material: ThermoelasticMaterial3D
) -> None:
    material = base_material.at_temperature(293.15)
    committed = initial_element_state(hexa, material)
    rng = np.random.default_rng(0)
    displacements = rng.normal(scale=1e-4, size=24)
    _, k_t, _ = hex8_internal_force_and_tangent(hexa, material, displacements, committed)
    assert_allclose(k_t, hexa.stiffness_matrix, rtol=1e-6, atol=1e-3)


def test_gauss_points_have_independent_states_under_nonuniform_displacement(
    hexa: Hex8Element3D, base_material: ThermoelasticMaterial3D
) -> None:
    material = base_material.at_temperature(393.15)
    committed = initial_element_state(hexa, material)
    displacements = np.zeros(24)
    for i in (4, 5, 6, 7):
        displacements[3 * i] = 0.001 * (1 if i % 2 == 0 else -1)

    _, _, trial_state = hex8_internal_force_and_tangent(hexa, material, displacements, committed)
    stresses = [tuple(np.round(s.stress, 6)) for s in trial_state.states]
    assert len(set(stresses)) > 1
