"""Tests confirming J2FiniteStrainPlasticity3D plugs into the HEX8 geometric-nonlinear
dispatch (femtoolkit.analysis.geometric_nonlinear) with zero changes to that module --
mirrors tests/test_geometric_tet4_finite_strain_plasticity.py for the 8-node solid element.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.geometric_nonlinear import (
    hex8_geometric_internal_force_and_tangent,
    hex8_geometric_stiffness_split,
    initial_element_state,
)
from femtoolkit.materials import J2FiniteStrainPlasticity3D, LinearElastic3D
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
def steel() -> J2FiniteStrainPlasticity3D:
    return J2FiniteStrainPlasticity3D(
        youngs_modulus=200e9, poisson_ratio=0.3, yield_stress=250e6, hardening_modulus=10e9
    )


def test_zero_displacement_gives_zero_force(
    hexa: Hex8Element3D, steel: J2FiniteStrainPlasticity3D
) -> None:
    committed = initial_element_state(hexa, steel)
    assert len(committed.states) == 8
    f_int, _, trial_state = hex8_geometric_internal_force_and_tangent(
        hexa, steel, np.zeros(24), committed
    )
    assert_allclose(f_int, np.zeros(24), atol=1e-5)
    for state in trial_state.states:
        assert not state.yielded


@pytest.mark.parametrize("translation", [(0.3, -0.1, 0.2), (1.5, 0.0, -0.5)])
def test_rigid_translation_gives_zero_force(
    hexa: Hex8Element3D, steel: J2FiniteStrainPlasticity3D, translation: tuple[float, float, float]
) -> None:
    committed = initial_element_state(hexa, steel)
    displacements = np.tile(translation, 8)
    f_int, _, trial_state = hex8_geometric_internal_force_and_tangent(
        hexa, steel, displacements, committed
    )
    assert_allclose(f_int, np.zeros(24), atol=1e-3)
    for state in trial_state.states:
        assert not state.yielded


@pytest.mark.parametrize("theta", [0.4, -0.9, np.pi / 4])
def test_rigid_rotation_gives_zero_force(
    hexa: Hex8Element3D, steel: J2FiniteStrainPlasticity3D, theta: float
) -> None:
    """Mandatory objectivity test at the element level."""
    committed = initial_element_state(hexa, steel)
    rotation = np.array(
        [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    )
    ref_coords = np.array(_UNIT_CUBE_COORDS)
    current = ref_coords @ rotation.T
    displacements = (current - ref_coords).flatten()

    f_int, _, trial_state = hex8_geometric_internal_force_and_tangent(
        hexa, steel, displacements, committed
    )
    assert_allclose(f_int, np.zeros(24), atol=1e-2)
    for state in trial_state.states:
        assert not state.yielded


def test_large_stretch_yields_at_every_gauss_point(
    hexa: Hex8Element3D, steel: J2FiniteStrainPlasticity3D
) -> None:
    committed = initial_element_state(hexa, steel)
    lam = 1.02
    ref_coords = np.array(_UNIT_CUBE_COORDS)
    current = ref_coords.copy()
    current[:, 0] *= lam
    displacements = (current - ref_coords).flatten()

    _, _, trial_state = hex8_geometric_internal_force_and_tangent(
        hexa, steel, displacements, committed
    )
    for state in trial_state.states:
        assert state.yielded


def test_tangent_symmetric(hexa: Hex8Element3D, steel: J2FiniteStrainPlasticity3D) -> None:
    committed = initial_element_state(hexa, steel)
    rng = np.random.default_rng(0)
    displacements = rng.normal(scale=0.02, size=24)
    _, k_t, _ = hex8_geometric_internal_force_and_tangent(hexa, steel, displacements, committed)
    assert_allclose(k_t, k_t.T, atol=1e2)


def test_material_and_geometric_split_sums_to_total(
    hexa: Hex8Element3D, steel: J2FiniteStrainPlasticity3D
) -> None:
    committed = initial_element_state(hexa, steel)
    rng = np.random.default_rng(1)
    displacements = rng.normal(scale=0.02, size=24)
    _, k_t, _ = hex8_geometric_internal_force_and_tangent(hexa, steel, displacements, committed)
    k_material, k_geometric = hex8_geometric_stiffness_split(hexa, steel, displacements, committed)
    assert_allclose(k_material + k_geometric, k_t, atol=1.0)


def test_gauss_points_have_independent_states_under_nonuniform_displacement(
    hexa: Hex8Element3D, steel: J2FiniteStrainPlasticity3D
) -> None:
    committed = initial_element_state(hexa, steel)
    displacements = np.zeros(24)
    for i in (4, 5, 6, 7):
        displacements[3 * i] = 0.05 * (1 if i % 2 == 0 else -1)

    _, _, trial_state = hex8_geometric_internal_force_and_tangent(
        hexa, steel, displacements, committed
    )
    stresses = [tuple(np.round(s.stress, 3)) for s in trial_state.states]
    assert len(set(stresses)) > 1
