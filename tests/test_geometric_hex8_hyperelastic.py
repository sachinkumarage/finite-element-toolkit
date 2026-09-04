"""Tests confirming NeoHookean3D/MooneyRivlin3D plug into the HEX8 geometric-nonlinear
dispatch (femtoolkit.analysis.geometric_nonlinear) with zero changes to that module --
mirrors tests/test_geometric_tet4_hyperelastic.py for the 8-node solid element.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.geometric_nonlinear import (
    hex8_geometric_internal_force_and_tangent,
    hex8_geometric_stiffness_split,
    initial_element_state,
)
from femtoolkit.materials import LinearElastic3D, MooneyRivlin3D, NeoHookean3D
from femtoolkit.mesh import Hex8Element3D, Node

_UNIT_CUBE_COORDS = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0),
]

_MATERIALS = [
    NeoHookean3D(youngs_modulus=5.0e6, poisson_ratio=0.45),
    MooneyRivlin3D(c10=0.4e6, c01=0.1e6, bulk_modulus=200e6),
]
_MATERIAL_IDS = ["neo_hookean", "mooney_rivlin"]


@pytest.fixture
def hexa() -> Hex8Element3D:
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_UNIT_CUBE_COORDS))
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    return Hex8Element3D(id=1, nodes=nodes, material=placeholder)


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
def test_zero_displacement_gives_zero_force(hexa: Hex8Element3D, material) -> None:
    committed = initial_element_state(hexa, material)
    assert len(committed.states) == 8
    f_int, _, trial_state = hex8_geometric_internal_force_and_tangent(
        hexa, material, np.zeros(24), committed
    )
    # atol=1e-3: MooneyRivlin3D's numerical-differentiation-of-energy stress default
    # leaves O(1e-5) noise per Gauss point, integrated across 8 points; see
    # tests/test_geometric_tet4_hyperelastic.py for the same effect at TET4 scale.
    assert_allclose(f_int, np.zeros(24), atol=1e-3)
    for state in trial_state.states:
        assert_allclose(state.strain, np.zeros(6), atol=1e-12)


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
def test_uniform_uniaxial_stretch_gives_expected_strain_at_every_gauss_point(
    hexa: Hex8Element3D, material
) -> None:
    committed = initial_element_state(hexa, material)
    lam = 1.15
    ref_coords = np.array(_UNIT_CUBE_COORDS)
    current = ref_coords.copy()
    current[:, 0] *= lam
    displacements = (current - ref_coords).flatten()

    _, _, trial_state = hex8_geometric_internal_force_and_tangent(
        hexa, material, displacements, committed
    )
    expected_exx = 0.5 * (lam**2 - 1.0)
    for state in trial_state.states:
        assert state.strain[0] == pytest.approx(expected_exx, rel=1e-8)
        assert_allclose(state.strain[1:], 0.0, atol=1e-10)


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
@pytest.mark.parametrize("translation", [(0.3, -0.1, 0.2), (1.5, 0.0, -0.5)])
def test_rigid_translation_gives_zero_force(
    hexa: Hex8Element3D, material, translation: tuple[float, float, float]
) -> None:
    committed = initial_element_state(hexa, material)
    displacements = np.tile(translation, 8)
    f_int, _, trial_state = hex8_geometric_internal_force_and_tangent(
        hexa, material, displacements, committed
    )
    assert_allclose(f_int, np.zeros(24), atol=1e-3)
    for state in trial_state.states:
        assert_allclose(state.strain, np.zeros(6), atol=1e-8)


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
@pytest.mark.parametrize("theta", [0.4, -0.9, np.pi / 4])
def test_rigid_rotation_gives_zero_force(hexa: Hex8Element3D, material, theta: float) -> None:
    """Mandatory objectivity test at the element level."""
    committed = initial_element_state(hexa, material)
    rotation = np.array(
        [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    )
    ref_coords = np.array(_UNIT_CUBE_COORDS)
    current = ref_coords @ rotation.T
    displacements = (current - ref_coords).flatten()

    f_int, _, trial_state = hex8_geometric_internal_force_and_tangent(
        hexa, material, displacements, committed
    )
    assert_allclose(f_int, np.zeros(24), atol=1e-2)
    for state in trial_state.states:
        assert_allclose(state.strain, np.zeros(6), atol=1e-6)


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
def test_tangent_symmetric(hexa: Hex8Element3D, material) -> None:
    committed = initial_element_state(hexa, material)
    rng = np.random.default_rng(0)
    displacements = rng.normal(scale=0.03, size=24)
    _, k_t, _ = hex8_geometric_internal_force_and_tangent(hexa, material, displacements, committed)
    assert_allclose(k_t, k_t.T, atol=1e2)


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
def test_material_and_geometric_split_sums_to_total(hexa: Hex8Element3D, material) -> None:
    committed = initial_element_state(hexa, material)
    rng = np.random.default_rng(1)
    displacements = rng.normal(scale=0.03, size=24)
    _, k_t, _ = hex8_geometric_internal_force_and_tangent(hexa, material, displacements, committed)
    k_material, k_geometric = hex8_geometric_stiffness_split(
        hexa, material, displacements, committed
    )
    assert_allclose(k_material + k_geometric, k_t, atol=1.0)


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
def test_gauss_points_have_independent_states_under_nonuniform_displacement(
    hexa: Hex8Element3D, material
) -> None:
    committed = initial_element_state(hexa, material)
    displacements = np.zeros(24)
    for i in (4, 5, 6, 7):
        displacements[3 * i] = 0.08 * (1 if i % 2 == 0 else -1)

    _, _, trial_state = hex8_geometric_internal_force_and_tangent(
        hexa, material, displacements, committed
    )
    strains = [tuple(np.round(s.strain, 6)) for s in trial_state.states]
    assert len(set(strains)) > 1
