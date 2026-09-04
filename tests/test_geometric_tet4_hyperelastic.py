"""Tests confirming NeoHookean3D/MooneyRivlin3D plug into the TET4 geometric-nonlinear
dispatch (femtoolkit.analysis.geometric_nonlinear) with zero changes to that module --
the same NonlinearMaterial interface already used by SaintVenantKirchhoff3D and
J2Plasticity3D (Versions 15/16), exercised here for the two new Version 17 materials.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.geometric_nonlinear import (
    initial_element_state,
    tet4_geometric_internal_force_and_tangent,
    tet4_geometric_stiffness_split,
)
from femtoolkit.exceptions import InvalidDeformationGradientError
from femtoolkit.materials import LinearElastic3D, MooneyRivlin3D, NeoHookean3D
from femtoolkit.mesh import Node, Tet4Element3D

_MATERIALS = [
    NeoHookean3D(youngs_modulus=5.0e6, poisson_ratio=0.45),
    MooneyRivlin3D(c10=0.4e6, c01=0.1e6, bulk_modulus=200e6),
]
_MATERIAL_IDS = ["neo_hookean", "mooney_rivlin"]


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


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
def test_zero_displacement_gives_zero_force(tet: Tet4Element3D, material) -> None:
    # atol=1e-4: MooneyRivlin3D's stress relies on the HyperelasticMaterial base
    # class's numerical-differentiation-of-energy default, which leaves O(1e-5)
    # central-difference noise at exact zero strain (see hyperelastic.py's
    # _FINITE_DIFFERENCE_ABSOLUTE_FLOOR docstring); NeoHookean3D's analytical
    # stress override has no such noise and passes at machine precision either way.
    committed = initial_element_state(tet, material)
    f_int, _, trial_state = tet4_geometric_internal_force_and_tangent(
        tet, material, np.zeros(12), committed
    )
    assert_allclose(f_int, np.zeros(12), atol=1e-4)
    assert_allclose(trial_state.states[0].strain, np.zeros(6), atol=1e-12)


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
@pytest.mark.parametrize("translation", [(0.2, -0.1, 0.3), (1.0, 1.0, 1.0)])
def test_rigid_translation_gives_zero_force(
    tet: Tet4Element3D, material, translation: tuple[float, float, float]
) -> None:
    committed = initial_element_state(tet, material)
    displacements = np.tile(translation, 4)
    f_int, _, trial_state = tet4_geometric_internal_force_and_tangent(
        tet, material, displacements, committed
    )
    assert_allclose(f_int, np.zeros(12), atol=1e-4)
    assert_allclose(trial_state.states[0].strain, np.zeros(6), atol=1e-9)


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
@pytest.mark.parametrize("theta", [0.5, -1.0, np.pi / 3])
def test_rigid_rotation_gives_zero_force(tet: Tet4Element3D, material, theta: float) -> None:
    """Mandatory objectivity test at the element level."""
    committed = initial_element_state(tet, material)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    ref_coords = np.array([[n.x, n.y, n.z] for n in tet.nodes])
    current = ref_coords @ rotation.T
    displacements = (current - ref_coords).flatten()

    f_int, _, trial_state = tet4_geometric_internal_force_and_tangent(
        tet, material, displacements, committed
    )
    assert_allclose(f_int, np.zeros(12), atol=1e-3)
    assert_allclose(trial_state.states[0].strain, np.zeros(6), atol=1e-7)


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
def test_uniform_uniaxial_stretch_gives_expected_strain(tet: Tet4Element3D, material) -> None:
    committed = initial_element_state(tet, material)
    lam = 1.2
    ref_coords = np.array([[n.x, n.y, n.z] for n in tet.nodes])
    current = ref_coords.copy()
    current[:, 0] *= lam
    displacements = (current - ref_coords).flatten()

    _, _, trial_state = tet4_geometric_internal_force_and_tangent(
        tet, material, displacements, committed
    )
    expected_exx = 0.5 * (lam**2 - 1.0)
    assert trial_state.states[0].strain[0] == pytest.approx(expected_exx)
    assert_allclose(trial_state.states[0].strain[1:], 0.0, atol=1e-12)


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
def test_tangent_symmetric(tet: Tet4Element3D, material) -> None:
    committed = initial_element_state(tet, material)
    displacements = np.array([0, 0, 0, 0.05, 0.02, -0.01, 0.01, 0.06, 0.0, -0.02, 0.0, 0.05])
    _, k_t, _ = tet4_geometric_internal_force_and_tangent(tet, material, displacements, committed)
    assert_allclose(k_t, k_t.T, atol=5.0)


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
def test_material_and_geometric_split_sums_to_total(tet: Tet4Element3D, material) -> None:
    committed = initial_element_state(tet, material)
    displacements = np.array([0, 0, 0, 0.05, 0.02, -0.01, 0.01, 0.06, 0.0, -0.02, 0.0, 0.05])
    _, k_t, _ = tet4_geometric_internal_force_and_tangent(tet, material, displacements, committed)
    k_material, k_geometric = tet4_geometric_stiffness_split(
        tet, material, displacements, committed
    )
    assert_allclose(k_material + k_geometric, k_t, atol=1.0)


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
def test_element_inversion_raises(tet: Tet4Element3D, material) -> None:
    committed = initial_element_state(tet, material)
    displacements = np.array([0, 0, 0, -5, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    with pytest.raises(InvalidDeformationGradientError):
        tet4_geometric_internal_force_and_tangent(tet, material, displacements, committed)
