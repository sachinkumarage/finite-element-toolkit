"""Tests confirming J2FiniteStrainPlasticity3D plugs into the TET4 geometric-nonlinear
dispatch (femtoolkit.analysis.geometric_nonlinear) with zero changes to that module --
the same NonlinearMaterial interface already used by SaintVenantKirchhoff3D,
J2Plasticity3D, NeoHookean3D, and MooneyRivlin3D, exercised here for the Version 18
finite-strain plastic material.
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
from femtoolkit.materials import J2FiniteStrainPlasticity3D, LinearElastic3D
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
def steel() -> J2FiniteStrainPlasticity3D:
    return J2FiniteStrainPlasticity3D(
        youngs_modulus=200e9, poisson_ratio=0.3, yield_stress=250e6, hardening_modulus=10e9
    )


def test_zero_displacement_gives_zero_force(
    tet: Tet4Element3D, steel: J2FiniteStrainPlasticity3D
) -> None:
    committed = initial_element_state(tet, steel)
    f_int, _, trial_state = tet4_geometric_internal_force_and_tangent(
        tet, steel, np.zeros(12), committed
    )
    assert_allclose(f_int, np.zeros(12), atol=1e-6)
    assert not trial_state.states[0].yielded


@pytest.mark.parametrize("translation", [(0.2, -0.1, 0.3), (1.0, 1.0, 1.0)])
def test_rigid_translation_gives_zero_force(
    tet: Tet4Element3D, steel: J2FiniteStrainPlasticity3D, translation: tuple[float, float, float]
) -> None:
    committed = initial_element_state(tet, steel)
    displacements = np.tile(translation, 4)
    f_int, _, trial_state = tet4_geometric_internal_force_and_tangent(
        tet, steel, displacements, committed
    )
    assert_allclose(f_int, np.zeros(12), atol=1e-4)
    assert not trial_state.states[0].yielded


@pytest.mark.parametrize("theta", [0.5, -1.0, np.pi / 3])
def test_rigid_rotation_gives_zero_force(
    tet: Tet4Element3D, steel: J2FiniteStrainPlasticity3D, theta: float
) -> None:
    """Mandatory objectivity test at the element level."""
    committed = initial_element_state(tet, steel)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    ref_coords = np.array([[n.x, n.y, n.z] for n in tet.nodes])
    current = ref_coords @ rotation.T
    displacements = (current - ref_coords).flatten()

    f_int, _, trial_state = tet4_geometric_internal_force_and_tangent(
        tet, steel, displacements, committed
    )
    assert_allclose(f_int, np.zeros(12), atol=1e-3)
    assert not trial_state.states[0].yielded


def test_moderate_stretch_stays_elastic(
    tet: Tet4Element3D, steel: J2FiniteStrainPlasticity3D
) -> None:
    committed = initial_element_state(tet, steel)
    lam = 1.0001
    ref_coords = np.array([[n.x, n.y, n.z] for n in tet.nodes])
    current = ref_coords.copy()
    current[:, 0] *= lam
    displacements = (current - ref_coords).flatten()

    _, _, trial_state = tet4_geometric_internal_force_and_tangent(
        tet, steel, displacements, committed
    )
    assert not trial_state.states[0].yielded


def test_large_stretch_yields(tet: Tet4Element3D, steel: J2FiniteStrainPlasticity3D) -> None:
    committed = initial_element_state(tet, steel)
    lam = 1.02
    ref_coords = np.array([[n.x, n.y, n.z] for n in tet.nodes])
    current = ref_coords.copy()
    current[:, 0] *= lam
    displacements = (current - ref_coords).flatten()

    _, _, trial_state = tet4_geometric_internal_force_and_tangent(
        tet, steel, displacements, committed
    )
    assert trial_state.states[0].yielded
    assert trial_state.states[0].hardening_variable > 0.0


def test_tangent_symmetric(tet: Tet4Element3D, steel: J2FiniteStrainPlasticity3D) -> None:
    committed = initial_element_state(tet, steel)
    displacements = np.array([0, 0, 0, 0.02, 0.005, -0.005, 0.005, 0.02, 0.0, -0.005, 0.0, 0.02])
    _, k_t, _ = tet4_geometric_internal_force_and_tangent(tet, steel, displacements, committed)
    assert_allclose(k_t, k_t.T, atol=5.0)


def test_material_and_geometric_split_sums_to_total(
    tet: Tet4Element3D, steel: J2FiniteStrainPlasticity3D
) -> None:
    committed = initial_element_state(tet, steel)
    displacements = np.array([0, 0, 0, 0.02, 0.005, -0.005, 0.005, 0.02, 0.0, -0.005, 0.0, 0.02])
    _, k_t, _ = tet4_geometric_internal_force_and_tangent(tet, steel, displacements, committed)
    k_material, k_geometric = tet4_geometric_stiffness_split(tet, steel, displacements, committed)
    assert_allclose(k_material + k_geometric, k_t, atol=1.0)


def test_element_inversion_raises(tet: Tet4Element3D, steel: J2FiniteStrainPlasticity3D) -> None:
    committed = initial_element_state(tet, steel)
    displacements = np.array([0, 0, 0, -5, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    with pytest.raises(InvalidDeformationGradientError):
        tet4_geometric_internal_force_and_tangent(tet, steel, displacements, committed)
