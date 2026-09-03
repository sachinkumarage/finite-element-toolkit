"""Validation: rigid-body motion must produce zero strain/stress (spec section 23-24).

The single most important correctness check for Version 16's finite-strain
kinematics: Green-Lagrange strain is *objective* by mathematical
construction (see femtoolkit.continuum.deformation's module docstring for
why), so a pure rigid-body translation, rotation, or combination of both
must give E = 0 (and therefore S = 0, F_int = 0) to floating-point
precision -- for every geometrically nonlinear element type (truss, TET4,
HEX8). This file is the consolidated, definitive version of the same
check exercised piecemeal in tests/test_geometric_truss.py,
tests/test_geometric_tet4.py, and tests/test_geometric_hex8.py.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.geometric_nonlinear import (
    hex8_geometric_internal_force_and_tangent,
    initial_element_state,
    tet4_geometric_internal_force_and_tangent,
    truss_geometric_internal_force_and_tangent,
)
from femtoolkit.materials import Material, SaintVenantKirchhoff1D, SaintVenantKirchhoff3D
from femtoolkit.mesh import Hex8Element3D, Node, Tet4Element3D, TrussElement2D
from femtoolkit.sections import CrossSection

_MOTIONS_2D = {
    "translation": (np.array([1.7, -0.9]), 0.0),
    "rotation": (np.array([0.0, 0.0]), 1.05),
    "translation_and_rotation": (np.array([-0.6, 2.1]), -0.8),
}

_MOTIONS_3D = {
    "translation": (np.array([1.3, -0.4, 0.9]), None),
    "rotation": (np.array([0.0, 0.0, 0.0]), 0.95),
    "translation_and_rotation": (np.array([0.5, -1.1, 0.3]), -0.6),
}


def _rotation_matrix_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("motion_name", list(_MOTIONS_2D))
def test_truss_rigid_motion(motion_name: str) -> None:
    translation, theta = _MOTIONS_2D[motion_name]
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.5, y=1.5, z=0.0)
    placeholder = Material(name="steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    truss = TrussElement2D(
        id=1, nodes=(node_1, node_2), material=placeholder, cross_section=CrossSection(area=0.002)
    )
    material = SaintVenantKirchhoff1D(youngs_modulus=200e9)
    committed = initial_element_state(truss, material)

    reference = np.array([[node_1.x, node_1.y], [node_2.x, node_2.y]])
    rotation_2d = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    current = reference @ rotation_2d.T + translation
    displacements = (current - reference).flatten()

    f_int, _, trial_state = truss_geometric_internal_force_and_tangent(
        truss, material, displacements, committed
    )
    assert trial_state.states[0].strain == pytest.approx(0.0, abs=1e-10)
    assert_allclose(f_int, np.zeros(4), atol=1e-5)


@pytest.mark.parametrize("motion_name", list(_MOTIONS_3D))
def test_tet4_rigid_motion(motion_name: str) -> None:
    translation, theta = _MOTIONS_3D[motion_name]
    nodes = (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=0.0, y=1.0, z=0.0),
        Node(id=4, x=0.0, y=0.0, z=1.0),
    )
    from femtoolkit.materials import LinearElastic3D

    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    tet = Tet4Element3D(id=1, nodes=nodes, material=placeholder)
    material = SaintVenantKirchhoff3D(youngs_modulus=200e9, poisson_ratio=0.3)
    committed = initial_element_state(tet, material)

    reference = np.array([[n.x, n.y, n.z] for n in nodes])
    rotation = _rotation_matrix_z(theta) if theta is not None else np.eye(3)
    current = reference @ rotation.T + translation
    displacements = (current - reference).flatten()

    f_int, _, trial_state = tet4_geometric_internal_force_and_tangent(
        tet, material, displacements, committed
    )
    assert_allclose(trial_state.states[0].strain, np.zeros(6), atol=1e-8)
    assert_allclose(f_int, np.zeros(12), atol=1e-4)


@pytest.mark.parametrize("motion_name", list(_MOTIONS_3D))
def test_hex8_rigid_motion(motion_name: str) -> None:
    translation, theta = _MOTIONS_3D[motion_name]
    coords = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ]
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(coords))
    from femtoolkit.materials import LinearElastic3D

    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)
    material = SaintVenantKirchhoff3D(youngs_modulus=200e9, poisson_ratio=0.3)
    committed = initial_element_state(hexa, material)

    reference = np.array(coords)
    rotation = _rotation_matrix_z(theta) if theta is not None else np.eye(3)
    current = reference @ rotation.T + translation
    displacements = (current - reference).flatten()

    f_int, _, trial_state = hex8_geometric_internal_force_and_tangent(
        hexa, material, displacements, committed
    )
    for state in trial_state.states:
        assert_allclose(state.strain, np.zeros(6), atol=1e-7)
    assert_allclose(f_int, np.zeros(24), atol=1e-4)
