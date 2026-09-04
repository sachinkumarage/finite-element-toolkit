"""Validation: hyperelastic rigid-body motion must give zero energy/stress (spec section 17).

The consolidated, definitive rigid-body objectivity check for the Version
17 hyperelastic materials, at both the material level (F = R directly)
and the element level (TET4/HEX8 under a rigid rotation + translation) --
mirroring tests/validation/test_rigid_body_motion.py's treatment of
Version 16's finite-strain kinematics. This is explicitly called out as
mandatory in the Version 17 specification: for any objective hyperelastic
material, W(C) depends on F only through C = F^T F, and C is exactly
invariant under F -> Q @ F for any rotation Q, so *any* pure rotation must
give W = 0 and S = 0 to floating-point precision, regardless of how
nonlinear W is.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.geometric_nonlinear import (
    hex8_geometric_internal_force_and_tangent,
    initial_element_state,
    tet4_geometric_internal_force_and_tangent,
)
from femtoolkit.materials import LinearElastic3D, MooneyRivlin3D, NeoHookean3D
from femtoolkit.mesh import Hex8Element3D, Node, Tet4Element3D

_MATERIALS = [
    NeoHookean3D(youngs_modulus=5.0e6, poisson_ratio=0.45),
    MooneyRivlin3D(c10=0.4e6, c01=0.1e6, bulk_modulus=200e6),
]
_MATERIAL_IDS = ["neo_hookean", "mooney_rivlin"]


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random proper rotation via QR decomposition of a random matrix."""
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    # Fix signs so det(q) = +1 (a proper rotation, not a reflection).
    d = np.sign(np.diag(r))
    q = q * d
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_material_level_random_rotation_gives_zero_energy_and_stress(material, seed: int) -> None:
    rng = np.random.default_rng(seed)
    r = _random_rotation_matrix(rng)
    assert_allclose(r.T @ r, np.eye(3), atol=1e-10)
    assert np.linalg.det(r) == pytest.approx(1.0)

    assert material.strain_energy_density(r) == pytest.approx(0.0, abs=1e-3)
    assert_allclose(material.second_piola_kirchhoff_stress(r), np.zeros(6), atol=10.0)
    assert_allclose(material.cauchy_stress(r), np.zeros((3, 3)), atol=10.0)


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
def test_tet4_rigid_translation_and_rotation_gives_zero_force(material) -> None:
    nodes = (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=0.0, y=1.0, z=0.0),
        Node(id=4, x=0.0, y=0.0, z=1.0),
    )
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    tet = Tet4Element3D(id=1, nodes=nodes, material=placeholder)
    committed = initial_element_state(tet, material)

    theta = 0.7
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    translation = np.array([1.3, -0.4, 0.9])
    reference = np.array([[n.x, n.y, n.z] for n in nodes])
    current = reference @ rotation.T + translation
    displacements = (current - reference).flatten()

    f_int, _, trial_state = tet4_geometric_internal_force_and_tangent(
        tet, material, displacements, committed
    )
    assert_allclose(trial_state.states[0].strain, np.zeros(6), atol=1e-6)
    assert_allclose(f_int, np.zeros(12), atol=1e-3)


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
def test_hex8_rigid_translation_and_rotation_gives_zero_force(material) -> None:
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
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)
    committed = initial_element_state(hexa, material)

    theta = -0.5
    rotation = np.array(
        [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    )
    translation = np.array([0.5, -1.1, 0.3])
    reference = np.array(coords)
    current = reference @ rotation.T + translation
    displacements = (current - reference).flatten()

    f_int, _, trial_state = hex8_geometric_internal_force_and_tangent(
        hexa, material, displacements, committed
    )
    for state in trial_state.states:
        assert_allclose(state.strain, np.zeros(6), atol=1e-5)
    assert_allclose(f_int, np.zeros(24), atol=1e-2)
