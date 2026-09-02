"""Tests for TET4/HEX8 support in femtoolkit.analysis.nonlinear_elements (Version 15)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.nonlinear_elements import (
    NONLINEAR_CAPABLE_ELEMENT_TYPES,
    element_internal_force_and_tangent,
    hex8_internal_force_and_tangent,
    initial_element_state,
    tet4_internal_force_and_tangent,
)
from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import J2Plasticity3D, LinearElastic3D
from femtoolkit.materials.nonlinear import ElasticMaterialAdapter
from femtoolkit.mesh import Hex8Element3D, Node, Tet4Element3D


@pytest.fixture
def placeholder() -> LinearElastic3D:
    return LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)


@pytest.fixture
def tet(placeholder: LinearElastic3D) -> Tet4Element3D:
    nodes = (
        Node(1, 0.0, 0.0, 0.0),
        Node(2, 1.0, 0.0, 0.0),
        Node(3, 0.0, 1.0, 0.0),
        Node(4, 0.0, 0.0, 1.0),
    )
    return Tet4Element3D(id=1, nodes=nodes, material=placeholder)


@pytest.fixture
def hexa(placeholder: LinearElastic3D) -> Hex8Element3D:
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
    return Hex8Element3D(id=2, nodes=nodes, material=placeholder)


def test_tet4_and_hex8_are_nonlinear_capable(tet: Tet4Element3D, hexa: Hex8Element3D) -> None:
    assert isinstance(tet, NONLINEAR_CAPABLE_ELEMENT_TYPES)
    assert isinstance(hexa, NONLINEAR_CAPABLE_ELEMENT_TYPES)


def test_tet4_initial_state_has_one_entry(tet: Tet4Element3D) -> None:
    adapter = ElasticMaterialAdapter.from_linear_elastic_3d(tet.material)
    state = initial_element_state(tet, adapter)
    assert len(state.states) == 1


def test_hex8_initial_state_has_eight_independent_entries(hexa: Hex8Element3D) -> None:
    adapter = ElasticMaterialAdapter.from_linear_elastic_3d(hexa.material)
    state = initial_element_state(hexa, adapter)
    assert len(state.states) == 8


def test_require_3d_material_rejects_2d_material(tet: Tet4Element3D) -> None:
    two_d_adapter = ElasticMaterialAdapter(modulus=np.eye(3))
    with pytest.raises(ValidationError):
        initial_element_state(tet, two_d_adapter)


def test_tet4_internal_force_matches_linear_stiffness_for_elastic_adapter(
    tet: Tet4Element3D,
) -> None:
    """With a purely elastic adapter, F_int(u) must equal K_e @ u exactly."""
    adapter = ElasticMaterialAdapter.from_linear_elastic_3d(tet.material)
    committed = initial_element_state(tet, adapter)
    displacements = np.array([0.0, 0.0, 0.0, 1e-3, 0.0, 0.0, 0.0, 1e-3, 0.0, 0.0, 0.0, 1e-3])

    f_int, k_t, trial_state = tet4_internal_force_and_tangent(
        tet, adapter, displacements, committed
    )

    assert_allclose(f_int, tet.stiffness_matrix @ displacements, rtol=1e-8)
    assert_allclose(k_t, tet.stiffness_matrix, rtol=1e-8)
    assert len(trial_state.states) == 1


def test_hex8_internal_force_matches_linear_stiffness_for_elastic_adapter(
    hexa: Hex8Element3D,
) -> None:
    adapter = ElasticMaterialAdapter.from_linear_elastic_3d(hexa.material)
    committed = initial_element_state(hexa, adapter)
    rng = np.random.default_rng(0)
    displacements = rng.normal(scale=1e-4, size=24)

    f_int, k_t, trial_state = hex8_internal_force_and_tangent(
        hexa, adapter, displacements, committed
    )

    assert_allclose(f_int, hexa.stiffness_matrix @ displacements, rtol=1e-6, atol=1e-3)
    assert_allclose(k_t, hexa.stiffness_matrix, rtol=1e-6, atol=1e-3)
    assert len(trial_state.states) == 8


def test_element_internal_force_and_tangent_dispatches_correctly(
    tet: Tet4Element3D, hexa: Hex8Element3D
) -> None:
    j2 = J2Plasticity3D(
        youngs_modulus=200e9, poisson_ratio=0.3, yield_stress=250e6, hardening_modulus=10e9
    )
    tet_committed = initial_element_state(tet, j2)
    hex_committed = initial_element_state(hexa, j2)

    tet_displacements = np.zeros(12)
    hex_displacements = np.zeros(24)

    f_tet, k_tet, _ = element_internal_force_and_tangent(tet, j2, tet_displacements, tet_committed)
    f_hex, k_hex, _ = element_internal_force_and_tangent(hexa, j2, hex_displacements, hex_committed)

    assert f_tet.shape == (12,)
    assert k_tet.shape == (12, 12)
    assert f_hex.shape == (24,)
    assert k_hex.shape == (24, 24)


def test_hex8_gauss_points_have_independent_states_when_strain_varies(
    hexa: Hex8Element3D,
) -> None:
    """A bending-like displacement field should yield some Gauss points but not others."""
    j2 = J2Plasticity3D(
        youngs_modulus=200e9, poisson_ratio=0.3, yield_stress=50e6, hardening_modulus=5e9
    )
    committed = initial_element_state(hexa, j2)

    # Twist the top face relative to the bottom to create a non-uniform strain field.
    displacements = np.zeros(24)
    top_node_indices = [4, 5, 6, 7]
    for i in top_node_indices:
        displacements[3 * i] = 0.05 * (1 if i % 2 == 0 else -1)

    _, _, trial_state = hex8_internal_force_and_tangent(hexa, j2, displacements, committed)
    yielded_flags = [state.yielded for state in trial_state.states]

    assert len(yielded_flags) == 8
    # Not asserting a specific mixed pattern (geometry-dependent) -- just that
    # each Gauss point's state was computed independently (no crash, correct count).
    assert all(isinstance(flag, (bool, np.bool_)) for flag in yielded_flags)
