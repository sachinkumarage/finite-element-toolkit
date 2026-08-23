"""Tests for natural_frequencies, natural_frequencies_of_system, and rigid-body mode handling."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, TranslationDOF
from femtoolkit.analysis.dynamic_system import build_dynamic_system
from femtoolkit.analysis.modal import natural_frequencies, natural_frequencies_of_system
from femtoolkit.exceptions import EigenvalueComputationError, ValidationError
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import create_quad_mesh

# --- Pure eigenvalue math: 2-DOF spring-mass, analytical golden-ratio reference ---


@pytest.fixture
def two_dof_system():
    """Two unit masses, springs k1=k2=1 (m1 to wall, m1 to m2).

    K = [[2,-1],[-1,1]], M = I. Analytical eigenvalues:
    lambda = (3 +/- sqrt(5)) / 2.
    """
    k = np.array([[2.0, -1.0], [-1.0, 1.0]])
    m = np.eye(2)
    return k, m


def test_two_dof_eigenvalues_match_analytical(two_dof_system) -> None:
    k, m = two_dof_system
    result = natural_frequencies(k, m)

    expected = np.array([(3 - math.sqrt(5)) / 2, (3 + math.sqrt(5)) / 2])
    assert_allclose(result.eigenvalues, expected, rtol=1e-9)


def test_two_dof_angular_frequencies_are_sqrt_eigenvalues(two_dof_system) -> None:
    k, m = two_dof_system
    result = natural_frequencies(k, m)
    assert_allclose(result.angular_frequencies, np.sqrt(result.eigenvalues))


def test_two_dof_frequencies_hz_conversion(two_dof_system) -> None:
    k, m = two_dof_system
    result = natural_frequencies(k, m)
    assert_allclose(result.frequencies, result.angular_frequencies / (2 * math.pi))


def test_two_dof_mode_shapes_satisfy_eigenvalue_equation(two_dof_system) -> None:
    k, m = two_dof_system
    result = natural_frequencies(k, m)

    for i in range(2):
        phi = result.mode_shapes[:, i]
        lam = result.eigenvalues[i]
        assert_allclose(k @ phi, lam * (m @ phi), atol=1e-9)


def test_two_dof_mode_shapes_normalized_to_max_abs_one(two_dof_system) -> None:
    k, m = two_dof_system
    result = natural_frequencies(k, m)

    for i in range(2):
        assert_allclose(np.max(np.abs(result.mode_shapes[:, i])), 1.0)


def test_two_dof_no_rigid_body_modes_when_supported(two_dof_system) -> None:
    k, m = two_dof_system
    result = natural_frequencies(k, m)
    assert not result.is_rigid_body_mode.any()


# --- Rigid-body modes ---


def test_free_free_system_flags_rigid_body_mode() -> None:
    k = np.array([[1.0, -1.0], [-1.0, 1.0]])
    m = np.eye(2)
    result = natural_frequencies(k, m)

    assert_allclose(result.eigenvalues[0], 0.0, atol=1e-9)
    assert result.is_rigid_body_mode[0]
    assert not result.is_rigid_body_mode[1]


def test_negative_eigenvalue_beyond_tolerance_raises() -> None:
    """A genuinely negative eigenvalue (non-positive-semi-definite K) is
    not a rigid-body mode -- it signals an invalid model.
    """
    k = np.array([[-1.0, 0.0], [0.0, 1.0]])
    m = np.eye(2)
    with pytest.raises(EigenvalueComputationError):
        natural_frequencies(k, m)


# --- num_modes truncation ---


def test_num_modes_truncates_to_lowest(two_dof_system) -> None:
    k, m = two_dof_system
    result = natural_frequencies(k, m, num_modes=1)

    assert result.eigenvalues.shape == (1,)
    assert result.mode_shapes.shape == (2, 1)
    assert_allclose(result.eigenvalues[0], (3 - math.sqrt(5)) / 2, rtol=1e-9)


def test_num_modes_rejects_out_of_range(two_dof_system) -> None:
    k, m = two_dof_system
    with pytest.raises(ValidationError):
        natural_frequencies(k, m, num_modes=5)


def test_num_modes_rejects_zero(two_dof_system) -> None:
    k, m = two_dof_system
    with pytest.raises(ValidationError):
        natural_frequencies(k, m, num_modes=0)


# --- Input validation ---


def test_rejects_non_square_stiffness() -> None:
    k = np.zeros((2, 3))
    m = np.eye(2)
    with pytest.raises(ValidationError):
        natural_frequencies(k, m)


def test_rejects_mismatched_shapes() -> None:
    k = np.eye(2)
    m = np.eye(3)
    with pytest.raises(ValidationError):
        natural_frequencies(k, m)


def test_rejects_negative_rigid_body_tolerance() -> None:
    k = np.eye(2)
    m = np.eye(2)
    with pytest.raises(ValidationError):
        natural_frequencies(k, m, rigid_body_tolerance=-1e-6)


# --- natural_frequencies_of_system (FEM, with boundary conditions) ---


@pytest.fixture
def cantilever_dynamic_system():
    material = LinearElastic2D(
        youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress", density=7850.0
    )
    domain = Rectangle(width=2.0, height=1.0)
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01)

    bcs = []
    for node in mesh.nodes_on_boundary(domain.boundary("left")):
        bcs.append(BoundaryCondition(node.id, TranslationDOF.X, 0.0))
        bcs.append(BoundaryCondition(node.id, TranslationDOF.Y, 0.0))

    return build_dynamic_system(mesh, bcs)


def test_fem_natural_frequencies_are_finite_and_positive(cantilever_dynamic_system) -> None:
    result = natural_frequencies_of_system(cantilever_dynamic_system, num_modes=5)

    assert np.isfinite(result.frequencies).all()
    assert (result.frequencies > 0).all()


def test_fem_natural_frequencies_ascending(cantilever_dynamic_system) -> None:
    result = natural_frequencies_of_system(cantilever_dynamic_system, num_modes=5)
    assert np.all(np.diff(result.frequencies) >= 0)


def test_fem_no_rigid_body_modes_for_cantilever(cantilever_dynamic_system) -> None:
    """A fully fixed-left cantilever has no rigid-body freedom left."""
    result = natural_frequencies_of_system(cantilever_dynamic_system, num_modes=5)
    assert not result.is_rigid_body_mode.any()


def test_fem_mode_shapes_are_zero_at_constrained_dofs(cantilever_dynamic_system) -> None:
    result = natural_frequencies_of_system(cantilever_dynamic_system, num_modes=3)

    system = cantilever_dynamic_system
    for bc in system.boundary_conditions:
        index = system.dof_map.global_index(bc.node_id, bc.dof)
        assert_allclose(result.mode_shapes[index, :], np.zeros(3), atol=1e-12)


def test_fem_mode_shapes_full_dof_space_shape(cantilever_dynamic_system) -> None:
    result = natural_frequencies_of_system(cantilever_dynamic_system, num_modes=3)
    assert result.mode_shapes.shape == (cantilever_dynamic_system.dof_map.total_dofs, 3)


def test_natural_frequencies_of_system_all_constrained_raises() -> None:
    material = LinearElastic2D(
        youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress", density=7850.0
    )
    mesh = create_quad_mesh(width=1.0, height=1.0, nx=1, ny=1, material=material, thickness=0.01)
    bcs = [
        BoundaryCondition(node.id, dof, 0.0)
        for node in mesh.nodes
        for dof in (TranslationDOF.X, TranslationDOF.Y)
    ]
    system = build_dynamic_system(mesh, bcs)

    with pytest.raises(ValidationError):
        natural_frequencies_of_system(system)
