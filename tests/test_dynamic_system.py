"""Tests for DynamicSystem and build_dynamic_system."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, TranslationDOF
from femtoolkit.analysis.damping import RayleighDamping
from femtoolkit.analysis.dof import DOFMap
from femtoolkit.analysis.dynamic_system import DynamicSystem, build_dynamic_system
from femtoolkit.exceptions import InvalidAnalysisError, InvalidElementError, ValidationError
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D, Material
from femtoolkit.mesh import BarElement, Mesh, Node, create_quad_mesh
from femtoolkit.sections import CrossSection

WIDTH = 2.0
HEIGHT = 1.0
THICKNESS = 0.01
DENSITY = 7850.0


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(
        youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress", density=DENSITY
    )


@pytest.fixture
def mesh(material: LinearElastic2D):
    return create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=4, ny=2, material=material, thickness=THICKNESS
    )


@pytest.fixture
def domain() -> Rectangle:
    return Rectangle(width=WIDTH, height=HEIGHT)


@pytest.fixture
def left_boundary_conditions(mesh, domain: Rectangle):
    conditions = []
    for node in mesh.nodes_on_boundary(domain.boundary("left")):
        conditions.append(BoundaryCondition(node.id, TranslationDOF.X, 0.0))
        conditions.append(BoundaryCondition(node.id, TranslationDOF.Y, 0.0))
    return conditions


# --- DynamicSystem validation ---


def test_dynamic_system_validates_shapes() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    bcs = [BoundaryCondition(1, TranslationDOF.X, 0.0)]
    with pytest.raises(ValidationError):
        DynamicSystem(
            dof_map=dof_map,
            mass=np.eye(3),
            damping=np.eye(2),
            stiffness=np.eye(2),
            boundary_conditions=bcs,
        )


def test_dynamic_system_requires_boundary_conditions() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    with pytest.raises(ValidationError):
        DynamicSystem(
            dof_map=dof_map,
            mass=np.eye(2),
            damping=np.eye(2),
            stiffness=np.eye(2),
            boundary_conditions=[],
        )


def test_dynamic_system_rejects_duplicate_boundary_conditions() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    bcs = [
        BoundaryCondition(1, TranslationDOF.X, 0.0),
        BoundaryCondition(1, TranslationDOF.X, 0.1),
    ]
    with pytest.raises(ValidationError):
        DynamicSystem(
            dof_map=dof_map,
            mass=np.eye(2),
            damping=np.eye(2),
            stiffness=np.eye(2),
            boundary_conditions=bcs,
        )


# --- build_dynamic_system ---


def test_build_dynamic_system_shapes(mesh, left_boundary_conditions) -> None:
    system = build_dynamic_system(mesh, left_boundary_conditions)

    n = system.dof_map.total_dofs
    assert system.mass.shape == (n, n)
    assert system.stiffness.shape == (n, n)
    assert system.damping.shape == (n, n)


def test_build_dynamic_system_default_damping_is_zero(mesh, left_boundary_conditions) -> None:
    system = build_dynamic_system(mesh, left_boundary_conditions)
    assert_allclose(system.damping, np.zeros_like(system.mass))


def test_build_dynamic_system_applies_rayleigh_damping(mesh, left_boundary_conditions) -> None:
    damping = RayleighDamping(alpha=0.1, beta=0.001)
    system = build_dynamic_system(mesh, left_boundary_conditions, damping=damping)

    expected = damping.damping_matrix(system.mass, system.stiffness)
    assert_allclose(system.damping, expected)


def test_build_dynamic_system_mass_and_stiffness_share_dof_numbering(
    mesh, left_boundary_conditions
) -> None:
    system = build_dynamic_system(mesh, left_boundary_conditions)
    nonzero_stiffness = set(zip(*np.nonzero(system.stiffness), strict=True))
    nonzero_mass = set(zip(*np.nonzero(system.mass), strict=True))
    # Every stiffness-coupled DOF pair must also be mass-coupled (both
    # come from the same element connectivity and dof_map).
    assert nonzero_stiffness.issubset(nonzero_mass) or nonzero_mass.issubset(nonzero_stiffness)


def test_build_dynamic_system_total_mass_matches_physical_mass(
    mesh, left_boundary_conditions
) -> None:
    system = build_dynamic_system(mesh, left_boundary_conditions, mass_matrix_type="consistent")

    n = system.dof_map.total_dofs
    x_indices = [i for i in range(n) if i % 2 == 0]
    x_block = system.mass[np.ix_(x_indices, x_indices)]
    expected = DENSITY * (WIDTH * HEIGHT) * THICKNESS
    assert_allclose(x_block.sum(), expected)


def test_build_dynamic_system_lumped_total_mass_matches_physical_mass(
    mesh, left_boundary_conditions
) -> None:
    system = build_dynamic_system(mesh, left_boundary_conditions, mass_matrix_type="lumped")

    n = system.dof_map.total_dofs
    x_indices = [i for i in range(n) if i % 2 == 0]
    expected = DENSITY * (WIDTH * HEIGHT) * THICKNESS
    assert_allclose(np.diag(system.mass)[x_indices].sum(), expected)


def test_build_dynamic_system_rejects_non_mass_capable_elements() -> None:
    steel = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    section = CrossSection(area=0.01)
    mesh = Mesh()
    mesh.add_node(Node(1, 0.0, 0.0, 0.0))
    mesh.add_node(Node(2, 1.0, 0.0, 0.0))
    mesh.add_element(
        BarElement(
            id=1, nodes=(mesh.get_node(1), mesh.get_node(2)), material=steel, cross_section=section
        )
    )
    bcs = [BoundaryCondition(1, TranslationDOF.X, 0.0)]

    with pytest.raises(InvalidElementError):
        build_dynamic_system(mesh, bcs)


def test_build_dynamic_system_rejects_empty_mesh() -> None:
    with pytest.raises(InvalidAnalysisError):
        build_dynamic_system(Mesh(), [])
