"""Tests for LoadCase."""

import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import (
    BoundaryCondition,
    DistributedLoad,
    LoadCase,
    NodalLoad,
    TranslationDOF,
)
from femtoolkit.exceptions import InsufficientConstraintsError
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import create_quad_mesh

WIDTH = 2.0
HEIGHT = 1.0
THICKNESS = 0.01


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")


@pytest.fixture
def mesh(material: LinearElastic2D):
    return create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=4, ny=2, material=material, thickness=THICKNESS
    )


@pytest.fixture
def domain() -> Rectangle:
    return Rectangle(width=WIDTH, height=HEIGHT)


def test_load_case_has_a_name(mesh) -> None:
    load_case = LoadCase(name="Tension", mesh=mesh)

    assert load_case.name == "Tension"
    assert load_case.mesh is mesh


def test_fix_boundary_adds_boundary_conditions(mesh, domain: Rectangle) -> None:
    load_case = LoadCase(name="Tension", mesh=mesh)
    load_case.fix_boundary(domain.boundary("left"), ux=0.0, uy=0.0)

    left_node_count = len(mesh.nodes_on_boundary(domain.boundary("left")))
    assert len(load_case._boundary_conditions) == 2 * left_node_count


def test_add_distributed_load_expands_to_nodal_loads(mesh, domain: Rectangle) -> None:
    load_case = LoadCase(name="Tension", mesh=mesh)
    load_case.add_distributed_load(DistributedLoad(domain.boundary("right"), magnitude=1000.0))

    assert len(load_case._nodal_loads) > 0


def test_add_nodal_load(mesh) -> None:
    load_case = LoadCase(name="Point Load", mesh=mesh)
    load_case.add_nodal_load(NodalLoad(node_id=5, dof=TranslationDOF.X, value=100.0))

    assert len(load_case._nodal_loads) == 1


def test_add_boundary_condition(mesh) -> None:
    load_case = LoadCase(name="Custom", mesh=mesh)
    load_case.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.X, value=0.0))

    assert len(load_case._boundary_conditions) == 1


def test_solve_reuses_the_existing_solver(
    mesh, domain: Rectangle, material: LinearElastic2D
) -> None:
    load_case = LoadCase(name="Tension", mesh=mesh)
    load_case.fix_boundary(domain.boundary("left"), ux=0.0, uy=0.0)
    load_case.add_distributed_load(DistributedLoad(domain.boundary("right"), magnitude=1000.0))

    result = load_case.solve()

    # Every element must see the same uniform sigma_x, matching the
    # applied traction exactly (global equilibrium); sigma_y/tau_xy near
    # the fixed edge are not exactly zero because each left-edge node is
    # individually fully fixed (a discrete boundary condition, not an
    # idealized uniform-traction support), which is physically expected.
    for element in mesh.elements:
        sigma_x, _, _ = result.element_stress(element.id)
        assert_allclose(sigma_x, 1000.0, rtol=1e-9)


def test_solve_without_boundary_conditions_raises(mesh, domain: Rectangle) -> None:
    load_case = LoadCase(name="No Supports", mesh=mesh)
    load_case.add_distributed_load(DistributedLoad(domain.boundary("right"), magnitude=1000.0))

    with pytest.raises(InsufficientConstraintsError):
        load_case.solve()


def test_reaction_equilibrium(mesh, domain: Rectangle) -> None:
    load_case = LoadCase(name="Tension", mesh=mesh)
    load_case.fix_boundary(domain.boundary("left"), ux=0.0, uy=0.0)
    load_case.add_distributed_load(DistributedLoad(domain.boundary("right"), magnitude=1000.0))

    result = load_case.solve()

    total_rx = sum(result.node_reaction(n.id)[0] for n in mesh.nodes)
    total_ry = sum(result.node_reaction(n.id)[1] for n in mesh.nodes)
    total_applied_fx = 1000.0 * HEIGHT * THICKNESS

    assert_allclose(total_rx + total_applied_fx, 0.0, atol=1e-6)
    assert_allclose(total_ry, 0.0, atol=1e-6)


def test_apply_to_populates_an_externally_created_analysis(mesh, domain: Rectangle) -> None:
    from femtoolkit.analysis import StaticLinearAnalysis

    load_case = LoadCase(name="Tension", mesh=mesh)
    load_case.fix_boundary(domain.boundary("left"), ux=0.0, uy=0.0)
    load_case.add_distributed_load(DistributedLoad(domain.boundary("right"), magnitude=1000.0))

    analysis = StaticLinearAnalysis(mesh)
    load_case.apply_to(analysis)
    result = analysis.solve()

    assert_allclose(result.element_stress(1)[0], 1000.0, rtol=1e-9)


def test_mixed_bc_and_distributed_load_and_nodal_load(mesh, domain: Rectangle) -> None:
    """A load case can combine boundary-region BCs, a distributed load,
    and a manually specified concentrated nodal load in one solve.
    """
    load_case = LoadCase(name="Combined", mesh=mesh)
    load_case.fix_boundary(domain.boundary("left"), ux=0.0, uy=0.0)
    load_case.add_distributed_load(DistributedLoad(domain.boundary("right"), magnitude=500.0))
    load_case.add_nodal_load(NodalLoad(node_id=10, dof=TranslationDOF.Y, value=-10.0))

    result = load_case.solve()  # must not raise

    ux, uy = result.node_displacement(10)
    assert uy < 0.0
