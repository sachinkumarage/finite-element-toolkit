"""Tests for LoadCombination."""

import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import (
    BoundaryCondition,
    DistributedLoad,
    LoadCase,
    LoadCombination,
    NodalLoad,
    TranslationDOF,
)
from femtoolkit.exceptions import ValidationError
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


def test_combination_requires_at_least_one_load_case() -> None:
    with pytest.raises(ValidationError):
        LoadCombination(name="Empty", factors={})


def test_combination_rejects_empty_name(mesh) -> None:
    load_case = LoadCase(name="Dead", mesh=mesh)
    with pytest.raises(ValidationError):
        LoadCombination(name="", factors={load_case: 1.0})


def test_combination_rejects_non_finite_factor(mesh) -> None:
    load_case = LoadCase(name="Dead", mesh=mesh)
    with pytest.raises(ValidationError):
        LoadCombination(name="Bad", factors={load_case: float("nan")})


def test_resolved_nodal_loads_are_scaled_by_factor(mesh, domain: Rectangle) -> None:
    dead = LoadCase(name="Dead", mesh=mesh)
    dead.add_nodal_load(NodalLoad(node_id=10, dof=TranslationDOF.Y, value=-100.0))
    live = LoadCase(name="Live", mesh=mesh)
    live.add_nodal_load(NodalLoad(node_id=10, dof=TranslationDOF.Y, value=-200.0))

    combo = LoadCombination(name="Ultimate", factors={dead: 1.2, live: 1.6})
    loads = combo.resolved_nodal_loads(mesh)

    values = [load.value for load in loads]
    assert -120.0 in values
    assert -320.0 in values


def test_resolved_boundary_conditions_are_unioned(mesh, domain: Rectangle) -> None:
    dead = LoadCase(name="Dead", mesh=mesh)
    dead.fix_boundary(domain.boundary("left"), ux=0.0, uy=0.0)
    live = LoadCase(name="Live", mesh=mesh)
    live.add_nodal_load(NodalLoad(node_id=10, dof=TranslationDOF.Y, value=-100.0))

    combo = LoadCombination(name="Ultimate", factors={dead: 1.2, live: 1.6})
    conditions = combo.resolved_boundary_conditions(mesh)

    left_node_count = len(mesh.nodes_on_boundary(domain.boundary("left")))
    assert len(conditions) == 2 * left_node_count


def test_conflicting_boundary_conditions_raise(mesh) -> None:
    case_a = LoadCase(name="A", mesh=mesh)
    case_a.add_boundary_condition(BoundaryCondition(1, TranslationDOF.X, 0.0))
    case_b = LoadCase(name="B", mesh=mesh)
    case_b.add_boundary_condition(BoundaryCondition(1, TranslationDOF.X, 0.05))

    combo = LoadCombination(name="Conflict", factors={case_a: 1.0, case_b: 1.0})

    with pytest.raises(ValidationError):
        combo.resolved_boundary_conditions(mesh)


def test_identical_boundary_conditions_do_not_conflict(mesh) -> None:
    case_a = LoadCase(name="A", mesh=mesh)
    case_a.add_boundary_condition(BoundaryCondition(1, TranslationDOF.X, 0.0))
    case_b = LoadCase(name="B", mesh=mesh)
    case_b.add_boundary_condition(BoundaryCondition(1, TranslationDOF.X, 0.0))

    combo = LoadCombination(name="Agree", factors={case_a: 1.0, case_b: 1.0})
    conditions = combo.resolved_boundary_conditions(mesh)

    assert len(conditions) == 1


def test_solve_matches_manual_superposition(mesh, domain: Rectangle) -> None:
    dead = LoadCase(name="Dead", mesh=mesh)
    dead.fix_boundary(domain.boundary("left"), ux=0.0, uy=0.0)
    dead.add_nodal_load(NodalLoad(node_id=15, dof=TranslationDOF.Y, value=-1000.0))

    live = LoadCase(name="Live", mesh=mesh)
    live.fix_boundary(domain.boundary("left"), ux=0.0, uy=0.0)
    live.add_distributed_load(DistributedLoad(domain.boundary("right"), magnitude=2000.0))

    combo = LoadCombination(name="Ultimate", factors={dead: 1.2, live: 1.6})
    result = combo.solve(mesh)

    dead_result = dead.solve()
    live_result = live.solve()

    for node in mesh.nodes:
        combined_ux, combined_uy = result.node_displacement(node.id)
        dead_ux, dead_uy = dead_result.node_displacement(node.id)
        live_ux, live_uy = live_result.node_displacement(node.id)
        assert_allclose(combined_ux, 1.2 * dead_ux + 1.6 * live_ux, atol=1e-12)
        assert_allclose(combined_uy, 1.2 * dead_uy + 1.6 * live_uy, atol=1e-12)


def test_solve_without_mesh_infers_from_bound_load_case(domain: Rectangle, mesh) -> None:
    dead = LoadCase(name="Dead", mesh=mesh)
    dead.fix_boundary(domain.boundary("left"), ux=0.0, uy=0.0)
    dead.add_nodal_load(NodalLoad(node_id=15, dof=TranslationDOF.Y, value=-1000.0))

    combo = LoadCombination(name="Solo", factors={dead: 1.0})
    result = combo.solve()  # no mesh passed; inferred from `dead.mesh`

    assert result is not None


def test_solve_without_any_mesh_raises() -> None:
    unbound = LoadCase(name="Unbound")
    unbound.add_nodal_load(NodalLoad(node_id=1, dof=TranslationDOF.X, value=10.0))
    combo = LoadCombination(name="NoMesh", factors={unbound: 1.0})

    with pytest.raises(ValidationError):
        combo.solve()
