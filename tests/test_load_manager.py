"""Tests for LoadManager and ResultSet."""

import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import (
    DistributedLoad,
    GravityLoad,
    LoadCase,
    LoadCombination,
    LoadManager,
    NodalLoad,
    TranslationDOF,
)
from femtoolkit.exceptions import EntityNotFoundError, ValidationError
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import create_quad_mesh

WIDTH = 2.0
HEIGHT = 1.0
THICKNESS = 0.01


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(
        youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress", density=7850.0
    )


@pytest.fixture
def mesh(material: LinearElastic2D):
    return create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=4, ny=2, material=material, thickness=THICKNESS
    )


@pytest.fixture
def domain() -> Rectangle:
    return Rectangle(width=WIDTH, height=HEIGHT)


def _dead_and_live(domain: Rectangle) -> tuple[LoadCase, LoadCase]:
    dead_load = LoadCase(name="Dead Load")
    dead_load.fix_boundary(domain.boundary("left"), ux=0.0, uy=0.0)
    dead_load.add_gravity_load(GravityLoad(g=9.81))

    live_load = LoadCase(name="Live Load")
    live_load.fix_boundary(domain.boundary("left"), ux=0.0, uy=0.0)
    live_load.add_distributed_load(DistributedLoad(domain.boundary("top"), magnitude=-5000.0))

    return dead_load, live_load


def test_add_load_case_rejects_duplicate_name(mesh) -> None:
    manager = LoadManager(mesh)
    manager.add_load_case(LoadCase(name="Dead"))
    with pytest.raises(ValidationError):
        manager.add_load_case(LoadCase(name="Dead"))


def test_add_combination_rejects_duplicate_name(mesh) -> None:
    manager = LoadManager(mesh)
    case = LoadCase(name="Dead", mesh=mesh)
    case.add_nodal_load(NodalLoad(1, TranslationDOF.X, 1.0))
    combo = LoadCombination(name="Ultimate", factors={case: 1.0})
    manager.add_combination(combo)
    with pytest.raises(ValidationError):
        manager.add_combination(LoadCombination(name="Ultimate", factors={case: 1.0}))


def test_solve_all_solves_every_load_case_and_combination(mesh, domain: Rectangle) -> None:
    dead_load, live_load = _dead_and_live(domain)

    manager = LoadManager(mesh)
    manager.add_load_case(dead_load)
    manager.add_load_case(live_load)

    ultimate = LoadCombination(name="Ultimate", factors={dead_load: 1.2, live_load: 1.6})
    manager.add_combination(ultimate)

    results = manager.solve_all()

    dead_result = results.for_load_case("Dead Load")
    live_result = results.for_load_case("Live Load")
    ultimate_result = results.for_combination("Ultimate")

    for node in mesh.nodes:
        dead_rx, dead_ry = dead_result.node_reaction(node.id)
        live_rx, live_ry = live_result.node_reaction(node.id)
        ult_rx, ult_ry = ultimate_result.node_reaction(node.id)
        assert_allclose(ult_rx, 1.2 * dead_rx + 1.6 * live_rx, atol=1e-6)
        assert_allclose(ult_ry, 1.2 * dead_ry + 1.6 * live_ry, atol=1e-6)


def test_for_load_case_unknown_name_raises(mesh, domain: Rectangle) -> None:
    dead_load, _ = _dead_and_live(domain)
    manager = LoadManager(mesh)
    manager.add_load_case(dead_load)
    results = manager.solve_all()

    with pytest.raises(EntityNotFoundError):
        results.for_load_case("Nonexistent")


def test_for_combination_unknown_name_raises(mesh, domain: Rectangle) -> None:
    dead_load, _ = _dead_and_live(domain)
    manager = LoadManager(mesh)
    manager.add_load_case(dead_load)
    results = manager.solve_all()

    with pytest.raises(EntityNotFoundError):
        results.for_combination("Nonexistent")


def test_solve_all_with_no_registrations_returns_empty_result_set(mesh) -> None:
    manager = LoadManager(mesh)
    results = manager.solve_all()

    assert results.load_case_results == {}
    assert results.combination_results == {}
