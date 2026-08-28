"""Tests for modal_superposition."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import (
    BoundaryCondition,
    ConstantLoad,
    RayleighDamping,
    SinusoidalLoad,
    TimeDependentNodalLoad,
    TranslationDOF,
)
from femtoolkit.analysis.damping import ModalDamping
from femtoolkit.analysis.dynamic_analysis import DynamicAnalysis
from femtoolkit.analysis.dynamic_system import build_dynamic_system
from femtoolkit.analysis.modal_superposition import modal_superposition
from femtoolkit.exceptions import ValidationError
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import create_quad_mesh

WIDTH = 0.5
HEIGHT = 0.1
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
        width=WIDTH, height=HEIGHT, nx=6, ny=2, material=material, thickness=THICKNESS
    )


@pytest.fixture
def domain() -> Rectangle:
    return Rectangle(width=WIDTH, height=HEIGHT)


@pytest.fixture
def boundary_conditions(mesh, domain: Rectangle):
    conditions = []
    for node in mesh.nodes_on_boundary(domain.boundary("left")):
        conditions.append(BoundaryCondition(node.id, TranslationDOF.X, 0.0))
        conditions.append(BoundaryCondition(node.id, TranslationDOF.Y, 0.0))
    return conditions


@pytest.fixture
def tip_node(mesh):
    return max((n for n in mesh.nodes if n.x == WIDTH), key=lambda n: n.y)


def test_rejects_non_positive_modes(mesh, boundary_conditions) -> None:
    system = build_dynamic_system(mesh, boundary_conditions)
    with pytest.raises(ValidationError):
        modal_superposition(system, modes=0, loads=[], time_step=0.001, total_time=0.01)


def test_rejects_non_positive_time_step(mesh, boundary_conditions) -> None:
    system = build_dynamic_system(mesh, boundary_conditions)
    with pytest.raises(ValidationError):
        modal_superposition(system, modes=3, loads=[], time_step=0.0, total_time=0.01)


def test_rejects_non_positive_total_time(mesh, boundary_conditions) -> None:
    system = build_dynamic_system(mesh, boundary_conditions)
    with pytest.raises(ValidationError):
        modal_superposition(system, modes=3, loads=[], time_step=0.001, total_time=0.0)


def test_rejects_nonzero_boundary_condition_value(mesh, domain: Rectangle) -> None:
    bcs = [BoundaryCondition(mesh.nodes[0].id, TranslationDOF.X, 0.01)]
    system = build_dynamic_system(mesh, bcs)
    with pytest.raises(ValidationError):
        modal_superposition(system, modes=3, loads=[], time_step=0.001, total_time=0.01)


def test_returns_correct_history_shapes(mesh, boundary_conditions, tip_node) -> None:
    system = build_dynamic_system(mesh, boundary_conditions)
    load = TimeDependentNodalLoad(tip_node.id, TranslationDOF.Y, ConstantLoad(-1000.0))

    result = modal_superposition(system, modes=5, loads=[load], time_step=0.001, total_time=0.01)

    n_steps = 10
    total_dofs = system.dof_map.total_dofs
    assert result.time.shape == (n_steps + 1,)
    assert result.displacement_history.shape == (n_steps + 1, total_dofs)
    assert result.velocity_history.shape == (n_steps + 1, total_dofs)
    assert result.acceleration_history.shape == (n_steps + 1, total_dofs)
    assert result.reaction_history.shape == (n_steps + 1, total_dofs)


def test_all_results_finite(mesh, boundary_conditions, tip_node) -> None:
    system = build_dynamic_system(mesh, boundary_conditions)
    load = TimeDependentNodalLoad(tip_node.id, TranslationDOF.Y, SinusoidalLoad(50.0, 500.0))

    result = modal_superposition(system, modes=5, loads=[load], time_step=2e-5, total_time=0.005)

    assert np.isfinite(result.displacement_history).all()
    assert np.isfinite(result.velocity_history).all()
    assert np.isfinite(result.acceleration_history).all()
    assert np.isfinite(result.reaction_history).all()


def test_starts_at_rest(mesh, boundary_conditions, tip_node) -> None:
    system = build_dynamic_system(mesh, boundary_conditions)
    load = TimeDependentNodalLoad(tip_node.id, TranslationDOF.Y, ConstantLoad(-1000.0))

    result = modal_superposition(system, modes=5, loads=[load], time_step=0.001, total_time=0.01)

    assert_allclose(result.displacement_history[0], 0.0, atol=1e-12)
    assert_allclose(result.velocity_history[0], 0.0, atol=1e-12)


def test_constrained_dofs_stay_zero(mesh, boundary_conditions, tip_node) -> None:
    system = build_dynamic_system(mesh, boundary_conditions)
    load = TimeDependentNodalLoad(tip_node.id, TranslationDOF.Y, ConstantLoad(-1000.0))

    result = modal_superposition(system, modes=5, loads=[load], time_step=0.001, total_time=0.01)

    for node in mesh.nodes_on_boundary(Rectangle(width=WIDTH, height=HEIGHT).boundary("left")):
        assert_allclose(result.displacement(node.id, TranslationDOF.X), 0.0, atol=1e-10)
        assert_allclose(result.displacement(node.id, TranslationDOF.Y), 0.0, atol=1e-10)


def test_explicit_modal_damping_overrides_system_damping(
    mesh, boundary_conditions, tip_node
) -> None:
    """An explicit ModalDamping should be used instead of deriving from
    system.damping (undamped here), so the response should decay.
    """
    system = build_dynamic_system(mesh, boundary_conditions)  # undamped
    load = TimeDependentNodalLoad(tip_node.id, TranslationDOF.Y, ConstantLoad(-1000.0))

    result = modal_superposition(
        system,
        modes=5,
        loads=[load],
        time_step=2e-5,
        total_time=0.05,
        damping=ModalDamping(damping_ratios=0.1),
    )
    uy = result.displacement(tip_node.id, TranslationDOF.Y)
    first_quarter = np.abs(uy[: len(uy) // 4]).max()
    last_quarter = np.abs(uy[-len(uy) // 4 :]).max()
    assert last_quarter < first_quarter


def test_derives_damping_from_system_when_none_given(mesh, boundary_conditions, tip_node) -> None:
    damping = RayleighDamping(alpha=200.0, beta=0.0)
    system = build_dynamic_system(mesh, boundary_conditions, damping=damping)
    load = TimeDependentNodalLoad(tip_node.id, TranslationDOF.Y, ConstantLoad(-1000.0))

    result = modal_superposition(system, modes=5, loads=[load], time_step=2e-5, total_time=0.05)
    uy = result.displacement(tip_node.id, TranslationDOF.Y)
    first_quarter = np.abs(uy[: len(uy) // 4]).max()
    last_quarter = np.abs(uy[-len(uy) // 4 :]).max()
    assert last_quarter < first_quarter


# --- Validation: modal superposition with all free modes matches direct Newmark ---


def test_all_modes_matches_direct_newmark_solution(mesh, boundary_conditions, tip_node) -> None:
    damping = RayleighDamping(alpha=10.0, beta=0.0)
    system = build_dynamic_system(mesh, boundary_conditions, damping=damping)
    n_free = system.dof_map.total_dofs - len(boundary_conditions)
    load = TimeDependentNodalLoad(tip_node.id, TranslationDOF.Y, SinusoidalLoad(50.0, 500.0))

    modal_result = modal_superposition(
        system, modes=n_free, loads=[load], time_step=2e-5, total_time=0.01
    )

    direct_analysis = DynamicAnalysis(mesh, damping=damping)
    for bc in boundary_conditions:
        direct_analysis.add_boundary_condition(bc)
    direct_analysis.add_time_dependent_load(load)
    direct_result = direct_analysis.solve(time_step=2e-5, total_time=0.01)

    modal_uy = modal_result.displacement(tip_node.id, TranslationDOF.Y)
    direct_uy = direct_result.displacement(tip_node.id, TranslationDOF.Y)
    assert_allclose(modal_uy, direct_uy, atol=1e-12)


def test_truncated_modes_is_a_close_approximation(mesh, boundary_conditions, tip_node) -> None:
    system = build_dynamic_system(mesh, boundary_conditions)
    load = TimeDependentNodalLoad(tip_node.id, TranslationDOF.Y, SinusoidalLoad(50.0, 500.0))

    modal_result = modal_superposition(
        system, modes=5, loads=[load], time_step=2e-5, total_time=0.01
    )

    direct_analysis = DynamicAnalysis(mesh)
    for bc in boundary_conditions:
        direct_analysis.add_boundary_condition(bc)
    direct_analysis.add_time_dependent_load(load)
    direct_result = direct_analysis.solve(time_step=2e-5, total_time=0.01)

    modal_uy = modal_result.displacement(tip_node.id, TranslationDOF.Y)
    direct_uy = direct_result.displacement(tip_node.id, TranslationDOF.Y)
    max_error = np.abs(modal_uy - direct_uy).max()
    assert max_error / np.abs(direct_uy).max() < 0.01
