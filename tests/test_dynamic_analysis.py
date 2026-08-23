"""Tests for DynamicAnalysis."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import (
    BoundaryCondition,
    ConstantLoad,
    NodalLoad,
    RayleighDamping,
    SinusoidalLoad,
    StaticLinearAnalysis,
    TimeDependentNodalLoad,
    TranslationDOF,
)
from femtoolkit.analysis.dynamic_analysis import DynamicAnalysis
from femtoolkit.exceptions import ValidationError
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import create_quad_mesh

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
def tip_node(mesh):
    return max((n for n in mesh.nodes if n.x == WIDTH), key=lambda n: n.y)


def _fix_left(analysis, mesh, domain) -> None:
    for node in mesh.nodes_on_boundary(domain.boundary("left")):
        analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.X, 0.0))
        analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.Y, 0.0))


def test_solve_returns_correct_history_shapes(mesh, domain, tip_node) -> None:
    analysis = DynamicAnalysis(mesh)
    _fix_left(analysis, mesh, domain)
    analysis.add_time_dependent_load(
        TimeDependentNodalLoad(tip_node.id, TranslationDOF.Y, ConstantLoad(-1000.0))
    )

    result = analysis.solve(time_step=0.001, total_time=0.01)

    n_steps = 10
    total_dofs = result.dof_map.total_dofs
    assert result.time.shape == (n_steps + 1,)
    assert result.displacement_history.shape == (n_steps + 1, total_dofs)
    assert result.velocity_history.shape == (n_steps + 1, total_dofs)
    assert result.acceleration_history.shape == (n_steps + 1, total_dofs)
    assert result.reaction_history.shape == (n_steps + 1, total_dofs)


def test_all_results_are_finite(mesh, domain, tip_node) -> None:
    analysis = DynamicAnalysis(mesh)
    _fix_left(analysis, mesh, domain)
    analysis.add_time_dependent_load(
        TimeDependentNodalLoad(tip_node.id, TranslationDOF.Y, SinusoidalLoad(1000.0, 200.0))
    )

    result = analysis.solve(time_step=0.0002, total_time=0.02)

    assert np.isfinite(result.displacement_history).all()
    assert np.isfinite(result.velocity_history).all()
    assert np.isfinite(result.acceleration_history).all()
    assert np.isfinite(result.reaction_history).all()


def test_constrained_dofs_stay_at_prescribed_value(mesh, domain, tip_node) -> None:
    analysis = DynamicAnalysis(mesh)
    _fix_left(analysis, mesh, domain)
    analysis.add_time_dependent_load(
        TimeDependentNodalLoad(tip_node.id, TranslationDOF.Y, ConstantLoad(-1000.0))
    )
    result = analysis.solve(time_step=0.0005, total_time=0.02)

    for node in mesh.nodes_on_boundary(domain.boundary("left")):
        assert_allclose(result.displacement(node.id, TranslationDOF.X), 0.0, atol=1e-12)
        assert_allclose(result.displacement(node.id, TranslationDOF.Y), 0.0, atol=1e-12)
        assert_allclose(result.velocity(node.id, TranslationDOF.X), 0.0, atol=1e-12)
        assert_allclose(result.acceleration(node.id, TranslationDOF.X), 0.0, atol=1e-12)


def test_zero_initial_conditions_at_t_zero(mesh, domain, tip_node) -> None:
    analysis = DynamicAnalysis(mesh)
    _fix_left(analysis, mesh, domain)
    analysis.add_time_dependent_load(
        TimeDependentNodalLoad(tip_node.id, TranslationDOF.Y, ConstantLoad(-1000.0))
    )
    result = analysis.solve(time_step=0.0005, total_time=0.02)

    assert_allclose(result.displacement_history[0], 0.0, atol=1e-12)
    assert_allclose(result.velocity_history[0], 0.0, atol=1e-12)


def test_rejects_non_positive_time_step(mesh, domain) -> None:
    analysis = DynamicAnalysis(mesh)
    _fix_left(analysis, mesh, domain)
    with pytest.raises(ValidationError):
        analysis.solve(time_step=0.0, total_time=1.0)


def test_rejects_non_positive_total_time(mesh, domain) -> None:
    analysis = DynamicAnalysis(mesh)
    _fix_left(analysis, mesh, domain)
    with pytest.raises(ValidationError):
        analysis.solve(time_step=0.001, total_time=-1.0)


def test_rejects_fully_constrained_system(mesh) -> None:
    analysis = DynamicAnalysis(mesh)
    for node in mesh.nodes:
        analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.X, 0.0))
        analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.Y, 0.0))
    with pytest.raises(ValidationError):
        analysis.solve(time_step=0.001, total_time=0.01)


# --- Static-limit validation: constant-load time-average matches statics ---


def test_constant_load_time_average_matches_static_solution(mesh, domain, tip_node) -> None:
    force = -1000.0

    dynamic = DynamicAnalysis(mesh)
    _fix_left(dynamic, mesh, domain)
    dynamic.add_time_dependent_load(
        TimeDependentNodalLoad(tip_node.id, TranslationDOF.Y, ConstantLoad(force))
    )
    dynamic_result = dynamic.solve(time_step=0.0001, total_time=0.5)

    static = StaticLinearAnalysis(mesh)
    _fix_left(static, mesh, domain)
    static.add_load(NodalLoad(tip_node.id, TranslationDOF.Y, force))
    static_result = static.solve()

    dynamic_uy = dynamic_result.displacement(tip_node.id, TranslationDOF.Y)
    static_uy = static_result.displacement(tip_node.id, TranslationDOF.Y)

    assert_allclose(dynamic_uy.mean(), static_uy, rtol=0.05)


def test_suddenly_applied_load_peaks_near_twice_static(mesh, domain, tip_node) -> None:
    """Classic step-load dynamic amplification: an undamped system's
    peak response to a suddenly applied constant force approaches twice
    its static displacement.
    """
    force = -1000.0

    dynamic = DynamicAnalysis(mesh)
    _fix_left(dynamic, mesh, domain)
    dynamic.add_time_dependent_load(
        TimeDependentNodalLoad(tip_node.id, TranslationDOF.Y, ConstantLoad(force))
    )
    dynamic_result = dynamic.solve(time_step=0.0001, total_time=0.5)

    static = StaticLinearAnalysis(mesh)
    _fix_left(static, mesh, domain)
    static.add_load(NodalLoad(tip_node.id, TranslationDOF.Y, force))
    static_result = static.solve()

    dynamic_uy = dynamic_result.displacement(tip_node.id, TranslationDOF.Y)
    static_uy = static_result.displacement(tip_node.id, TranslationDOF.Y)

    assert_allclose(dynamic_uy.min(), 2.0 * static_uy, rtol=0.1)


def test_damping_reduces_late_time_amplitude(mesh, domain, tip_node) -> None:
    force = -1000.0

    def build_and_solve(damping):
        analysis = DynamicAnalysis(mesh, damping=damping)
        _fix_left(analysis, mesh, domain)
        analysis.add_time_dependent_load(
            TimeDependentNodalLoad(tip_node.id, TranslationDOF.Y, ConstantLoad(force))
        )
        return analysis.solve(time_step=0.0001, total_time=0.5)

    undamped = build_and_solve(None)
    damped = build_and_solve(RayleighDamping(alpha=5.0, beta=0.0))

    late = slice(-1250, None)
    undamped_amplitude = np.abs(undamped.displacement(tip_node.id, TranslationDOF.Y)[late]).max()
    damped_amplitude = np.abs(damped.displacement(tip_node.id, TranslationDOF.Y)[late]).max()

    assert damped_amplitude < undamped_amplitude


def test_lumped_mass_option_solves_successfully(mesh, domain, tip_node) -> None:
    analysis = DynamicAnalysis(mesh, mass_matrix_type="lumped")
    _fix_left(analysis, mesh, domain)
    analysis.add_time_dependent_load(
        TimeDependentNodalLoad(tip_node.id, TranslationDOF.Y, ConstantLoad(-1000.0))
    )
    result = analysis.solve(time_step=0.001, total_time=0.01)
    assert np.isfinite(result.displacement_history).all()
