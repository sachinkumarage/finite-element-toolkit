"""Tests for NonlinearSolverSettings and NonlinearAnalysis."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.exceptions import (
    InsufficientConstraintsError,
    InvalidAnalysisError,
    InvalidElementError,
    NonlinearConvergenceError,
    ValidationError,
)
from femtoolkit.materials import ElasticMaterialAdapter, LinearElastic2D
from femtoolkit.materials.material import Material
from femtoolkit.mesh.bar_element import BarElement
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.mesh.node import Node
from femtoolkit.sections import CrossSection

X = TranslationDOF.X
Y = TranslationDOF.Y

LINEAR_MATERIAL = LinearElastic2D(
    youngs_modulus=210e9, poisson_ratio=0.3, formulation="plane_stress"
)


def _two_triangle_mesh() -> tuple[Mesh, CSTElement2D, CSTElement2D]:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=1.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=1.0, z=0.0)
    lower = CSTElement2D(
        id=1, nodes=(node_1, node_2, node_3), material=LINEAR_MATERIAL, thickness=0.01
    )
    upper = CSTElement2D(
        id=2, nodes=(node_1, node_3, node_4), material=LINEAR_MATERIAL, thickness=0.01
    )

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    mesh.add_element(lower)
    mesh.add_element(upper)
    return mesh, lower, upper


def _fix_left_edge(analysis: NonlinearAnalysis | StaticLinearAnalysis) -> None:
    analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, Y, 0.0))


# --- NonlinearSolverSettings ---


def test_settings_defaults() -> None:
    settings = NonlinearSolverSettings()

    assert settings.max_iterations == 25
    assert settings.tolerance == pytest.approx(1e-8)
    assert settings.load_steps == 10
    assert settings.convergence == "residual"


def test_settings_rejects_non_positive_max_iterations() -> None:
    with pytest.raises(ValidationError):
        NonlinearSolverSettings(max_iterations=0)


def test_settings_rejects_non_positive_tolerance() -> None:
    with pytest.raises(ValidationError):
        NonlinearSolverSettings(tolerance=0.0)


def test_settings_rejects_non_positive_load_steps() -> None:
    with pytest.raises(ValidationError):
        NonlinearSolverSettings(load_steps=0)


def test_settings_rejects_unknown_convergence_criterion() -> None:
    with pytest.raises(ValidationError):
        NonlinearSolverSettings(convergence="bogus")


# --- NonlinearAnalysis: setup validation ---


def test_solve_rejects_empty_mesh() -> None:
    analysis = NonlinearAnalysis(Mesh(), materials={})

    with pytest.raises(InvalidAnalysisError):
        analysis.solve()


def test_solve_rejects_mesh_with_nodes_but_no_elements() -> None:
    mesh = Mesh()
    mesh.add_node(Node(id=1, x=0.0, y=0.0, z=0.0))
    analysis = NonlinearAnalysis(mesh, materials={})

    with pytest.raises(InvalidAnalysisError):
        analysis.solve()


def test_solve_rejects_non_continuum_elements() -> None:
    steel = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    bar = BarElement(
        id=1, nodes=(node_1, node_2), material=steel, cross_section=CrossSection(area=0.01)
    )
    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_element(bar)

    analysis = NonlinearAnalysis(mesh, materials={1: ElasticMaterialAdapter(modulus=200e9)})
    analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))

    with pytest.raises(InvalidElementError):
        analysis.solve()


def test_solve_rejects_missing_material_for_an_element() -> None:
    mesh, lower, upper = _two_triangle_mesh()
    materials = {lower.id: ElasticMaterialAdapter.from_linear_elastic_2d(LINEAR_MATERIAL)}
    # upper.id has no entry.
    analysis = NonlinearAnalysis(mesh, materials)
    _fix_left_edge(analysis)
    analysis.add_load(NodalLoad(2, X, 1000.0))

    with pytest.raises(ValidationError):
        analysis.solve()


def test_solve_requires_boundary_conditions() -> None:
    mesh, lower, upper = _two_triangle_mesh()
    materials = {
        e.id: ElasticMaterialAdapter.from_linear_elastic_2d(LINEAR_MATERIAL) for e in (lower, upper)
    }
    analysis = NonlinearAnalysis(mesh, materials)
    analysis.add_load(NodalLoad(2, X, 1000.0))

    with pytest.raises(InsufficientConstraintsError):
        analysis.solve()


def test_solve_rejects_duplicate_boundary_conditions_on_same_dof() -> None:
    mesh, lower, upper = _two_triangle_mesh()
    materials = {
        e.id: ElasticMaterialAdapter.from_linear_elastic_2d(LINEAR_MATERIAL) for e in (lower, upper)
    }
    analysis = NonlinearAnalysis(mesh, materials)
    _fix_left_edge(analysis)
    analysis.add_boundary_condition(BoundaryCondition(1, X, 0.1))  # duplicate target
    analysis.add_load(NodalLoad(2, X, 1000.0))

    with pytest.raises(ValidationError):
        analysis.solve()


# --- Critical validation: linear regression (Section 23) ---


def test_nonlinear_solve_matches_linear_solve_with_elastic_adapter() -> None:
    mesh, lower, upper = _two_triangle_mesh()

    linear_analysis = StaticLinearAnalysis(mesh)
    _fix_left_edge(linear_analysis)
    linear_analysis.add_load(NodalLoad(2, X, 1.0e6))
    linear_analysis.add_load(NodalLoad(3, X, 1.0e6))
    linear_result = linear_analysis.solve()

    materials = {
        e.id: ElasticMaterialAdapter.from_linear_elastic_2d(LINEAR_MATERIAL) for e in (lower, upper)
    }
    nonlinear_analysis = NonlinearAnalysis(
        mesh, materials, NonlinearSolverSettings(load_steps=5, tolerance=1e-10)
    )
    _fix_left_edge(nonlinear_analysis)
    nonlinear_analysis.add_load(NodalLoad(2, X, 1.0e6))
    nonlinear_analysis.add_load(NodalLoad(3, X, 1.0e6))
    nonlinear_result = nonlinear_analysis.solve()

    assert nonlinear_result.converged
    assert_allclose(
        nonlinear_result.displacement(2, X), linear_result.displacement(2, X), rtol=1e-8
    )
    assert_allclose(
        nonlinear_result.displacement(3, X), linear_result.displacement(3, X), rtol=1e-8
    )
    # A linear problem converges immediately: one correction, one verification.
    assert np.all(nonlinear_result.iteration_counts() <= 2)


def test_nonlinear_solve_reactions_are_in_equilibrium() -> None:
    mesh, lower, upper = _two_triangle_mesh()
    materials = {
        e.id: ElasticMaterialAdapter.from_linear_elastic_2d(LINEAR_MATERIAL) for e in (lower, upper)
    }
    analysis = NonlinearAnalysis(mesh, materials, NonlinearSolverSettings(load_steps=4))
    _fix_left_edge(analysis)
    analysis.add_load(NodalLoad(2, X, 5.0e5))
    analysis.add_load(NodalLoad(3, X, 5.0e5))
    result = analysis.solve()

    for step_result in result.step_results:
        applied = step_result.load_factor * 1.0e6
        assert np.sum(step_result.reactions) == pytest.approx(-applied, abs=1e-3)


def test_load_stepping_reaches_full_load_at_final_step() -> None:
    mesh, lower, upper = _two_triangle_mesh()
    materials = {
        e.id: ElasticMaterialAdapter.from_linear_elastic_2d(LINEAR_MATERIAL) for e in (lower, upper)
    }
    analysis = NonlinearAnalysis(mesh, materials, NonlinearSolverSettings(load_steps=5))
    _fix_left_edge(analysis)
    analysis.add_load(NodalLoad(2, X, 1.0e6))
    result = analysis.solve()

    assert result.load_factors()[-1] == pytest.approx(1.0)
    assert_allclose(result.load_factors(), np.array([0.2, 0.4, 0.6, 0.8, 1.0]))


def test_displacement_controlled_boundary_condition_is_scaled_by_load_factor() -> None:
    mesh, lower, upper = _two_triangle_mesh()
    materials = {
        e.id: ElasticMaterialAdapter.from_linear_elastic_2d(LINEAR_MATERIAL) for e in (lower, upper)
    }
    settings = NonlinearSolverSettings(load_steps=4, tolerance=1e-10)
    analysis = NonlinearAnalysis(mesh, materials, settings)
    _fix_left_edge(analysis)
    analysis.add_boundary_condition(BoundaryCondition(2, X, 0.001))
    result = analysis.solve()

    assert_allclose(
        result.displacement_history(2, X), np.array([0.00025, 0.0005, 0.00075, 0.001])
    )
    assert result.displacement(2, X) == pytest.approx(0.001)


# --- Newton-Raphson iteration limits (Section 28) ---


def test_convergence_failure_raises_with_partial_step_results() -> None:
    mesh, lower, upper = _two_triangle_mesh()
    materials = {
        e.id: ElasticMaterialAdapter.from_linear_elastic_2d(LINEAR_MATERIAL) for e in (lower, upper)
    }
    analysis = NonlinearAnalysis(
        mesh, materials, NonlinearSolverSettings(max_iterations=1, tolerance=1e-12, load_steps=3)
    )
    _fix_left_edge(analysis)
    analysis.add_load(NodalLoad(2, X, 1.0e6))

    with pytest.raises(NonlinearConvergenceError) as exc_info:
        analysis.solve()

    step_results = exc_info.value.step_results
    assert len(step_results) == 1
    assert step_results[0].converged is False
    assert step_results[0].iterations == 1


def test_max_iterations_is_respected() -> None:
    mesh, lower, upper = _two_triangle_mesh()
    materials = {
        e.id: ElasticMaterialAdapter.from_linear_elastic_2d(LINEAR_MATERIAL) for e in (lower, upper)
    }
    analysis = NonlinearAnalysis(
        mesh, materials, NonlinearSolverSettings(max_iterations=1, tolerance=1e-12, load_steps=1)
    )
    _fix_left_edge(analysis)
    analysis.add_load(NodalLoad(2, X, 1.0e6))

    with pytest.raises(NonlinearConvergenceError) as exc_info:
        analysis.solve()

    assert exc_info.value.step_results[0].iterations == 1


def test_generous_tolerance_converges_where_strict_tolerance_fails() -> None:
    mesh, lower, upper = _two_triangle_mesh()
    materials = {
        e.id: ElasticMaterialAdapter.from_linear_elastic_2d(LINEAR_MATERIAL) for e in (lower, upper)
    }

    strict_analysis = NonlinearAnalysis(
        mesh, materials, NonlinearSolverSettings(max_iterations=1, tolerance=1e-14, load_steps=1)
    )
    _fix_left_edge(strict_analysis)
    strict_analysis.add_load(NodalLoad(2, X, 1.0e6))
    with pytest.raises(NonlinearConvergenceError):
        strict_analysis.solve()

    generous_analysis = NonlinearAnalysis(
        mesh, materials, NonlinearSolverSettings(max_iterations=2, tolerance=1e-8, load_steps=1)
    )
    _fix_left_edge(generous_analysis)
    generous_analysis.add_load(NodalLoad(2, X, 1.0e6))
    result = generous_analysis.solve()
    assert result.converged


def test_displacement_convergence_criterion_also_converges() -> None:
    mesh, lower, upper = _two_triangle_mesh()
    materials = {
        e.id: ElasticMaterialAdapter.from_linear_elastic_2d(LINEAR_MATERIAL) for e in (lower, upper)
    }
    analysis = NonlinearAnalysis(
        mesh,
        materials,
        NonlinearSolverSettings(load_steps=4, tolerance=1e-9, convergence="displacement"),
    )
    _fix_left_edge(analysis)
    analysis.add_load(NodalLoad(2, X, 1.0e6))
    result = analysis.solve()

    assert result.converged
