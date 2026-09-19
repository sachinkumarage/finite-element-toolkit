"""Tests for SteadyStateThermalAnalysis's Version 26 solver-strategy integration."""

import pytest
from numpy.testing import assert_allclose

from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.generator import create_triangular_mesh
from femtoolkit.solvers import ConjugateGradientSolver, SparseDirectSolver
from femtoolkit.thermal import PrescribedTemperature, SteadyStateThermalAnalysis, ThermalMaterial

T0 = 373.15
T1 = 293.15


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")


@pytest.fixture
def thermal_material() -> ThermalMaterial:
    return ThermalMaterial(thermal_conductivity=25.0, density=7850.0, specific_heat=460.0)


def _heated_plate_analysis(
    material: LinearElastic2D, thermal_material: ThermalMaterial, solver=None
) -> SteadyStateThermalAnalysis:
    mesh = create_triangular_mesh(
        width=2.0, height=1.0, nx=6, ny=4, material=material, thickness=0.01
    )
    materials = {e.id: thermal_material for e in mesh.elements}
    analysis = SteadyStateThermalAnalysis(mesh, materials, solver=solver)
    for node in mesh.nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, T0))
        elif node.x == 2.0:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, T1))
    return analysis


def test_default_solver_is_none(
    material: LinearElastic2D, thermal_material: ThermalMaterial
) -> None:
    analysis = _heated_plate_analysis(material, thermal_material)
    analysis.solve()
    assert analysis.last_solver_result is None


def test_sparse_direct_matches_default_dense_path(
    material: LinearElastic2D, thermal_material: ThermalMaterial
) -> None:
    default_result = _heated_plate_analysis(material, thermal_material).solve()
    sparse_analysis = _heated_plate_analysis(
        material, thermal_material, solver=SparseDirectSolver()
    )
    sparse_result = sparse_analysis.solve()

    assert_allclose(default_result.temperatures, sparse_result.temperatures, atol=1e-8)
    assert sparse_analysis.last_solver_result is not None
    assert sparse_analysis.last_solver_result.solver_name == "Sparse Direct"


def test_conjugate_gradient_matches_default_dense_path(
    material: LinearElastic2D, thermal_material: ThermalMaterial
) -> None:
    default_result = _heated_plate_analysis(material, thermal_material).solve()
    cg_analysis = _heated_plate_analysis(
        material,
        thermal_material,
        solver=ConjugateGradientSolver(tolerance=1e-12, max_iterations=2000),
    )
    cg_result = cg_analysis.solve()

    assert_allclose(default_result.temperatures, cg_result.temperatures, atol=1e-6)
    assert cg_analysis.last_solver_result.iterations is not None


def test_sparse_solve_reproduces_analytical_linear_field(
    material: LinearElastic2D, thermal_material: ThermalMaterial
) -> None:
    """T(x) = T0 + (T1-T0)/L * x is the exact 1D steady-conduction solution."""
    analysis = _heated_plate_analysis(material, thermal_material, solver=SparseDirectSolver())
    result = analysis.solve()

    for node in analysis.mesh.nodes:
        expected = T0 + (T1 - T0) / 2.0 * node.x
        assert result.node_temperature(node.id) == pytest.approx(expected, abs=1e-6)


def test_solver_diagnostics_report_correct_dof_count(
    material: LinearElastic2D, thermal_material: ThermalMaterial
) -> None:
    analysis = _heated_plate_analysis(material, thermal_material, solver=SparseDirectSolver())
    analysis.solve()

    diagnostics = analysis.last_solver_result
    assert diagnostics.diagnostics["dofs"] == len(analysis.mesh.nodes)
