"""Tests for SteadyStateThermalAnalysis's Version 27 execution-strategy integration."""

from __future__ import annotations

import pytest
from numpy.testing import assert_allclose

from femtoolkit.execution import ExecutionConfig
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.generator import create_triangular_mesh
from femtoolkit.solvers import SparseDirectSolver
from femtoolkit.thermal import PrescribedTemperature, SteadyStateThermalAnalysis, ThermalMaterial
from femtoolkit.thermal.thermal_boundary_conditions import RadiationBoundaryCondition
from femtoolkit.thermal.thermal_surfaces import ThermalSurface

T0 = 373.15
T1 = 293.15


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")


@pytest.fixture
def thermal_material() -> ThermalMaterial:
    return ThermalMaterial(thermal_conductivity=25.0, density=7850.0, specific_heat=460.0)


def _heated_plate_analysis(
    material: LinearElastic2D,
    thermal_material: ThermalMaterial,
    execution=None,
    solver=None,
) -> SteadyStateThermalAnalysis:
    mesh = create_triangular_mesh(
        width=2.0, height=1.0, nx=5, ny=3, material=material, thickness=0.01
    )
    materials = {e.id: thermal_material for e in mesh.elements}
    analysis = SteadyStateThermalAnalysis(mesh, materials, solver=solver, execution=execution)
    for node in mesh.nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, T0))
        elif node.x == 2.0:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, T1))
    return analysis


def test_default_execution_is_none_and_reports_serial(
    material: LinearElastic2D, thermal_material: ThermalMaterial
) -> None:
    analysis = _heated_plate_analysis(material, thermal_material)
    analysis.solve()

    report = analysis.last_performance_report
    assert report is not None
    assert report.execution_mode == "serial"
    assert report.element_time is not None
    assert report.assembly_time is not None
    assert report.solve_time is not None


def test_parallel_execution_matches_serial_temperatures(
    material: LinearElastic2D, thermal_material: ThermalMaterial
) -> None:
    serial_result = _heated_plate_analysis(material, thermal_material).solve()
    parallel_analysis = _heated_plate_analysis(
        material,
        thermal_material,
        execution=ExecutionConfig(mode="parallel", workers=2, backend="process"),
    )
    parallel_result = parallel_analysis.solve()

    assert_allclose(serial_result.temperatures, parallel_result.temperatures)
    assert parallel_analysis.last_performance_report.execution_mode == "parallel"
    assert parallel_analysis.last_performance_report.workers == 2


def test_parallel_execution_combined_with_sparse_solver(
    material: LinearElastic2D, thermal_material: ThermalMaterial
) -> None:
    dense_result = _heated_plate_analysis(material, thermal_material).solve()
    combined_analysis = _heated_plate_analysis(
        material,
        thermal_material,
        execution=ExecutionConfig(mode="parallel", workers=2, backend="process"),
        solver=SparseDirectSolver(),
    )
    combined_result = combined_analysis.solve()

    assert_allclose(dense_result.temperatures, combined_result.temperatures, atol=1e-8)
    assert combined_analysis.last_solver_result.solver_name == "Sparse Direct"


def test_nonlinear_radiation_path_ignores_execution_setting(
    material: LinearElastic2D, thermal_material: ThermalMaterial
) -> None:
    """The nonlinear Newton-Raphson path is unaffected by `execution` -- no
    performance report is populated for it, matching how `last_solver_result`
    is also left untouched on that path."""
    mesh = create_triangular_mesh(
        width=1.0, height=1.0, nx=3, ny=3, material=material, thickness=0.01
    )
    materials = {e.id: thermal_material for e in mesh.elements}
    analysis = SteadyStateThermalAnalysis(
        mesh,
        materials,
        execution=ExecutionConfig(mode="parallel", workers=2, backend="process"),
    )
    for node in mesh.nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, 373.15))
    analysis.add_radiation(
        RadiationBoundaryCondition(
            surface=ThermalSurface(element_id=mesh.elements[0].id, local_face_index=0),
            emissivity=0.8,
            surrounding_temperature=293.15,
        )
    )

    analysis.solve()
    assert analysis.last_performance_report is None
