"""Tests for femtoolkit.application.results_service."""

import pytest

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.application.results_service import ResultsService
from femtoolkit.application.simulation_service import SimulationService


def _run_mechanical() -> object:
    project = Project(name="Beam", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.mesh.width = 2.0
    project.mesh.height = 0.4
    project.mesh.nx = 4
    project.mesh.ny = 2
    project.mesh.thickness = 0.02
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-500.0)]
    return SimulationService().run(project).simulation


def test_summary_returns_engineering_summary() -> None:
    simulation = _run_mechanical()
    summary = ResultsService().summary(simulation)
    assert summary.maximum_displacement is not None
    assert summary.maximum_von_mises_stress is not None


def test_available_nodal_fields() -> None:
    simulation = _run_mechanical()
    fields = ResultsService().available_nodal_fields(simulation)
    assert "displacement" in fields
    assert "displacement_magnitude" in fields
    assert "reaction" in fields


def test_available_element_fields() -> None:
    simulation = _run_mechanical()
    fields = ResultsService().available_element_fields(simulation)
    assert "von_mises_stress" in fields
    assert "stress" in fields
    assert "strain" in fields


def test_field_range_nodal() -> None:
    simulation = _run_mechanical()
    field_range = ResultsService().field_range(simulation, "displacement_magnitude", kind="nodal")
    assert field_range.minimum == pytest.approx(0.0, abs=1e-9)
    assert field_range.maximum > field_range.minimum


def test_field_range_element() -> None:
    simulation = _run_mechanical()
    field_range = ResultsService().field_range(simulation, "von_mises_stress", kind="element")
    assert field_range.maximum >= field_range.minimum >= 0.0


def test_field_range_invalid_kind_raises() -> None:
    simulation = _run_mechanical()
    with pytest.raises(ValueError):
        ResultsService().field_range(simulation, "displacement_magnitude", kind="bogus")
