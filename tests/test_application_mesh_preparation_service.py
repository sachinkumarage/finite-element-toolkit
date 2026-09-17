"""Tests for femtoolkit.application.mesh_preparation_service.MeshPreparationService (Version 25)."""

import pytest

from femtoolkit.application.mesh_preparation_service import MeshPreparationService
from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.application.simulation_service import SimulationService
from femtoolkit.application.validation import validate_project
from femtoolkit.mesh.sizing import MeshSizingParameters


def _project() -> Project:
    project = Project(name="Beam", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.mesh.width = 2.0
    project.mesh.height = 0.4
    project.mesh.nx = 10
    project.mesh.ny = 2
    project.mesh.thickness = 0.02
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-500.0)]
    return project


def test_apply_sizing_writes_back_subdivisions() -> None:
    project = _project()
    service = MeshPreparationService()
    service.apply_sizing(project, MeshSizingParameters(target_size=0.2))

    assert project.mesh.nx == 10
    assert project.mesh.ny == 2


def test_preview_matches_configured_mesh() -> None:
    project = _project()
    service = MeshPreparationService()
    mesh = service.preview(project)

    assert len(mesh.nodes) == 11 * 3
    assert len(mesh.elements) == 10 * 2


def test_validation_report_on_preview() -> None:
    project = _project()
    service = MeshPreparationService()
    mesh = service.preview(project)
    report = service.validation_report(mesh)

    assert report.status == "OK"
    assert report.is_valid


def test_quality_report_on_preview() -> None:
    project = _project()
    service = MeshPreparationService()
    mesh = service.preview(project)
    report = service.quality_report(mesh)

    assert report.minimum_quality == pytest.approx(1.0)
    assert not report.has_poor_quality_elements


def test_statistics_on_preview() -> None:
    project = _project()
    service = MeshPreparationService()
    mesh = service.preview(project)
    stats = service.statistics(mesh)

    assert stats.num_nodes == 33
    assert stats.num_elements == 20
    assert stats.dimension == "2D"


def test_refine_increments_and_preview_reflects_it() -> None:
    project = _project()
    service = MeshPreparationService()
    baseline = service.preview(project)

    service.refine(project)
    assert project.mesh.refinement_passes == 1

    refined = service.preview(project)
    assert len(refined.elements) == 4 * len(baseline.elements)


def test_refine_twice_compounds() -> None:
    project = _project()
    service = MeshPreparationService()
    baseline = service.preview(project)

    service.refine(project)
    service.refine(project)
    assert project.mesh.refinement_passes == 2

    twice_refined = service.preview(project)
    assert len(twice_refined.elements) == 16 * len(baseline.elements)


def test_reset_refinement() -> None:
    project = _project()
    service = MeshPreparationService()
    service.refine(project)
    service.refine(project)
    service.reset_refinement(project)

    assert project.mesh.refinement_passes == 0


def test_refinement_flows_through_to_a_real_solve() -> None:
    project = _project()
    prep_service = MeshPreparationService()
    prep_service.refine(project)

    assert validate_project(project).is_valid
    result = SimulationService().run(project)

    assert result.succeeded
    assert result.summary.maximum_displacement is not None


def test_negative_refinement_passes_rejected_by_validation() -> None:
    project = _project()
    project.mesh.refinement_passes = -1
    validation = validate_project(project)
    assert not validation.is_valid
    assert any("refinement_passes" in error for error in validation.errors)
