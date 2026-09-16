"""Tests for femtoolkit.application.model_service."""

import pytest

from femtoolkit.analysis.boundary_conditions import BoundaryCondition
from femtoolkit.analysis.loads import NodalLoad
from femtoolkit.application.model_service import ModelService
from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.exceptions import ValidationError
from femtoolkit.thermal import PrescribedHeatFlux, PrescribedTemperature


def _mechanical_project() -> Project:
    project = Project(name="Beam", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.width = 2.0
    project.mesh.height = 0.4
    project.mesh.nx = 4
    project.mesh.ny = 2
    project.mesh.thickness = 0.02
    return project


def _thermal_project() -> Project:
    project = Project(name="Plate", analysis_type="thermal_steady_state")
    project.material.thermal_conductivity = 45.0
    project.material.specific_heat = 460.0
    project.material.density = 7850.0
    project.mesh.width = 1.0
    project.mesh.height = 1.0
    project.mesh.nx = 4
    project.mesh.ny = 4
    project.mesh.element_type = "cst"
    return project


def test_build_mesh_quad_element_type() -> None:
    service = ModelService()
    mesh = service.build_mesh(_mechanical_project())
    assert len(mesh.nodes) == 5 * 3
    assert len(mesh.elements) == 4 * 2


def test_build_mesh_cst_element_type() -> None:
    project = _mechanical_project()
    project.mesh.element_type = "cst"
    service = ModelService()
    mesh = service.build_mesh(project)
    assert len(mesh.elements) == 4 * 2 * 2


def test_build_mesh_unknown_element_type_raises() -> None:
    project = _mechanical_project()
    project.mesh.element_type = "hex8"
    service = ModelService()
    with pytest.raises(ValidationError):
        service.build_mesh(project)


def test_build_mesh_for_thermal_uses_placeholder_geometry_material() -> None:
    service = ModelService()
    mesh = service.build_mesh(_thermal_project())
    assert len(mesh.nodes) > 0
    assert len(mesh.elements) > 0


def test_mesh_summary_reports_correct_counts() -> None:
    service = ModelService()
    mesh = service.build_mesh(_mechanical_project())
    summary = service.mesh_summary(mesh)

    assert summary.num_nodes == 15
    assert summary.num_elements == 8
    assert summary.dimension == "2D"
    assert summary.element_type == "QuadElement2D"
    assert summary.quality.num_invalid_elements == 0


def test_build_thermal_materials_assigns_every_element() -> None:
    service = ModelService()
    project = _thermal_project()
    mesh = service.build_mesh(project)
    materials = service.build_thermal_materials(project, mesh)

    assert set(materials) == {element.id for element in mesh.elements}
    for material in materials.values():
        assert material.thermal_conductivity == pytest.approx(45.0)


def test_build_boundary_conditions_mechanical() -> None:
    project = _mechanical_project()
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    service = ModelService()
    mesh = service.build_mesh(project)
    boundary_conditions = service.build_boundary_conditions(project, mesh)

    assert boundary_conditions
    assert all(isinstance(bc, BoundaryCondition) for bc in boundary_conditions)
    left_node_count = len(mesh.nodes_on_boundary(service.build_rectangle(project).boundary("left")))
    assert len(boundary_conditions) == left_node_count * 2


def test_build_boundary_conditions_thermal() -> None:
    project = _thermal_project()
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="TEMPERATURE", value=373.15)
    ]
    service = ModelService()
    mesh = service.build_mesh(project)
    boundary_conditions = service.build_boundary_conditions(project, mesh)

    assert boundary_conditions
    assert all(isinstance(bc, PrescribedTemperature) for bc in boundary_conditions)
    assert all(bc.value == pytest.approx(373.15) for bc in boundary_conditions)


def test_build_loads_mechanical() -> None:
    project = _mechanical_project()
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-500.0)]
    service = ModelService()
    mesh = service.build_mesh(project)
    loads = service.build_loads(project, mesh)

    assert loads
    assert all(isinstance(load, NodalLoad) for load in loads)
    assert all(load.value == pytest.approx(-500.0) for load in loads)


def test_build_loads_thermal() -> None:
    project = _thermal_project()
    project.loads = [LoadConfig(region="right", dof="HEAT_FLUX", magnitude=25.0)]
    service = ModelService()
    mesh = service.build_mesh(project)
    loads = service.build_loads(project, mesh)

    assert loads
    assert all(isinstance(load, PrescribedHeatFlux) for load in loads)


def test_build_loads_empty_when_no_loads_configured() -> None:
    project = _mechanical_project()
    project.loads = []
    service = ModelService()
    mesh = service.build_mesh(project)
    assert service.build_loads(project, mesh) == []
