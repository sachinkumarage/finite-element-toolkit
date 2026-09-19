"""Tests for femtoolkit.application.validation."""

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.application.validation import (
    validate_analysis_type,
    validate_boundary_conditions,
    validate_loads,
    validate_material,
    validate_mesh,
    validate_project,
    validate_solver,
)


def _valid_mechanical_project() -> Project:
    project = Project(name="Valid Mechanical", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.boundary_conditions = [BoundaryConditionConfig(region="left", dof="X", value=0.0)]
    project.loads = [LoadConfig(region="right", dof="X", magnitude=100.0)]
    return project


def _valid_thermal_project() -> Project:
    project = Project(name="Valid Thermal", analysis_type="thermal_steady_state")
    project.material.thermal_conductivity = 45.0
    project.material.specific_heat = 460.0
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="TEMPERATURE", value=373.15)
    ]
    return project


def test_valid_mechanical_project_has_no_errors() -> None:
    result = validate_project(_valid_mechanical_project())
    assert result.is_valid
    assert result.errors == []


def test_valid_thermal_project_has_no_errors() -> None:
    result = validate_project(_valid_thermal_project())
    assert result.is_valid


def test_unknown_analysis_type_invalid() -> None:
    project = Project(analysis_type="does_not_exist")
    errors = validate_analysis_type(project)
    assert errors
    assert "Unknown analysis type" in errors[0]


def test_unavailable_analysis_type_invalid() -> None:
    project = Project(analysis_type="nonlinear_static")
    errors = validate_analysis_type(project)
    assert errors
    assert "not yet available" in errors[0]


def test_invalid_material_non_positive_youngs_modulus() -> None:
    project = _valid_mechanical_project()
    project.material.youngs_modulus = -1.0
    errors = validate_material(project)
    assert any("Young's modulus" in error for error in errors)


def test_invalid_material_poisson_ratio_out_of_range() -> None:
    project = _valid_mechanical_project()
    project.material.poisson_ratio = 0.5
    errors = validate_material(project)
    assert any("Poisson's ratio" in error for error in errors)


def test_invalid_material_missing_youngs_modulus() -> None:
    project = _valid_mechanical_project()
    project.material.youngs_modulus = None
    errors = validate_material(project)
    assert any("Young's modulus" in error for error in errors)


def test_invalid_thermal_material_non_positive_conductivity() -> None:
    project = _valid_thermal_project()
    project.material.thermal_conductivity = 0.0
    errors = validate_material(project)
    assert any("Thermal conductivity" in error for error in errors)


def test_invalid_material_negative_density() -> None:
    project = _valid_mechanical_project()
    project.material.density = -10.0
    errors = validate_material(project)
    assert any("Density" in error for error in errors)


def test_invalid_mesh_non_positive_width() -> None:
    project = _valid_mechanical_project()
    project.mesh.width = 0.0
    errors = validate_mesh(project)
    assert any("width" in error for error in errors)


def test_invalid_mesh_zero_subdivisions() -> None:
    project = _valid_mechanical_project()
    project.mesh.nx = 0
    errors = validate_mesh(project)
    assert any("nx" in error for error in errors)


def test_invalid_mesh_bad_element_type() -> None:
    project = _valid_mechanical_project()
    project.mesh.element_type = "hex8"
    errors = validate_mesh(project)
    assert any("element_type" in error for error in errors)


def test_invalid_mesh_non_positive_thickness_mechanical() -> None:
    project = _valid_mechanical_project()
    project.mesh.thickness = 0.0
    errors = validate_mesh(project)
    assert any("thickness" in error for error in errors)


def test_missing_boundary_conditions_invalid() -> None:
    project = _valid_mechanical_project()
    project.boundary_conditions = []
    errors = validate_boundary_conditions(project)
    assert any("No boundary conditions" in error for error in errors)


def test_boundary_condition_wrong_dof_for_analysis_type() -> None:
    project = _valid_mechanical_project()
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="TEMPERATURE", value=0.0)
    ]
    errors = validate_boundary_conditions(project)
    assert any("dof" in error for error in errors)


def test_boundary_condition_unknown_region() -> None:
    project = _valid_mechanical_project()
    project.boundary_conditions = [BoundaryConditionConfig(region="middle", dof="X", value=0.0)]
    errors = validate_boundary_conditions(project)
    assert any("unknown region" in error for error in errors)


def test_boundary_condition_non_finite_value() -> None:
    project = _valid_mechanical_project()
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=float("nan"))
    ]
    errors = validate_boundary_conditions(project)
    assert any("finite" in error for error in errors)


def test_loads_are_optional() -> None:
    project = _valid_mechanical_project()
    project.loads = []
    errors = validate_loads(project)
    assert errors == []


def test_invalid_load_wrong_dof() -> None:
    project = _valid_mechanical_project()
    project.loads = [LoadConfig(region="right", dof="HEAT_FLUX", magnitude=1.0)]
    errors = validate_loads(project)
    assert any("dof" in error for error in errors)


def test_invalid_load_non_finite_magnitude() -> None:
    project = _valid_mechanical_project()
    project.loads = [LoadConfig(region="right", dof="X", magnitude=float("inf"))]
    errors = validate_loads(project)
    assert any("finite" in error for error in errors)


def test_invalid_solver_non_positive_tolerance() -> None:
    project = _valid_mechanical_project()
    project.solver.tolerance = 0.0
    errors = validate_solver(project)
    assert any("tolerance" in error for error in errors)


def test_invalid_solver_zero_max_iterations() -> None:
    project = _valid_mechanical_project()
    project.solver.max_iterations = 0
    errors = validate_solver(project)
    assert any("max_iterations" in error for error in errors)


def test_valid_solver_matrix_and_solver_types() -> None:
    project = _valid_mechanical_project()
    for matrix_type, solver_type in [
        ("dense", "direct"),
        ("sparse", "direct"),
        ("sparse", "conjugate_gradient"),
    ]:
        project.solver.matrix_type = matrix_type
        project.solver.solver_type = solver_type
        assert validate_solver(project) == []


def test_invalid_solver_unknown_matrix_type() -> None:
    project = _valid_mechanical_project()
    project.solver.matrix_type = "bogus"
    errors = validate_solver(project)
    assert any("matrix_type" in error for error in errors)


def test_invalid_solver_unknown_solver_type() -> None:
    project = _valid_mechanical_project()
    project.solver.solver_type = "bogus"
    errors = validate_solver(project)
    assert any("solver_type" in error for error in errors)


def test_invalid_solver_dense_conjugate_gradient_combo() -> None:
    project = _valid_mechanical_project()
    project.solver.matrix_type = "dense"
    project.solver.solver_type = "conjugate_gradient"
    errors = validate_solver(project)
    assert any("Conjugate Gradient" in error for error in errors)


def test_validate_project_aggregates_multiple_categories() -> None:
    project = Project(analysis_type="linear_static")
    project.material.youngs_modulus = -1.0
    project.mesh.width = -1.0
    project.boundary_conditions = []
    project.solver.tolerance = -1.0

    result = validate_project(project)
    assert not result.is_valid
    assert len(result.errors) >= 4
