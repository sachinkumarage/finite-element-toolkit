"""Tests for femtoolkit.gui.visualization (Version 23 3D integration glue,
extended in Version 25 for mesh-quality visualization).
"""

import pytest

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.application.simulation_service import SimulationService
from femtoolkit.exceptions import ValidationError
from femtoolkit.gui.visualization import (
    available_scalar_fields,
    is_pyvista_available,
    render_mesh_quality_screenshot,
    render_result_screenshot,
)
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.generator import create_quad_mesh
from femtoolkit.mesh.quality import QualityEvaluator


def _run_mechanical():
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


def _quad_mesh():
    material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    return create_quad_mesh(width=2.0, height=0.4, nx=4, ny=2, material=material, thickness=0.02)


def test_is_pyvista_available_returns_bool() -> None:
    assert isinstance(is_pyvista_available(), bool)


def test_available_scalar_fields_includes_known_fields() -> None:
    simulation = _run_mechanical()
    fields = available_scalar_fields(simulation)
    assert "von_mises_stress" in fields
    assert "displacement_magnitude" in fields
    assert "temperature" not in fields


@pytest.mark.skipif(not is_pyvista_available(), reason="PyVista (viz3d extra) not installed")
def test_render_result_screenshot_creates_file(tmp_path) -> None:
    simulation = _run_mechanical()
    output_path = tmp_path / "beam.png"

    result_path = render_result_screenshot(
        simulation, output_path, scalar_field="von_mises_stress"
    )

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


@pytest.mark.skipif(not is_pyvista_available(), reason="PyVista (viz3d extra) not installed")
def test_render_result_screenshot_deformed(tmp_path) -> None:
    simulation = _run_mechanical()
    output_path = tmp_path / "beam_deformed.png"

    render_result_screenshot(
        simulation,
        output_path,
        scalar_field="displacement_magnitude",
        deformed=True,
        deformation_scale=500.0,
    )

    assert output_path.exists()


@pytest.mark.skipif(not is_pyvista_available(), reason="PyVista (viz3d extra) not installed")
def test_render_result_screenshot_invalid_field_raises(tmp_path) -> None:
    simulation = _run_mechanical()
    output_path = tmp_path / "invalid.png"

    with pytest.raises(ValidationError):
        render_result_screenshot(simulation, output_path, scalar_field="does_not_exist")


@pytest.mark.skipif(not is_pyvista_available(), reason="PyVista (viz3d extra) not installed")
def test_render_mesh_quality_screenshot_creates_file(tmp_path) -> None:
    mesh = _quad_mesh()
    report = QualityEvaluator().evaluate(mesh)
    output_path = tmp_path / "quality.png"

    result_path = render_mesh_quality_screenshot(mesh, report, output_path, metric="quality")

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


@pytest.mark.skipif(not is_pyvista_available(), reason="PyVista (viz3d extra) not installed")
def test_render_mesh_quality_screenshot_aspect_ratio_metric(tmp_path) -> None:
    mesh = _quad_mesh()
    report = QualityEvaluator().evaluate(mesh)
    output_path = tmp_path / "aspect_ratio.png"

    render_mesh_quality_screenshot(mesh, report, output_path, metric="aspect_ratio")
    assert output_path.exists()


@pytest.mark.skipif(not is_pyvista_available(), reason="PyVista (viz3d extra) not installed")
def test_render_mesh_quality_screenshot_invalid_metric_raises(tmp_path) -> None:
    mesh = _quad_mesh()
    report = QualityEvaluator().evaluate(mesh)
    output_path = tmp_path / "invalid.png"

    with pytest.raises(ValueError):
        render_mesh_quality_screenshot(mesh, report, output_path, metric="does_not_exist")


@pytest.mark.skipif(not is_pyvista_available(), reason="PyVista (viz3d extra) not installed")
def test_render_mesh_quality_screenshot_invalid_camera_view_raises(tmp_path) -> None:
    mesh = _quad_mesh()
    report = QualityEvaluator().evaluate(mesh)
    output_path = tmp_path / "invalid_camera.png"

    with pytest.raises(ValueError):
        render_mesh_quality_screenshot(mesh, report, output_path, camera_view="does_not_exist")
