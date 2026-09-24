"""Tests for femtoolkit.studies.plots."""

import pytest
from matplotlib.figure import Figure

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.exceptions import ValidationError
from femtoolkit.studies.extractors import get_extractor
from femtoolkit.studies.parameter_sweep import ParameterDefinition
from femtoolkit.studies.plots import plot_study_quantity
from femtoolkit.studies.runner import SimulationStudy, StudyRunner


def _base_project() -> Project:
    project = Project(name="Beam", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.mesh.width, project.mesh.height = 2.0, 0.4
    project.mesh.nx, project.mesh.ny, project.mesh.thickness = 6, 2, 0.02
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-1000.0)]
    return project


def test_plot_study_quantity_returns_figure() -> None:
    parameter = ParameterDefinition(
        path="loads.0.magnitude", label="Load (N)", values=[-1000.0, -2000.0, -3000.0]
    )
    study = SimulationStudy(
        study_id="s1", name="Load Study", base_project=_base_project(), parameters=[parameter]
    )
    result = StudyRunner().run(study)

    figure = plot_study_quantity(
        result, parameter, get_extractor("maximum_displacement"), "Maximum displacement (m)"
    )

    assert isinstance(figure, Figure)
    axes = figure.axes[0]
    assert axes.get_xlabel() == "Load (N)"
    assert axes.get_ylabel() == "Maximum displacement (m)"


def test_plot_study_quantity_no_successful_runs_raises() -> None:
    parameter = ParameterDefinition(path="material.youngs_modulus", label="E", values=[-1.0, -2.0])
    study = SimulationStudy(
        study_id="s1", name="All Invalid", base_project=_base_project(), parameters=[parameter]
    )
    result = StudyRunner().run(study)

    extractor = get_extractor("maximum_displacement")
    with pytest.raises(ValidationError):
        plot_study_quantity(result, parameter, extractor, "Displacement")
