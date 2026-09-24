"""Tests for femtoolkit.studies.runner and femtoolkit.studies.results."""

import pytest

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.exceptions import DuplicateScenarioIdError, StudySizeExceededError
from femtoolkit.runs.models import RunStatus
from femtoolkit.studies.extractors import get_extractor
from femtoolkit.studies.parameter_sweep import ParameterDefinition
from femtoolkit.studies.runner import SimulationStudy, StudyRunner
from femtoolkit.studies.scenarios import Scenario


def _base_project() -> Project:
    project = Project(name="Cantilever", analysis_type="linear_static")
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


def _load_parameter(values: list[float]) -> ParameterDefinition:
    return ParameterDefinition(path="loads.0.magnitude", label="Tip load (N)", values=values)


def test_study_runner_single_parameter_sweep_all_succeed() -> None:
    base = _base_project()
    study = SimulationStudy(
        study_id="study-1",
        name="Load Study",
        base_project=base,
        parameters=[_load_parameter([-1000.0, -2000.0, -3000.0])],
    )
    result = StudyRunner().run(study)

    assert len(result.runs) == 3
    assert len(result.successful_runs) == 3
    assert len(result.failed_runs) == 0
    assert result.base_project_id == base.project_id
    assert all(run.status == RunStatus.COMPLETED for run in result.runs)


def test_study_runner_verification_summary_reflects_real_checks() -> None:
    study = SimulationStudy(
        study_id="study-1",
        name="Load Study",
        base_project=_base_project(),
        parameters=[_load_parameter([-1000.0, -2000.0])],
    )
    result = StudyRunner().run(study)
    assert result.verification_summary() == {"pass": 2}
    assert result.validation_summary() == {"not_validated": 2}


def test_study_runner_compare_and_sensitivity() -> None:
    parameter = _load_parameter([-1000.0, -2000.0, -3000.0])
    study = SimulationStudy(
        study_id="study-1", name="Load Study", base_project=_base_project(), parameters=[parameter]
    )
    result = StudyRunner().run(study)
    extractor = get_extractor("maximum_displacement")

    comparison = result.compare(extractor, "Maximum displacement")
    assert len(comparison.entries) == 2
    assert comparison.entries[0].percentage_change == pytest.approx(100.0, rel=1e-6)

    sensitivities = result.sensitivity(parameter, extractor, "Maximum displacement")
    assert len(sensitivities) == 2
    assert all(abs(s.sensitivity) == pytest.approx(1.0, rel=1e-6) for s in sensitivities)


def test_study_runner_explicit_scenarios_and_parameters_combine() -> None:
    explicit = Scenario(
        scenario_id="explicit-1",
        name="Explicit override",
        parameter_overrides={"material.youngs_modulus": 70e9},
    )
    study = SimulationStudy(
        study_id="study-1",
        name="Combined",
        base_project=_base_project(),
        parameters=[_load_parameter([-1000.0, -2000.0])],
        scenarios=[explicit],
    )
    result = StudyRunner().run(study)

    assert len(result.runs) == 3
    assert {s.scenario_id for s in result.scenarios} == {"explicit-1", "Combined-0", "Combined-1"}


def test_study_runner_stops_before_execution_when_over_limit() -> None:
    study = SimulationStudy(
        study_id="study-1",
        name="Too Big",
        base_project=_base_project(),
        parameters=[ParameterDefinition(path="mesh.nx", label="nx", values=list(range(1, 20)))],
        max_scenarios=5,
    )
    with pytest.raises(StudySizeExceededError):
        StudyRunner().run(study)


def test_study_runner_duplicate_explicit_and_generated_ids_raise() -> None:
    duplicate = Scenario(scenario_id="Dup-0", name="Manual duplicate")
    study = SimulationStudy(
        study_id="study-1",
        name="Dup",
        base_project=_base_project(),
        parameters=[_load_parameter([-1000.0])],
        scenarios=[duplicate],
    )
    with pytest.raises(DuplicateScenarioIdError):
        StudyRunner().run(study)


def test_study_runner_records_failed_scenarios_without_aborting_study() -> None:
    valid_parameter = _load_parameter([-1000.0])
    bad_material = Scenario(
        scenario_id="bad-material",
        name="Invalid material",
        parameter_overrides={"material.youngs_modulus": -1.0},
    )
    study = SimulationStudy(
        study_id="study-1",
        name="Mixed",
        base_project=_base_project(),
        parameters=[valid_parameter],
        scenarios=[bad_material],
    )
    result = StudyRunner().run(study)

    assert len(result.runs) == 2
    assert len(result.successful_runs) == 1
    assert len(result.failed_runs) == 1
    assert result.failed_runs[0].error_stage == "validation"
