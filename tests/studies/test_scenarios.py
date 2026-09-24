"""Tests for femtoolkit.studies.scenarios."""

import pytest

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.exceptions import DuplicateScenarioIdError, ValidationError
from femtoolkit.studies.scenarios import Scenario, apply_scenario, validate_unique_scenario_ids


def _base_project() -> Project:
    project = Project(name="Base", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-1000.0)]
    project.boundary_conditions = [BoundaryConditionConfig(region="left", dof="X", value=0.0)]
    return project


def test_apply_scenario_overrides_top_level_field() -> None:
    base = _base_project()
    scenario = Scenario(
        scenario_id="s1", name="Aluminum", parameter_overrides={"material.youngs_modulus": 70e9}
    )
    varied = apply_scenario(base, scenario)

    assert varied.material.youngs_modulus == 70e9
    assert base.material.youngs_modulus == 200e9


def test_apply_scenario_overrides_list_element_field() -> None:
    base = _base_project()
    scenario = Scenario(
        scenario_id="s1", name="Bigger load", parameter_overrides={"loads.0.magnitude": -5000.0}
    )
    varied = apply_scenario(base, scenario)

    assert varied.loads[0].magnitude == -5000.0
    assert base.loads[0].magnitude == -1000.0


def test_apply_scenario_returns_independent_deep_copy() -> None:
    base = _base_project()
    scenario = Scenario(scenario_id="s1", name="No-op")
    varied = apply_scenario(base, scenario)

    assert varied is not base
    assert varied.loads is not base.loads
    varied.loads[0].magnitude = -9999.0
    assert base.loads[0].magnitude == -1000.0


def test_apply_scenario_solver_overrides() -> None:
    base = _base_project()
    scenario = Scenario(scenario_id="s1", name="Tighter tol", solver_overrides={"tolerance": 1e-9})
    varied = apply_scenario(base, scenario)

    assert varied.solver.tolerance == 1e-9
    assert base.solver.tolerance != 1e-9


def test_apply_scenario_invalid_field_path_raises() -> None:
    base = _base_project()
    scenario = Scenario(
        scenario_id="s1", name="Bad", parameter_overrides={"material.not_a_real_field": 1.0}
    )
    with pytest.raises(ValidationError):
        apply_scenario(base, scenario)


def test_apply_scenario_out_of_range_list_index_raises() -> None:
    base = _base_project()
    scenario = Scenario(
        scenario_id="s1", name="Bad index", parameter_overrides={"loads.5.magnitude": -1.0}
    )
    with pytest.raises(ValidationError):
        apply_scenario(base, scenario)


def test_validate_unique_scenario_ids_passes_for_unique_ids() -> None:
    validate_unique_scenario_ids(
        [Scenario(scenario_id="a", name="A"), Scenario(scenario_id="b", name="B")]
    )


def test_validate_unique_scenario_ids_raises_for_duplicate() -> None:
    with pytest.raises(DuplicateScenarioIdError):
        validate_unique_scenario_ids(
            [Scenario(scenario_id="dup", name="A"), Scenario(scenario_id="dup", name="B")]
        )
