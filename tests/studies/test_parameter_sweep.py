"""Tests for femtoolkit.studies.parameter_sweep."""

import pytest

from femtoolkit.exceptions import StudySizeExceededError, ValidationError
from femtoolkit.studies.parameter_sweep import (
    DEFAULT_MAX_SCENARIOS,
    ParameterDefinition,
    count_combinations,
    generate_scenarios,
)
from femtoolkit.studies.scenarios import validate_unique_scenario_ids


def test_parameter_definition_rejects_empty_values() -> None:
    with pytest.raises(ValidationError):
        ParameterDefinition(path="x", label="X", values=[])


def test_generate_scenarios_single_parameter_sweep() -> None:
    parameter = ParameterDefinition(
        path="loads.0.magnitude", label="Load", values=[-1000.0, -2000.0, -3000.0]
    )
    scenarios = generate_scenarios("Load Study", [parameter])

    assert len(scenarios) == 3
    assert [s.parameter_overrides["loads.0.magnitude"] for s in scenarios] == [
        -1000.0,
        -2000.0,
        -3000.0,
    ]
    validate_unique_scenario_ids(scenarios)


def test_generate_scenarios_multi_parameter_cartesian_product() -> None:
    load = ParameterDefinition(path="loads.0.magnitude", label="Load", values=[-1000.0, -2000.0])
    material = ParameterDefinition(
        path="material.youngs_modulus", label="E", values=[70e9, 200e9, 110e9]
    )
    scenarios = generate_scenarios("Multi", [load, material])

    assert len(scenarios) == count_combinations([load, material]) == 6
    combinations = {
        (
            s.parameter_overrides["loads.0.magnitude"],
            s.parameter_overrides["material.youngs_modulus"],
        )
        for s in scenarios
    }
    assert len(combinations) == 6
    validate_unique_scenario_ids(scenarios)


def test_generate_scenarios_requires_at_least_one_parameter() -> None:
    with pytest.raises(ValidationError):
        generate_scenarios("Empty", [])


def test_generate_scenarios_stops_before_execution_when_over_limit() -> None:
    parameters = [
        ParameterDefinition(path="mesh.nx", label=f"p{i}", values=list(range(5)))
        for i in range(4)
    ]
    assert count_combinations(parameters) == 625

    with pytest.raises(StudySizeExceededError):
        generate_scenarios("Too Big", parameters, max_scenarios=100)


def test_generate_scenarios_default_max_scenarios_constant() -> None:
    assert DEFAULT_MAX_SCENARIOS == 100
