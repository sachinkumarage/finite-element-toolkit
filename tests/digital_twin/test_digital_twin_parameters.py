"""Tests for femtoolkit.digital_twin.parameters."""

from __future__ import annotations

import pytest

from femtoolkit.digital_twin.parameters import ModelParameter, validate_unique_parameter_names
from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.variables import DesignVariable


def test_valid_parameter() -> None:
    parameter = ModelParameter(
        name="youngs_modulus", current_value=200e9, lower_bound=180e9, upper_bound=220e9,
        units="Pa", updatable=True,
    )
    assert parameter.path == "youngs_modulus"
    assert parameter.updatable is True


def test_parameter_path_defaults_to_name() -> None:
    parameter = ModelParameter(
        name="thickness", current_value=0.01, lower_bound=0.005, upper_bound=0.02
    )
    assert parameter.path == "thickness"


def test_parameter_explicit_path() -> None:
    parameter = ModelParameter(
        name="youngs_modulus", current_value=200e9, lower_bound=180e9, upper_bound=220e9,
        path="material.youngs_modulus",
    )
    assert parameter.path == "material.youngs_modulus"


def test_parameter_requires_non_empty_name() -> None:
    with pytest.raises(ValidationError):
        ModelParameter(name="", current_value=1.0, lower_bound=0.0, upper_bound=2.0)


def test_parameter_rejects_lower_bound_not_less_than_upper_bound() -> None:
    with pytest.raises(ValidationError):
        ModelParameter(name="x", current_value=1.0, lower_bound=2.0, upper_bound=1.0)
    with pytest.raises(ValidationError):
        ModelParameter(name="x", current_value=1.0, lower_bound=1.0, upper_bound=1.0)


def test_parameter_rejects_current_value_outside_bounds() -> None:
    with pytest.raises(ValidationError):
        ModelParameter(name="x", current_value=5.0, lower_bound=0.0, upper_bound=2.0)


def test_parameter_rejects_non_finite_values() -> None:
    with pytest.raises(ValidationError):
        ModelParameter(name="x", current_value=float("nan"), lower_bound=0.0, upper_bound=2.0)
    with pytest.raises(ValidationError):
        ModelParameter(name="x", current_value=1.0, lower_bound=float("-inf"), upper_bound=2.0)


def test_fixed_parameter_cannot_become_a_design_variable() -> None:
    parameter = ModelParameter(
        name="density", current_value=7850.0, lower_bound=7000.0, upper_bound=8500.0,
        updatable=False,
    )
    with pytest.raises(ValidationError):
        parameter.to_design_variable()


def test_updatable_parameter_to_design_variable() -> None:
    parameter = ModelParameter(
        name="youngs_modulus", current_value=200e9, lower_bound=180e9, upper_bound=220e9,
        path="material.youngs_modulus",
    )
    variable = parameter.to_design_variable()
    assert isinstance(variable, DesignVariable)
    assert variable.name == "youngs_modulus"
    assert variable.path == "material.youngs_modulus"
    assert variable.lower_bound == 180e9
    assert variable.upper_bound == 220e9
    assert variable.default_value == 200e9


def test_validate_unique_parameter_names_detects_duplicates() -> None:
    parameters = [
        ModelParameter(name="x", current_value=1.0, lower_bound=0.0, upper_bound=2.0),
        ModelParameter(name="x", current_value=1.0, lower_bound=0.0, upper_bound=2.0),
    ]
    with pytest.raises(ValidationError):
        validate_unique_parameter_names(parameters)


def test_validate_unique_parameter_names_accepts_unique() -> None:
    parameters = [
        ModelParameter(name="x", current_value=1.0, lower_bound=0.0, upper_bound=2.0),
        ModelParameter(name="y", current_value=1.0, lower_bound=0.0, upper_bound=2.0),
    ]
    validate_unique_parameter_names(parameters)
