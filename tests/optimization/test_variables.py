"""Tests for femtoolkit.optimization.variables."""

import numpy as np
import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.variables import (
    DesignVariable,
    DesignVariableType,
    validate_unique_variable_names,
)

# --- continuous ---


def test_continuous_default_is_midpoint() -> None:
    variable = DesignVariable(
        name="t", path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=0.005, upper_bound=0.020,
    )
    assert variable.default_value == pytest.approx(0.0125)


def test_continuous_requires_strict_lower_less_than_upper() -> None:
    with pytest.raises(ValidationError):
        DesignVariable(
            name="t", path="x", variable_type=DesignVariableType.CONTINUOUS,
            lower_bound=1.0, upper_bound=1.0,
        )


def test_continuous_is_valid_value() -> None:
    variable = DesignVariable(
        name="t", path="x", variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=0.0, upper_bound=1.0,
    )
    assert variable.is_valid_value(0.5)
    assert variable.is_valid_value(0.0)
    assert variable.is_valid_value(1.0)
    assert not variable.is_valid_value(1.5)
    assert not variable.is_valid_value(-0.1)
    assert not variable.is_valid_value(float("nan"))
    assert not variable.is_valid_value("0.5")
    assert not variable.is_valid_value(True)


def test_continuous_clip() -> None:
    variable = DesignVariable(
        name="t", path="x", variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=0.0, upper_bound=1.0,
    )
    assert variable.clip(-5.0) == 0.0
    assert variable.clip(5.0) == 1.0
    assert variable.clip(0.3) == 0.3


def test_continuous_sample_stays_within_bounds() -> None:
    variable = DesignVariable(
        name="t", path="x", variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=-2.0, upper_bound=5.0,
    )
    rng = np.random.default_rng(0)
    samples = [variable.sample(rng) for _ in range(2000)]
    assert all(-2.0 <= s <= 5.0 for s in samples)


# --- integer ---


def test_integer_default_is_floor_midpoint() -> None:
    variable = DesignVariable(
        name="n", path="mesh.nx", variable_type=DesignVariableType.INTEGER,
        lower_bound=1, upper_bound=10,
    )
    assert variable.default_value == 5


def test_integer_requires_integer_bounds() -> None:
    with pytest.raises(ValidationError):
        DesignVariable(
            name="n", path="x", variable_type=DesignVariableType.INTEGER,
            lower_bound=1.5, upper_bound=10,
        )


def test_integer_allows_equal_bounds() -> None:
    variable = DesignVariable(
        name="n", path="x", variable_type=DesignVariableType.INTEGER, lower_bound=5, upper_bound=5
    )
    assert variable.default_value == 5


def test_integer_is_valid_value() -> None:
    variable = DesignVariable(
        name="n", path="x", variable_type=DesignVariableType.INTEGER, lower_bound=1, upper_bound=10
    )
    assert variable.is_valid_value(5)
    assert not variable.is_valid_value(5.5)
    assert not variable.is_valid_value(11)
    assert not variable.is_valid_value(True)


def test_integer_sample_and_clip_are_integers_in_range() -> None:
    variable = DesignVariable(
        name="n", path="x", variable_type=DesignVariableType.INTEGER, lower_bound=1, upper_bound=5
    )
    rng = np.random.default_rng(1)
    samples = [variable.sample(rng) for _ in range(500)]
    assert all(isinstance(s, int) and 1 <= s <= 5 for s in samples)
    assert variable.clip(100) == 5
    assert variable.clip(-100) == 1


# --- categorical ---


def test_categorical_default_is_first_category() -> None:
    variable = DesignVariable(
        name="m", path="material.name", variable_type=DesignVariableType.CATEGORICAL,
        categories=["Steel", "Aluminum", "Titanium"],
    )
    assert variable.default_value == "Steel"


def test_categorical_requires_nonempty_categories() -> None:
    with pytest.raises(ValidationError):
        DesignVariable(
            name="m", path="x", variable_type=DesignVariableType.CATEGORICAL, categories=[]
        )


def test_categorical_requires_unique_categories() -> None:
    with pytest.raises(ValidationError):
        DesignVariable(
            name="m", path="x", variable_type=DesignVariableType.CATEGORICAL,
            categories=["Steel", "Steel"],
        )


def test_categorical_is_valid_value() -> None:
    variable = DesignVariable(
        name="m", path="x", variable_type=DesignVariableType.CATEGORICAL,
        categories=["Steel", "Aluminum"],
    )
    assert variable.is_valid_value("Steel")
    assert not variable.is_valid_value("Copper")


def test_categorical_clip_rejects_unknown_value() -> None:
    variable = DesignVariable(
        name="m", path="x", variable_type=DesignVariableType.CATEGORICAL,
        categories=["Steel", "Aluminum"],
    )
    assert variable.clip("Aluminum") == "Aluminum"
    with pytest.raises(ValidationError):
        variable.clip("Copper")


def test_categorical_sample_returns_a_category() -> None:
    variable = DesignVariable(
        name="m", path="x", variable_type=DesignVariableType.CATEGORICAL,
        categories=["Steel", "Aluminum", "Titanium"],
    )
    rng = np.random.default_rng(2)
    samples = {variable.sample(rng) for _ in range(200)}
    assert samples <= {"Steel", "Aluminum", "Titanium"}


# --- default value validation and uniqueness ---


def test_default_value_outside_domain_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DesignVariable(
            name="t", path="x", variable_type=DesignVariableType.CONTINUOUS,
            lower_bound=0.0, upper_bound=1.0, default_value=5.0,
        )


def _continuous(name: str, path: str) -> DesignVariable:
    return DesignVariable(
        name=name, path=path, variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=0, upper_bound=1,
    )


def test_validate_unique_variable_names_passes_for_unique_names() -> None:
    validate_unique_variable_names([_continuous("a", "x"), _continuous("b", "y")])


def test_validate_unique_variable_names_rejects_duplicates() -> None:
    with pytest.raises(ValidationError):
        validate_unique_variable_names([_continuous("a", "x"), _continuous("a", "y")])
