"""Tests for femtoolkit.optimization.algorithms._encoding.

The shared real-valued box encoding every population-based algorithm
(differential evolution, genetic algorithm, particle swarm, NSGA-II)
uses to represent a mixed-type design as one vector.
"""

from __future__ import annotations

import numpy as np

from femtoolkit.optimization.algorithms._encoding import (
    bounds_arrays,
    decode_vector,
    encode_values,
    sample_vector,
)
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType

_CONTINUOUS = DesignVariable(
    name="thickness", path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
    lower_bound=0.005, upper_bound=0.02,
)
_INTEGER = DesignVariable(
    name="count", path="mesh.nx", variable_type=DesignVariableType.INTEGER,
    lower_bound=1, upper_bound=10,
)
_CATEGORICAL = DesignVariable(
    name="element_type", path="mesh.element_type", variable_type=DesignVariableType.CATEGORICAL,
    categories=["quad", "cst"],
)
_VARIABLES = [_CONTINUOUS, _INTEGER, _CATEGORICAL]


def test_bounds_arrays_continuous_and_integer_use_their_own_bounds() -> None:
    lower, upper = bounds_arrays(_VARIABLES)
    assert lower[0] == 0.005
    assert upper[0] == 0.02
    assert lower[1] == 1
    assert upper[1] == 10


def test_bounds_arrays_categorical_uses_index_range() -> None:
    lower, upper = bounds_arrays(_VARIABLES)
    assert lower[2] == 0.0
    assert upper[2] == 1.0  # two categories -> indices 0, 1


def test_encode_then_decode_round_trips_continuous() -> None:
    values = {"thickness": 0.012, "count": 5, "element_type": "cst"}
    vector = encode_values(_VARIABLES, values)
    decoded = decode_vector(_VARIABLES, vector)
    assert decoded["thickness"] == 0.012
    assert decoded["count"] == 5
    assert decoded["element_type"] == "cst"


def test_decode_vector_rounds_integer_component() -> None:
    vector = np.array([0.01, 4.6, 0.0])
    decoded = decode_vector(_VARIABLES, vector)
    assert decoded["count"] == 5  # 4.6 rounds to 5


def test_decode_vector_clips_out_of_range_continuous() -> None:
    vector = np.array([100.0, 5.0, 0.0])
    decoded = decode_vector(_VARIABLES, vector)
    assert decoded["thickness"] == 0.02  # clipped to upper bound


def test_decode_vector_clamps_categorical_index_to_valid_range() -> None:
    vector = np.array([0.01, 5.0, 5.0])  # index 5 is out of range for 2 categories
    decoded = decode_vector(_VARIABLES, vector)
    assert decoded["element_type"] in ("quad", "cst")
    assert decoded["element_type"] == "cst"  # clamped to the last valid index


def test_decode_vector_every_value_passes_variable_validation() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        vector = sample_vector(_VARIABLES, rng)
        decoded = decode_vector(_VARIABLES, vector)
        for variable in _VARIABLES:
            assert variable.is_valid_value(decoded[variable.name])


def test_sample_vector_is_reproducible_with_same_rng_seed() -> None:
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    vector_a = sample_vector(_VARIABLES, rng_a)
    vector_b = sample_vector(_VARIABLES, rng_b)
    assert np.array_equal(vector_a, vector_b)


def test_sample_vector_stays_within_bounds() -> None:
    rng = np.random.default_rng(1)
    lower, upper = bounds_arrays(_VARIABLES)
    for _ in range(50):
        vector = sample_vector(_VARIABLES, rng)
        assert np.all(vector >= lower)
        assert np.all(vector <= upper)
