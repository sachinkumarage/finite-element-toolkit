"""Shared real-valued encoding for population-based algorithms (Version 33).

Differential evolution, the genetic algorithm, particle swarm
optimization, and NSGA-II all search a "box" ``[lower, upper]`` in
real-valued vector space -- the natural representation for continuous
mutation/crossover arithmetic. This module is the one place that
representation is defined, so every population-based algorithm encodes
and decodes a design the same way instead of repeating the logic four
times:

- **Continuous** variables map directly to their own ``[lower, upper]``.
- **Integer** variables also map to ``[lower, upper]`` as a real
  number; decoding rounds to the nearest integer and clips back into
  range.
- **Categorical** variables map to ``[0, len(categories) - 1]``, the
  index into the category list; decoding rounds to the nearest valid
  index. This is the same "treat a category as a position in its own
  list" idea :mod:`~femtoolkit.optimization.algorithms.coordinate_search`
  already uses for stepping, generalized here to continuous arithmetic.

This module is private to :mod:`femtoolkit.optimization.algorithms` --
it is an implementation detail of how DE/GA/PSO/NSGA-II represent a
design internally, not part of this package's public interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from femtoolkit.optimization.variables import DesignVariable, DesignVariableType


def bounds_arrays(design_variables: list[DesignVariable]) -> tuple[np.ndarray, np.ndarray]:
    """The lower- and upper-bound vectors for every design variable's encoded range.

    Args:
        design_variables: The problem's design variable definitions.

    Returns:
        ``(lower, upper)``, each a 1D array with one entry per
        design variable, in the same order.
    """
    lower = np.empty(len(design_variables))
    upper = np.empty(len(design_variables))
    for index, variable in enumerate(design_variables):
        if variable.variable_type is DesignVariableType.CATEGORICAL:
            lower[index] = 0.0
            upper[index] = float(len(variable.categories) - 1)
        else:
            lower[index] = float(variable.lower_bound)
            upper[index] = float(variable.upper_bound)
    return lower, upper


def encode_values(design_variables: list[DesignVariable], values: dict[str, Any]) -> np.ndarray:
    """Encode a design's variable values as a real-valued vector.

    Args:
        design_variables: The problem's design variable definitions.
        values: The design's values, keyed by variable name.

    Returns:
        A 1D array with one entry per design variable, in the same order.
    """
    vector = np.empty(len(design_variables))
    for index, variable in enumerate(design_variables):
        value = values[variable.name]
        if variable.variable_type is DesignVariableType.CATEGORICAL:
            vector[index] = float(variable.categories.index(value))
        else:
            vector[index] = float(value)
    return vector


def decode_vector(design_variables: list[DesignVariable], vector: np.ndarray) -> dict[str, Any]:
    """Decode a real-valued vector back into a design's variable values.

    Rounds and clips as needed so the result always lies within every
    variable's own domain (:meth:`~femtoolkit.optimization.variables.DesignVariable.is_valid_value`
    is guaranteed true for every returned value).

    Args:
        design_variables: The problem's design variable definitions.
        vector: A 1D array with one entry per design variable, in the
            same order (typically produced by arithmetic on an
            :func:`encode_values`/:func:`sample_vector` result).

    Returns:
        The decoded values, keyed by variable name.
    """
    values: dict[str, Any] = {}
    for variable, component in zip(design_variables, vector, strict=True):
        if variable.variable_type is DesignVariableType.CONTINUOUS:
            values[variable.name] = variable.clip(float(component))
        elif variable.variable_type is DesignVariableType.INTEGER:
            values[variable.name] = variable.clip(round(float(component)))
        else:
            index = int(round(float(component)))
            index = min(max(index, 0), len(variable.categories) - 1)
            values[variable.name] = variable.categories[index]
    return values


def sample_vector(design_variables: list[DesignVariable], rng: np.random.Generator) -> np.ndarray:
    """Draw one real-valued vector uniformly at random within every variable's encoded range.

    Args:
        design_variables: The problem's design variable definitions.
        rng: The random generator to draw from.

    Returns:
        A 1D array with one entry per design variable, in the same order.
    """
    lower, upper = bounds_arrays(design_variables)
    return lower + rng.random(len(design_variables)) * (upper - lower)


__all__ = ["bounds_arrays", "decode_vector", "encode_values", "sample_vector"]
