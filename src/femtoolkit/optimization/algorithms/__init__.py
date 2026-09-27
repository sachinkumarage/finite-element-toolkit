"""Optimization algorithms: transparent, derivative-free design search (Version 32).

Two algorithms are implemented, both deliberately simple enough that an
engineer can trace exactly why they moved from one design to the next
-- no gradient, no probabilistic model, no black box:

- :class:`~femtoolkit.optimization.algorithms.random_search.BoundedRandomSearch`
  -- draws each candidate uniformly at random from the design space.
- :class:`~femtoolkit.optimization.algorithms.coordinate_search.CoordinateSearch`
  -- improves one design variable at a time from a starting point.

Neither claims to find a global optimum; see each class's own
docstring and ``docs/optimization.md`` for what they do and do not
guarantee. A larger algorithm collection (differential evolution,
genetic algorithms, particle swarm) is explicitly out of scope for this
version -- see the Version 33 preview in ``README.md``.
"""

from __future__ import annotations

from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.algorithms.base import (
    DEFAULT_EVALUATION_LIMIT,
    DEFAULT_MAX_EVALUATIONS,
    SUPPORTED_ALGORITHMS,
    SUPPORTED_CONSTRAINT_HANDLING,
    OptimizationAlgorithm,
    OptimizationConfig,
    StopReason,
    should_stop,
)
from femtoolkit.optimization.algorithms.coordinate_search import CoordinateSearch
from femtoolkit.optimization.algorithms.random_search import BoundedRandomSearch

_ALGORITHM_REGISTRY: dict[str, type[OptimizationAlgorithm]] = {
    "random_search": BoundedRandomSearch,
    "coordinate_search": CoordinateSearch,
}


def build_algorithm(name: str) -> OptimizationAlgorithm:
    """Construct the named algorithm.

    Args:
        name: One of :data:`~femtoolkit.optimization.algorithms.base.SUPPORTED_ALGORITHMS`.

    Returns:
        A fresh :class:`~femtoolkit.optimization.algorithms.base.OptimizationAlgorithm` instance.

    Raises:
        ValidationError: If ``name`` is not a known algorithm.
    """
    algorithm_cls = _ALGORITHM_REGISTRY.get(name)
    if algorithm_cls is None:
        raise ValidationError(
            f"Unknown algorithm {name!r}; expected one of {SUPPORTED_ALGORITHMS}."
        )
    return algorithm_cls()


__all__ = [
    "DEFAULT_EVALUATION_LIMIT",
    "DEFAULT_MAX_EVALUATIONS",
    "SUPPORTED_ALGORITHMS",
    "SUPPORTED_CONSTRAINT_HANDLING",
    "BoundedRandomSearch",
    "CoordinateSearch",
    "OptimizationAlgorithm",
    "OptimizationConfig",
    "StopReason",
    "build_algorithm",
    "should_stop",
]
