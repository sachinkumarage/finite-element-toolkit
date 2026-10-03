"""Optimization algorithms: transparent, derivative-free design search (Version 32/33).

Two single-point algorithms (Version 32) are deliberately simple enough
that an engineer can trace exactly why they moved from one design to
the next -- no gradient, no probabilistic model, no black box:

- :class:`~femtoolkit.optimization.algorithms.random_search.BoundedRandomSearch`
  -- draws each candidate uniformly at random from the design space.
- :class:`~femtoolkit.optimization.algorithms.coordinate_search.CoordinateSearch`
  -- improves one design variable at a time from a starting point.

Four population-based algorithms (Version 33) add exploration/exploitation
dynamics a single-point search cannot express, while reusing exactly the
same :class:`~femtoolkit.optimization.algorithms.base.OptimizationAlgorithm`
interface, feasibility-first comparison, and evaluation pipeline:

- :class:`~femtoolkit.optimization.algorithms.differential_evolution.DifferentialEvolution`
- :class:`~femtoolkit.optimization.algorithms.genetic_algorithm.GeneticAlgorithm`
- :class:`~femtoolkit.optimization.algorithms.particle_swarm.ParticleSwarmOptimization`
- :class:`~femtoolkit.optimization.algorithms.nsga2.NSGA2` -- multi-objective

None of these six algorithms claims to find a global optimum; see each
class's own docstring and ``docs/optimization.md`` for what they do and
do not guarantee.
"""

from __future__ import annotations

from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.algorithms.base import (
    DEFAULT_EVALUATION_LIMIT,
    DEFAULT_MAX_EVALUATIONS,
    DEFAULT_MAX_GENERATIONS,
    SUPPORTED_ALGORITHMS,
    SUPPORTED_CONSTRAINT_HANDLING,
    OptimizationAlgorithm,
    OptimizationConfig,
    StopReason,
    should_stop,
)
from femtoolkit.optimization.algorithms.coordinate_search import CoordinateSearch
from femtoolkit.optimization.algorithms.differential_evolution import DifferentialEvolution
from femtoolkit.optimization.algorithms.genetic_algorithm import GeneticAlgorithm
from femtoolkit.optimization.algorithms.nsga2 import NSGA2
from femtoolkit.optimization.algorithms.particle_swarm import ParticleSwarmOptimization
from femtoolkit.optimization.algorithms.random_search import BoundedRandomSearch

_ALGORITHM_REGISTRY: dict[str, type[OptimizationAlgorithm]] = {
    "random_search": BoundedRandomSearch,
    "coordinate_search": CoordinateSearch,
    "differential_evolution": DifferentialEvolution,
    "genetic_algorithm": GeneticAlgorithm,
    "particle_swarm": ParticleSwarmOptimization,
    "nsga2": NSGA2,
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
    "DEFAULT_MAX_GENERATIONS",
    "SUPPORTED_ALGORITHMS",
    "SUPPORTED_CONSTRAINT_HANDLING",
    "NSGA2",
    "BoundedRandomSearch",
    "CoordinateSearch",
    "DifferentialEvolution",
    "GeneticAlgorithm",
    "OptimizationAlgorithm",
    "OptimizationConfig",
    "ParticleSwarmOptimization",
    "StopReason",
    "build_algorithm",
    "should_stop",
]
