"""Objectives: what an optimization is trying to minimize or maximize (Version 32).

An objective is a name, a direction (never left implicit -- see
:class:`ObjectiveDirection`), and an evaluation function,
``DesignContext -> float | None``. Nothing about what quantity an
objective measures is hard-coded into the optimization engine itself;
:func:`from_result_extractor` and :func:`rectangular_mass` are provided
as small, optional convenience builders for two very common cases (an
FEA result quantity, and mass of a rectangular mesh domain), not as
special cases the engine treats differently from any other objective a
caller defines.

**Picklability (Version 34).** :func:`from_result_extractor` and
:func:`robust_objective_mean` return small, module-level, frozen
``@dataclass`` callables (:class:`_ExtractorObjective`,
:class:`_RobustMeanObjective`) rather than nested-function closures.
This matters only for parallel optimization evaluation
(:mod:`femtoolkit.optimization.batch`): a :class:`~multiprocessing`
worker process can only receive a task payload that
:mod:`pickle` can serialize, and Python cannot pickle a closure (a
function defined inside another function) -- only a plain module-level
function or callable object. ``rectangular_mass`` needed no change; it
was already a plain module-level function. A caller-supplied ``lambda``
or closure passed directly as an :class:`Objective`'s ``evaluate``
remains perfectly valid for serial execution, but will raise
:exc:`~femtoolkit.exceptions.TaskSerializationError` if parallel
evaluation is requested for it -- an inherent property of Python
multiprocessing, not a toolkit defect.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.context import DesignContext
from femtoolkit.studies.extractors import Extractor

if TYPE_CHECKING:
    from femtoolkit.uncertainty.parameters import UncertainParameter

ObjectiveFunction = Callable[[DesignContext], float | None]


class ObjectiveDirection(Enum):
    """Whether an objective should be minimized or maximized.

    Never left implicit: every :class:`Objective` states its direction
    explicitly, and every report/plot labels it.
    """

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class Objective:
    """One quantity an optimization is trying to minimize or maximize.

    Attributes:
        name: A unique, human-readable identifier.
        direction: Whether to minimize or maximize this objective.
        evaluate: A callable computing this objective's value from a
            :class:`~femtoolkit.optimization.context.DesignContext`,
            returning ``None`` if it cannot be computed (e.g. the run
            did not complete, or a required result field is absent) --
            never raises for an ordinary "not available" case.
        units: A units string for display.
        description: A longer, free-text description.
    """

    name: str
    direction: ObjectiveDirection
    evaluate: ObjectiveFunction
    units: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("Objective requires a non-empty name.")


def validate_unique_objective_names(objectives: list[Objective]) -> None:
    """Check that every objective in ``objectives`` has a unique name.

    Args:
        objectives: The objectives to check.

    Raises:
        ValidationError: If any ``name`` appears more than once.
    """
    seen: set[str] = set()
    for objective in objectives:
        if objective.name in seen:
            raise ValidationError(
                f"Duplicate objective name {objective.name!r}; every objective must have "
                "a unique name."
            )
        seen.add(objective.name)


@dataclass(frozen=True)
class _ExtractorObjective:
    """Picklable callable wrapping one extractor as an objective function.

    A plain module-level class rather than a closure so that an
    :class:`Objective` built via :func:`from_result_extractor` can be
    sent to a worker process (see the module docstring's "Picklability"
    section). ``extractor`` itself must also be picklable -- true for
    every named extractor in :data:`~femtoolkit.studies.extractors.EXTRACTORS`,
    which are plain module-level functions.
    """

    extractor: Extractor

    def __call__(self, context: DesignContext) -> float | None:
        return self.extractor(context.run)


def from_result_extractor(extractor: Extractor) -> ObjectiveFunction:
    """Wrap an existing Version 30 result extractor as an objective evaluation function.

    Args:
        extractor: A named extractor from
            :data:`~femtoolkit.studies.extractors.EXTRACTORS`, or any
            callable with the same ``SimulationRun -> float | None`` signature.

    Returns:
        An :data:`ObjectiveFunction` reading the extractor from ``context.run``.
    """
    return _ExtractorObjective(extractor)


def rectangular_mass(context: DesignContext) -> float | None:
    """Mass of a rectangular mesh domain: ``width * height * thickness * density``.

    A ready-to-use :data:`ObjectiveFunction` for the common "minimize
    mass" objective on this toolkit's rectangular 2D domains -- purely
    a convenience built from the project's own configuration, not a
    quantity the optimization engine treats specially.

    Args:
        context: The design's evaluation context.

    Returns:
        The mass in kilograms, or ``None`` if the project's material
        has no configured density.
    """
    mesh = context.project.mesh
    density = context.project.material.density
    if density is None:
        return None
    return mesh.width * mesh.height * mesh.thickness * density


@dataclass(frozen=True)
class _RobustMeanObjective:
    """Picklable callable implementing :func:`robust_objective_mean`.

    A plain module-level class rather than a closure (see the module
    docstring's "Picklability" section). Sending this to a worker
    process additionally requires ``build_parameters`` itself to be
    picklable -- true for a plain module-level function, not for a
    ``lambda`` or another closure.
    """

    build_parameters: Callable[[DesignContext], list[UncertainParameter]]
    quantity_name: str
    n_samples: int = 20
    seed: int | None = None

    def __call__(self, context: DesignContext) -> float | None:
        from femtoolkit.studies.extractors import get_extractor
        from femtoolkit.uncertainty.monte_carlo import MonteCarloConfig, MonteCarloRunner

        parameters = self.build_parameters(context)
        if not parameters:
            return None
        config = MonteCarloConfig(
            study_id="robust-objective-sample",
            name="Robust objective sample",
            base_project=context.project,
            parameters=parameters,
            output_quantities=[self.quantity_name],
            n_samples=self.n_samples,
            seed=self.seed,
        )
        result = MonteCarloRunner().run(config)
        values = result.output_values(get_extractor(self.quantity_name))
        if values.size == 0:
            return None
        return float(values.mean())


def robust_objective_mean(
    build_parameters: Callable[[DesignContext], list[UncertainParameter]],
    quantity_name: str,
    n_samples: int = 20,
    seed: int | None = None,
) -> ObjectiveFunction:
    """Build an objective evaluating the mean of a quantity over a small uncertainty study.

    A lightweight, optional building block toward uncertainty-aware
    ("robust") optimization (see
    :attr:`~femtoolkit.optimization.problems.OptimizationMode.UNCERTAINTY_AWARE`):
    at each design point, this runs a small Version 31 Monte Carlo
    study (reusing :class:`~femtoolkit.uncertainty.monte_carlo.MonteCarloRunner`
    directly, not a second execution path) and returns the mean of the
    requested output quantity across it, instead of a single
    deterministic value. This is **not** a robust-optimization
    algorithm -- it only changes what one objective evaluation means;
    see the Version 33 preview in ``README.md`` for genuine robust
    optimization and reliability constraints.

    Args:
        build_parameters: Given the current design's
            :class:`~femtoolkit.optimization.context.DesignContext`,
            returns the list of
            :class:`~femtoolkit.uncertainty.parameters.UncertainParameter`
            to sample around that design point (e.g. material property
            scatter around the design's own nominal values).
        quantity_name: A named extractor from
            :data:`~femtoolkit.studies.extractors.EXTRACTORS`.
        n_samples: How many Monte Carlo samples to draw per design evaluation.
        seed: A random seed for reproducibility.

    Returns:
        An :data:`ObjectiveFunction` returning the sampled mean, or
        ``None`` if no successful sample was obtained.
    """
    return _RobustMeanObjective(build_parameters, quantity_name, n_samples, seed)


__all__ = [
    "Objective",
    "ObjectiveDirection",
    "ObjectiveFunction",
    "from_result_extractor",
    "rectangular_mass",
    "robust_objective_mean",
    "validate_unique_objective_names",
]
