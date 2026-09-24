"""Named result-quantity extractors for comparison and plotting (Version 30).

Every extractor here is a thin accessor over fields that already exist
on a :class:`~femtoolkit.runs.models.SimulationRun` -- the Version 22
:class:`~femtoolkit.postprocessing.field_calculator.EngineeringSummary`,
the Version 26
:class:`~femtoolkit.solvers.results.SolverResult` diagnostics, and the
run's own recorded execution time. No extractor computes a new physical
quantity; this module exists only so
:func:`~femtoolkit.studies.comparison.compare_runs` and study plots can
refer to a quantity by a stable name instead of every caller writing its
own ``lambda run: ...``.
"""

from __future__ import annotations

from collections.abc import Callable

from femtoolkit.exceptions import ValidationError
from femtoolkit.runs.models import SimulationRun

Extractor = Callable[[SimulationRun], float | None]


def maximum_displacement(run: SimulationRun) -> float | None:
    """The run's maximum displacement magnitude, or ``None`` if unavailable."""
    if run.result is None or run.result.summary is None:
        return None
    return run.result.summary.maximum_displacement


def maximum_von_mises_stress(run: SimulationRun) -> float | None:
    """The run's maximum von Mises stress, or ``None`` if unavailable."""
    if run.result is None or run.result.summary is None:
        return None
    return run.result.summary.maximum_von_mises_stress


def maximum_temperature(run: SimulationRun) -> float | None:
    """The run's maximum nodal temperature, or ``None`` if unavailable."""
    if run.result is None or run.result.summary is None:
        return None
    return run.result.summary.maximum_temperature


def maximum_heat_flux(run: SimulationRun) -> float | None:
    """The run's maximum heat-flux magnitude, or ``None`` if unavailable."""
    if run.result is None or run.result.summary is None:
        return None
    return run.result.summary.maximum_heat_flux


def solver_iterations(run: SimulationRun) -> float | None:
    """The run's solver iteration count, or ``None`` for a direct solve (no diagnostics)."""
    if run.result is None or run.result.solver_diagnostics is None:
        return None
    iterations = run.result.solver_diagnostics.iterations
    return float(iterations) if iterations is not None else None


def execution_time(run: SimulationRun) -> float | None:
    """The run's wall-clock execution time in seconds, or ``None`` if the run never finished."""
    return run.execution_time_seconds


EXTRACTORS: dict[str, Extractor] = {
    "maximum_displacement": maximum_displacement,
    "maximum_von_mises_stress": maximum_von_mises_stress,
    "maximum_temperature": maximum_temperature,
    "maximum_heat_flux": maximum_heat_flux,
    "solver_iterations": solver_iterations,
    "execution_time": execution_time,
}
"""Every named extractor, keyed by the name :func:`get_extractor` accepts."""


def get_extractor(name: str) -> Extractor:
    """Look up a named extractor from :data:`EXTRACTORS`.

    Args:
        name: One of :data:`EXTRACTORS`' keys.

    Returns:
        The matching extractor callable.

    Raises:
        ValidationError: If ``name`` is not a known extractor.
    """
    try:
        return EXTRACTORS[name]
    except KeyError as exc:
        available = ", ".join(sorted(EXTRACTORS))
        raise ValidationError(
            f"Unknown result quantity {name!r}. Available quantities: {available}."
        ) from exc
