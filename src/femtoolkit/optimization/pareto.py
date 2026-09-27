"""Pareto dominance and non-dominated filtering for multi-objective problems (Version 32).

**Engineering concept.** For two objectives to minimize, ``m`` (mass)
and ``u_max`` (tip displacement):

.. code-block:: text

    min [m(x), u_max(x)]

Design *A* **dominates** design *B* if *A* is no worse than *B* on
every objective, and strictly better on at least one. The
**non-dominated set** (the Pareto front) is every design no other
design dominates -- each one represents a different, equally valid
trade-off between the objectives; **this module never ranks the
non-dominated set into one "best" design** -- that judgment needs
engineering requirements (cost, manufacturability, ...) this toolkit
has no way to know, exactly the same principle Version 30's material
comparison already established.
"""

from __future__ import annotations

from femtoolkit.optimization.evaluation import DesignEvaluation, DesignStatus
from femtoolkit.optimization.objectives import Objective, ObjectiveDirection


def dominates(
    a: dict[str, float], b: dict[str, float], objectives: list[Objective]
) -> bool:
    """Whether objective vector ``a`` dominates objective vector ``b``.

    ``a`` dominates ``b`` if ``a`` is at least as good as ``b`` on
    every objective and strictly better on at least one -- accounting
    for each objective's own minimize/maximize direction.

    Args:
        a: The first design's objective values, by objective name.
        b: The second design's objective values, by objective name.
        objectives: The objectives to compare over (defines both which
            names to compare and each one's direction).

    Returns:
        ``True`` if ``a`` dominates ``b``.
    """
    at_least_as_good = True
    strictly_better_once = False
    for objective in objectives:
        a_value = a[objective.name]
        b_value = b[objective.name]
        if objective.direction is ObjectiveDirection.MINIMIZE:
            better = a_value < b_value
            as_good = a_value <= b_value
        else:
            better = a_value > b_value
            as_good = a_value >= b_value
        if not as_good:
            at_least_as_good = False
        if better:
            strictly_better_once = True
    return at_least_as_good and strictly_better_once


def pareto_front(
    evaluations: list[DesignEvaluation], objectives: list[Objective]
) -> list[DesignEvaluation]:
    """The non-dominated subset of every feasible evaluation, by objective vector.

    Infeasible, failed, and invalid evaluations are excluded -- a
    Pareto front is only meaningful among designs that actually satisfy
    every constraint. Two designs with identical objective vectors do
    not dominate each other (dominance requires a *strict* improvement
    on at least one objective) and both remain in the front.

    Args:
        evaluations: The evaluations to filter (typically an
            :class:`~femtoolkit.optimization.history.OptimizationHistory`'s
            ``evaluations``).
        objectives: The objectives defining the objective vector.

    Returns:
        Every non-dominated feasible evaluation, in the order they
        appeared in ``evaluations``. Never sorted or ranked into a
        single preferred solution.
    """
    feasible = [
        evaluation for evaluation in evaluations if evaluation.status is DesignStatus.FEASIBLE
    ]
    non_dominated: list[DesignEvaluation] = []
    for candidate in feasible:
        dominated = any(
            other is not candidate
            and dominates(other.objective_values, candidate.objective_values, objectives)
            for other in feasible
        )
        if not dominated:
            non_dominated.append(candidate)
    return non_dominated


__all__ = ["dominates", "pareto_front"]
