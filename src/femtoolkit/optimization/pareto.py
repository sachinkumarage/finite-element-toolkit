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

import math

from femtoolkit.optimization.evaluation import DesignEvaluation, DesignStatus, feasibility_rank
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


def constrained_dominates(
    a: DesignEvaluation, b: DesignEvaluation, objectives: list[Objective]
) -> bool:
    """Whether ``a`` dominates ``b`` once feasibility is accounted for (Version 33).

    Extends plain Pareto :func:`dominates` with the same feasibility-first
    principle :func:`~femtoolkit.optimization.evaluation.is_better_evaluation`
    already uses for single-objective comparisons, generalized to NSGA-II's
    constrained-domination rule:

    1. A design with a better :func:`~femtoolkit.optimization.evaluation.feasibility_rank`
       (lower is better: feasible < infeasible < failed/invalid) always
       dominates one with a worse rank, regardless of objective values.
    2. Between two designs tied on feasibility rank:

       - Both ``FEASIBLE``: ordinary Pareto dominance over the objective
         vectors (see :func:`dominates`).
       - Both ``INFEASIBLE``: the smaller total constraint violation
         dominates (never combined into the objective comparison).
       - Both ``FAILED``/``INVALID``: neither dominates the other --
         there is no usable objective value or violation to compare.

    Args:
        a: The first design's evaluation.
        b: The second design's evaluation.
        objectives: The objectives defining the objective vector.

    Returns:
        ``True`` if ``a`` dominates ``b``.
    """
    rank_a = feasibility_rank(a.status)
    rank_b = feasibility_rank(b.status)
    if rank_a != rank_b:
        return rank_a < rank_b
    if a.status is DesignStatus.FEASIBLE:
        return dominates(a.objective_values, b.objective_values, objectives)
    if a.status is DesignStatus.INFEASIBLE:
        return a.total_violation < b.total_violation
    return False


def fast_non_dominated_sort(
    evaluations: list[DesignEvaluation], objectives: list[Objective]
) -> list[list[DesignEvaluation]]:
    """Partition ``evaluations`` into successive non-dominated fronts (NSGA-II, Version 33).

    **Engineering/algorithmic concept.** The first front is every design
    no other design dominates (see :func:`constrained_dominates`,
    accounting for feasibility as well as the objective vector). The
    second front is every design dominated only by designs in the first
    front, and so on -- this is the classic "non-dominated sorting"
    step of NSGA-II (Deb et al., 2002), generalized here to rank
    infeasible and failed designs below feasible ones rather than
    silently discarding them, so a population-based search still has a
    meaningful ordering to select against even before any design is
    feasible.

    Args:
        evaluations: The evaluations to sort (typically one generation's
            combined parent+offspring population).
        objectives: The objectives defining the objective vector for
            feasible-vs-feasible comparisons.

    Returns:
        A list of fronts, each a list of evaluations, in rank order
        (index 0 is the best front). Every input evaluation appears in
        exactly one front.
    """
    n = len(evaluations)
    dominated_by: list[list[int]] = [[] for _ in range(n)]
    domination_count = [0] * n
    fronts: list[list[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if constrained_dominates(evaluations[i], evaluations[j], objectives):
                dominated_by[i].append(j)
            elif constrained_dominates(evaluations[j], evaluations[i], objectives):
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)

    rank = 0
    while fronts[rank]:
        next_front: list[int] = []
        for i in fronts[rank]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        rank += 1
        fronts.append(next_front)

    return [[evaluations[i] for i in front] for front in fronts if front]


def crowding_distance(
    front: list[DesignEvaluation], objectives: list[Objective]
) -> dict[str, float]:
    """NSGA-II crowding distance: how isolated each design is within its own front.

    **Engineering/algorithmic concept.** Within one non-dominated front,
    two designs are equally "good" by dominance alone -- crowding
    distance breaks ties in favor of designs that sit in a sparser part
    of objective space, which keeps the population spread across the
    whole front rather than clustering in one region. For each
    objective, the front is sorted by that objective's value; the two
    boundary designs (best and worst) are assigned infinite distance
    (always preferred, to preserve the extremes of the trade-off), and
    every interior design accumulates the normalized gap between its
    two neighbors.

    Args:
        front: One non-dominated front (e.g. one entry from
            :func:`fast_non_dominated_sort`'s result). Only feasible
            designs are meaningfully ranked this way; a front containing
            fewer than three designs returns infinite distance for all
            of them (nothing to compare against).
        objectives: The objectives defining the objective vector.

    Returns:
        Each design's crowding distance, keyed by
        :attr:`~femtoolkit.optimization.evaluation.DesignEvaluation.design_id`.
        Larger is less crowded (more preferred as a tie-breaker).
    """
    distances = {evaluation.design_id: 0.0 for evaluation in front}
    n = len(front)
    if n <= 2:
        return {evaluation.design_id: math.inf for evaluation in front}

    for objective in objectives:
        ordered = sorted(front, key=lambda e: e.objective_values[objective.name])
        low = ordered[0].objective_values[objective.name]
        high = ordered[-1].objective_values[objective.name]
        distances[ordered[0].design_id] = math.inf
        distances[ordered[-1].design_id] = math.inf
        span = high - low
        if span <= 0.0:
            continue
        for index in range(1, n - 1):
            previous_value = ordered[index - 1].objective_values[objective.name]
            next_value = ordered[index + 1].objective_values[objective.name]
            design_id = ordered[index].design_id
            if distances[design_id] != math.inf:
                distances[design_id] += (next_value - previous_value) / span

    return distances


__all__ = [
    "constrained_dominates",
    "crowding_distance",
    "dominates",
    "fast_non_dominated_sort",
    "pareto_front",
]
