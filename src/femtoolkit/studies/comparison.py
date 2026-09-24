"""Result comparison: "how did the result change?" (Version 30).

This is a deliberately different question from
:mod:`femtoolkit.verification.metrics` ("how wrong is this result
compared to a reference?"), which is why this module does not reuse
those functions. Verification error metrics are unsigned (``abs(...)``)
because a numerical error's *sign* is not meaningful -- only its
magnitude matters. A parameter study's result *change* is the opposite:
the sign is the whole point (did displacement go up or down when the
load increased?). :func:`absolute_difference`/:func:`relative_difference`/
:func:`percentage_change` are therefore signed: ``value2 - value1``, not
``|value2 - value1|``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from femtoolkit.exceptions import ValidationError
from femtoolkit.runs.models import SimulationRun

DEFAULT_EPSILON = 1e-12
"""Floor applied to a denominator so a relative/percentage comparison
against a near-zero baseline does not divide by (near) zero."""


def absolute_difference(value1: float, value2: float) -> float:
    """Return the signed absolute difference ``value2 - value1``."""
    return value2 - value1


def relative_difference(value1: float, value2: float, epsilon: float = DEFAULT_EPSILON) -> float:
    """Return the signed relative difference ``(value2 - value1) / max(|value1|, epsilon)``."""
    return (value2 - value1) / max(abs(value1), epsilon)


def percentage_change(value1: float, value2: float, epsilon: float = DEFAULT_EPSILON) -> float:
    """Return the signed percentage change ``relative_difference(...) * 100``."""
    return relative_difference(value1, value2, epsilon) * 100.0


@dataclass(frozen=True)
class ComparisonEntry:
    """One pairwise result comparison between two runs.

    Attributes:
        quantity_label: The compared quantity's name (e.g. ``"Maximum
            displacement"``).
        run_id_1: The baseline run's ID.
        run_id_2: The comparison run's ID.
        value1: The baseline run's extracted value.
        value2: The comparison run's extracted value.
        absolute_difference: ``value2 - value1``.
        relative_difference: ``(value2 - value1) / max(|value1|, eps)``.
        percentage_change: ``relative_difference * 100``.
    """

    quantity_label: str
    run_id_1: str
    run_id_2: str
    value1: float
    value2: float
    absolute_difference: float
    relative_difference: float
    percentage_change: float


@dataclass(frozen=True)
class ComparisonResult:
    """Every pairwise comparison of one quantity across a sequence of runs.

    Attributes:
        quantity_label: The compared quantity's name.
        baseline_run_id: The run every other run was compared against.
        entries: One :class:`ComparisonEntry` per non-baseline run, in
            the same order the runs were supplied.
    """

    quantity_label: str
    baseline_run_id: str
    entries: list[ComparisonEntry]


def compare_runs(
    runs: list[SimulationRun],
    extractor: Callable[[SimulationRun], float | None],
    quantity_label: str,
    epsilon: float = DEFAULT_EPSILON,
) -> ComparisonResult:
    """Compare a quantity across ``runs`` against the first run as the baseline.

    Args:
        runs: The runs to compare, in order. ``runs[0]`` is the
            baseline every other run is compared against.
        extractor: A callable extracting the quantity to compare from
            one run (e.g. ``lambda run:
            run.result.summary.maximum_displacement``). Reuses whatever
            scalar the caller already has -- this function never
            re-derives a result quantity itself.
        quantity_label: A human-readable name for the compared quantity.
        epsilon: The denominator floor for relative/percentage
            comparisons (see :data:`DEFAULT_EPSILON`).

    Returns:
        A :class:`ComparisonResult` with one entry per run after the
        baseline.

    Raises:
        ValidationError: If ``runs`` has fewer than two runs, or if the
            extractor returns ``None`` for any run (the quantity is not
            available on that run's result -- e.g. a failed run, or a
            quantity this analysis type never produces).
    """
    if len(runs) < 2:
        raise ValidationError("compare_runs requires at least two runs to compare.")

    baseline = runs[0]
    baseline_value = extractor(baseline)
    if baseline_value is None:
        raise ValidationError(
            f"Quantity {quantity_label!r} is not available on baseline run {baseline.run_id!r}."
        )

    entries: list[ComparisonEntry] = []
    for run in runs[1:]:
        value = extractor(run)
        if value is None:
            raise ValidationError(
                f"Quantity {quantity_label!r} is not available on run {run.run_id!r}."
            )
        entries.append(
            ComparisonEntry(
                quantity_label=quantity_label,
                run_id_1=baseline.run_id,
                run_id_2=run.run_id,
                value1=baseline_value,
                value2=value,
                absolute_difference=absolute_difference(baseline_value, value),
                relative_difference=relative_difference(baseline_value, value, epsilon),
                percentage_change=percentage_change(baseline_value, value, epsilon),
            )
        )

    return ComparisonResult(
        quantity_label=quantity_label, baseline_run_id=baseline.run_id, entries=entries
    )
