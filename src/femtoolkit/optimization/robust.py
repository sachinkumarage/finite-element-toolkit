"""Robust, uncertainty-aware design optimization (Version 33).

**Deterministic vs. robust optimization.** Every algorithm in
:mod:`femtoolkit.optimization.algorithms` solves

.. code-block:: text

    min f(x)

using one fixed set of input parameters per design -- a *deterministic*
optimization. A real engineering input is often uncertain (material
scatter, load variability); a **robust** optimization instead accounts
for that uncertainty while still searching over the same design
variables ``x``:

.. code-block:: text

    min E[f(x, xi)]              (minimize the expected objective)
    Q_0.95(u(x, xi)) <= u_max     (control a high percentile of a response)
    P(sigma(x, xi) > sigma_allow) <= p_max   (limit an estimated exceedance probability)

where ``xi`` is an uncertain parameter (Version 31:
:class:`~femtoolkit.uncertainty.parameters.UncertainParameter`). This
module does not implement a new optimization *algorithm* for this --
every algorithm in :mod:`femtoolkit.optimization.algorithms` already
works unchanged with any objective/constraint function, deterministic
or not. What this module provides is the **robust objective/constraint
building blocks** (statistics computed from a small Monte Carlo study
at each design point, reusing Version 31's
:class:`~femtoolkit.uncertainty.monte_carlo.MonteCarloRunner` directly)
and the **cost-control configuration** that keeps a nested
optimization-inside-Monte-Carlo-inside-FEA study from silently becoming
unboundedly expensive.

**Empirical, not rigorous, probabilities.** Every statistic here
(including the exceedance probability from
:func:`femtoolkit.uncertainty.reliability.exceedance_probability`)
is an *empirical* estimate from a finite Monte Carlo sample at one
design point -- never a rigorous reliability index (no FORM/SORM is
implemented anywhere in this toolkit). A design that appears to satisfy
``P(sigma > sigma_allow) <= p_max`` based on 20 samples carries real
sampling uncertainty the report must not hide.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from femtoolkit.exceptions import StudySizeExceededError, ValidationError
from femtoolkit.optimization.constraints import ConstraintFunction
from femtoolkit.optimization.context import DesignContext
from femtoolkit.optimization.objectives import ObjectiveFunction
from femtoolkit.uncertainty.sampling import LATIN_HYPERCUBE_METHOD, RANDOM_METHOD

if TYPE_CHECKING:
    from femtoolkit.uncertainty.parameters import UncertainParameter

SUPPORTED_SAMPLING_METHODS = (RANDOM_METHOD, LATIN_HYPERCUBE_METHOD)
SUPPORTED_STATISTICS = (
    "mean",
    "median",
    "std",
    "coefficient_of_variation",
    "min",
    "max",
    "percentile",
    "exceedance_probability",
)
SUPPORTED_FAILURE_POLICIES = ("fail_fast", "continue")

DEFAULT_SAMPLE_COUNT = 20
"""Deliberately modest -- each sample is itself a real FEA run, and every
optimization evaluation draws this many samples."""

DEFAULT_MAXIMUM_TOTAL_EVALUATIONS = 20000
"""The hard ceiling on `max_evaluations * sample_count` (the estimated
total FEA solve count for a robust run) -- mirrors every prior "stop
before execution" safety pattern in this toolkit (Version 30's
`StudySizeExceededError`, Version 31's `max_samples`, Version 32's
`evaluation_limit`)."""


@dataclass
class RobustDesignConfig:
    """Configuration for uncertainty-aware ("robust") objectives and constraints.

    Attributes:
        uncertainty_enabled: Whether this run uses robust (uncertainty-
            aware) objective/constraint evaluation at all. ``False``
            (the default) means every objective/constraint is evaluated
            deterministically, exactly as in Version 32 -- robust
            optimization is strictly opt-in.
        sampling_method: ``"random"`` or ``"latin_hypercube"`` (see
            :mod:`femtoolkit.uncertainty.sampling`).
        sample_count: How many Monte Carlo samples to draw per design
            evaluation -- the inner loop of
            "optimization evaluations -> Monte Carlo samples -> FEA
            simulations" (see spec: nested simulation cost control).
        random_seed: A random seed; the same seed reproduces the same
            sample sequence at every design point.
        objective_statistic: Which statistic an objective built by
            :func:`robust_objective_statistic` reports (see
            :data:`SUPPORTED_STATISTICS`).
        constraint_statistic: Which statistic a constraint built by
            :func:`robust_constraint_statistic` reports.
        percentile: The percentile (0-100) used when
            ``objective_statistic``/``constraint_statistic`` is
            ``"percentile"``.
        maximum_total_evaluations: The hard ceiling on the estimated
            total FEA solve count (``max_evaluations * sample_count``),
            checked by :func:`validate_robust_budget` before any
            evaluation is executed.
        failure_policy: ``"fail_fast"`` stops a design's inner Monte
            Carlo study at its first failed sample (not a rejected
            invalid sample); ``"continue"`` (the default) runs every
            valid sample regardless of earlier failures, matching
            Version 31's `MonteCarloConfig.fail_fast` convention.
    """

    uncertainty_enabled: bool = False
    sampling_method: str = RANDOM_METHOD
    sample_count: int = DEFAULT_SAMPLE_COUNT
    random_seed: int | None = None
    objective_statistic: str = "mean"
    constraint_statistic: str = "mean"
    percentile: float = 95.0
    maximum_total_evaluations: int = DEFAULT_MAXIMUM_TOTAL_EVALUATIONS
    failure_policy: str = "continue"

    def __post_init__(self) -> None:
        if self.sampling_method not in SUPPORTED_SAMPLING_METHODS:
            raise ValidationError(
                f"Unknown sampling_method {self.sampling_method!r}; expected one of "
                f"{SUPPORTED_SAMPLING_METHODS}."
            )
        if self.sample_count < 1:
            raise ValidationError(f"sample_count must be at least 1, got {self.sample_count}.")
        if self.objective_statistic not in SUPPORTED_STATISTICS:
            raise ValidationError(
                f"Unknown objective_statistic {self.objective_statistic!r}; expected one of "
                f"{SUPPORTED_STATISTICS}."
            )
        if self.constraint_statistic not in SUPPORTED_STATISTICS:
            raise ValidationError(
                f"Unknown constraint_statistic {self.constraint_statistic!r}; expected one of "
                f"{SUPPORTED_STATISTICS}."
            )
        if not (0.0 < self.percentile < 100.0):
            raise ValidationError(f"percentile must lie in (0, 100), got {self.percentile}.")
        if self.maximum_total_evaluations < 1:
            raise ValidationError(
                f"maximum_total_evaluations must be at least 1, got "
                f"{self.maximum_total_evaluations}."
            )
        if self.failure_policy not in SUPPORTED_FAILURE_POLICIES:
            raise ValidationError(
                f"Unknown failure_policy {self.failure_policy!r}; expected one of "
                f"{SUPPORTED_FAILURE_POLICIES}."
            )


def estimate_total_fea_count(max_evaluations: int, robust_config: RobustDesignConfig | None) -> int:
    """How many FEA solves a run may attempt, including nested Monte Carlo sampling.

    Args:
        max_evaluations: The optimization's own evaluation budget
            (:attr:`~femtoolkit.optimization.algorithms.base.OptimizationConfig.max_evaluations`).
        robust_config: The run's robust-design configuration, or
            ``None``/disabled for a purely deterministic run.

    Returns:
        ``max_evaluations`` for a deterministic run, or
        ``max_evaluations * robust_config.sample_count`` when robust
        evaluation is enabled -- an upper bound, since not every
        objective/constraint in a problem necessarily draws a fresh
        Monte Carlo sample.
    """
    if robust_config is None or not robust_config.uncertainty_enabled:
        return max_evaluations
    return max_evaluations * robust_config.sample_count


def validate_robust_budget(max_evaluations: int, robust_config: RobustDesignConfig | None) -> None:
    """Reject an oversized robust-optimization budget before any evaluation runs.

    Args:
        max_evaluations: The optimization's own evaluation budget.
        robust_config: The run's robust-design configuration.

    Raises:
        StudySizeExceededError: If :func:`estimate_total_fea_count`
            exceeds ``robust_config.maximum_total_evaluations``.
    """
    if robust_config is None or not robust_config.uncertainty_enabled:
        return
    estimated = estimate_total_fea_count(max_evaluations, robust_config)
    if estimated > robust_config.maximum_total_evaluations:
        raise StudySizeExceededError(
            f"Robust optimization requests up to {max_evaluations} optimization "
            f"evaluations x {robust_config.sample_count} uncertainty samples = "
            f"{estimated} estimated FEA evaluations, exceeding "
            f"maximum_total_evaluations={robust_config.maximum_total_evaluations}. Reduce "
            "max_evaluations or sample_count, or explicitly raise maximum_total_evaluations "
            "if this many simulations is intentional. No evaluation was executed."
        )


def _compute_statistic(
    values: np.ndarray, statistic: str, percentile: float, threshold: float | None, direction: str
) -> float | None:
    if values.size == 0:
        return None
    if statistic == "mean":
        return float(values.mean())
    if statistic == "median":
        return float(np.median(values))
    if statistic == "std":
        return float(values.std(ddof=1)) if values.size >= 2 else None
    if statistic == "coefficient_of_variation":
        if values.size < 2:
            return None
        mean = float(values.mean())
        if mean == 0.0:
            return None
        return float(values.std(ddof=1) / abs(mean))
    if statistic == "min":
        return float(values.min())
    if statistic == "max":
        return float(values.max())
    if statistic == "percentile":
        return float(np.percentile(values, percentile))
    if statistic == "exceedance_probability":
        if threshold is None:
            raise ValidationError(
                "statistic='exceedance_probability' requires a threshold."
            )
        from femtoolkit.uncertainty.reliability import exceedance_probability

        result = exceedance_probability(values, threshold, "robust", direction)
        return result.exceedance_frequency
    raise ValidationError(
        f"Unknown statistic {statistic!r}; expected one of {SUPPORTED_STATISTICS}."
    )


def _run_inner_study(
    context: DesignContext,
    build_parameters: Callable[[DesignContext], list[UncertainParameter]],
    quantity_name: str,
    robust_config: RobustDesignConfig,
) -> np.ndarray:
    from femtoolkit.studies.extractors import get_extractor
    from femtoolkit.uncertainty.monte_carlo import MonteCarloConfig, MonteCarloRunner

    parameters = build_parameters(context)
    if not parameters:
        return np.array([])
    config = MonteCarloConfig(
        study_id="robust-sample",
        name="Robust evaluation sample",
        base_project=context.project,
        parameters=parameters,
        output_quantities=[quantity_name],
        n_samples=robust_config.sample_count,
        seed=robust_config.random_seed,
        method=robust_config.sampling_method,
        fail_fast=robust_config.failure_policy == "fail_fast",
    )
    result = MonteCarloRunner().run(config)
    return result.output_values(get_extractor(quantity_name))


def robust_objective_statistic(
    build_parameters: Callable[[DesignContext], list[UncertainParameter]],
    quantity_name: str,
    robust_config: RobustDesignConfig,
    threshold: float | None = None,
    direction: str = "above",
) -> ObjectiveFunction:
    """Build an objective evaluating an uncertainty-aware statistic of a quantity.

    At each design point, runs a Version 31 Monte Carlo study (reusing
    :class:`~femtoolkit.uncertainty.monte_carlo.MonteCarloRunner`
    directly -- no second execution path) and reduces the sampled
    output to one statistic, per ``robust_config.objective_statistic``
    (see :data:`SUPPORTED_STATISTICS`). Records the statistic, its
    value, and the underlying sample count into
    :attr:`~femtoolkit.optimization.context.DesignContext.metadata`
    (copied into the resulting
    :class:`~femtoolkit.optimization.evaluation.DesignEvaluation.metadata`).

    Args:
        build_parameters: Given the current design's
            :class:`~femtoolkit.optimization.context.DesignContext`,
            returns the uncertain parameters to sample around that
            design point.
        quantity_name: A named extractor from
            :data:`~femtoolkit.studies.extractors.EXTRACTORS`.
        robust_config: The run's robust-design configuration
            (``objective_statistic``, ``sample_count``, ``percentile``, ...).
        threshold: Required only when ``robust_config.objective_statistic
            == "exceedance_probability"``.
        direction: ``"above"`` or ``"below"``, for the
            ``"exceedance_probability"`` statistic only.

    Returns:
        An :data:`~femtoolkit.optimization.objectives.ObjectiveFunction`.
    """

    def _evaluate(context: DesignContext) -> float | None:
        values = _run_inner_study(context, build_parameters, quantity_name, robust_config)
        statistic_value = _compute_statistic(
            values,
            robust_config.objective_statistic,
            robust_config.percentile,
            threshold,
            direction,
        )
        if statistic_value is not None:
            context.metadata.setdefault("robust_objectives", {})[quantity_name] = {
                "statistic": robust_config.objective_statistic,
                "value": statistic_value,
                "n_samples": int(values.size),
            }
        return statistic_value

    return _evaluate


def robust_constraint_statistic(
    build_parameters: Callable[[DesignContext], list[UncertainParameter]],
    quantity_name: str,
    robust_config: RobustDesignConfig,
    threshold: float | None = None,
    direction: str = "above",
) -> ConstraintFunction:
    """Build a constraint evaluating an uncertainty-aware statistic of a quantity.

    The same inner Monte Carlo evaluation as
    :func:`robust_objective_statistic`, but returning a value for a
    :class:`~femtoolkit.optimization.constraints.Constraint` to compare
    against its limit -- e.g. "95th percentile stress <= allowable
    stress" (``statistic="percentile"``, ``percentile=95``) or
    "estimated exceedance probability <= permitted probability"
    (``statistic="exceedance_probability"``).

    Args:
        build_parameters: See :func:`robust_objective_statistic`.
        quantity_name: A named extractor from
            :data:`~femtoolkit.studies.extractors.EXTRACTORS`.
        robust_config: The run's robust-design configuration (uses
            ``constraint_statistic``, ``sample_count``, ``percentile``, ...).
        threshold: Required only when ``robust_config.constraint_statistic
            == "exceedance_probability"``.
        direction: ``"above"`` or ``"below"``, for the
            ``"exceedance_probability"`` statistic only.

    Returns:
        A :data:`~femtoolkit.optimization.constraints.ConstraintFunction`.
    """

    def _evaluate(context: DesignContext) -> float | None:
        values = _run_inner_study(context, build_parameters, quantity_name, robust_config)
        statistic_value = _compute_statistic(
            values,
            robust_config.constraint_statistic,
            robust_config.percentile,
            threshold,
            direction,
        )
        if statistic_value is not None:
            context.metadata.setdefault("robust_constraints", {})[quantity_name] = {
                "statistic": robust_config.constraint_statistic,
                "value": statistic_value,
                "n_samples": int(values.size),
            }
        return statistic_value

    return _evaluate


__all__ = [
    "DEFAULT_MAXIMUM_TOTAL_EVALUATIONS",
    "DEFAULT_SAMPLE_COUNT",
    "SUPPORTED_FAILURE_POLICIES",
    "SUPPORTED_SAMPLING_METHODS",
    "SUPPORTED_STATISTICS",
    "RobustDesignConfig",
    "estimate_total_fea_count",
    "robust_constraint_statistic",
    "robust_objective_statistic",
    "validate_robust_budget",
]
