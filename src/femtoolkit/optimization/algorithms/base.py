"""The optimization-algorithm interface, configuration, and stop reasons (Version 32).

Keeps *problem definition* (:mod:`femtoolkit.optimization.problems`)
separate from *how to search* (this package): any
:class:`OptimizationAlgorithm` can be handed any
:class:`~femtoolkit.optimization.problems.OptimizationProblem`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from femtoolkit.exceptions import StudySizeExceededError, ValidationError
from femtoolkit.optimization.evaluation import DesignEvaluation
from femtoolkit.optimization.history import OptimizationHistory, compute_convergence
from femtoolkit.optimization.objectives import Objective
from femtoolkit.optimization.problems import OptimizationProblem
from femtoolkit.runs.manager import SimulationRunManager

DEFAULT_MAX_EVALUATIONS = 50
"""The default evaluation budget -- deliberately modest, since each
evaluation is a real FEA run."""

DEFAULT_EVALUATION_LIMIT = 1000
"""The default hard ceiling on `max_evaluations`, mirroring Version 30/31's
"stop before execution" safety pattern for a study that could otherwise
request an unreasonable number of simulations."""

SUPPORTED_ALGORITHMS = ("random_search", "coordinate_search")
SUPPORTED_CONSTRAINT_HANDLING = ("feasibility_first",)


class StopReason(Enum):
    """Why an optimization run stopped.

    Attributes:
        COMPLETED: The algorithm finished its own search logic (e.g.
            coordinate search found no further improvement in a full
            pass over every design variable) without hitting another
            stopping condition first.
        MAX_EVALUATIONS: The configured evaluation budget was reached.
        CONVERGED: The best-feasible objective value's relative
            improvement stayed below ``tolerance`` for ``patience``
            consecutive evaluations. This describes an empirically
            *observed* plateau in a derivative-free heuristic search --
            it is never evidence of a mathematically guaranteed global
            optimum.
        CANCELLED: The run was cancelled before finishing (see the GUI
            integration; a cancelled run's history up to that point is
            preserved).
        FAILED: Too many consecutive evaluations failed in a row (see
            ``max_consecutive_failures``) -- the search was abandoned
            rather than continuing to spend evaluations on a
            configuration that cannot solve.
    """

    COMPLETED = "completed"
    MAX_EVALUATIONS = "max_evaluations"
    CONVERGED = "converged"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class OptimizationConfig:
    """Configuration for one optimization run.

    Attributes:
        algorithm: ``"random_search"`` or ``"coordinate_search"``.
        max_evaluations: The evaluation budget -- checked before
            execution against ``evaluation_limit``.
        evaluation_limit: The hard ceiling ``max_evaluations`` may not
            exceed without being deliberately raised.
        tolerance: The relative-improvement threshold below which the
            best-feasible objective value is considered to have
            plateaued (see :attr:`StopReason.CONVERGED`).
        patience: How many consecutive evaluations' relative
            improvement must stay below ``tolerance`` before declaring
            convergence.
        seed: A random seed; the same seed reproduces the same search
            (subject to the algorithm's own use of randomness --
            coordinate search is otherwise fully deterministic).
        step_size: For coordinate search, the step taken per move as a
            fraction of a continuous variable's ``[lower, upper]``
            range (or the number of integer steps, rounded to at least
            one step); must lie in ``(0, 1]``.
        max_consecutive_failures: How many evaluations may fail in a
            row before the run stops with :attr:`StopReason.FAILED`.
        constraint_handling: The constraint-handling strategy. Only
            ``"feasibility_first"`` is implemented in this version (see
            :func:`~femtoolkit.optimization.evaluation.is_better_evaluation`).
    """

    algorithm: str = "random_search"
    max_evaluations: int = DEFAULT_MAX_EVALUATIONS
    evaluation_limit: int = DEFAULT_EVALUATION_LIMIT
    tolerance: float = 1e-6
    patience: int = 10
    seed: int | None = None
    step_size: float = 0.1
    max_consecutive_failures: int = 10
    constraint_handling: str = "feasibility_first"

    def __post_init__(self) -> None:
        if self.algorithm not in SUPPORTED_ALGORITHMS:
            raise ValidationError(
                f"Unknown algorithm {self.algorithm!r}; expected one of {SUPPORTED_ALGORITHMS}."
            )
        if self.max_evaluations < 1:
            raise ValidationError(
                f"max_evaluations must be at least 1, got {self.max_evaluations}."
            )
        if self.tolerance <= 0:
            raise ValidationError(f"tolerance must be positive, got {self.tolerance}.")
        if self.patience < 1:
            raise ValidationError(f"patience must be at least 1, got {self.patience}.")
        if not (0.0 < self.step_size <= 1.0):
            raise ValidationError(f"step_size must lie in (0, 1], got {self.step_size}.")
        if self.max_consecutive_failures < 1:
            raise ValidationError(
                f"max_consecutive_failures must be at least 1, got {self.max_consecutive_failures}."
            )
        if self.constraint_handling not in SUPPORTED_CONSTRAINT_HANDLING:
            raise ValidationError(
                f"Unknown constraint_handling {self.constraint_handling!r}; this version "
                f"only implements {SUPPORTED_CONSTRAINT_HANDLING}."
            )
        if self.max_evaluations > self.evaluation_limit:
            raise StudySizeExceededError(
                f"Optimization requests max_evaluations={self.max_evaluations}, which "
                f"exceeds evaluation_limit={self.evaluation_limit}. Reduce max_evaluations, "
                "or explicitly raise evaluation_limit if this many simulations is "
                "intentional. The run was rejected before any evaluation was executed."
            )

    @property
    def estimated_simulation_count(self) -> int:
        """How many FEA solves this run may attempt.

        Each evaluation is exactly one FEA run in this version's
        deterministic algorithms -- except an
        :func:`~femtoolkit.optimization.objectives.robust_objective_mean`
        objective, which itself runs several FEA solves per evaluation
        and is not reflected in this simple estimate.
        """
        return self.max_evaluations


class OptimizationAlgorithm(ABC):
    """The interface every optimization algorithm implements."""

    @abstractmethod
    def optimize(
        self,
        problem: OptimizationProblem,
        config: OptimizationConfig,
        run_manager: SimulationRunManager,
        history: OptimizationHistory,
        starting_evaluation: DesignEvaluation | None = None,
    ) -> StopReason:
        """Search ``problem``'s design space, appending every evaluation to ``history``.

        Args:
            problem: The problem to optimize.
            config: The run's configuration.
            run_manager: The run manager to execute each design's
                scenario with (the same one every evaluation shares).
            history: The history to append every evaluated design to,
                in evaluation order, as the search proceeds.
            starting_evaluation: An already-evaluated design (typically
                the problem's baseline) an algorithm may reuse as its
                starting point instead of evaluating it again. An
                algorithm that does not need a starting point (e.g.
                pure random sampling) ignores this.

        Returns:
            The :class:`StopReason` the run stopped for.
        """


def should_stop(
    history: OptimizationHistory,
    objective: Objective,
    config: OptimizationConfig,
    consecutive_failures: int,
) -> StopReason | None:
    """Check the shared stopping conditions every algorithm in this package uses.

    Args:
        history: The evaluations recorded so far.
        objective: The objective driving convergence tracking (for a
            multi-objective problem, the first-listed objective is used
            as a practical single-objective proxy for this purpose
            only -- the actual multi-objective result is the full
            Pareto front computed from the complete history afterward,
            not this internal proxy).
        config: The run's configuration.
        consecutive_failures: How many evaluations have failed in a row
            immediately before this check.

    Returns:
        The :class:`StopReason` to stop for, or ``None`` to continue.
    """
    if history.n_evaluations >= config.max_evaluations:
        return StopReason.MAX_EVALUATIONS
    if consecutive_failures >= config.max_consecutive_failures:
        return StopReason.FAILED
    steps = compute_convergence(history, objective) if history.n_evaluations > 0 else []
    if len(steps) >= config.patience:
        recent = steps[-config.patience :]
        if all(
            step.relative_improvement is not None and step.relative_improvement < config.tolerance
            for step in recent
        ):
            return StopReason.CONVERGED
    return None


__all__ = [
    "DEFAULT_EVALUATION_LIMIT",
    "DEFAULT_MAX_EVALUATIONS",
    "SUPPORTED_ALGORITHMS",
    "SUPPORTED_CONSTRAINT_HANDLING",
    "OptimizationAlgorithm",
    "OptimizationConfig",
    "StopReason",
    "should_stop",
]
