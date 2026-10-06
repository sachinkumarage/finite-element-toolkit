"""The optimization-algorithm interface, configuration, and stop reasons (Version 32).

Keeps *problem definition* (:mod:`femtoolkit.optimization.problems`)
separate from *how to search* (this package): any
:class:`OptimizationAlgorithm` can be handed any
:class:`~femtoolkit.optimization.problems.OptimizationProblem`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from femtoolkit.exceptions import StudySizeExceededError, ValidationError
from femtoolkit.optimization.evaluation import DesignEvaluation
from femtoolkit.optimization.history import OptimizationHistory, compute_convergence
from femtoolkit.optimization.objectives import Objective, ObjectiveDirection
from femtoolkit.optimization.problems import OptimizationProblem
from femtoolkit.orchestration.config import OrchestrationConfig
from femtoolkit.runs.manager import SimulationRunManager

DEFAULT_MAX_EVALUATIONS = 50
"""The default evaluation budget -- deliberately modest, since each
evaluation is a real FEA run."""

DEFAULT_EVALUATION_LIMIT = 1000
"""The default hard ceiling on `max_evaluations`, mirroring Version 30/31's
"stop before execution" safety pattern for a study that could otherwise
request an unreasonable number of simulations."""

DEFAULT_MAX_GENERATIONS = 100
"""The default generation/iteration budget for a population-based
algorithm (Version 33). `max_evaluations` remains the binding,
always-enforced budget; this is a secondary, independent stopping
condition (see spec: "maximum generations/iterations" as a distinct
criterion from "maximum evaluations")."""

SUPPORTED_ALGORITHMS = (
    "random_search",
    "coordinate_search",
    "differential_evolution",
    "genetic_algorithm",
    "particle_swarm",
    "nsga2",
)
_POPULATION_ALGORITHMS = (
    "differential_evolution",
    "genetic_algorithm",
    "particle_swarm",
    "nsga2",
)
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
        MAX_GENERATIONS: A population-based algorithm (Version 33:
            differential evolution, genetic algorithm, particle swarm,
            NSGA-II) reached its configured ``max_generations`` before
            any other stopping condition was hit.
        TARGET_REACHED: The best-feasible objective value reached or
            passed a user-supplied ``target_objective`` (Version 33).
            Reaching a target value says nothing about optimality --
            only that the search's own stated goal was met.
    """

    COMPLETED = "completed"
    MAX_EVALUATIONS = "max_evaluations"
    CONVERGED = "converged"
    CANCELLED = "cancelled"
    FAILED = "failed"
    MAX_GENERATIONS = "max_generations"
    TARGET_REACHED = "target_reached"


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
        max_generations: For a population-based algorithm (Version 33),
            the generation/iteration budget -- a stopping condition
            independent of (and usually reached before)
            ``max_evaluations``. Ignored by ``random_search``/
            ``coordinate_search``. ``None`` disables this check.
        target_objective: An optional target value for the primary
            objective; once the best-feasible value reaches or passes
            it, the run stops with :attr:`StopReason.TARGET_REACHED`.
            ``None`` disables this check. Reaching a target is a
            user-defined stopping goal, never evidence of optimality.
        population_size: The number of candidate designs maintained per
            generation by a population-based algorithm. Differential
            evolution requires at least 4 (its mutation step needs
            three other distinct population members); every other
            population-based algorithm requires at least 2.
        mutation_factor: Differential evolution's scale factor ``F`` in
            ``v = x_r1 + F * (x_r2 - x_r3)``; conventionally in
            ``(0, 2]``.
        crossover_probability: The probability a candidate's component
            is taken from the mutant/other-parent vector rather than
            the target/current vector -- differential evolution's
            ``CR``, and the genetic algorithm's/NSGA-II's per-pair
            crossover probability. Must lie in ``[0, 1]``.
        mutation_probability: The genetic algorithm's/NSGA-II's
            per-gene probability of a random mutation. Must lie in
            ``[0, 1]``.
        elite_count: How many of the current generation's best
            individuals the genetic algorithm copies unchanged into the
            next generation before filling the remainder via
            selection/crossover/mutation. Must satisfy
            ``0 <= elite_count < population_size``.
        tournament_size: How many individuals compete in one tournament
            selection draw (genetic algorithm, NSGA-II). Must satisfy
            ``2 <= tournament_size <= population_size``.
        inertia_weight: Particle swarm's velocity-retention coefficient
            ``w``. Must lie in ``[0, 2]``.
        cognitive_coefficient: Particle swarm's personal-best
            attraction coefficient ``c1``. Must be non-negative.
        social_coefficient: Particle swarm's global-best attraction
            coefficient ``c2``. Must be non-negative.
        velocity_limit: Particle swarm's per-step velocity clamp, as a
            fraction of each variable's ``[lower, upper]`` span (the
            same convention ``step_size`` uses for coordinate search).
            Must lie in ``(0, 1]``.
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
    max_generations: int | None = DEFAULT_MAX_GENERATIONS
    target_objective: float | None = None
    population_size: int = 20
    mutation_factor: float = 0.8
    crossover_probability: float = 0.9
    mutation_probability: float = 0.1
    elite_count: int = 1
    tournament_size: int = 3
    inertia_weight: float = 0.7
    cognitive_coefficient: float = 1.5
    social_coefficient: float = 1.5
    velocity_limit: float = 0.2

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
        if self.max_generations is not None and self.max_generations < 1:
            raise ValidationError(
                f"max_generations must be at least 1 or None, got {self.max_generations}."
            )
        if self.target_objective is not None and not math.isfinite(self.target_objective):
            raise ValidationError("target_objective must be finite or None.")
        if self.algorithm in _POPULATION_ALGORITHMS:
            minimum_population = 4 if self.algorithm == "differential_evolution" else 2
            if self.population_size < minimum_population:
                raise ValidationError(
                    f"population_size must be at least {minimum_population} for "
                    f"{self.algorithm!r}, got {self.population_size}."
                )
        elif self.population_size < 1:
            raise ValidationError(
                f"population_size must be at least 1, got {self.population_size}."
            )
        if not (0.0 < self.mutation_factor <= 2.0):
            raise ValidationError(
                f"mutation_factor must lie in (0, 2], got {self.mutation_factor}."
            )
        if not (0.0 <= self.crossover_probability <= 1.0):
            raise ValidationError(
                f"crossover_probability must lie in [0, 1], got {self.crossover_probability}."
            )
        if not (0.0 <= self.mutation_probability <= 1.0):
            raise ValidationError(
                f"mutation_probability must lie in [0, 1], got {self.mutation_probability}."
            )
        if self.algorithm in ("genetic_algorithm", "nsga2"):
            if not (0 <= self.elite_count < self.population_size):
                raise ValidationError(
                    f"elite_count must satisfy 0 <= elite_count < population_size "
                    f"(population_size={self.population_size}), got {self.elite_count}."
                )
            if not (2 <= self.tournament_size <= self.population_size):
                raise ValidationError(
                    f"tournament_size must satisfy 2 <= tournament_size <= population_size "
                    f"(population_size={self.population_size}), got {self.tournament_size}."
                )
        else:
            if self.elite_count < 0:
                raise ValidationError(f"elite_count must be non-negative, got {self.elite_count}.")
            if self.tournament_size < 2:
                raise ValidationError(
                    f"tournament_size must be at least 2, got {self.tournament_size}."
                )
        if not (0.0 <= self.inertia_weight <= 2.0):
            raise ValidationError(f"inertia_weight must lie in [0, 2], got {self.inertia_weight}.")
        if self.cognitive_coefficient < 0.0:
            raise ValidationError("cognitive_coefficient must be non-negative.")
        if self.social_coefficient < 0.0:
            raise ValidationError("social_coefficient must be non-negative.")
        if not (0.0 < self.velocity_limit <= 1.0):
            raise ValidationError(f"velocity_limit must lie in (0, 1], got {self.velocity_limit}.")
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
        orchestration_config: OrchestrationConfig | None = None,
    ) -> StopReason:
        """Search ``problem``'s design space, appending every evaluation to ``history``.

        Args:
            problem: The problem to optimize.
            config: The run's configuration.
            run_manager: The run manager to execute each design's
                scenario with (the same one every evaluation shares).
                Only used for evaluations this algorithm still performs
                one at a time -- a batched evaluation point (see
                ``orchestration_config``) constructs its own run
                managers inside worker processes instead (Version 34).
            history: The history to append every evaluated design to,
                in evaluation order, as the search proceeds.
            starting_evaluation: An already-evaluated design (typically
                the problem's baseline) an algorithm may reuse as its
                starting point instead of evaluating it again. An
                algorithm that does not need a starting point (e.g.
                pure random sampling) ignores this.
            orchestration_config: An optional Version 34 orchestration
                configuration. ``None`` (the default) evaluates every
                design one at a time, in the calling process --
                identical to every prior version's behavior for every
                algorithm. A population-based algorithm
                (:mod:`femtoolkit.optimization.algorithms.differential_evolution`/
                ``genetic_algorithm``/``particle_swarm``/``nsga2``) that
                can safely batch some of its evaluations uses this to
                run them through
                :func:`~femtoolkit.optimization.batch.evaluate_design_batch`
                instead -- see each such algorithm's own module
                docstring for exactly which evaluations it batches and
                why. An algorithm with no independent-candidate
                evaluation to batch (random search, coordinate search)
                ignores this.

        Returns:
            The :class:`StopReason` the run stopped for.
        """


def should_stop(
    history: OptimizationHistory,
    objective: Objective,
    config: OptimizationConfig,
    consecutive_failures: int,
    generation: int | None = None,
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
        generation: The current generation/iteration index, for a
            population-based algorithm checking ``config.max_generations``
            (Version 33). ``None`` (the default) skips this check --
            random search and coordinate search have no generation
            concept and never pass it.

    Returns:
        The :class:`StopReason` to stop for, or ``None`` to continue.
    """
    if history.n_evaluations >= config.max_evaluations:
        return StopReason.MAX_EVALUATIONS
    if (
        generation is not None
        and config.max_generations is not None
        and generation >= config.max_generations
    ):
        return StopReason.MAX_GENERATIONS
    if consecutive_failures >= config.max_consecutive_failures:
        return StopReason.FAILED
    if config.target_objective is not None:
        best = history.best_feasible(objective)
        if best is not None:
            value = best.objective_values[objective.name]
            reached = (
                value <= config.target_objective
                if objective.direction is ObjectiveDirection.MINIMIZE
                else value >= config.target_objective
            )
            if reached:
                return StopReason.TARGET_REACHED
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
    "DEFAULT_MAX_GENERATIONS",
    "SUPPORTED_ALGORITHMS",
    "SUPPORTED_CONSTRAINT_HANDLING",
    "OptimizationAlgorithm",
    "OptimizationConfig",
    "StopReason",
    "should_stop",
]
