"""Tests for femtoolkit.optimization.algorithms.nsga2.

Covers non-dominated sorting, crowding distance, the Pareto archive,
multi-objective optimization mechanics, and constraint handling.
"""

from __future__ import annotations

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.optimization.algorithms import OptimizationConfig, StopReason
from femtoolkit.optimization.algorithms.nsga2 import NSGA2
from femtoolkit.optimization.constraints import Constraint, ConstraintRelation
from femtoolkit.optimization.evaluation import DesignStatus
from femtoolkit.optimization.history import OptimizationHistory
from femtoolkit.optimization.objectives import (
    Objective,
    ObjectiveDirection,
    from_result_extractor,
    rectangular_mass,
)
from femtoolkit.optimization.pareto import pareto_front
from femtoolkit.optimization.problems import OptimizationProblem
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType
from femtoolkit.runs.manager import SimulationRunManager
from femtoolkit.studies.extractors import get_extractor


def _base_project() -> Project:
    project = Project(name="NSGA2 Test", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.width, project.mesh.height = 2.0, 0.4
    project.mesh.nx, project.mesh.ny = 8, 2
    project.mesh.thickness = 0.01
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-2000.0)]
    return project


def _thickness_variable() -> DesignVariable:
    return DesignVariable(
        name="thickness", path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=0.005, upper_bound=0.02, default_value=0.01,
    )


def _mass_displacement_problem() -> OptimizationProblem:
    displacement = Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")),
    )
    mass = Objective(name="mass", direction=ObjectiveDirection.MINIMIZE, evaluate=rectangular_mass)
    return OptimizationProblem(
        name="nsga2-beam", base_project=_base_project(), design_variables=[_thickness_variable()],
        objectives=[displacement, mass],
    )


def test_nsga2_produces_a_non_empty_non_dominated_front() -> None:
    problem = _mass_displacement_problem()
    config = OptimizationConfig(
        algorithm="nsga2", population_size=10, max_evaluations=80, max_generations=10, seed=1
    )
    history = OptimizationHistory()
    NSGA2().optimize(problem, config, SimulationRunManager(), history)
    front = pareto_front(history.evaluations, problem.objectives)
    assert len(front) >= 1
    assert all(e.is_feasible for e in front)


def test_nsga2_front_is_mass_displacement_trade_off() -> None:
    """A genuine trade-off curve: sorted by mass ascending, displacement must be
    non-increasing is false in general, but no single front member can dominate
    another -- directly verified here."""
    problem = _mass_displacement_problem()
    config = OptimizationConfig(
        algorithm="nsga2", population_size=12, max_evaluations=100, max_generations=12, seed=2
    )
    history = OptimizationHistory()
    NSGA2().optimize(problem, config, SimulationRunManager(), history)
    front = pareto_front(history.evaluations, problem.objectives)

    from femtoolkit.optimization.pareto import dominates

    for a in front:
        for b in front:
            if a is b:
                continue
            assert not dominates(a.objective_values, b.objective_values, problem.objectives)


def test_nsga2_never_exceeds_max_evaluations() -> None:
    problem = _mass_displacement_problem()
    config = OptimizationConfig(
        algorithm="nsga2", population_size=8, max_evaluations=20, seed=3
    )
    history = OptimizationHistory()
    NSGA2().optimize(problem, config, SimulationRunManager(), history)
    assert history.n_evaluations <= 20


def test_nsga2_is_reproducible_with_same_seed() -> None:
    problem = _mass_displacement_problem()
    config = OptimizationConfig(
        algorithm="nsga2", population_size=8, max_evaluations=40, seed=11
    )

    history_a = OptimizationHistory()
    NSGA2().optimize(problem, config, SimulationRunManager(), history_a)
    history_b = OptimizationHistory()
    NSGA2().optimize(problem, config, SimulationRunManager(), history_b)

    values_a = [e.design_variables for e in history_a.evaluations]
    values_b = [e.design_variables for e in history_b.evaluations]
    assert values_a == values_b


def test_nsga2_respects_bounds() -> None:
    problem = _mass_displacement_problem()
    config = OptimizationConfig(
        algorithm="nsga2", population_size=8, max_evaluations=40, seed=4
    )
    history = OptimizationHistory()
    NSGA2().optimize(problem, config, SimulationRunManager(), history)
    for evaluation in history.evaluations:
        assert evaluation.status != DesignStatus.INVALID
        thickness = evaluation.design_variables["thickness"]
        assert 0.005 <= thickness <= 0.02


def test_nsga2_stops_at_max_generations() -> None:
    problem = _mass_displacement_problem()
    config = OptimizationConfig(
        algorithm="nsga2",
        population_size=4,
        max_evaluations=1000,
        max_generations=2,
        tolerance=1e-300,
        patience=100000,
        seed=5,
    )
    history = OptimizationHistory()
    stop_reason = NSGA2().optimize(problem, config, SimulationRunManager(), history)
    assert stop_reason == StopReason.MAX_GENERATIONS


def test_nsga2_handles_constraints_feasibility_first() -> None:
    problem = _mass_displacement_problem()
    stress_constraint = Constraint(
        name="stress_limit",
        evaluate=from_result_extractor(get_extractor("maximum_von_mises_stress")),
        relation=ConstraintRelation.LESS_EQUAL, limit=100e6,
    )
    problem.constraints.append(stress_constraint)
    config = OptimizationConfig(
        algorithm="nsga2", population_size=10, max_evaluations=80, max_generations=10, seed=6
    )
    history = OptimizationHistory()
    NSGA2().optimize(problem, config, SimulationRunManager(), history)
    front = pareto_front(history.evaluations, problem.objectives)
    assert all(e.is_feasible for e in front)
    assert history.n_evaluations > 0


# --- Version 34: parallel (batched) evaluation ---


def test_nsga2_parallel_matches_serial_at_max_evaluations() -> None:
    # Forcing a MAX_EVALUATIONS stop avoids the documented mid-generation-
    # stopping-granularity difference between serial and batched offspring
    # evaluation (see the algorithm's module docstring).
    def _run(orchestration_config):
        problem = _mass_displacement_problem()
        config = OptimizationConfig(
            algorithm="nsga2", population_size=6, max_evaluations=18, max_generations=10,
            seed=9, patience=1000, tolerance=1e-15,
        )
        history = OptimizationHistory()
        stop_reason = NSGA2().optimize(
            problem, config, SimulationRunManager(), history,
            orchestration_config=orchestration_config,
        )
        return history, stop_reason

    from femtoolkit.orchestration.config import OrchestrationConfig

    serial_history, serial_stop = _run(None)
    parallel_history, parallel_stop = _run(
        OrchestrationConfig(execution_mode="parallel", max_workers=2)
    )

    assert serial_stop == StopReason.MAX_EVALUATIONS == parallel_stop
    assert serial_history.n_evaluations == parallel_history.n_evaluations == 18
    serial_values = [e.objective_values["maximum_displacement"] for e in serial_history.evaluations]
    parallel_values = [
        e.objective_values["maximum_displacement"] for e in parallel_history.evaluations
    ]
    assert serial_values == parallel_values


def test_nsga2_parallel_respects_max_evaluations_ceiling() -> None:
    from femtoolkit.orchestration.config import OrchestrationConfig

    problem = _mass_displacement_problem()
    config = OptimizationConfig(
        algorithm="nsga2", population_size=8, max_evaluations=16, max_generations=20, seed=4,
    )
    history = OptimizationHistory()
    NSGA2().optimize(
        problem, config, SimulationRunManager(), history,
        orchestration_config=OrchestrationConfig(execution_mode="parallel", max_workers=2),
    )
    assert history.n_evaluations <= 16
