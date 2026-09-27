"""Tests for femtoolkit.optimization.evaluation."""

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.optimization.constraints import Constraint, ConstraintRelation
from femtoolkit.optimization.evaluation import (
    DesignEvaluation,
    DesignStatus,
    evaluate_design,
    feasibility_rank,
    is_better_evaluation,
)
from femtoolkit.optimization.objectives import (
    Objective,
    ObjectiveDirection,
    from_result_extractor,
    rectangular_mass,
)
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType
from femtoolkit.runs.manager import SimulationRunManager
from femtoolkit.studies.extractors import get_extractor


def _base_project() -> Project:
    project = Project(name="Evaluation Test", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.width, project.mesh.height = 2.0, 0.4
    project.mesh.nx, project.mesh.ny = 8, 2
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-2000.0)]
    return project


def _thickness_variable(default: float = 0.005) -> DesignVariable:
    return DesignVariable(
        name="thickness", path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=0.005, upper_bound=0.020, default_value=default,
    )


def _displacement_objective() -> Objective:
    return Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")),
    )


def test_evaluate_design_feasible_when_no_constraints() -> None:
    variable = _thickness_variable()
    objective = _displacement_objective()
    evaluation = evaluate_design(
        "d1", {"thickness": 0.01}, [variable], _base_project(), [objective], [],
        SimulationRunManager(),
    )
    assert evaluation.status is DesignStatus.FEASIBLE
    assert evaluation.objective_values["maximum_displacement"] > 0
    assert evaluation.total_violation == 0.0
    assert evaluation.is_feasible


def test_evaluate_design_infeasible_when_constraint_violated() -> None:
    variable = _thickness_variable()
    objective = _displacement_objective()
    tight_stress_constraint = Constraint(
        name="stress_limit",
        evaluate=from_result_extractor(get_extractor("maximum_von_mises_stress")),
        relation=ConstraintRelation.LESS_EQUAL,
        limit=1.0,  # impossibly tight -- guarantees violation
    )
    evaluation = evaluate_design(
        "d1", {"thickness": 0.01}, [variable], _base_project(), [objective],
        [tight_stress_constraint], SimulationRunManager(),
    )
    assert evaluation.status is DesignStatus.INFEASIBLE
    assert evaluation.total_violation > 0.0
    assert not evaluation.constraint_evaluations[0].satisfied


def test_evaluate_design_invalid_for_out_of_bounds_value() -> None:
    variable = _thickness_variable()
    objective = _displacement_objective()
    evaluation = evaluate_design(
        "d1", {"thickness": 999.0}, [variable], _base_project(), [objective], [],
        SimulationRunManager(),
    )
    assert evaluation.status is DesignStatus.INVALID
    assert evaluation.run_id is None
    assert evaluation.error_message is not None


def test_evaluate_design_failed_for_unsolvable_project() -> None:
    bad_variable = DesignVariable(
        name="E", path="material.youngs_modulus", variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=-1e11, upper_bound=1.0, default_value=-1e11,
    )
    objective = _displacement_objective()
    evaluation = evaluate_design(
        "d1", {"E": -5e10}, [bad_variable], _base_project(), [objective], [],
        SimulationRunManager(),
    )
    assert evaluation.status is DesignStatus.FAILED
    assert evaluation.error_message is not None


def test_evaluate_design_failed_when_objective_unavailable() -> None:
    variable = _thickness_variable()
    objective_with_no_density = Objective(
        name="mass", direction=ObjectiveDirection.MINIMIZE, evaluate=rectangular_mass
    )
    project = _base_project()
    project.material.density = None
    evaluation = evaluate_design(
        "d1", {"thickness": 0.01}, [variable], project, [objective_with_no_density], [],
        SimulationRunManager(),
    )
    assert evaluation.status is DesignStatus.FAILED
    assert "mass" in evaluation.error_message


def test_evaluate_design_records_execution_time_and_run_id() -> None:
    variable = _thickness_variable()
    objective = _displacement_objective()
    evaluation = evaluate_design(
        "d1", {"thickness": 0.01}, [variable], _base_project(), [objective], [],
        SimulationRunManager(),
    )
    assert evaluation.execution_time_seconds is not None
    assert evaluation.execution_time_seconds >= 0.0
    assert evaluation.run_id is not None


def test_thinner_beam_displaces_more_than_thicker_beam() -> None:
    variable = _thickness_variable()
    objective = _displacement_objective()
    run_manager = SimulationRunManager()
    thin = evaluate_design(
        "thin", {"thickness": 0.005}, [variable], _base_project(), [objective], [], run_manager
    )
    thick = evaluate_design(
        "thick", {"thickness": 0.020}, [variable], _base_project(), [objective], [], run_manager
    )
    thin_disp = thin.objective_values["maximum_displacement"]
    thick_disp = thick.objective_values["maximum_displacement"]
    assert thin_disp > thick_disp


# --- feasibility_rank / is_better_evaluation ---


def test_feasibility_rank_order() -> None:
    assert feasibility_rank(DesignStatus.FEASIBLE) < feasibility_rank(DesignStatus.INFEASIBLE)
    assert feasibility_rank(DesignStatus.INFEASIBLE) < feasibility_rank(DesignStatus.FAILED)
    assert feasibility_rank(DesignStatus.FAILED) == feasibility_rank(DesignStatus.INVALID)


def _objective(direction: ObjectiveDirection) -> Objective:
    return Objective(name="f", direction=direction, evaluate=lambda ctx: None)


def _evaluation(status: DesignStatus, objective_values=None, total_violation: float = 0.0):
    return DesignEvaluation(
        design_id="x", design_variables={}, run_id=None, status=status,
        objective_values=objective_values or {}, total_violation=total_violation,
    )


def test_is_better_evaluation_feasible_beats_infeasible() -> None:
    objective = _objective(ObjectiveDirection.MINIMIZE)
    feasible = _evaluation(DesignStatus.FEASIBLE, {"f": 100.0})
    infeasible = _evaluation(DesignStatus.INFEASIBLE, {}, total_violation=0.1)
    assert is_better_evaluation(feasible, infeasible, objective)
    assert not is_better_evaluation(infeasible, feasible, objective)


def test_is_better_evaluation_infeasible_beats_failed() -> None:
    objective = _objective(ObjectiveDirection.MINIMIZE)
    infeasible = _evaluation(DesignStatus.INFEASIBLE, {}, total_violation=0.1)
    failed = _evaluation(DesignStatus.FAILED)
    assert is_better_evaluation(infeasible, failed, objective)
    assert not is_better_evaluation(failed, infeasible, objective)


def test_is_better_evaluation_compares_objective_for_minimize() -> None:
    objective = _objective(ObjectiveDirection.MINIMIZE)
    lower = _evaluation(DesignStatus.FEASIBLE, {"f": 10.0})
    higher = _evaluation(DesignStatus.FEASIBLE, {"f": 20.0})
    assert is_better_evaluation(lower, higher, objective)
    assert not is_better_evaluation(higher, lower, objective)


def test_is_better_evaluation_compares_objective_for_maximize() -> None:
    objective = _objective(ObjectiveDirection.MAXIMIZE)
    lower = _evaluation(DesignStatus.FEASIBLE, {"f": 10.0})
    higher = _evaluation(DesignStatus.FEASIBLE, {"f": 20.0})
    assert is_better_evaluation(higher, lower, objective)
    assert not is_better_evaluation(lower, higher, objective)


def test_is_better_evaluation_uses_total_violation_among_infeasible() -> None:
    objective = _objective(ObjectiveDirection.MINIMIZE)
    less_violated = _evaluation(DesignStatus.INFEASIBLE, {}, total_violation=0.1)
    more_violated = _evaluation(DesignStatus.INFEASIBLE, {}, total_violation=5.0)
    assert is_better_evaluation(less_violated, more_violated, objective)
    assert not is_better_evaluation(more_violated, less_violated, objective)


def test_is_better_evaluation_never_prefers_between_two_failed() -> None:
    objective = _objective(ObjectiveDirection.MINIMIZE)
    a = _evaluation(DesignStatus.FAILED)
    b = _evaluation(DesignStatus.INVALID)
    assert not is_better_evaluation(a, b, objective)
    assert not is_better_evaluation(b, a, objective)
