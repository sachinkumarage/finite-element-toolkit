"""Tests for femtoolkit.optimization.batch (Version 34)."""

from __future__ import annotations

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.optimization.batch import DesignEvaluationTask, evaluate_design_batch
from femtoolkit.optimization.constraints import Constraint, ConstraintRelation
from femtoolkit.optimization.evaluation import DesignStatus
from femtoolkit.optimization.objectives import Objective, ObjectiveDirection, from_result_extractor
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType
from femtoolkit.orchestration.config import OrchestrationConfig
from femtoolkit.studies.extractors import get_extractor


def _base_project() -> Project:
    project = Project(name="Beam", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.width, project.mesh.height = 2.0, 0.4
    project.mesh.nx, project.mesh.ny = 10, 3
    project.mesh.thickness = 0.01
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-3000.0)]
    return project


def _thickness_variable() -> DesignVariable:
    return DesignVariable(
        name="thickness", path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=0.005, upper_bound=0.02, default_value=0.01,
    )


def _objective() -> Objective:
    return Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")),
    )


def _constraint() -> Constraint:
    return Constraint(
        name="stress_limit",
        evaluate=from_result_extractor(get_extractor("maximum_von_mises_stress")),
        relation=ConstraintRelation.LESS_EQUAL, limit=250e6,
    )


def _tasks(thicknesses: list[float]) -> list[DesignEvaluationTask]:
    variable = _thickness_variable()
    objective = _objective()
    constraint = _constraint()
    tasks = []
    for i, thickness in enumerate(thicknesses):
        tasks.append(DesignEvaluationTask(
            task_id=f"candidate-{i}",
            values={"thickness": thickness},
            design_variables=[variable],
            base_project=_base_project(),
            objectives=[objective],
            constraints=[constraint],
            generation=0,
        ))
    return tasks


def test_serial_batch_matches_direct_evaluate_design() -> None:
    from femtoolkit.optimization.evaluation import evaluate_design
    from femtoolkit.runs.manager import SimulationRunManager

    task = _tasks([0.01])[0]
    direct = evaluate_design(
        task.task_id, task.values, task.design_variables, task.base_project,
        task.objectives, task.constraints, SimulationRunManager(), generation=task.generation,
    )
    batched = evaluate_design_batch([task])[0]
    assert batched.status == direct.status
    assert batched.objective_values == direct.objective_values


def test_evaluate_design_batch_default_order() -> None:
    evaluations = evaluate_design_batch(_tasks([0.006, 0.01, 0.014, 0.018]))
    assert [e.design_id for e in evaluations] == [f"candidate-{i}" for i in range(4)]
    assert all(e.status is DesignStatus.FEASIBLE for e in evaluations)


def test_parallel_matches_serial() -> None:
    thicknesses = [0.006, 0.01, 0.014, 0.018]
    serial = evaluate_design_batch(_tasks(thicknesses))
    parallel = evaluate_design_batch(
        _tasks(thicknesses), config=OrchestrationConfig(execution_mode="parallel", max_workers=2)
    )
    serial_values = [e.objective_values.get("maximum_displacement") for e in serial]
    parallel_values = [e.objective_values.get("maximum_displacement") for e in parallel]
    assert serial_values == parallel_values
    assert [e.design_id for e in parallel] == [f"candidate-{i}" for i in range(4)]


def test_objectives_and_constraints_are_picklable_for_parallel_use() -> None:
    import pickle

    objective = _objective()
    constraint = _constraint()
    pickle.dumps(objective.evaluate)
    pickle.dumps(constraint.evaluate)
