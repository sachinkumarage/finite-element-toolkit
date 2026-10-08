"""Integration tests: surrogate framework <-> Version 30/31/33/34 infrastructure.

These tests run real (small) FEA simulations through the existing
infrastructure -- they exercise the full
"parameter space -> FEA -> snapshot dataset -> surrogate -> verification"
pipeline end to end, not just the surrogate math in isolation.
"""

from __future__ import annotations

from unittest import mock

import pytest

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.optimization.objectives import Objective, ObjectiveDirection, from_result_extractor
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType
from femtoolkit.orchestration.config import OrchestrationConfig
from femtoolkit.orchestration.simulation import evaluate_simulation_batch
from femtoolkit.studies.extractors import get_extractor
from femtoolkit.studies.parameter_sweep import ParameterDefinition
from femtoolkit.surrogate.workflows.evaluator import EvaluationBackend, SurrogateEvaluator
from femtoolkit.surrogate.workflows.recommendation import recommend_candidates
from femtoolkit.surrogate.workflows.training import (
    TrainingConfig,
    generate_training_dataset,
    train_surrogate,
)
from femtoolkit.surrogate.workflows.verification import (
    AcceptanceStatus,
    verify_against_high_fidelity,
)


def _cantilever_project() -> Project:
    project = Project(name="Surrogate Integration Cantilever", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.width = 2.0
    project.mesh.height = 0.4
    project.mesh.nx = 8
    project.mesh.ny = 2
    project.mesh.thickness = 0.01
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-2000.0)]
    return project


def _thickness_parameter(values: list[float]) -> ParameterDefinition:
    return ParameterDefinition(path="mesh.thickness", label="Plate thickness", values=values)


@pytest.mark.slow
def test_generate_training_dataset_from_parameter_sweep() -> None:
    base_project = _cantilever_project()
    thickness = _thickness_parameter([0.006, 0.008, 0.010, 0.012, 0.014, 0.016])
    dataset = generate_training_dataset(
        base_project, [thickness], {"maximum_displacement": get_extractor("maximum_displacement")}
    )
    assert dataset.n_samples == 6
    assert dataset.feature_names == ["mesh.thickness"]
    assert dataset.response_names == ["maximum_displacement"]


@pytest.mark.slow
def test_generate_training_dataset_forwards_orchestration_config_to_v34() -> None:
    """``generate_training_dataset`` must hand the caller's ``OrchestrationConfig`` straight to
    :func:`~femtoolkit.orchestration.simulation.evaluate_simulation_batch` unchanged (Version 34
    integration) -- that the resulting execution is bit-identical to serial regardless of worker
    count is already exhaustively covered by the dedicated Version 34 orchestration test suite, so
    this test only needs to confirm the pass-through itself, not re-create a real worker pool for
    every surrogate test run.
    """
    base_project = _cantilever_project()
    thickness = _thickness_parameter([0.006, 0.008, 0.010, 0.012])
    requested_config = OrchestrationConfig(execution_mode="parallel", max_workers=2)
    received_configs: list[OrchestrationConfig | None] = []

    real_evaluate_simulation_batch = evaluate_simulation_batch

    def spy(tasks, config=None):
        received_configs.append(config)
        return real_evaluate_simulation_batch(tasks, config=None)

    with mock.patch(
        "femtoolkit.surrogate.workflows.training.evaluate_simulation_batch", side_effect=spy
    ):
        dataset = generate_training_dataset(
            base_project,
            [thickness],
            {"maximum_displacement": get_extractor("maximum_displacement")},
            orchestration_config=requested_config,
        )

    assert received_configs == [requested_config]
    assert dataset.n_samples == 4


@pytest.mark.slow
def test_full_training_and_high_fidelity_verification_workflow() -> None:
    base_project = _cantilever_project()
    # A larger sweep than the other tests in this module: the R^2 assertion below
    # is computed on the held-out test split, and with only a handful of snapshots
    # that split is just 1-2 points -- too few for R^2 to be a stable statistic.
    thickness = _thickness_parameter([0.005 + 0.001 * i for i in range(16)])
    dataset = generate_training_dataset(
        base_project, [thickness], {"maximum_displacement": get_extractor("maximum_displacement")}
    )

    model, report = train_surrogate(
        dataset,
        TrainingConfig(model_type="polynomial", model_kwargs={"degree": 2}, split_seed=0),
    )
    assert report.test_metrics["maximum_displacement"].r2 > 0.85

    held_out_points = [{"mesh.thickness": 0.01}, {"mesh.thickness": 0.018}]
    records = verify_against_high_fidelity(
        model,
        base_project,
        held_out_points,
        {"maximum_displacement": get_extractor("maximum_displacement")},
    )
    assert len(records) == 2
    for record in records:
        assert record.actual
        assert record.acceptance in (
            AcceptanceStatus.ACCEPT,
            AcceptanceStatus.REVIEW,
            AcceptanceStatus.REJECT,
        )


@pytest.mark.slow
def test_recommend_candidates_flags_out_of_domain_points() -> None:
    base_project = _cantilever_project()
    thickness = _thickness_parameter([0.008, 0.010, 0.012, 0.014])
    dataset = generate_training_dataset(
        base_project, [thickness], {"maximum_displacement": get_extractor("maximum_displacement")}
    )
    model, _report = train_surrogate(dataset, TrainingConfig(model_type="polynomial", split_seed=0))

    candidates = recommend_candidates(
        model,
        [
            {"mesh.thickness": 0.011},
            {"mesh.thickness": 0.10},
            {"mesh.thickness": 0.0081},
        ],
    )
    assert any(c.point["mesh.thickness"] == 0.10 for c in candidates)
    assert all(c.point["mesh.thickness"] != 0.011 for c in candidates)


@pytest.mark.slow
def test_surrogate_evaluator_marks_results_as_surrogate_derived() -> None:
    base_project = _cantilever_project()
    thickness = _thickness_parameter([0.006, 0.008, 0.010, 0.012, 0.014, 0.016])
    dataset = generate_training_dataset(
        base_project, [thickness], {"maximum_displacement": get_extractor("maximum_displacement")}
    )
    model, _report = train_surrogate(dataset, TrainingConfig(model_type="polynomial", split_seed=0))

    design_variable = DesignVariable(
        name="maximum_displacement",
        path="mesh.thickness",
        variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=0.006,
        upper_bound=0.016,
    )
    objective = Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")),
    )
    evaluator = SurrogateEvaluator({"maximum_displacement": model})
    assert evaluator.backend is EvaluationBackend.SURROGATE

    evaluation = evaluator.evaluate(
        "candidate-1", {"maximum_displacement": 0.01}, [design_variable], [objective], []
    )
    assert evaluation.run_id is None
    assert evaluation.metadata["surrogate_derived"] is True
    assert "maximum_displacement" in evaluation.objective_values
