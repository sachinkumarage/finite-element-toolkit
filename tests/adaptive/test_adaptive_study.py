"""Integration tests: femtoolkit.adaptive <-> real (small) FEA simulations.

These tests run the full "initial samples -> surrogate -> candidate search ->
high-fidelity verification -> retrain" pipeline end to end against a real cantilever
beam model, not synthetic data.
"""

from __future__ import annotations

import pytest

from femtoolkit.adaptive.refinement import (
    RefinementConfig,
    SurrogateAcceptanceState,
    run_refinement_step,
)
from femtoolkit.adaptive.sampling import SamplingStrategy
from femtoolkit.adaptive.study import AdaptiveStudy, generate_initial_samples
from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.exceptions import InsufficientInitialSamplesError
from femtoolkit.optimization.constraints import Constraint, ConstraintRelation
from femtoolkit.optimization.objectives import Objective, ObjectiveDirection, from_result_extractor
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType
from femtoolkit.studies.extractors import get_extractor
from femtoolkit.surrogate.workflows.training import TrainingConfig, train_surrogate


def _cantilever_project() -> Project:
    project = Project(name="Adaptive Cantilever", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.width = 2.0
    project.mesh.height = 0.4
    project.mesh.nx = 8
    project.mesh.ny = 2
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-4000.0)]
    return project


def _thickness_variable() -> DesignVariable:
    return DesignVariable(
        name="thickness", path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=0.006, upper_bound=0.020,
    )


def _displacement_objective() -> Objective:
    return Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")),
    )


def _stress_constraint() -> Constraint:
    return Constraint(
        name="maximum_von_mises_stress",
        evaluate=from_result_extractor(get_extractor("maximum_von_mises_stress")),
        relation=ConstraintRelation.LESS_EQUAL,
        limit=1.5e8,
    )


def _response_extractors() -> dict:
    return {
        "maximum_displacement": get_extractor("maximum_displacement"),
        "maximum_von_mises_stress": get_extractor("maximum_von_mises_stress"),
    }


@pytest.mark.slow
def test_generate_initial_samples_builds_a_valid_dataset() -> None:
    dataset = generate_initial_samples(
        _cantilever_project(), [_thickness_variable()], 6, _response_extractors(), seed=0
    )
    assert dataset.n_samples == 6
    assert dataset.feature_names == ["mesh.thickness"]
    assert set(dataset.response_names) == {"maximum_displacement", "maximum_von_mises_stress"}
    for snapshot in dataset.snapshots:
        assert 0.006 <= snapshot.inputs["mesh.thickness"] <= 0.020


@pytest.mark.slow
def test_run_refinement_step_adds_sample_and_retrains() -> None:
    base_project = _cantilever_project()
    design_variables = [_thickness_variable()]
    objective = _displacement_objective()
    dataset = generate_initial_samples(
        base_project, design_variables, 6, _response_extractors(), seed=0
    )
    model, _report = train_surrogate(dataset, TrainingConfig(model_type="polynomial", split_seed=0))

    config = RefinementConfig(n_candidates=10, seed=0)
    updated_dataset, updated_model, report, step = run_refinement_step(
        dataset, model, design_variables, objective, [], base_project,
        _response_extractors(), config, iteration=0,
    )

    assert updated_dataset.n_samples == dataset.n_samples + 1
    assert updated_dataset.dataset_version == dataset.dataset_version + 1
    assert updated_model.is_fitted
    assert report is not None
    assert step.verification.actual  # the high-fidelity run completed
    assert step.status in (
        SurrogateAcceptanceState.VERIFIED,
        SurrogateAcceptanceState.REQUIRES_REFINEMENT,
    )


@pytest.mark.slow
def test_adaptive_study_runs_end_to_end() -> None:
    study = AdaptiveStudy(
        base_project=_cantilever_project(),
        design_variables=[_thickness_variable()],
        objective=_displacement_objective(),
        constraints=[_stress_constraint()],
        response_extractors=_response_extractors(),
        refinement_config=RefinementConfig(
            max_iterations=3, n_candidates=15, sampling_strategy=SamplingStrategy.HYBRID, seed=0
        ),
        random_seed=0,
    )
    result = study.run(n_initial_samples=6)

    assert result.initial_sample_count == 6
    assert result.iteration_count <= 3
    assert result.total_high_fidelity_evaluations >= result.initial_sample_count
    assert result.best_verified_design is not None
    assert result.best_verified_objective is not None
    assert len(result.convergence_history) == result.iteration_count
    assert len(result.iteration_history) == result.iteration_count
    assert result.final_surrogate_metrics is not None
    assert result.stopping_reason


@pytest.mark.slow
def test_adaptive_study_stops_at_high_fidelity_evaluation_budget() -> None:
    study = AdaptiveStudy(
        base_project=_cantilever_project(),
        design_variables=[_thickness_variable()],
        objective=_displacement_objective(),
        response_extractors={"maximum_displacement": get_extractor("maximum_displacement")},
        refinement_config=RefinementConfig(
            max_iterations=10, max_high_fidelity_evaluations=1, n_candidates=10, seed=0
        ),
        random_seed=0,
    )
    result = study.run(n_initial_samples=5)
    assert result.total_high_fidelity_evaluations <= 5 + 1
    assert "high-fidelity evaluations" in result.stopping_reason


@pytest.mark.slow
def test_adaptive_study_requires_minimum_initial_samples() -> None:
    study = AdaptiveStudy(
        base_project=_cantilever_project(),
        design_variables=[_thickness_variable()],
        objective=_displacement_objective(),
        response_extractors={"maximum_displacement": get_extractor("maximum_displacement")},
    )
    with pytest.raises(InsufficientInitialSamplesError):
        study.run(n_initial_samples=2)
