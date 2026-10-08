"""Tests for femtoolkit.adaptive.sampling."""

from __future__ import annotations

import pytest

from femtoolkit.adaptive.candidates import generate_candidate_pool
from femtoolkit.adaptive.sampling import (
    SamplingStrategy,
    design_variable_bounds,
    error_score,
    exploration_score,
    normalized_distance,
    rank_candidates,
)
from femtoolkit.optimization.objectives import Objective, ObjectiveDirection, from_result_extractor
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType
from femtoolkit.surrogate.datasets import Snapshot, SnapshotDataset
from femtoolkit.surrogate.domain import DomainStatus
from femtoolkit.surrogate.models.polynomial import PolynomialRegressionSurrogate
from femtoolkit.surrogate.workflows.evaluator import SurrogateEvaluator


def _design_variables() -> list[DesignVariable]:
    return [
        DesignVariable(
            name="thickness", path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
            lower_bound=0.0, upper_bound=10.0,
        ),
        DesignVariable(
            name="width", path="mesh.width", variable_type=DesignVariableType.CONTINUOUS,
            lower_bound=0.0, upper_bound=100.0,
        ),
    ]


def _thickness_variable(lower_bound: float = 1.0, upper_bound: float = 5.0) -> list[DesignVariable]:
    return [
        DesignVariable(
            name="thickness", path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
            lower_bound=lower_bound, upper_bound=upper_bound,
        )
    ]


def _displacement_objective() -> Objective:
    return Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(lambda run: None),
    )


def test_normalized_distance_rescales_each_dimension() -> None:
    bounds = {"mesh.thickness": (0.0, 10.0), "mesh.width": (0.0, 100.0)}
    a = {"mesh.thickness": 0.0, "mesh.width": 0.0}
    b = {"mesh.thickness": 10.0, "mesh.width": 0.0}
    c = {"mesh.thickness": 0.0, "mesh.width": 100.0}
    # Both b and c are one full normalized unit away from a along one axis, despite
    # the raw physical magnitudes differing by 10x -- the whole point of rescaling.
    assert normalized_distance(a, b, bounds) == pytest.approx(1.0)
    assert normalized_distance(a, c, bounds) == pytest.approx(1.0)


def test_normalized_distance_handles_zero_width_range() -> None:
    bounds = {"x": (5.0, 5.0)}
    assert normalized_distance({"x": 5.0}, {"x": 5.0}, bounds) == pytest.approx(0.0)


def test_normalized_distance_ignores_dimensions_missing_from_a_point() -> None:
    bounds = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
    assert normalized_distance({"x": 0.0}, {"x": 1.0, "y": 1.0}, bounds) == pytest.approx(1.0)


def test_design_variable_bounds_omits_categorical() -> None:
    variables = _design_variables() + [
        DesignVariable(
            name="material", path="material.name", variable_type=DesignVariableType.CATEGORICAL,
            categories=["steel", "aluminum"],
        )
    ]
    bounds = design_variable_bounds(variables)
    assert bounds == {"mesh.thickness": (0.0, 10.0), "mesh.width": (0.0, 100.0)}


def test_exploration_score_is_zero_for_empty_dataset() -> None:
    dataset = SnapshotDataset(feature_names=["x"], response_names=["y"])
    assert exploration_score({"x": 0.5}, dataset, {"x": (0.0, 1.0)}) == pytest.approx(0.0)


def test_exploration_score_prefers_points_far_from_training_data() -> None:
    dataset = SnapshotDataset(feature_names=["x"], response_names=["y"])
    dataset.add_snapshot(Snapshot(snapshot_id="a", inputs={"x": 0.0}, outputs={"y": 0.0}))
    bounds = {"x": (0.0, 1.0)}
    near = exploration_score({"x": 0.1}, dataset, bounds)
    far = exploration_score({"x": 0.9}, dataset, bounds)
    assert far > near


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (DomainStatus.WITHIN_TRAINING_DOMAIN, 0.0),
        (DomainStatus.BOUNDARY, 0.5),
        (DomainStatus.OUTSIDE_TRAINING_DOMAIN, 1.0),
        (DomainStatus.INVALID, 1.0),
    ],
)
def test_error_score_ranks_domain_status(status: DomainStatus, expected: float) -> None:
    assert error_score(status) == pytest.approx(expected)


def _fitted_model_and_dataset() -> tuple[PolynomialRegressionSurrogate, SnapshotDataset]:
    dataset = SnapshotDataset(
        feature_names=["mesh.thickness"], response_names=["maximum_displacement"]
    )
    for i, x in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
        dataset.add_snapshot(
            Snapshot(
                snapshot_id=str(i),
                inputs={"mesh.thickness": x},
                outputs={"maximum_displacement": 1.0 / x},
            )
        )
    model = PolynomialRegressionSurrogate(degree=2)
    x, y = dataset.to_arrays()
    model.fit(
        x, y, feature_names=dataset.feature_names, response_names=dataset.response_names,
        dataset_id="d1",
    )
    return model, dataset


def test_rank_candidates_reproducible_with_same_seed() -> None:
    model, dataset = _fitted_model_and_dataset()
    design_variables = _thickness_variable()
    objective = _displacement_objective()
    evaluator = SurrogateEvaluator({"maximum_displacement": model})

    candidates_a = generate_candidate_pool(design_variables, 10, seed=42)
    candidates_b = generate_candidate_pool(design_variables, 10, seed=42)
    assert candidates_a == candidates_b

    ranked_a = rank_candidates(candidates_a, dataset, design_variables, evaluator, objective)
    ranked_b = rank_candidates(candidates_b, dataset, design_variables, evaluator, objective)
    assert [c.point for c in ranked_a] == [c.point for c in ranked_b]
    assert ranked_a[0].score == pytest.approx(ranked_b[0].score)


def test_rank_candidates_distance_strategy_prefers_unexplored_point() -> None:
    model, dataset = _fitted_model_and_dataset()
    design_variables = _thickness_variable()
    objective = _displacement_objective()
    evaluator = SurrogateEvaluator({"maximum_displacement": model})

    # Training points sit at integers 1..5; 3.0 lands exactly on one (distance 0),
    # 1.01 is almost on top of training point 1, and 1.5 sits at the midpoint between
    # training points 1 and 2 -- the most poorly covered of the three.
    candidates = [{"thickness": 1.01}, {"thickness": 3.0}, {"thickness": 1.5}]
    ranked = rank_candidates(
        candidates, dataset, design_variables, evaluator, objective,
        strategy=SamplingStrategy.DISTANCE,
    )
    assert ranked[0].point["thickness"] == pytest.approx(1.5)
    assert ranked[-1].point["thickness"] == pytest.approx(3.0)


def test_rank_candidates_objective_strategy_prefers_best_predicted_value() -> None:
    model, dataset = _fitted_model_and_dataset()
    design_variables = _thickness_variable()
    objective = _displacement_objective()
    evaluator = SurrogateEvaluator({"maximum_displacement": model})

    candidates = [{"thickness": 1.1}, {"thickness": 4.9}]
    ranked = rank_candidates(
        candidates, dataset, design_variables, evaluator, objective,
        strategy=SamplingStrategy.OBJECTIVE,
    )
    # Minimizing displacement ~ 1/x: the larger thickness should predict the smaller
    # (better) displacement and therefore rank first.
    assert ranked[0].point["thickness"] == pytest.approx(4.9)


def test_rank_candidates_hybrid_strategy_combines_scores() -> None:
    model, dataset = _fitted_model_and_dataset()
    design_variables = _thickness_variable()
    objective = _displacement_objective()
    evaluator = SurrogateEvaluator({"maximum_displacement": model})
    candidates = generate_candidate_pool(design_variables, 15, seed=1)

    ranked = rank_candidates(
        candidates, dataset, design_variables, evaluator, objective,
        strategy=SamplingStrategy.HYBRID, exploration_weight=0.5, exploitation_weight=0.5,
    )
    assert len(ranked) == 15
    assert all(ranked[i].score >= ranked[i + 1].score for i in range(len(ranked) - 1))
