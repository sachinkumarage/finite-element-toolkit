"""Tests for femtoolkit.optimization.robust.

Covers RobustDesignConfig validation, the evaluation-budget safeguard,
the statistic computation against small analytically-known datasets,
and end-to-end robust objective/constraint evaluation through the real
Monte Carlo + FEA pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.exceptions import StudySizeExceededError, ValidationError
from femtoolkit.optimization.algorithms import OptimizationConfig
from femtoolkit.optimization.constraints import Constraint, ConstraintRelation
from femtoolkit.optimization.objectives import Objective, ObjectiveDirection, rectangular_mass
from femtoolkit.optimization.problems import OptimizationMode, OptimizationProblem
from femtoolkit.optimization.robust import (
    RobustDesignConfig,
    _compute_statistic,
    estimate_total_fea_count,
    robust_constraint_statistic,
    robust_objective_statistic,
    validate_robust_budget,
)
from femtoolkit.optimization.runner import OptimizationRunner
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType
from femtoolkit.uncertainty.distributions import NormalDistribution
from femtoolkit.uncertainty.parameters import UncertainParameter, UncertaintyCategory


def _base_project() -> Project:
    project = Project(name="Robust Test", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.width, project.mesh.height = 2.0, 0.4
    project.mesh.nx, project.mesh.ny = 6, 2
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


def _build_params(context):
    return [
        UncertainParameter(
            path="material.youngs_modulus", label="E", distribution=NormalDistribution(200e9, 10e9),
            units="Pa", category=UncertaintyCategory.ALEATORY, physical_lower_bound=1.0,
        )
    ]


# --- RobustDesignConfig validation ---


def test_robust_design_config_defaults_are_valid() -> None:
    config = RobustDesignConfig()
    assert not config.uncertainty_enabled


def test_robust_design_config_rejects_zero_sample_count() -> None:
    with pytest.raises(ValidationError):
        RobustDesignConfig(sample_count=0)


def test_robust_design_config_rejects_percentile_out_of_range() -> None:
    with pytest.raises(ValidationError):
        RobustDesignConfig(percentile=0.0)
    with pytest.raises(ValidationError):
        RobustDesignConfig(percentile=100.0)


def test_robust_design_config_rejects_unknown_sampling_method() -> None:
    with pytest.raises(ValidationError):
        RobustDesignConfig(sampling_method="not_a_method")


def test_robust_design_config_rejects_unknown_statistic() -> None:
    with pytest.raises(ValidationError):
        RobustDesignConfig(objective_statistic="not_a_statistic")
    with pytest.raises(ValidationError):
        RobustDesignConfig(constraint_statistic="not_a_statistic")


def test_robust_design_config_rejects_unknown_failure_policy() -> None:
    with pytest.raises(ValidationError):
        RobustDesignConfig(failure_policy="not_a_policy")


def test_robust_design_config_rejects_zero_maximum_total_evaluations() -> None:
    with pytest.raises(ValidationError):
        RobustDesignConfig(maximum_total_evaluations=0)


# --- Evaluation budget protection ---


def test_estimate_total_fea_count_deterministic_equals_max_evaluations() -> None:
    assert estimate_total_fea_count(50, None) == 50
    disabled = RobustDesignConfig(uncertainty_enabled=False, sample_count=20)
    assert estimate_total_fea_count(50, disabled) == 50


def test_estimate_total_fea_count_robust_multiplies_by_sample_count() -> None:
    config = RobustDesignConfig(uncertainty_enabled=True, sample_count=20)
    assert estimate_total_fea_count(50, config) == 1000


def test_validate_robust_budget_raises_before_any_execution() -> None:
    config = RobustDesignConfig(
        uncertainty_enabled=True, sample_count=100, maximum_total_evaluations=500
    )
    with pytest.raises(StudySizeExceededError):
        validate_robust_budget(10, config)


def test_validate_robust_budget_accepts_within_limit() -> None:
    config = RobustDesignConfig(
        uncertainty_enabled=True, sample_count=10, maximum_total_evaluations=500
    )
    validate_robust_budget(10, config)  # must not raise


def test_runner_rejects_oversized_robust_budget_before_baseline_runs() -> None:
    variable = _thickness_variable()
    objective = Objective(
        name="mean_disp", direction=ObjectiveDirection.MINIMIZE,
        evaluate=robust_objective_statistic(
            _build_params,
            "maximum_displacement",
            RobustDesignConfig(
                uncertainty_enabled=True, sample_count=50, maximum_total_evaluations=10
            ),
        ),
    )
    problem = OptimizationProblem(
        name="robust-budget", base_project=_base_project(), design_variables=[variable],
        objectives=[objective], mode=OptimizationMode.UNCERTAINTY_AWARE,
    )
    config = OptimizationConfig(algorithm="random_search", max_evaluations=5)
    robust_config = RobustDesignConfig(
        uncertainty_enabled=True, sample_count=50, maximum_total_evaluations=10
    )
    with pytest.raises(StudySizeExceededError):
        OptimizationRunner().run(problem, config, robust_config=robust_config)


# --- Statistic computation against known small datasets ---


def test_compute_statistic_mean() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _compute_statistic(values, "mean", 95.0, None, "above") == pytest.approx(3.0)


def test_compute_statistic_median() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    assert _compute_statistic(values, "median", 95.0, None, "above") == pytest.approx(3.0)


def test_compute_statistic_std_requires_two_samples() -> None:
    assert _compute_statistic(np.array([5.0]), "std", 95.0, None, "above") is None
    values = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert _compute_statistic(values, "std", 95.0, None, "above") == pytest.approx(
        float(values.std(ddof=1))
    )


def test_compute_statistic_coefficient_of_variation() -> None:
    values = np.array([10.0, 10.0, 10.0, 10.0])
    assert _compute_statistic(values, "coefficient_of_variation", 95.0, None, "above") == 0.0


def test_compute_statistic_min_max() -> None:
    values = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    assert _compute_statistic(values, "min", 95.0, None, "above") == pytest.approx(1.0)
    assert _compute_statistic(values, "max", 95.0, None, "above") == pytest.approx(5.0)


def test_compute_statistic_percentile() -> None:
    values = np.arange(1.0, 101.0)  # 1..100
    result = _compute_statistic(values, "percentile", 50.0, None, "above")
    assert result == pytest.approx(np.percentile(values, 50.0))


def test_compute_statistic_exceedance_probability() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = _compute_statistic(values, "exceedance_probability", 95.0, 3.0, "above")
    assert result == pytest.approx(2.0 / 5.0)  # 4, 5 exceed 3.0


def test_compute_statistic_exceedance_probability_requires_threshold() -> None:
    values = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValidationError):
        _compute_statistic(values, "exceedance_probability", 95.0, None, "above")


def test_compute_statistic_empty_values_returns_none() -> None:
    assert _compute_statistic(np.array([]), "mean", 95.0, None, "above") is None


# --- End-to-end robust objective/constraint evaluation ---


def test_robust_objective_statistic_mean_end_to_end() -> None:
    variable = _thickness_variable()
    robust_config = RobustDesignConfig(uncertainty_enabled=True, sample_count=10, random_seed=1)
    objective = Objective(
        name="mean_disp", direction=ObjectiveDirection.MINIMIZE,
        evaluate=robust_objective_statistic(_build_params, "maximum_displacement", robust_config),
    )
    problem = OptimizationProblem(
        name="robust-mean", base_project=_base_project(), design_variables=[variable],
        objectives=[objective], mode=OptimizationMode.UNCERTAINTY_AWARE,
    )
    config = OptimizationConfig(algorithm="random_search", max_evaluations=3, seed=2)
    result = OptimizationRunner().run(problem, config, robust_config=robust_config)
    assert result.baseline.is_feasible
    assert result.baseline.objective_values["mean_disp"] > 0.0
    assert "robust_objectives" in result.baseline.metadata
    diagnostics = result.baseline.metadata["robust_objectives"]["maximum_displacement"]
    assert diagnostics["statistic"] == "mean"
    assert diagnostics["n_samples"] == 10


def test_robust_objective_statistic_percentile() -> None:
    variable = _thickness_variable()
    robust_config = RobustDesignConfig(
        uncertainty_enabled=True, sample_count=15, random_seed=3,
        objective_statistic="percentile", percentile=95.0,
    )
    objective = Objective(
        name="p95_disp", direction=ObjectiveDirection.MINIMIZE,
        evaluate=robust_objective_statistic(_build_params, "maximum_displacement", robust_config),
    )
    problem = OptimizationProblem(
        name="robust-p95", base_project=_base_project(), design_variables=[variable],
        objectives=[objective], mode=OptimizationMode.UNCERTAINTY_AWARE,
    )
    config = OptimizationConfig(algorithm="random_search", max_evaluations=2, seed=4)
    result = OptimizationRunner().run(problem, config, robust_config=robust_config)
    assert result.baseline.metadata["robust_objectives"]["maximum_displacement"]["statistic"] == (
        "percentile"
    )


def test_robust_constraint_statistic_exceedance_probability() -> None:
    variable = _thickness_variable()
    robust_config = RobustDesignConfig(
        uncertainty_enabled=True, sample_count=15, random_seed=5,
        constraint_statistic="exceedance_probability",
    )
    mass_objective = Objective(
        name="mass", direction=ObjectiveDirection.MINIMIZE, evaluate=rectangular_mass
    )
    stress_constraint = Constraint(
        name="exceedance_stress",
        evaluate=robust_constraint_statistic(
            _build_params, "maximum_von_mises_stress", robust_config, threshold=250e6
        ),
        relation=ConstraintRelation.LESS_EQUAL, limit=0.1,
    )
    problem = OptimizationProblem(
        name="robust-exceedance", base_project=_base_project(), design_variables=[variable],
        objectives=[mass_objective],
        constraints=[stress_constraint], mode=OptimizationMode.UNCERTAINTY_AWARE,
    )
    config = OptimizationConfig(algorithm="random_search", max_evaluations=2, seed=6)
    result = OptimizationRunner().run(problem, config, robust_config=robust_config)
    assert "robust_constraints" in result.baseline.metadata
    diagnostics = result.baseline.metadata["robust_constraints"]["maximum_von_mises_stress"]
    assert diagnostics["statistic"] == "exceedance_probability"
    assert 0.0 <= diagnostics["value"] <= 1.0


def test_robust_objective_is_reproducible_with_same_seed() -> None:
    variable = _thickness_variable()

    def _run():
        robust_config = RobustDesignConfig(uncertainty_enabled=True, sample_count=8, random_seed=42)
        objective = Objective(
            name="mean_disp", direction=ObjectiveDirection.MINIMIZE,
            evaluate=robust_objective_statistic(
                _build_params, "maximum_displacement", robust_config
            ),
        )
        problem = OptimizationProblem(
            name="robust-repro", base_project=_base_project(), design_variables=[variable],
            objectives=[objective], mode=OptimizationMode.UNCERTAINTY_AWARE,
        )
        config = OptimizationConfig(algorithm="random_search", max_evaluations=3, seed=7)
        return OptimizationRunner().run(problem, config, robust_config=robust_config)

    result_a = _run()
    result_b = _run()
    values_a = [e.objective_values["mean_disp"] for e in result_a.history.evaluations]
    values_b = [e.objective_values["mean_disp"] for e in result_b.history.evaluations]
    assert values_a == values_b


# --- Picklability (Version 34: required for parallel optimization evaluation) ---


def test_robust_objective_statistic_is_picklable() -> None:
    import pickle

    robust_config = RobustDesignConfig(uncertainty_enabled=True, sample_count=5)
    objective_fn = robust_objective_statistic(_build_params, "maximum_displacement", robust_config)
    pickle.dumps(objective_fn)


def test_robust_constraint_statistic_is_picklable() -> None:
    import pickle

    robust_config = RobustDesignConfig(uncertainty_enabled=True, sample_count=5)
    constraint_fn = robust_constraint_statistic(
        _build_params, "maximum_von_mises_stress", robust_config
    )
    pickle.dumps(constraint_fn)
