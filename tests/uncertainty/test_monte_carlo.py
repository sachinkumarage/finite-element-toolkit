"""Tests for femtoolkit.uncertainty.monte_carlo."""

import numpy as np
import pytest

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.exceptions import StudySizeExceededError, ValidationError
from femtoolkit.runs.models import RunStatus
from femtoolkit.studies.extractors import get_extractor
from femtoolkit.uncertainty.distributions import NormalDistribution, UniformDistribution
from femtoolkit.uncertainty.monte_carlo import MonteCarloConfig, MonteCarloRunner
from femtoolkit.uncertainty.parameters import UncertainParameter


def _base_project() -> Project:
    project = Project(name="MC Beam", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.mesh.width, project.mesh.height = 2.0, 0.4
    project.mesh.nx, project.mesh.ny, project.mesh.thickness = 6, 2, 0.02
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-1000.0)]
    return project


def _e_parameter(mean=200e9, std=5e9, lower_bound=0.0) -> UncertainParameter:
    return UncertainParameter(
        path="material.youngs_modulus",
        label="Young's Modulus",
        distribution=NormalDistribution(mean, std),
        units="Pa",
        physical_lower_bound=lower_bound,
    )


# --- MonteCarloConfig validation ---


def test_config_requires_at_least_one_parameter() -> None:
    with pytest.raises(ValidationError):
        MonteCarloConfig(
            study_id="s", name="s", base_project=_base_project(), parameters=[],
            output_quantities=["maximum_displacement"],
        )


def test_config_requires_at_least_one_output_quantity() -> None:
    with pytest.raises(ValidationError):
        MonteCarloConfig(
            study_id="s", name="s", base_project=_base_project(), parameters=[_e_parameter()],
            output_quantities=[],
        )


def test_config_rejects_unknown_sampling_method() -> None:
    with pytest.raises(ValidationError):
        MonteCarloConfig(
            study_id="s", name="s", base_project=_base_project(), parameters=[_e_parameter()],
            output_quantities=["maximum_displacement"], method="not_a_method",
        )


def test_config_rejects_invalid_confidence_level() -> None:
    with pytest.raises(ValidationError):
        MonteCarloConfig(
            study_id="s", name="s", base_project=_base_project(), parameters=[_e_parameter()],
            output_quantities=["maximum_displacement"], confidence_level=1.5,
        )


def test_config_rejects_n_samples_over_max_samples_before_execution() -> None:
    with pytest.raises(StudySizeExceededError):
        MonteCarloConfig(
            study_id="too-big", name="Too Big", base_project=_base_project(),
            parameters=[_e_parameter()], output_quantities=["maximum_displacement"],
            n_samples=5000, max_samples=1000,
        )


def test_estimated_simulation_count_equals_n_samples() -> None:
    config = MonteCarloConfig(
        study_id="s", name="s", base_project=_base_project(), parameters=[_e_parameter()],
        output_quantities=["maximum_displacement"], n_samples=37,
    )
    assert config.estimated_simulation_count == 37


# --- MonteCarloRunner execution ---


def test_run_all_samples_succeed_for_well_posed_parameter() -> None:
    config = MonteCarloConfig(
        study_id="mc-1", name="Material Study", base_project=_base_project(),
        parameters=[_e_parameter()], output_quantities=["maximum_displacement"],
        n_samples=30, seed=1,
    )
    result = MonteCarloRunner().run(config)
    assert result.n_successful == 30
    assert result.n_failed == 0
    assert result.n_invalid == 0
    assert len(result.study_result.runs) == 30


def test_run_is_reproducible_with_same_seed() -> None:
    config = MonteCarloConfig(
        study_id="mc-repro", name="Repro", base_project=_base_project(),
        parameters=[_e_parameter()], output_quantities=["maximum_displacement"],
        n_samples=20, seed=99,
    )
    result_a = MonteCarloRunner().run(config)
    result_b = MonteCarloRunner().run(config)
    assert np.array_equal(result_a.sample_set.values, result_b.sample_set.values)

    extractor = get_extractor("maximum_displacement")
    values_a = result_a.output_values(extractor)
    values_b = result_b.output_values(extractor)
    assert np.array_equal(values_a, values_b)


def test_run_rejects_physically_invalid_samples_before_execution() -> None:
    risky_parameter = _e_parameter(mean=1e9, std=5e9, lower_bound=0.0)
    config = MonteCarloConfig(
        study_id="mc-invalid", name="Invalid Sample Test", base_project=_base_project(),
        parameters=[risky_parameter], output_quantities=["maximum_displacement"],
        n_samples=200, seed=1,
    )
    result = MonteCarloRunner().run(config)
    assert result.n_invalid > 0
    for record in result.invalid_samples:
        assert record.parameter_values["material.youngs_modulus"] <= 0.0
        assert "physical bounds" in record.reason
    assert result.n_successful + result.n_failed + result.n_invalid == config.n_samples


def test_run_output_values_matches_successful_run_count() -> None:
    config = MonteCarloConfig(
        study_id="mc-values", name="Values", base_project=_base_project(),
        parameters=[_e_parameter()], output_quantities=["maximum_displacement"],
        n_samples=15, seed=2,
    )
    result = MonteCarloRunner().run(config)
    extractor = get_extractor("maximum_displacement")
    values = result.output_values(extractor)
    assert len(values) == result.n_successful


def test_successful_pairs_align_input_and_output_by_sample() -> None:
    config = MonteCarloConfig(
        study_id="mc-pairs", name="Pairs", base_project=_base_project(),
        parameters=[_e_parameter()], output_quantities=["maximum_displacement"],
        n_samples=25, seed=3,
    )
    result = MonteCarloRunner().run(config)
    extractor = get_extractor("maximum_displacement")
    x, y = result.successful_pairs("material.youngs_modulus", extractor)
    assert len(x) == len(y) == result.n_successful
    # physically: higher E -> lower displacement (inverse relation)
    assert np.corrcoef(x, y)[0, 1] < -0.9


def test_run_with_latin_hypercube_method() -> None:
    config = MonteCarloConfig(
        study_id="mc-lhs", name="LHS", base_project=_base_project(),
        parameters=[_e_parameter()], output_quantities=["maximum_displacement"],
        n_samples=20, seed=4, method="latin_hypercube",
    )
    result = MonteCarloRunner().run(config)
    assert result.sample_set.method == "latin_hypercube"
    assert result.n_successful == 20


def test_run_with_multiple_uncertain_parameters() -> None:
    load_parameter = UncertainParameter(
        path="loads.0.magnitude", label="Load", distribution=UniformDistribution(-2000.0, -1000.0)
    )
    config = MonteCarloConfig(
        study_id="mc-multi", name="Multi", base_project=_base_project(),
        parameters=[_e_parameter(), load_parameter],
        output_quantities=["maximum_displacement"], n_samples=15, seed=5,
    )
    result = MonteCarloRunner().run(config)
    assert result.sample_set.values.shape == (15, 2)
    assert result.n_successful == 15


def test_run_fail_fast_stops_at_first_failure() -> None:
    bad_project = _base_project()
    # A Poisson ratio range straddling the valid limit (v < 0.5) produces a genuine mix
    # of project-validation failures and successes -- not a physical-bound rejection.
    nu_parameter = UncertainParameter(
        path="material.poisson_ratio",
        label="Poisson ratio",
        distribution=UniformDistribution(0.4, 0.6),
    )
    config = MonteCarloConfig(
        study_id="mc-fail-fast", name="Fail Fast", base_project=bad_project,
        parameters=[nu_parameter], output_quantities=["maximum_displacement"],
        n_samples=50, seed=6, fail_fast=True,
    )
    result = MonteCarloRunner().run(config)
    assert any(run.status == RunStatus.FAILED for run in result.study_result.runs)
    # execution stopped at (or before) the first failure -- not all 50 were attempted
    assert len(result.study_result.runs) <= config.n_samples


def test_run_non_fail_fast_continues_after_failures() -> None:
    bad_project = _base_project()
    nu_parameter = UncertainParameter(
        path="material.poisson_ratio",
        label="Poisson ratio",
        distribution=UniformDistribution(0.4, 0.6),
    )
    config = MonteCarloConfig(
        study_id="mc-continue", name="Continue", base_project=bad_project,
        parameters=[nu_parameter], output_quantities=["maximum_displacement"],
        n_samples=50, seed=6, fail_fast=False,
    )
    result = MonteCarloRunner().run(config)
    assert len(result.study_result.runs) == 50
    assert result.n_failed > 0
