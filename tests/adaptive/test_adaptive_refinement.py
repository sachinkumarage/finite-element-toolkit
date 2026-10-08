"""Tests for femtoolkit.adaptive.refinement (non-FEA units; see test_adaptive_study.py
for the full real-FEA integration test of ``run_refinement_step`` itself)."""

from __future__ import annotations

import pytest

from femtoolkit.adaptive.refinement import RefinementConfig, prediction_agreement
from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.objectives import ObjectiveDirection


def test_prediction_agreement_matches_for_minimize() -> None:
    # Actual improvement 2.0, predicted improvement 2.0 -> perfect agreement.
    rho = prediction_agreement(10.0, 8.0, 10.0, 8.0, direction=ObjectiveDirection.MINIMIZE)
    assert rho == pytest.approx(1.0)


def test_prediction_agreement_partial_for_minimize() -> None:
    # Actual improvement 1.0, predicted improvement 2.0 -> the real result improved
    # only half as much as the surrogate predicted.
    rho = prediction_agreement(10.0, 9.0, 10.0, 8.0, direction=ObjectiveDirection.MINIMIZE)
    assert rho == pytest.approx(0.5)


def test_prediction_agreement_for_maximize() -> None:
    rho = prediction_agreement(5.0, 7.0, 5.0, 7.0, direction=ObjectiveDirection.MAXIMIZE)
    assert rho == pytest.approx(1.0)


def test_prediction_agreement_handles_near_zero_denominator() -> None:
    rho = prediction_agreement(10.0, 9.0, 10.0, 10.0 - 1e-14, direction=ObjectiveDirection.MINIMIZE)
    assert rho is None


def test_refinement_config_defaults_are_valid() -> None:
    config = RefinementConfig()
    assert config.max_iterations > 0
    assert config.max_high_fidelity_evaluations > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iterations": 0},
        {"max_high_fidelity_evaluations": 0},
        {"error_tolerance": 0.0},
        {"n_candidates": 0},
    ],
)
def test_refinement_config_rejects_invalid_values(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        RefinementConfig(**kwargs)
