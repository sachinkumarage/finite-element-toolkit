"""Tests for femtoolkit.digital_twin.results."""

from __future__ import annotations

import json

import numpy as np

from femtoolkit.digital_twin.results import ModelUpdateResult, ModelUpdateStatus


def _result(initial_objective: float | None, final_objective: float | None) -> ModelUpdateResult:
    return ModelUpdateResult(
        initial_parameters={"youngs_modulus": 200e9},
        updated_parameters={"youngs_modulus": 180e9},
        initial_objective=initial_objective,
        final_objective=final_objective,
        initial_predictions=np.array([0.0037]),
        updated_predictions=np.array([0.0041]),
        measured_values=np.array([0.0041]),
        residuals_before=np.array([-0.0004]),
        residuals_after=np.array([0.0]),
        n_evaluations=5,
        status=ModelUpdateStatus.CONVERGED,
        stopping_reason="converged",
    )


def test_initial_and_final_values_preserved() -> None:
    result = _result(1e-6, 1e-8)
    assert result.initial_parameters == {"youngs_modulus": 200e9}
    assert result.updated_parameters == {"youngs_modulus": 180e9}
    assert result.initial_objective == 1e-6
    assert result.final_objective == 1e-8


def test_residual_calculation() -> None:
    result = _result(1e-6, 1e-8)
    assert result.residuals_before[0] == -0.0004
    assert result.residuals_after[0] == 0.0


def test_improved_true_when_final_objective_smaller() -> None:
    result = _result(1e-6, 1e-8)
    assert result.improved is True


def test_improved_false_when_final_objective_not_smaller() -> None:
    result = _result(1e-8, 1e-6)
    assert result.improved is False


def test_improved_false_when_objectives_unavailable() -> None:
    result = _result(None, None)
    assert result.improved is False


def test_to_dict_is_json_serializable() -> None:
    result = _result(1e-6, 1e-8)
    payload = result.to_dict()
    json.dumps(payload)  # raises if anything is not JSON-serializable
    assert payload["status"] == "converged"
    assert payload["improved"] is True
    assert payload["initial_predictions"] == [0.0037]
