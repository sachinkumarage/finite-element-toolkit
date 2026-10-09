"""Tests for femtoolkit.digital_twin.validation."""

from __future__ import annotations

import numpy as np
import pytest

from femtoolkit.digital_twin.results import ModelUpdateResult, ModelUpdateStatus
from femtoolkit.digital_twin.validation import compare_model_accuracy


def _result(initial_predictions, updated_predictions, measured) -> ModelUpdateResult:
    initial_predictions = np.array(initial_predictions)
    updated_predictions = np.array(updated_predictions)
    measured = np.array(measured)
    return ModelUpdateResult(
        initial_parameters={"x": 1.0},
        updated_parameters={"x": 1.0},
        initial_objective=1.0,
        final_objective=0.1,
        initial_predictions=initial_predictions,
        updated_predictions=updated_predictions,
        measured_values=measured,
        residuals_before=initial_predictions - measured,
        residuals_after=updated_predictions - measured,
        n_evaluations=3,
        status=ModelUpdateStatus.CONVERGED,
    )


def test_compare_model_accuracy_shows_improvement() -> None:
    result = _result(
        initial_predictions=[0.0037, 0.0075], updated_predictions=[0.0041, 0.0082],
        measured=[0.0041, 0.0082],
    )
    comparison = compare_model_accuracy(result)
    assert comparison.after.rmse < comparison.before.rmse
    assert comparison.improved is True


def test_compare_model_accuracy_reports_max_absolute_error() -> None:
    result = _result(
        initial_predictions=[0.0, 1.0], updated_predictions=[0.0, 0.1], measured=[0.0, 0.0]
    )
    comparison = compare_model_accuracy(result)
    assert comparison.before_max_absolute_error == 1.0
    assert comparison.after_max_absolute_error == pytest.approx(0.1)


def test_compare_model_accuracy_does_not_assume_improvement() -> None:
    # Updated predictions are deliberately worse than the baseline.
    result = _result(
        initial_predictions=[0.0040], updated_predictions=[0.0100], measured=[0.0041]
    )
    comparison = compare_model_accuracy(result)
    assert comparison.improved is False
