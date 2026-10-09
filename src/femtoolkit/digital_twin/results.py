"""Structured outcomes of digital twin model updating (Version 38)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ModelUpdateStatus(Enum):
    """Why a :func:`~femtoolkit.digital_twin.updating.run_model_update` call stopped.

    Attributes:
        CONVERGED: The calibration's best-found objective value plateaued
            (an empirically observed improvement plateau, never evidence of a
            global optimum) or the underlying search reported its own
            completion.
        MAX_EVALUATIONS: The configured evaluation budget was reached.
        FAILED: The baseline evaluation itself did not complete, or the
            search stopped after too many consecutive simulation failures.
        INVALID: The problem could not be run at all (e.g. no updatable
            parameter was provided).
    """

    CONVERGED = "converged"
    MAX_EVALUATIONS = "max_evaluations"
    FAILED = "failed"
    INVALID = "invalid"


class DigitalTwinStatus(Enum):
    """The engineering lifecycle state of one digital twin model-updating run.

    Attributes:
        BASELINE: Only the baseline (unmodified) model has been evaluated.
        CALIBRATING: Model updating is in progress (not observed by a caller
            of the synchronous :func:`~femtoolkit.digital_twin.updating.run_model_update`,
            which returns only after calibration finishes or stops).
        UPDATED: Model updating finished and produced new parameter values.
        VALIDATED: The updated model's accuracy was confirmed to improve on
            the baseline (see :func:`~femtoolkit.digital_twin.validation.compare_model_accuracy`).
        FAILED: Model updating did not produce a usable result.
    """

    BASELINE = "baseline"
    CALIBRATING = "calibrating"
    UPDATED = "updated"
    VALIDATED = "validated"
    FAILED = "failed"


@dataclass
class ModelUpdateResult:
    """The complete, structured outcome of one model-updating run.

    Attributes:
        initial_parameters: Every parameter's value before updating, keyed by name.
        updated_parameters: Every parameter's value after updating, keyed by
            name (unchanged from ``initial_parameters`` for non-updatable
            parameters).
        initial_objective: The calibration objective :math:`J(\\theta)`
            (mean squared residual) at the initial parameter values.
        final_objective: :math:`J(\\theta)` at the updated parameter values.
        initial_predictions: The simulation's prediction at every measurement
            point before updating, in the same order as the measurement data.
        updated_predictions: The simulation's prediction at every measurement
            point after updating.
        measured_values: The measured value at every measurement point, in
            the same order (carried alongside the predictions for
            convenience).
        residuals_before: ``initial_predictions - measured_values``.
        residuals_after: ``updated_predictions - measured_values``.
        n_evaluations: How many simulation evaluations the search performed
            (baseline included).
        status: Why the search stopped.
        stopping_reason: A short, human-readable description.
        convergence_history: The best-found objective value after each
            evaluation, in evaluation order (reusing
            :func:`~femtoolkit.optimization.history.compute_convergence`
            directly) -- for a convergence plot, not a new metric.
    """

    initial_parameters: dict[str, float]
    updated_parameters: dict[str, float]
    initial_objective: float | None
    final_objective: float | None
    initial_predictions: np.ndarray
    updated_predictions: np.ndarray
    measured_values: np.ndarray
    residuals_before: np.ndarray
    residuals_after: np.ndarray
    n_evaluations: int
    status: ModelUpdateStatus
    stopping_reason: str = ""
    convergence_history: list[float | None] = field(default_factory=list)

    @property
    def improved(self) -> bool:
        """Whether the final objective is strictly smaller than the initial objective."""
        if self.initial_objective is None or self.final_objective is None:
            return False
        return self.final_objective < self.initial_objective

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable representation of this result."""
        return {
            "initial_parameters": dict(self.initial_parameters),
            "updated_parameters": dict(self.updated_parameters),
            "initial_objective": self.initial_objective,
            "final_objective": self.final_objective,
            "initial_predictions": self.initial_predictions.tolist(),
            "updated_predictions": self.updated_predictions.tolist(),
            "measured_values": self.measured_values.tolist(),
            "residuals_before": self.residuals_before.tolist(),
            "residuals_after": self.residuals_after.tolist(),
            "n_evaluations": self.n_evaluations,
            "status": self.status.value,
            "stopping_reason": self.stopping_reason,
            "improved": self.improved,
            "convergence_history": list(self.convergence_history),
        }


__all__ = ["DigitalTwinStatus", "ModelUpdateResult", "ModelUpdateStatus"]
