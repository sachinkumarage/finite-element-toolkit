"""The fused prediction `y_hat_H(x) = y_L(x) + delta_hat(x)` and its verification (Version 37).

.. math::

    \\hat y_H(x) = y_L(x) + \\hat\\delta(x)

``y_L`` (the low-fidelity result), ``delta_hat`` (the predicted discrepancy), and
``y_hat_H`` (the fused prediction) are kept as three separate fields throughout
this module -- never collapsed into one number -- and **the fused prediction is
never the same thing as an actual high-fidelity FEA result**. Use
:func:`verify_fused_prediction` to check it against one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from femtoolkit.exceptions import ValidationError
from femtoolkit.multifidelity.fidelity import FidelityModel
from femtoolkit.surrogate.metrics import relative_error
from femtoolkit.surrogate.models.base import SurrogateModel


@dataclass(frozen=True)
class FusedPrediction:
    """One multi-fidelity prediction, with its three components kept separate.

    Attributes:
        input_point: The design point's input values, keyed by name.
        low_fidelity_result: :math:`y_L(x)`, the low-fidelity model's result.
        predicted_discrepancy: :math:`\\hat\\delta(x)`, the discrepancy surrogate's
            prediction.
        fused_prediction: :math:`\\hat y_H(x) = y_L(x) + \\hat\\delta(x)`.
        is_fused_prediction: Always ``True`` -- an explicit, named flag so a report
            or log line can never mistake this for a direct high-fidelity FEA
            result (mirrors
            :attr:`~femtoolkit.surrogate.models.base.SurrogatePrediction.is_surrogate_prediction`).
    """

    input_point: dict[str, float]
    low_fidelity_result: dict[str, float]
    predicted_discrepancy: dict[str, float]
    fused_prediction: dict[str, float]
    is_fused_prediction: bool = True


class MultiFidelityModel:
    """Combines a low-fidelity model with a discrepancy surrogate into a fused prediction.

    Attributes:
        low_fidelity_model: The cheap model evaluated directly at every query point.
        discrepancy_model: The Version 35 surrogate trained on
            ``x -> delta(x) = y_H(x) - y_L(x)`` (see
            :func:`~femtoolkit.multifidelity.discrepancy.train_discrepancy_surrogate`).
    """

    def __init__(
        self, low_fidelity_model: FidelityModel, discrepancy_model: SurrogateModel
    ) -> None:
        """Create a multi-fidelity model.

        Args:
            low_fidelity_model: The cheap model to query directly.
            discrepancy_model: The fitted discrepancy surrogate.

        Raises:
            ValidationError: If ``discrepancy_model`` has not been fit.
        """
        if not discrepancy_model.is_fitted:
            raise ValidationError(
                "MultiFidelityModel requires an already-fitted discrepancy_model."
            )
        self.low_fidelity_model = low_fidelity_model
        self.discrepancy_model = discrepancy_model

    def predict(self, point: dict[str, float]) -> FusedPrediction:
        """Compute the fused prediction at one design point.

        Args:
            point: The design point's input values, keyed by name.

        Returns:
            A :class:`FusedPrediction` with ``low_fidelity_result``,
            ``predicted_discrepancy``, and ``fused_prediction`` kept separate.
        """
        low_result = self.low_fidelity_model.evaluate(point)
        discrepancy_prediction = self.discrepancy_model.predict_point(point)
        fused = {
            name: low_result[name] + discrepancy_prediction.values[name]
            for name in low_result
            if name in discrepancy_prediction.values
        }
        return FusedPrediction(
            input_point=dict(point),
            low_fidelity_result=low_result,
            predicted_discrepancy=dict(discrepancy_prediction.values),
            fused_prediction=fused,
        )


class FusionAcceptanceStatus(Enum):
    """Whether a fused prediction's high-fidelity verification improved on the low-fidelity result.

    Attributes:
        IMPROVED: The fused prediction's error against the real high-fidelity
            result is smaller than the low-fidelity model's own error.
        NOT_IMPROVED: The fused prediction did not improve on (or made worse) the
            raw low-fidelity error -- the correction is not assumed to always help.
        VERIFICATION_FAILED: The high-fidelity run itself did not complete; no
            comparison could be made.
    """

    IMPROVED = "improved"
    NOT_IMPROVED = "not_improved"
    VERIFICATION_FAILED = "verification_failed"


@dataclass(frozen=True)
class FusionVerificationRecord:
    """One fused-prediction-vs-real-high-fidelity-FEA comparison.

    Attributes:
        design_point: The design point that was checked.
        low_fidelity_result: :math:`y_L(x)`.
        fused_prediction: :math:`\\hat y_H(x)`.
        high_fidelity_result: The real high-fidelity result (empty if the
            high-fidelity evaluation did not produce a usable value).
        low_fidelity_absolute_error: ``|y_H - y_L|``, per response.
        low_fidelity_relative_error: ``|y_H - y_L| / (|y_H| + epsilon)``, per response.
        fused_absolute_error: ``|y_H - y_hat_H|``, per response.
        fused_relative_error: ``|y_H - y_hat_H| / (|y_H| + epsilon)``, per response.
        status: This record's :class:`FusionAcceptanceStatus`.
    """

    design_point: dict[str, float]
    low_fidelity_result: dict[str, float]
    fused_prediction: dict[str, float]
    high_fidelity_result: dict[str, float]
    low_fidelity_absolute_error: dict[str, float] = field(default_factory=dict)
    low_fidelity_relative_error: dict[str, float] = field(default_factory=dict)
    fused_absolute_error: dict[str, float] = field(default_factory=dict)
    fused_relative_error: dict[str, float] = field(default_factory=dict)
    status: FusionAcceptanceStatus = FusionAcceptanceStatus.VERIFICATION_FAILED


def verify_fused_prediction(
    model: MultiFidelityModel, high_fidelity_model: FidelityModel, points: list[dict[str, float]]
) -> list[FusionVerificationRecord]:
    """Compare the fused prediction against a freshly-evaluated real high-fidelity result.

    Every point is both predicted by ``model`` (cheap: a low-fidelity evaluation
    plus a discrepancy-surrogate prediction) and actually evaluated by
    ``high_fidelity_model`` -- the fused prediction is never trusted on its own.

    Args:
        model: The fitted multi-fidelity model to verify.
        high_fidelity_model: The real high-fidelity model to verify against.
        points: The design points to check.

    Returns:
        One :class:`FusionVerificationRecord` per point, in order.
    """
    records: list[FusionVerificationRecord] = []
    for point in points:
        fused = model.predict(point)
        high_result = high_fidelity_model.evaluate(point)

        if not high_result:
            records.append(
                FusionVerificationRecord(
                    design_point=dict(point),
                    low_fidelity_result=fused.low_fidelity_result,
                    fused_prediction=fused.fused_prediction,
                    high_fidelity_result={},
                    status=FusionAcceptanceStatus.VERIFICATION_FAILED,
                )
            )
            continue

        low_abs: dict[str, float] = {}
        low_rel: dict[str, float] = {}
        fused_abs: dict[str, float] = {}
        fused_rel: dict[str, float] = {}
        for name, high_value in high_result.items():
            high_array = np.array([high_value])
            if name in fused.low_fidelity_result:
                low_value = fused.low_fidelity_result[name]
                low_abs[name] = abs(high_value - low_value)
                low_rel[name] = float(relative_error(high_array, np.array([low_value]))[0])
            if name in fused.fused_prediction:
                fused_value = fused.fused_prediction[name]
                fused_abs[name] = abs(high_value - fused_value)
                fused_rel[name] = float(relative_error(high_array, np.array([fused_value]))[0])

        comparable = [name for name in fused_rel if name in low_rel]
        if comparable and all(fused_rel[name] <= low_rel[name] for name in comparable):
            status = FusionAcceptanceStatus.IMPROVED
        else:
            status = FusionAcceptanceStatus.NOT_IMPROVED

        records.append(
            FusionVerificationRecord(
                design_point=dict(point),
                low_fidelity_result=fused.low_fidelity_result,
                fused_prediction=fused.fused_prediction,
                high_fidelity_result=dict(high_result),
                low_fidelity_absolute_error=low_abs,
                low_fidelity_relative_error=low_rel,
                fused_absolute_error=fused_abs,
                fused_relative_error=fused_rel,
                status=status,
            )
        )
    return records


__all__ = [
    "FusedPrediction",
    "FusionAcceptanceStatus",
    "FusionVerificationRecord",
    "MultiFidelityModel",
    "verify_fused_prediction",
]
