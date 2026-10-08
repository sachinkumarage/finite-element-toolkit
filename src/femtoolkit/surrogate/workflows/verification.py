"""Verifying surrogate predictions against high-fidelity FEA (Version 35).

A core engineering requirement for any surrogate: before a prediction
(or a surrogate-optimized design) is trusted, it must be checked
against the real FEA model on held-out points it was never trained on.
This module never reports a surrogate-only result as verified -- every
:class:`SurrogateVerificationRecord` here carries both the surrogate's
prediction *and* an actual high-fidelity FEA result obtained by
actually running :class:`~femtoolkit.runs.manager.SimulationRunManager`.

.. code-block:: text

    Surrogate optimization
            |
            v
    Candidate designs
            |
            v
    High-fidelity FEA verification
            |
            v
    Compare
            |
            v
    Accept / Reject / Review
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from femtoolkit.application.project import Project
from femtoolkit.runs.manager import SimulationRunManager
from femtoolkit.runs.models import RunStatus
from femtoolkit.studies.extractors import Extractor
from femtoolkit.studies.scenarios import Scenario, apply_scenario
from femtoolkit.surrogate.metrics import relative_error
from femtoolkit.surrogate.models.base import PredictionStatus, SurrogateModel
from femtoolkit.surrogate.validation import EngineeringTolerance


class AcceptanceStatus(Enum):
    """The outcome of comparing a surrogate-derived design against high-fidelity verification.

    Attributes:
        ACCEPT: The high-fidelity result confirms the surrogate's
            prediction within every configured engineering tolerance,
            and the design point was within its training domain.
        REVIEW: The high-fidelity result is available and was computed
            (so the design *is* verified), but either no tolerance was
            configured, or the design point was outside/at the boundary
            of the training domain -- a human should review this case
            rather than treating it as automatically acceptable.
        REJECT: The high-fidelity result violates at least one
            configured engineering tolerance.
        FAILED: The high-fidelity verification run itself did not
            complete; no comparison could be made.
    """

    ACCEPT = "accept"
    REVIEW = "review"
    REJECT = "reject"
    FAILED = "failed"


@dataclass(frozen=True)
class SurrogateVerificationRecord:
    """One surrogate-prediction-vs-high-fidelity-FEA comparison.

    Attributes:
        design_id: A unique identifier for this verification (also used
            as the underlying verification run's scenario ID).
        design_point: The input/design-variable values that were checked.
        predicted: The surrogate's predicted response value(s).
        actual: The high-fidelity FEA's actual response value(s) (empty
            if the run did not complete).
        absolute_error: ``|actual - predicted|``, per response (empty if unavailable).
        relative_error: The relative error, per response (empty if unavailable).
        domain_status: The design point's applicability-domain status.
        acceptance: This record's :class:`AcceptanceStatus`.
        run_id: The verification run's ID, or ``None`` if it never ran.
        error_message: A human-readable failure description, or ``None``.
    """

    design_id: str
    design_point: dict[str, float]
    predicted: dict[str, float]
    actual: dict[str, float]
    absolute_error: dict[str, float]
    relative_error: dict[str, float]
    domain_status: str
    acceptance: AcceptanceStatus
    run_id: str | None
    error_message: str | None = None


def verify_against_high_fidelity(
    model: SurrogateModel,
    base_project: Project,
    design_points: list[dict[str, float]],
    response_extractors: dict[str, Extractor],
    tolerances: list[EngineeringTolerance] | None = None,
    run_manager: SimulationRunManager | None = None,
) -> list[SurrogateVerificationRecord]:
    """Compare a surrogate's predictions against real, freshly-run high-fidelity FEA.

    Each design point is both predicted by ``model`` and actually
    simulated through the existing high-fidelity pipeline -- the
    surrogate is never trusted on its own. Intended for held-out design
    points the surrogate's training dataset never saw.

    Args:
        model: The fitted surrogate model to verify.
        base_project: The unmodified base project every design point overrides.
        design_points: The design points to check, keyed by feature name.
        response_extractors: The named result-quantity extractors to
            apply to each high-fidelity run, keyed by response name
            (should match the surrogate's response names for a
            meaningful comparison).
        tolerances: Engineering tolerances used to decide
            :class:`AcceptanceStatus` for each point.
        run_manager: The run manager to execute each verification point
            with. ``None`` constructs a fresh
            :class:`~femtoolkit.runs.manager.SimulationRunManager`.

    Returns:
        One :class:`SurrogateVerificationRecord` per design point, in order.
    """
    manager = run_manager or SimulationRunManager()
    tolerances = tolerances or []
    tolerance_by_response = {tolerance.response_name: tolerance for tolerance in tolerances}
    records: list[SurrogateVerificationRecord] = []

    for index, point in enumerate(design_points):
        design_id = f"surrogate-verification-{index}"
        prediction = model.predict_point(point)

        scenario = Scenario(
            scenario_id=design_id,
            name=f"Verification {design_id}",
            parameter_overrides=dict(point),
        )
        project = apply_scenario(base_project, scenario)
        run = manager.execute(project, scenario_id=design_id)

        if run.status != RunStatus.COMPLETED:
            records.append(
                SurrogateVerificationRecord(
                    design_id=design_id,
                    design_point=dict(point),
                    predicted=prediction.values,
                    actual={},
                    absolute_error={},
                    relative_error={},
                    domain_status=prediction.domain_status.value,
                    acceptance=AcceptanceStatus.FAILED,
                    run_id=run.run_id,
                    error_message=run.error_message,
                )
            )
            continue

        actual: dict[str, float] = {}
        for name, extractor in response_extractors.items():
            value = extractor(run)
            if value is not None:
                actual[name] = value

        absolute_error = {
            name: abs(actual[name] - prediction.values[name])
            for name in actual
            if name in prediction.values
        }
        relative = {
            name: float(
                relative_error(np.array([actual[name]]), np.array([prediction.values[name]]))[0]
            )
            for name in absolute_error
        }

        acceptance = AcceptanceStatus.ACCEPT
        if prediction.status is not PredictionStatus.OK or not absolute_error:
            acceptance = AcceptanceStatus.FAILED
        else:
            tolerance_violated = False
            for name, tolerance in tolerance_by_response.items():
                if name not in absolute_error:
                    continue
                if (
                    tolerance.max_absolute_error is not None
                    and absolute_error[name] > tolerance.max_absolute_error
                ) or (
                    tolerance.max_relative_error is not None
                    and relative[name] > tolerance.max_relative_error
                ):
                    tolerance_violated = True
            if tolerance_violated:
                acceptance = AcceptanceStatus.REJECT
            elif not tolerances or prediction.domain_status.value != "within_training_domain":
                acceptance = AcceptanceStatus.REVIEW

        records.append(
            SurrogateVerificationRecord(
                design_id=design_id,
                design_point=dict(point),
                predicted=prediction.values,
                actual=actual,
                absolute_error=absolute_error,
                relative_error=relative,
                domain_status=prediction.domain_status.value,
                acceptance=acceptance,
                run_id=run.run_id,
            )
        )

    return records


__all__ = ["AcceptanceStatus", "SurrogateVerificationRecord", "verify_against_high_fidelity"]
