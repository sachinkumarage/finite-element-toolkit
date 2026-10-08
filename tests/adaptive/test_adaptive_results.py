"""Tests for femtoolkit.adaptive.results."""

from __future__ import annotations

from femtoolkit.adaptive.refinement import SurrogateAcceptanceState
from femtoolkit.adaptive.results import AdaptiveStudyResult


def test_adaptive_study_result_to_dict_is_json_serializable() -> None:
    import json

    result = AdaptiveStudyResult(
        initial_sample_count=6,
        total_high_fidelity_evaluations=9,
        surrogate_evaluations=120,
        iteration_count=3,
        best_verified_design={"mesh.thickness": 0.018},
        best_verified_objective=0.0012,
        best_surrogate_predicted_design={"thickness": 0.018},
        best_surrogate_predicted_objective=0.0011,
        prediction_error={"maximum_displacement": 0.05},
        convergence_history=[0.002, 0.0015, 0.0012],
        status=SurrogateAcceptanceState.VERIFIED,
        stopping_reason="Reached the maximum number of refinement iterations.",
    )
    payload = result.to_dict()
    json.dumps(payload)  # raises if anything is not JSON-serializable
    assert payload["status"] == "verified"
    assert payload["iteration_count"] == 3
    assert payload["convergence_history"] == [0.002, 0.0015, 0.0012]


def test_adaptive_study_result_defaults_to_surrogate_only_status() -> None:
    result = AdaptiveStudyResult(
        initial_sample_count=4,
        total_high_fidelity_evaluations=4,
        surrogate_evaluations=0,
        iteration_count=0,
        best_verified_design=None,
        best_verified_objective=None,
        best_surrogate_predicted_design=None,
        best_surrogate_predicted_objective=None,
    )
    assert result.status is SurrogateAcceptanceState.SURROGATE_ONLY
    assert result.iteration_history == []
