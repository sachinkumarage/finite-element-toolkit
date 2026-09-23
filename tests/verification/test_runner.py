"""Tests for femtoolkit.verification.cases and .runner (Version 29)."""

from __future__ import annotations

import numpy as np
import pytest

from femtoolkit.verification.cases import VerificationCase
from femtoolkit.verification.runner import VerificationRunner
from femtoolkit.verification.status import VerificationStatus
from femtoolkit.verification.tolerance import Tolerance


def _scalar_case(numerical: float, reference: float = 1.0, tolerance=None) -> VerificationCase:
    return VerificationCase(
        name="Scalar case",
        description="A synthetic scalar case for runner tests.",
        analysis_type="unit_test",
        quantity="Test quantity",
        reference_value=reference,
        tolerance=tolerance or Tolerance(absolute=1e-9, relative=1e-6),
        run=lambda: numerical,
    )


def test_runner_passes_matching_scalar() -> None:
    result = VerificationRunner().run(_scalar_case(1.0000001, tolerance=Tolerance(relative=1e-3)))
    assert result.status is VerificationStatus.PASS
    assert result.absolute_error == pytest.approx(0.0000001)


def test_runner_fails_mismatched_scalar() -> None:
    result = VerificationRunner().run(_scalar_case(2.0, tolerance=Tolerance(relative=1e-6)))
    assert result.status is VerificationStatus.FAIL


def test_runner_reports_not_available_when_run_raises() -> None:
    case = VerificationCase(
        name="Broken case",
        description="A case whose run() always raises.",
        analysis_type="unit_test",
        quantity="Test quantity",
        reference_value=1.0,
        tolerance=Tolerance(),
        run=lambda: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )
    result = VerificationRunner().run(case)
    assert result.status is VerificationStatus.NOT_AVAILABLE
    assert result.numerical_value is None
    assert "synthetic failure" in result.message


def test_runner_handles_vector_quantities() -> None:
    reference = np.array([1.0, 2.0, 3.0])
    case = VerificationCase(
        name="Vector case",
        description="A synthetic vector case.",
        analysis_type="unit_test",
        quantity="Vector quantity",
        reference_value=reference,
        tolerance=Tolerance(absolute=1e-6, relative=1e-3),
        run=lambda: reference + 1e-9,
    )
    result = VerificationRunner().run(case)
    assert result.status is VerificationStatus.PASS
    assert result.absolute_error is not None


def test_runner_preserves_case_metadata() -> None:
    case = _scalar_case(1.0)
    result = VerificationRunner().run(case)
    assert result.case_name == case.name
    assert result.description == case.description
    assert result.analysis_type == case.analysis_type
    assert result.quantity == case.quantity


def test_run_all_collects_every_result_in_order() -> None:
    cases = [_scalar_case(1.0, tolerance=Tolerance(relative=1e-3)), _scalar_case(5.0)]
    report = VerificationRunner().run_all(cases)
    assert len(report.results) == 2
    assert report.results[0].status is VerificationStatus.PASS
    assert report.results[1].status is VerificationStatus.FAIL


def test_report_counts_and_all_passed() -> None:
    report = VerificationRunner().run_all(
        [_scalar_case(1.0, tolerance=Tolerance(relative=1e-3)), _scalar_case(5.0)]
    )
    assert report.passed == 1
    assert report.failed == 1
    assert report.warnings == 0
    assert report.not_available == 0
    assert not report.all_passed


def test_report_all_passed_true_for_all_pass() -> None:
    report = VerificationRunner().run_all([_scalar_case(1.0, tolerance=Tolerance(relative=1e-3))])
    assert report.all_passed


def test_report_all_passed_vacuously_true_when_empty() -> None:
    report = VerificationRunner().run_all([])
    assert report.all_passed
    assert report.results == []
