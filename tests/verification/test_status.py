"""Tests for femtoolkit.verification.status (Version 29)."""

from __future__ import annotations

from femtoolkit.verification.status import VerificationStatus


def test_status_has_exactly_five_members() -> None:
    assert {status.value for status in VerificationStatus} == {
        "pass",
        "fail",
        "warning",
        "not_available",
        "not_run",
    }


def test_only_pass_is_favorable() -> None:
    assert VerificationStatus.PASS.is_favorable
    assert not VerificationStatus.FAIL.is_favorable
    assert not VerificationStatus.WARNING.is_favorable
    assert not VerificationStatus.NOT_AVAILABLE.is_favorable
    assert not VerificationStatus.NOT_RUN.is_favorable
