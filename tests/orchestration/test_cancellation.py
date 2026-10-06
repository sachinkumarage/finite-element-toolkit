"""Tests for femtoolkit.orchestration.cancellation (Version 34)."""

from __future__ import annotations

from femtoolkit.orchestration.cancellation import CancellationToken


def test_starts_not_cancelled() -> None:
    token = CancellationToken()
    assert not token.is_cancelled


def test_cancel_sets_is_cancelled() -> None:
    token = CancellationToken()
    token.cancel()
    assert token.is_cancelled


def test_cancel_is_idempotent() -> None:
    token = CancellationToken()
    token.cancel()
    token.cancel()
    assert token.is_cancelled
