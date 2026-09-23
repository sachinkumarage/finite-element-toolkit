"""Tests for femtoolkit.verification.tolerance (Version 29)."""

from __future__ import annotations

import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.verification.tolerance import Tolerance


def test_tolerance_defaults_are_positive() -> None:
    tolerance = Tolerance()
    assert tolerance.absolute > 0.0
    assert tolerance.relative > 0.0


def test_tolerance_rejects_negative_absolute() -> None:
    with pytest.raises(ValidationError):
        Tolerance(absolute=-1e-9, relative=1e-6)


def test_tolerance_rejects_negative_relative() -> None:
    with pytest.raises(ValidationError):
        Tolerance(absolute=1e-9, relative=-1e-6)


def test_tolerance_rejects_both_zero() -> None:
    with pytest.raises(ValidationError):
        Tolerance(absolute=0.0, relative=0.0)


def test_tolerance_allows_one_component_zero() -> None:
    Tolerance(absolute=0.0, relative=1e-6)
    Tolerance(absolute=1e-9, relative=0.0)


def test_allowed_error_combines_absolute_and_relative() -> None:
    tolerance = Tolerance(absolute=1e-3, relative=1e-2)
    assert tolerance.allowed_error(100.0) == pytest.approx(1e-3 + 1e-2 * 100.0)


def test_is_satisfied_true_within_tolerance() -> None:
    tolerance = Tolerance(absolute=1e-6, relative=1e-3)
    assert tolerance.is_satisfied(numerical=1.0005, reference=1.0)


def test_is_satisfied_false_outside_tolerance() -> None:
    tolerance = Tolerance(absolute=1e-6, relative=1e-6)
    assert not tolerance.is_satisfied(numerical=1.1, reference=1.0)


def test_is_satisfied_handles_near_zero_reference_via_absolute_term() -> None:
    tolerance = Tolerance(absolute=1e-6, relative=1e-3)
    # Relative term contributes ~0 here; absolute term must still allow this.
    assert tolerance.is_satisfied(numerical=1e-9, reference=0.0)


def test_is_satisfied_never_uses_exact_floating_point_equality() -> None:
    tolerance = Tolerance(absolute=0.0, relative=1e-12)
    assert tolerance.is_satisfied(numerical=0.1 + 0.2, reference=0.3)
