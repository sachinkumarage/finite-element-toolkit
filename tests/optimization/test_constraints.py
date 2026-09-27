"""Tests for femtoolkit.optimization.constraints."""

import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.constraints import (
    Constraint,
    ConstraintRelation,
    validate_unique_constraint_names,
)


def _constraint(relation: ConstraintRelation, limit: float, tolerance: float = 1e-9) -> Constraint:
    return Constraint(
        name="c", evaluate=lambda ctx: None, relation=relation, limit=limit, tolerance=tolerance
    )


def test_constraint_requires_nonempty_name() -> None:
    with pytest.raises(ValidationError):
        Constraint(
            name="", evaluate=lambda ctx: None, relation=ConstraintRelation.LESS_EQUAL, limit=1.0
        )


def test_constraint_rejects_negative_tolerance() -> None:
    with pytest.raises(ValidationError):
        _constraint(ConstraintRelation.LESS_EQUAL, 1.0, tolerance=-1.0)


def test_less_equal_violation_known_values() -> None:
    constraint = _constraint(ConstraintRelation.LESS_EQUAL, 250e6)
    assert constraint.violation(200e6) == 0.0
    assert constraint.violation(250e6) == 0.0
    assert constraint.violation(260e6) == pytest.approx(10e6, rel=1e-6)
    assert constraint.is_satisfied(240e6)
    assert not constraint.is_satisfied(260e6)


def test_greater_equal_violation_known_values() -> None:
    constraint = _constraint(ConstraintRelation.GREATER_EQUAL, 100.0)
    assert constraint.violation(150.0) == 0.0
    assert constraint.violation(80.0) == pytest.approx(20.0, rel=1e-6)
    assert constraint.is_satisfied(120.0)
    assert not constraint.is_satisfied(80.0)


def test_equal_violation_known_values() -> None:
    constraint = _constraint(ConstraintRelation.EQUAL, 5.0, tolerance=0.1)
    assert constraint.violation(5.05) == 0.0
    assert constraint.violation(5.5) == pytest.approx(0.4, rel=1e-6)
    assert constraint.is_satisfied(4.95)
    assert not constraint.is_satisfied(5.5)


def test_validate_unique_constraint_names_rejects_duplicates() -> None:
    a = _constraint(ConstraintRelation.LESS_EQUAL, 1.0)
    dup = Constraint(
        name="c", evaluate=lambda ctx: None, relation=ConstraintRelation.GREATER_EQUAL, limit=2.0
    )
    with pytest.raises(ValidationError):
        validate_unique_constraint_names([a, dup])


def test_validate_unique_constraint_names_passes_for_unique_names() -> None:
    a = _constraint(ConstraintRelation.LESS_EQUAL, 1.0)
    b = Constraint(
        name="d", evaluate=lambda ctx: None, relation=ConstraintRelation.GREATER_EQUAL, limit=2.0
    )
    validate_unique_constraint_names([a, b])
