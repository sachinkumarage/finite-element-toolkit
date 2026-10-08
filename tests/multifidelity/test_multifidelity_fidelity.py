"""Tests for femtoolkit.multifidelity.fidelity."""

from __future__ import annotations

import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.multifidelity.fidelity import (
    HIGH_FIDELITY,
    LOW_FIDELITY,
    AnalyticalFidelityModel,
    FidelityLevel,
    summarize_costs,
)


def test_low_and_high_fidelity_levels_are_distinct() -> None:
    assert LOW_FIDELITY.name == "LOW"
    assert HIGH_FIDELITY.name == "HIGH"
    assert LOW_FIDELITY.estimated_cost < HIGH_FIDELITY.estimated_cost


def test_fidelity_level_metadata() -> None:
    level = FidelityLevel(name="MEDIUM", description="A mid-cost model.", estimated_cost=10.0)
    assert level.name == "MEDIUM"
    assert level.description == "A mid-cost model."
    assert level.estimated_cost == 10.0


def test_fidelity_level_rejects_empty_name() -> None:
    with pytest.raises(ValidationError):
        FidelityLevel(name="", description="x", estimated_cost=1.0)


def test_fidelity_level_rejects_negative_cost() -> None:
    with pytest.raises(ValidationError):
        FidelityLevel(name="X", description="x", estimated_cost=-1.0)


def test_analytical_fidelity_model_evaluates_and_reports_cost() -> None:
    model = AnalyticalFidelityModel(
        name="square", evaluate_fn=lambda point: {"y": point["x"] ** 2}, level=LOW_FIDELITY
    )
    assert model.level is LOW_FIDELITY
    assert model.name == "square"
    assert model.estimated_cost == LOW_FIDELITY.estimated_cost
    assert model.evaluate({"x": 3.0}) == {"y": 9.0}


def test_analytical_fidelity_model_estimated_cost_override() -> None:
    model = AnalyticalFidelityModel(
        name="square", evaluate_fn=lambda point: {"y": point["x"] ** 2},
        level=LOW_FIDELITY, estimated_cost=5.0,
    )
    assert model.estimated_cost == 5.0


def test_summarize_costs_reports_every_model() -> None:
    low = AnalyticalFidelityModel(name="low", evaluate_fn=lambda p: {"y": 0.0}, level=LOW_FIDELITY)
    high = AnalyticalFidelityModel(
        name="high", evaluate_fn=lambda p: {"y": 0.0}, level=HIGH_FIDELITY
    )
    summary = summarize_costs([low, high])
    assert [entry["name"] for entry in summary] == ["low", "high"]
    assert summary[0]["estimated_cost"] < summary[1]["estimated_cost"]
