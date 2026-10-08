"""Tests for femtoolkit.adaptive.candidates."""

from __future__ import annotations

import pytest

from femtoolkit.adaptive.candidates import clip_to_bounds, generate_candidate_pool
from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType


def _design_variables() -> list[DesignVariable]:
    return [
        DesignVariable(
            name="thickness", path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
            lower_bound=0.005, upper_bound=0.02,
        ),
        DesignVariable(
            name="count", path="mesh.count", variable_type=DesignVariableType.INTEGER,
            lower_bound=1, upper_bound=10,
        ),
        DesignVariable(
            name="material", path="material.name", variable_type=DesignVariableType.CATEGORICAL,
            categories=["steel", "aluminum", "titanium"],
        ),
    ]


def test_generate_candidate_pool_respects_bounds() -> None:
    variables = _design_variables()
    candidates = generate_candidate_pool(variables, 50, seed=0)
    assert len(candidates) == 50
    for point in candidates:
        assert 0.005 <= point["thickness"] <= 0.02
        assert 1 <= point["count"] <= 10
        assert isinstance(point["count"], int)
        assert point["material"] in ("steel", "aluminum", "titanium")


def test_generate_candidate_pool_reproducible_with_same_seed() -> None:
    variables = _design_variables()
    pool_a = generate_candidate_pool(variables, 10, seed=7)
    pool_b = generate_candidate_pool(variables, 10, seed=7)
    assert pool_a == pool_b


def test_generate_candidate_pool_different_seeds_differ() -> None:
    variables = _design_variables()
    pool_a = generate_candidate_pool(variables, 10, seed=1)
    pool_b = generate_candidate_pool(variables, 10, seed=2)
    assert pool_a != pool_b


def test_generate_candidate_pool_requires_design_variables() -> None:
    with pytest.raises(ValidationError):
        generate_candidate_pool([], 10)


def test_generate_candidate_pool_requires_positive_count() -> None:
    with pytest.raises(ValidationError):
        generate_candidate_pool(_design_variables(), 0)


def test_clip_to_bounds_clamps_out_of_range_values() -> None:
    variables = _design_variables()
    point = {"thickness": 0.1, "count": 50, "material": "steel"}
    clipped = clip_to_bounds(point, variables)
    assert clipped["thickness"] == pytest.approx(0.02)
    assert clipped["count"] == 10
    assert clipped["material"] == "steel"


def test_clip_to_bounds_passes_through_unknown_names() -> None:
    variables = _design_variables()
    clipped = clip_to_bounds({"unrelated": 123.0}, variables)
    assert clipped == {"unrelated": 123.0}
