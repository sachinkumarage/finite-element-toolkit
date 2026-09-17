"""Tests for femtoolkit.mesh.sizing.MeshSizingParameters (Version 25)."""

import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.mesh.sizing import MeshSizingParameters


def test_default_construction() -> None:
    sizing = MeshSizingParameters(target_size=0.1)
    assert sizing.target_size == pytest.approx(0.1)
    assert sizing.minimum_size is None
    assert sizing.maximum_size is None
    assert sizing.regional_sizes == {}


def test_subdivisions_exact_division() -> None:
    sizing = MeshSizingParameters(target_size=0.1)
    assert sizing.subdivisions(width=2.0, height=0.4) == (20, 4)


def test_subdivisions_rounds_and_enforces_minimum_of_one() -> None:
    sizing = MeshSizingParameters(target_size=5.0)
    nx, ny = sizing.subdivisions(width=1.0, height=1.0)
    assert nx >= 1
    assert ny >= 1


def test_subdivisions_rounds_to_nearest() -> None:
    sizing = MeshSizingParameters(target_size=0.3)
    nx, ny = sizing.subdivisions(width=1.0, height=1.0)
    assert nx == round(1.0 / 0.3)
    assert ny == round(1.0 / 0.3)


def test_non_positive_target_size_raises() -> None:
    with pytest.raises(ValidationError):
        MeshSizingParameters(target_size=0.0)
    with pytest.raises(ValidationError):
        MeshSizingParameters(target_size=-1.0)


def test_minimum_size_exceeding_target_raises() -> None:
    with pytest.raises(ValidationError):
        MeshSizingParameters(target_size=0.1, minimum_size=0.5)


def test_maximum_size_below_target_raises() -> None:
    with pytest.raises(ValidationError):
        MeshSizingParameters(target_size=0.5, maximum_size=0.1)


def test_valid_min_max_range_accepted() -> None:
    sizing = MeshSizingParameters(target_size=0.1, minimum_size=0.01, maximum_size=1.0)
    assert sizing.minimum_size == pytest.approx(0.01)
    assert sizing.maximum_size == pytest.approx(1.0)


def test_non_positive_minimum_size_raises() -> None:
    with pytest.raises(ValidationError):
        MeshSizingParameters(target_size=0.1, minimum_size=0.0)


def test_non_positive_maximum_size_raises() -> None:
    with pytest.raises(ValidationError):
        MeshSizingParameters(target_size=0.1, maximum_size=-1.0)


def test_regional_sizes_validated() -> None:
    with pytest.raises(ValidationError):
        MeshSizingParameters(target_size=0.1, regional_sizes={"left": -0.05})


def test_regional_sizes_accepted_but_not_consumed() -> None:
    sizing = MeshSizingParameters(target_size=0.1, regional_sizes={"left": 0.02, "right": 0.2})
    assert sizing.regional_sizes == {"left": 0.02, "right": 0.2}
    # Structured generators do not vary size by region in this version.
    assert sizing.subdivisions(width=1.0, height=1.0) == (10, 10)


def test_subdivisions_invalid_domain_raises() -> None:
    sizing = MeshSizingParameters(target_size=0.1)
    with pytest.raises(ValidationError):
        sizing.subdivisions(width=0.0, height=1.0)
    with pytest.raises(ValidationError):
        sizing.subdivisions(width=1.0, height=-1.0)
