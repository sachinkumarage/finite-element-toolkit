"""Tests for the CrossSection data model."""

import math

import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.sections import CrossSection


def test_valid_cross_section_creation() -> None:
    section = CrossSection(area=0.01)

    assert section.area == 0.01
    assert section.second_moment_of_area is None
    assert section.extreme_fiber_distance is None


@pytest.mark.parametrize("area", [0.0, -0.01, math.nan, math.inf, -math.inf])
def test_invalid_area_raises(area: float) -> None:
    with pytest.raises(ValidationError):
        CrossSection(area=area)


def test_valid_cross_section_with_bending_properties() -> None:
    section = CrossSection(area=0.01, second_moment_of_area=8.333e-6, extreme_fiber_distance=0.05)

    assert section.area == 0.01
    assert section.second_moment_of_area == 8.333e-6
    assert section.extreme_fiber_distance == 0.05


@pytest.mark.parametrize("second_moment_of_area", [0.0, -8.333e-6, math.nan, math.inf, -math.inf])
def test_invalid_second_moment_of_area_raises(second_moment_of_area: float) -> None:
    with pytest.raises(ValidationError):
        CrossSection(area=0.01, second_moment_of_area=second_moment_of_area)


@pytest.mark.parametrize("extreme_fiber_distance", [0.0, -0.05, math.nan, math.inf, -math.inf])
def test_invalid_extreme_fiber_distance_raises(extreme_fiber_distance: float) -> None:
    with pytest.raises(ValidationError):
        CrossSection(
            area=0.01, second_moment_of_area=8.333e-6, extreme_fiber_distance=extreme_fiber_distance
        )
