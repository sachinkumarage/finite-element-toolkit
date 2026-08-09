"""Tests for the CrossSection data model."""

import math

import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.sections import CrossSection


def test_valid_cross_section_creation() -> None:
    section = CrossSection(area=0.01)

    assert section.area == 0.01


@pytest.mark.parametrize("area", [0.0, -0.01, math.nan, math.inf, -math.inf])
def test_invalid_area_raises(area: float) -> None:
    with pytest.raises(ValidationError):
        CrossSection(area=area)
