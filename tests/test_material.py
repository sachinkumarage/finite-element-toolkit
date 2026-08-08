"""Tests for the Material data model."""

import math

import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import Material


def test_valid_material_creation() -> None:
    steel = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)

    assert steel.name == "Steel"
    assert steel.density == 7850.0
    assert steel.youngs_modulus == 200e9
    assert steel.poissons_ratio == 0.3


@pytest.mark.parametrize("name", ["", "   "])
def test_invalid_name_raises(name: str) -> None:
    with pytest.raises(ValidationError):
        Material(name=name, density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)


@pytest.mark.parametrize("density", [0.0, -1.0, math.nan, math.inf])
def test_invalid_density_raises(density: float) -> None:
    with pytest.raises(ValidationError):
        Material(name="Steel", density=density, youngs_modulus=200e9, poissons_ratio=0.3)


@pytest.mark.parametrize("youngs_modulus", [0.0, -200e9, math.nan, math.inf])
def test_invalid_youngs_modulus_raises(youngs_modulus: float) -> None:
    with pytest.raises(ValidationError):
        Material(name="Steel", density=7850.0, youngs_modulus=youngs_modulus, poissons_ratio=0.3)


@pytest.mark.parametrize("poissons_ratio", [-1.0, 0.5, -1.5, 0.6, math.nan])
def test_invalid_poissons_ratio_raises(poissons_ratio: float) -> None:
    with pytest.raises(ValidationError):
        Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=poissons_ratio)
