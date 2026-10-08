"""Tests for femtoolkit.adaptive.trust_region."""

from __future__ import annotations

import pytest

from femtoolkit.adaptive.trust_region import TrustRegion
from femtoolkit.exceptions import InvalidTrustRegionConfigurationError

_BOUNDS = {"x": (0.0, 10.0)}


def test_trust_region_contains_point_within_radius() -> None:
    region = TrustRegion(center={"x": 5.0}, radius=0.2)
    assert region.contains({"x": 5.5}, _BOUNDS)  # 0.05 normalized
    assert not region.contains({"x": 8.0}, _BOUNDS)  # 0.3 normalized


def test_trust_region_expand_grows_radius_up_to_max() -> None:
    region = TrustRegion(center={"x": 5.0}, radius=1.0, max_radius=1.5, expansion_factor=2.0)
    expanded = region.expand()
    assert expanded.radius == pytest.approx(1.5)
    assert expanded.center == region.center


def test_trust_region_contract_shrinks_radius_down_to_min() -> None:
    region = TrustRegion(center={"x": 5.0}, radius=0.05, min_radius=0.04, contraction_factor=0.5)
    contracted = region.contract()
    assert contracted.radius == pytest.approx(0.04)


def test_trust_region_expand_contract_are_normal_within_bounds() -> None:
    region = TrustRegion(center={"x": 5.0}, radius=0.1, min_radius=0.01, max_radius=1.0)
    expanded = region.expand()
    assert expanded.radius == pytest.approx(0.2)
    contracted = region.contract()
    assert contracted.radius == pytest.approx(0.05)


def test_trust_region_recenter_preserves_radius() -> None:
    region = TrustRegion(center={"x": 5.0}, radius=0.3)
    recentered = region.recenter({"x": 7.0})
    assert recentered.center == {"x": 7.0}
    assert recentered.radius == pytest.approx(0.3)


def test_trust_region_rejects_non_positive_radius() -> None:
    with pytest.raises(InvalidTrustRegionConfigurationError):
        TrustRegion(center={"x": 5.0}, radius=0.0)


def test_trust_region_rejects_min_radius_greater_than_max() -> None:
    with pytest.raises(InvalidTrustRegionConfigurationError):
        TrustRegion(center={"x": 5.0}, radius=0.1, min_radius=1.0, max_radius=0.5)


def test_trust_region_rejects_invalid_factors() -> None:
    with pytest.raises(InvalidTrustRegionConfigurationError):
        TrustRegion(center={"x": 5.0}, radius=0.1, expansion_factor=0.5)
    with pytest.raises(InvalidTrustRegionConfigurationError):
        TrustRegion(center={"x": 5.0}, radius=0.1, contraction_factor=1.5)


def test_trust_region_boundary_point_is_contained() -> None:
    region = TrustRegion(center={"x": 0.0}, radius=0.5)
    assert region.contains({"x": 5.0}, _BOUNDS)
