"""Validation: multilinear isotropic hardening curve evaluation (Section 8/13).

Independently re-derives the expected stress/tangent at points inside
each segment and at segment boundaries, rather than re-using the
material's own internal formulas, so this is a genuine check of the
implementation rather than a tautological one (matching this project's
established validation style, e.g.
``tests/validation/test_single_triangle.py``).
"""

import pytest

from femtoolkit.materials import MultilinearIsotropicHardeningMaterial1D

STRAIN_POINTS = (0.0, 0.001, 0.005, 0.02)
STRESS_POINTS = (0.0, 200e6, 250e6, 300e6)


def _independent_piecewise_evaluation(strain: float) -> tuple[float, float]:
    """Hand-rolled piecewise-linear evaluation, independent of the material class.

    Uses the same "forward" convention as the material at an exact
    segment-boundary strain: the segment *starting* at that point applies
    (matching :meth:`MultilinearIsotropicHardeningMaterial1D._segment_index`'s
    ``side="right"`` lookup), not the one ending there.
    """
    magnitude = abs(strain)
    sign = 1.0 if strain >= 0.0 else -1.0

    for index in range(len(STRAIN_POINTS) - 1):
        left_strain, right_strain = STRAIN_POINTS[index], STRAIN_POINTS[index + 1]
        left_stress, right_stress = STRESS_POINTS[index], STRESS_POINTS[index + 1]
        slope = (right_stress - left_stress) / (right_strain - left_strain)
        if magnitude < right_strain or index == len(STRAIN_POINTS) - 2:
            stress_magnitude = left_stress + slope * (magnitude - left_strain)
            return sign * stress_magnitude, slope

    raise AssertionError("unreachable")


@pytest.fixture
def curve() -> MultilinearIsotropicHardeningMaterial1D:
    return MultilinearIsotropicHardeningMaterial1D(
        strain_points=STRAIN_POINTS, stress_points=STRESS_POINTS
    )


@pytest.mark.parametrize(
    "strain", [0.0, 0.0003, 0.001, 0.0025, 0.005, 0.012, 0.02, 0.023, -0.0007, -0.003, -0.015]
)
def test_interpolation_matches_independent_evaluation(
    curve: MultilinearIsotropicHardeningMaterial1D, strain: float
) -> None:
    expected_stress, expected_tangent = _independent_piecewise_evaluation(strain)

    state = curve.trial_state(strain, curve.initial_state())

    assert state.stress == pytest.approx(expected_stress, rel=1e-9)
    assert curve.tangent_modulus(state) == pytest.approx(expected_tangent, rel=1e-9)


def test_segment_transitions_are_continuous(curve: MultilinearIsotropicHardeningMaterial1D) -> None:
    """Stress must be continuous across every segment boundary (no jumps),
    even though the tangent (slope) is discontinuous there.
    """
    for boundary_strain in STRAIN_POINTS[1:-1]:
        just_below = curve.trial_state(boundary_strain - 1e-9, curve.initial_state())
        just_above = curve.trial_state(boundary_strain + 1e-9, curve.initial_state())
        assert just_below.stress == pytest.approx(just_above.stress, abs=1000.0)


def test_final_hardening_region_uses_last_segment_slope(
    curve: MultilinearIsotropicHardeningMaterial1D,
) -> None:
    rise = STRESS_POINTS[-1] - STRESS_POINTS[-2]
    run = STRAIN_POINTS[-1] - STRAIN_POINTS[-2]
    expected_slope = rise / run

    state = curve.trial_state(0.015, curve.initial_state())

    assert curve.tangent_modulus(state) == pytest.approx(expected_slope, rel=1e-9)
    assert state.yielded is True


def test_elastic_region_uses_first_segment_slope(
    curve: MultilinearIsotropicHardeningMaterial1D,
) -> None:
    state = curve.trial_state(0.0005, curve.initial_state())

    assert curve.tangent_modulus(state) == pytest.approx(curve.youngs_modulus, rel=1e-9)
    assert state.yielded is False
