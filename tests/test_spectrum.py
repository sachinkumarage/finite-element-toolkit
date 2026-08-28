"""Tests for ResponseSpectrum and modal_spectral_response."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.modal import modal_analysis
from femtoolkit.analysis.spectrum import ResponseSpectrum, modal_spectral_response
from femtoolkit.exceptions import ValidationError


@pytest.fixture
def spectrum() -> ResponseSpectrum:
    return ResponseSpectrum(
        periods=[0.0, 0.2, 0.5, 1.0, 2.0],
        accelerations=[0.4, 1.0, 0.8, 0.4, 0.2],
    )


# --- ResponseSpectrum construction ---


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValidationError):
        ResponseSpectrum(periods=[0.0, 1.0], accelerations=[0.4, 0.5, 0.6])


def test_rejects_fewer_than_two_points() -> None:
    with pytest.raises(ValidationError):
        ResponseSpectrum(periods=[0.0], accelerations=[0.4])


def test_rejects_unsorted_periods() -> None:
    with pytest.raises(ValidationError):
        ResponseSpectrum(periods=[0.0, 1.0, 0.5], accelerations=[0.4, 0.5, 0.6])


def test_rejects_duplicate_periods() -> None:
    with pytest.raises(ValidationError):
        ResponseSpectrum(periods=[0.0, 0.5, 0.5, 1.0], accelerations=[0.4, 0.5, 0.6, 0.7])


def test_rejects_negative_period() -> None:
    with pytest.raises(ValidationError):
        ResponseSpectrum(periods=[-0.1, 0.5, 1.0], accelerations=[0.4, 0.5, 0.6])


def test_rejects_negative_acceleration() -> None:
    with pytest.raises(ValidationError):
        ResponseSpectrum(periods=[0.0, 0.5, 1.0], accelerations=[0.4, -0.5, 0.6])


def test_rejects_non_finite_values() -> None:
    with pytest.raises(ValidationError):
        ResponseSpectrum(periods=[0.0, 0.5, float("nan")], accelerations=[0.4, 0.5, 0.6])


def test_accepts_valid_data(spectrum: ResponseSpectrum) -> None:
    assert spectrum.periods.shape == (5,)
    assert spectrum.accelerations.shape == (5,)


# --- evaluate() interpolation ---


def test_evaluate_at_exact_point(spectrum: ResponseSpectrum) -> None:
    assert_allclose(spectrum.evaluate(0.5), 0.8)


def test_evaluate_linear_interpolation(spectrum: ResponseSpectrum) -> None:
    # Between (0.2, 1.0) and (0.5, 0.8): linear interpolation at 0.35.
    expected = 1.0 + (0.35 - 0.2) / (0.5 - 0.2) * (0.8 - 1.0)
    assert_allclose(spectrum.evaluate(0.35), expected)


def test_evaluate_at_lower_boundary(spectrum: ResponseSpectrum) -> None:
    assert_allclose(spectrum.evaluate(0.0), 0.4)


def test_evaluate_at_upper_boundary(spectrum: ResponseSpectrum) -> None:
    assert_allclose(spectrum.evaluate(2.0), 0.2)


def test_evaluate_rejects_negative_period(spectrum: ResponseSpectrum) -> None:
    with pytest.raises(ValidationError):
        spectrum.evaluate(-0.1)


def test_evaluate_rejects_period_above_range(spectrum: ResponseSpectrum) -> None:
    with pytest.raises(ValidationError):
        spectrum.evaluate(2.5)


def test_evaluate_rejects_non_finite_period(spectrum: ResponseSpectrum) -> None:
    with pytest.raises(ValidationError):
        spectrum.evaluate(float("nan"))


def test_evaluate_returns_finite_values_across_range(spectrum: ResponseSpectrum) -> None:
    for period in np.linspace(0.0, 2.0, 50):
        assert np.isfinite(spectrum.evaluate(float(period)))


# --- modal_spectral_response ---


@pytest.fixture
def wide_spectrum() -> ResponseSpectrum:
    return ResponseSpectrum(
        periods=[0.0, 0.5, 1.0, 5.0, 20.0], accelerations=[0.4, 1.0, 0.8, 0.4, 0.2]
    )


def test_modal_spectral_response_requires_participation_factors(wide_spectrum) -> None:
    k = np.array([[2.0, -1.0], [-1.0, 1.0]])
    m = np.eye(2)
    modal_result = modal_analysis(k, m)  # no direction given

    with pytest.raises(ValidationError):
        modal_spectral_response(modal_result, wide_spectrum)


def test_modal_spectral_response_matches_manual_formula(wide_spectrum) -> None:
    k = np.array([[2.0, -1.0], [-1.0, 1.0]])
    m = np.eye(2)
    r = np.array([1.0, 0.0])
    modal_result = modal_analysis(k, m, direction=r)

    response = modal_spectral_response(modal_result, wide_spectrum)

    for i in range(2):
        gamma = modal_result.participation_factors[i]
        omega = modal_result.angular_frequencies[i]
        sa = wide_spectrum.evaluate(float(modal_result.periods[i]))
        expected_q = gamma * sa / omega**2
        assert_allclose(response.modal_displacements[i], expected_q, rtol=1e-9)

        expected_force = modal_result.effective_modal_mass[i] * sa
        assert_allclose(response.equivalent_static_forces[i], expected_force, rtol=1e-9)


def test_modal_spectral_response_period_out_of_range_raises() -> None:
    narrow_spectrum = ResponseSpectrum(periods=[0.0, 0.01], accelerations=[0.4, 0.5])
    k = np.array([[2.0, -1.0], [-1.0, 1.0]])
    m = np.eye(2)
    modal_result = modal_analysis(k, m, direction=np.array([1.0, 0.0]))

    with pytest.raises(ValidationError):
        modal_spectral_response(modal_result, narrow_spectrum)
