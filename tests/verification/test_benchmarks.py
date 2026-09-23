"""Tests for femtoolkit.verification.benchmarks (Version 29).

Every benchmark's own element formulation is exact for the field it
represents, so each case is expected to pass to floating-point
precision -- these tests assert PASS with a tight tolerance, not a loose
"close enough" bound, deliberately mirroring the tightness already
proven in ``tests/validation/`` for the same underlying element/analysis
combinations.
"""

from __future__ import annotations

from femtoolkit.verification.benchmarks import (
    axial_bar_cases,
    cantilever_beam_cases,
    cst_patch_case,
    hex8_patch_case,
    quad_patch_case,
    standard_benchmark_suite,
    thermal_conduction_cases,
    truss_cases,
)
from femtoolkit.verification.runner import VerificationRunner
from femtoolkit.verification.status import VerificationStatus


def _assert_all_pass(cases) -> None:
    report = VerificationRunner().run_all(cases)
    for result in report.results:
        assert result.status is VerificationStatus.PASS, result.message


def test_axial_bar_cases_pass() -> None:
    _assert_all_pass(axial_bar_cases())


def test_axial_bar_cases_scale_with_parameters() -> None:
    default_case = axial_bar_cases()[0]
    scaled_case = axial_bar_cases(load=2000.0)[0]
    assert scaled_case.reference_value == default_case.reference_value * 2.0


def test_truss_cases_pass() -> None:
    _assert_all_pass(truss_cases())


def test_truss_member_force_is_compression_under_downward_load() -> None:
    force_case = truss_cases()[1]
    assert force_case.reference_value < 0.0


def test_cantilever_beam_cases_pass() -> None:
    _assert_all_pass(cantilever_beam_cases())


def test_cantilever_beam_deflection_is_negative_downward() -> None:
    deflection_case = cantilever_beam_cases()[0]
    assert deflection_case.reference_value < 0.0


def test_thermal_conduction_cases_pass() -> None:
    _assert_all_pass(thermal_conduction_cases())


def test_thermal_conduction_midpoint_is_average_of_end_temperatures() -> None:
    temperature_case = thermal_conduction_cases(hot_temperature=400.0, cold_temperature=300.0)[0]
    assert temperature_case.reference_value == 350.0


def test_quad_patch_case_passes() -> None:
    _assert_all_pass([quad_patch_case()])


def test_cst_patch_case_passes() -> None:
    _assert_all_pass([cst_patch_case()])


def test_hex8_patch_case_passes() -> None:
    _assert_all_pass([hex8_patch_case()])


def test_standard_benchmark_suite_contains_every_category() -> None:
    suite = standard_benchmark_suite()
    names = {case.name for case in suite}
    assert any("Axial bar" in name for name in names)
    assert any("truss" in name for name in names)
    assert any("Cantilever" in name for name in names)
    assert any("conduction" in name for name in names)
    assert any("Q4 patch" in name for name in names)
    assert any("CST patch" in name for name in names)
    assert any("HEX8 patch" in name for name in names)


def test_standard_benchmark_suite_all_pass() -> None:
    report = VerificationRunner().run_all(standard_benchmark_suite())
    assert report.all_passed
    assert len(report.results) == len(standard_benchmark_suite())
