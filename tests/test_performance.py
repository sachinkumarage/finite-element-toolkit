"""Tests for femtoolkit.performance (Version 27)."""

from __future__ import annotations

import time

import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.performance import (
    ASSEMBLY,
    ELEMENT,
    MESH,
    BenchmarkResult,
    PerformanceReport,
    Profiler,
    amdahl_speedup,
    format_report,
    parallel_efficiency,
    run_benchmark,
    speedup,
    time_callable,
)

# --- Profiler / PerformanceReport ------------------------------------------


def test_profiler_report_has_none_for_unmeasured_stages() -> None:
    report = Profiler().report()
    assert report.mesh_time is None
    assert report.element_time is None
    assert report.assembly_time is None
    assert report.boundary_condition_time is None
    assert report.solve_time is None
    assert report.post_processing_time is None
    assert report.total_time == 0.0


def test_profiler_measures_named_stages() -> None:
    profiler = Profiler()
    with profiler.stage(MESH):
        time.sleep(0.001)
    with profiler.stage(ELEMENT):
        time.sleep(0.001)

    report = profiler.report()
    assert report.mesh_time is not None and report.mesh_time > 0.0
    assert report.element_time is not None and report.element_time > 0.0
    assert report.assembly_time is None


def test_profiler_accumulates_repeated_stage_calls() -> None:
    profiler = Profiler()
    with profiler.stage(ASSEMBLY):
        time.sleep(0.001)
    with profiler.stage(ASSEMBLY):
        time.sleep(0.001)

    report = profiler.report()
    assert report.assembly_time is not None
    assert report.assembly_time >= 0.002


def test_profiler_context_manager_total_time_is_positive() -> None:
    profiler = Profiler()
    with profiler, profiler.stage(ELEMENT):
        time.sleep(0.001)

    report = profiler.report()
    assert report.total_time > 0.0
    assert report.total_time >= (report.element_time or 0.0)


def test_profiler_report_without_context_manager_sums_stages() -> None:
    profiler = Profiler()
    with profiler.stage(ELEMENT):
        time.sleep(0.001)
    with profiler.stage(ASSEMBLY):
        time.sleep(0.001)

    report = profiler.report()
    expected = (report.element_time or 0.0) + (report.assembly_time or 0.0)
    assert report.total_time == pytest.approx(expected)


def test_profiler_report_carries_metadata() -> None:
    report = Profiler().report(
        model_name="Test Model", node_count=10, element_count=5, execution_mode="serial"
    )
    assert report.model_name == "Test Model"
    assert report.node_count == 10
    assert report.element_count == 5
    assert report.execution_mode == "serial"


def test_profiler_stage_time_returns_none_when_unmeasured() -> None:
    assert Profiler().stage_time(ELEMENT) is None


# --- format_report -----------------------------------------------------------


def test_format_report_omits_unmeasured_stages() -> None:
    profiler = Profiler()
    with profiler.stage(ELEMENT):
        pass
    report = profiler.report()
    text = format_report(report)

    assert "Element calculations" in text
    assert "Assembly" not in text
    assert "Solve" not in text


def test_format_report_includes_metadata() -> None:
    report = PerformanceReport(
        total_time=1.0, model_name="Cantilever", node_count=4, element_count=2
    )
    text = format_report(report)
    assert "Cantilever" in text
    assert "Nodes: 4" in text
    assert "Elements: 2" in text


# --- benchmark ---------------------------------------------------------------


def test_run_benchmark_wraps_report() -> None:
    def build_report() -> PerformanceReport:
        with Profiler() as profiler, profiler.stage(ELEMENT):
            time.sleep(0.001)
        return profiler.report()

    result = run_benchmark("case-1", build_report)
    assert isinstance(result, BenchmarkResult)
    assert result.name == "case-1"
    assert result.report.total_time > 0.0


def test_time_callable_measures_positive_duration() -> None:
    elapsed = time_callable(lambda: time.sleep(0.001))
    assert elapsed > 0.0


def test_speedup_basic() -> None:
    assert speedup(4.0, 2.0) == pytest.approx(2.0)


def test_speedup_rejects_non_positive_timings() -> None:
    with pytest.raises(ValidationError):
        speedup(0.0, 1.0)
    with pytest.raises(ValidationError):
        speedup(1.0, -1.0)


def test_parallel_efficiency_basic() -> None:
    assert parallel_efficiency(4.0, 4) == pytest.approx(1.0)


def test_parallel_efficiency_rejects_non_positive_workers() -> None:
    with pytest.raises(ValidationError):
        parallel_efficiency(2.0, 0)


def test_amdahl_speedup_fully_serial_gives_no_speedup() -> None:
    assert amdahl_speedup(0.0, 8) == pytest.approx(1.0)


def test_amdahl_speedup_fully_parallel_gives_linear_speedup() -> None:
    assert amdahl_speedup(1.0, 8) == pytest.approx(8.0)


def test_amdahl_speedup_rejects_invalid_fraction() -> None:
    with pytest.raises(ValidationError):
        amdahl_speedup(1.5, 4)
    with pytest.raises(ValidationError):
        amdahl_speedup(-0.1, 4)


def test_amdahl_speedup_rejects_non_positive_workers() -> None:
    with pytest.raises(ValidationError):
        amdahl_speedup(0.5, 0)


def test_amdahl_speedup_bounds_measured_speedup() -> None:
    # A measured speedup should never exceed Amdahl's theoretical ceiling
    # for the same parallelizable fraction and worker count.
    theoretical = amdahl_speedup(0.9, 4)
    measured = speedup(serial_time=1.0, parallel_time=0.4)
    assert measured <= theoretical
