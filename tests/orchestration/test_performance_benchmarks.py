"""Performance benchmarks for the orchestration layer (Version 34).

Kept separate from the correctness tests in this package, per this
version's own "do not make tests dependent on exact execution time"
requirement -- these benchmarks exercise
:mod:`femtoolkit.performance.benchmark`'s speedup/efficiency formulas
against a real parallel batch and assert only loose, structural
properties (a value was computed, its sign/range is sane), never a
specific numeric speedup threshold, which would be flaky across the
range of machines this toolkit runs on.
"""

from __future__ import annotations

import time

from femtoolkit.orchestration.config import OrchestrationConfig
from femtoolkit.orchestration.manager import build_execution_manager


def _busy_work(seconds: float) -> float:
    # A CPU-bound (not I/O-bound) workload, to actually benefit from a
    # process pool bypassing the GIL -- a plain time.sleep would
    # "parallelize" even on a single core and prove nothing.
    deadline = time.perf_counter() + seconds
    total = 0.0
    while time.perf_counter() < deadline:
        total += 1.0
    return total


def test_parallel_batch_produces_a_speedup_estimate() -> None:
    manager = build_execution_manager(OrchestrationConfig(execution_mode="parallel", max_workers=2))
    _outcomes, summary = manager.run_batch(_busy_work, [0.05] * 8)
    assert summary.parallel_speedup is not None
    assert summary.parallel_efficiency is not None
    assert summary.parallel_speedup > 0.0
    assert summary.parallel_efficiency > 0.0


def test_serial_batch_reports_no_speedup_estimate() -> None:
    manager = build_execution_manager(OrchestrationConfig(execution_mode="serial"))
    _outcomes, summary = manager.run_batch(_busy_work, [0.01] * 4)
    assert summary.parallel_speedup is None
    assert summary.parallel_efficiency is None


def test_benchmark_more_workers_is_not_assumed_better() -> None:
    # Spec requirement: never assume more workers is always faster.
    # This benchmark only records both summaries' timings for a human
    # to compare -- it makes no pass/fail assertion comparing them,
    # since a 2-worker vs 4-worker comparison's outcome is genuinely
    # machine-dependent (and this toolkit's own documentation says so).
    for workers in (2, 4):
        manager = build_execution_manager(
            OrchestrationConfig(execution_mode="parallel", max_workers=workers)
        )
        _outcomes, summary = manager.run_batch(_busy_work, [0.02] * 6)
        assert summary.total_elapsed_seconds >= 0.0
        assert summary.worker_count == workers
