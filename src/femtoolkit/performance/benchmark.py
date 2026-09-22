"""A reusable benchmarking framework for FEA workloads (Version 27).

:func:`run_benchmark` times a zero-argument callable that produces a
:class:`~femtoolkit.performance.profiler.PerformanceReport` (typically
built with a :class:`~femtoolkit.performance.profiler.Profiler`) and
attaches a human-readable label, so several runs -- dense vs. sparse,
serial vs. parallel, small vs. large mesh -- can be collected into one
list and compared. :func:`speedup` and :func:`parallel_efficiency`
implement the standard parallel-computing metrics (spec section 21).

This module is deliberately free of any specific mesh size, element type,
or analysis workflow -- those live in ``examples/performance/`` (spec
section 30), which import this module rather than the other way around,
keeping the reusable framework independent of any one benchmark scenario.
Benchmark execution is not part of the normal test suite (spec section
6): ``tests/test_benchmark.py`` exercises this module's *machinery* with
trivial, fast callables, never a full FEA solve.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from femtoolkit.exceptions import ValidationError
from femtoolkit.performance.profiler import PerformanceReport


@dataclass(frozen=True)
class BenchmarkResult:
    """One named benchmark run's measured performance.

    Attributes:
        name: A human-readable label for this run (e.g. ``"Dense,
            902 DOF"``, ``"Sparse + CG, 4 workers"``).
        report: The :class:`~femtoolkit.performance.profiler.PerformanceReport`
            this run produced.
    """

    name: str
    report: PerformanceReport


def run_benchmark(name: str, func: Callable[[], PerformanceReport]) -> BenchmarkResult:
    """Run one benchmark case and label its resulting report.

    Args:
        name: A human-readable label for this run.
        func: A zero-argument callable that performs the work to be
            benchmarked and returns a
            :class:`~femtoolkit.performance.profiler.PerformanceReport`
            describing it (see
            :class:`~femtoolkit.performance.profiler.Profiler`).

    Returns:
        A :class:`BenchmarkResult` pairing ``name`` with ``func``'s report.
    """
    return BenchmarkResult(name=name, report=func())


def time_callable(func: Callable[[], object]) -> float:
    """Time one invocation of a zero-argument callable, in seconds.

    A minimal alternative to :class:`~femtoolkit.performance.profiler.Profiler`
    for the common case of just wanting one overall elapsed time (e.g.
    timing an entire ``analysis.solve()`` call for a serial-vs-parallel
    comparison) without breaking it into named stages.

    Args:
        func: The callable to time. Its return value is discarded.

    Returns:
        Elapsed wall-clock time, in seconds, via :func:`time.perf_counter`.
    """
    start = time.perf_counter()
    func()
    return time.perf_counter() - start


def speedup(serial_time: float, parallel_time: float) -> float:
    """Compute parallel speedup ``S_p = T_1 / T_p``.

    Args:
        serial_time: ``T_1``, the serial (one-worker) execution time, in
            seconds.
        parallel_time: ``T_p``, the parallel execution time, in seconds.

    Returns:
        The speedup ratio. Greater than 1 means parallel execution was
        faster; less than 1 means it was slower (common for small
        workloads, where process-startup and serialization overhead
        outweigh the work itself -- see Amdahl's Law in
        :mod:`docs/performance.md`).

    Raises:
        ValidationError: If ``serial_time`` or ``parallel_time`` is not
            positive.
    """
    if serial_time <= 0.0 or parallel_time <= 0.0:
        raise ValidationError(
            f"speedup() requires positive timings, got serial_time={serial_time}, "
            f"parallel_time={parallel_time}."
        )
    return serial_time / parallel_time


def parallel_efficiency(speedup_value: float, workers: int) -> float:
    """Compute parallel efficiency ``E_p = S_p / p``.

    Args:
        speedup_value: ``S_p``, the speedup (see :func:`speedup`).
        workers: ``p``, the number of workers used to achieve it.

    Returns:
        The efficiency ratio, typically in ``(0, 1]`` for a real workload
        (1.0 would be ideal linear scaling; real efficiency is lower due
        to serial portions, synchronization, and communication overhead
        -- see Amdahl's Law in :mod:`docs/performance.md`).

    Raises:
        ValidationError: If ``workers`` is not positive.
    """
    if workers <= 0:
        raise ValidationError(f"parallel_efficiency() requires workers > 0, got {workers}.")
    return speedup_value / workers


def amdahl_speedup(parallelizable_fraction: float, workers: int) -> float:
    """Predict speedup from Amdahl's Law: ``S(N) = 1 / ((1 - P) + P / N)``.

    Args:
        parallelizable_fraction: ``P``, the fraction of total work that
            can run in parallel (between 0 and 1 inclusive). The
            remaining ``1 - P`` is an inherently serial portion (e.g.
            boundary-condition application or the solve itself in this
            toolkit's Version 27 scope) that caps the achievable speedup
            no matter how many workers are added.
        workers: ``N``, the number of workers.

    Returns:
        The theoretical maximum speedup -- an upper bound a *measured*
        :func:`speedup` should never exceed by more than measurement
        noise, used to sanity-check benchmark results rather than to
        predict them exactly (spec section 22).

    Raises:
        ValidationError: If ``parallelizable_fraction`` is not in
            ``[0, 1]`` or ``workers`` is not positive.
    """
    if not 0.0 <= parallelizable_fraction <= 1.0:
        raise ValidationError(
            "amdahl_speedup() requires parallelizable_fraction in [0, 1], got "
            f"{parallelizable_fraction}."
        )
    if workers <= 0:
        raise ValidationError(f"amdahl_speedup() requires workers > 0, got {workers}.")
    serial_fraction = 1.0 - parallelizable_fraction
    return 1.0 / (serial_fraction + parallelizable_fraction / workers)


__all__ = [
    "BenchmarkResult",
    "amdahl_speedup",
    "parallel_efficiency",
    "run_benchmark",
    "speedup",
    "time_callable",
]
