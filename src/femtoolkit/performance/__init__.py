"""Performance profiling and benchmarking for FEA workflows (Version 27).

See :mod:`femtoolkit.performance.profiler` for stage-level timing and
:mod:`femtoolkit.performance.benchmark` for comparing runs (serial vs.
parallel, dense vs. sparse, mesh size scaling).
"""

from __future__ import annotations

from femtoolkit.performance.benchmark import (
    BenchmarkResult,
    amdahl_speedup,
    parallel_efficiency,
    run_benchmark,
    speedup,
    time_callable,
)
from femtoolkit.performance.profiler import (
    ASSEMBLY,
    BOUNDARY_CONDITION,
    ELEMENT,
    MESH,
    POST_PROCESSING,
    SOLVE,
    PerformanceReport,
    Profiler,
)
from femtoolkit.performance.report import format_report

__all__ = [
    "ASSEMBLY",
    "BOUNDARY_CONDITION",
    "ELEMENT",
    "MESH",
    "POST_PROCESSING",
    "SOLVE",
    "BenchmarkResult",
    "PerformanceReport",
    "Profiler",
    "amdahl_speedup",
    "format_report",
    "parallel_efficiency",
    "run_benchmark",
    "speedup",
    "time_callable",
]
