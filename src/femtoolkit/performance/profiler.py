"""Performance profiling infrastructure for FEA workflows (Version 27).

An FEA solve has several distinct stages -- preparing the mesh, computing
each element's local matrices, assembling them into a global system,
applying boundary conditions, solving the linear system, and recovering
results (stresses, strains, reactions) -- and each has a different
computational character (see :mod:`femtoolkit.execution.executor`'s
module docstring for which of these are naturally parallelizable).
:class:`Profiler` measures how long each stage actually takes, using
:func:`time.perf_counter` (a monotonic, high-resolution timer unaffected
by system clock adjustments -- the standard choice for wall-clock
benchmarking in Python), so that optimization effort (parallel execution,
sparse assembly, batching) can be guided by evidence rather than guesswork
(spec section 17: "only optimize operations supported by profiling
evidence").

Profiling a stage costs one :func:`time.perf_counter` call on entry and
exit -- a few hundred nanoseconds -- so :meth:`Profiler.stage` is safe to
wrap around a whole assembly loop or solve call, but is not intended to be
called once per element inside a hot loop (spec section 4: "do not add
profiling overhead to hot loops unnecessarily"); use
:class:`~femtoolkit.performance.benchmark.BenchmarkResult` and a coarser
measurement for that.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from femtoolkit.solvers.results import SolverResult

MESH = "mesh"
ELEMENT = "element"
ASSEMBLY = "assembly"
BOUNDARY_CONDITION = "boundary_condition"
SOLVE = "solve"
POST_PROCESSING = "post_processing"
"""Canonical stage names, matching :class:`PerformanceReport`'s fields.

Using these constants (rather than typing the stage name as a raw string
at every call site) makes a stage name typo a ``NameError`` at import
time instead of a silently-empty field in the resulting report.
"""


@dataclass(frozen=True)
class PerformanceReport:
    """Measured timing (and, where available, model/solver context) for one analysis run.

    Every ``*_time`` field is ``None`` if that stage was never profiled --
    this toolkit never fabricates a timing it did not measure (spec
    section 4/20).

    Attributes:
        total_time: Total wall-clock time for the profiled run, in
            seconds. If the :class:`Profiler` that produced this report
            was used as a context manager, this is the actual elapsed
            time from entering to exiting it (which may exceed the sum of
            the individual stages, if any untimed work happened between
            stages); otherwise it is the sum of the measured stage times.
        mesh_time: Time spent preparing the mesh, in seconds.
        element_time: Time spent computing element-local matrices
            (stiffness, mass, conductivity, ...), in seconds.
        assembly_time: Time spent scattering element contributions into
            the global matrix, in seconds.
        boundary_condition_time: Time spent applying boundary conditions,
            in seconds.
        solve_time: Time spent in the linear solve itself, in seconds.
            Distinct from ``solver_result.solve_time`` when
            ``solver_result`` is given: this is measured by the profiler
            around the whole solve call (may include, e.g., building the
            :class:`~femtoolkit.analysis.system.LinearSystem`), while
            ``solver_result.solve_time`` (Version 26) is measured
            entirely inside the solver's own numerical routine.
        post_processing_time: Time spent recovering results (stress,
            strain, reactions, ...), in seconds.
        solver_result: The full Version 26
            :class:`~femtoolkit.solvers.results.SolverResult` for this
            run, if a non-default solver was used -- carries DOF count,
            non-zero count, matrix density, iteration count, and residual
            (in ``solver_result.diagnostics``) without this report
            duplicating any of them.
        execution_mode: ``"serial"`` or ``"parallel"``, if element
            computation went through an
            :class:`~femtoolkit.execution.executor.ElementExecutor`.
        workers: Worker count used, if ``execution_mode == "parallel"``.
        model_name: A human-readable label for the model this run solved
            (e.g. ``"Cantilever Beam"``), for display purposes only.
        node_count: Number of mesh nodes.
        element_count: Number of mesh elements.
    """

    total_time: float
    mesh_time: float | None = None
    element_time: float | None = None
    assembly_time: float | None = None
    boundary_condition_time: float | None = None
    solve_time: float | None = None
    post_processing_time: float | None = None
    solver_result: SolverResult | None = None
    execution_mode: str | None = None
    workers: int | None = None
    model_name: str | None = None
    node_count: int | None = None
    element_count: int | None = None


class Profiler:
    """Measures named stage durations and an overall wall-clock total.

    Example:
        >>> profiler = Profiler()
        >>> with profiler:
        ...     with profiler.stage(MESH):
        ...         mesh = create_quad_mesh(...)
        ...     with profiler.stage(ELEMENT):
        ...         contributions = [...]
        >>> report = profiler.report(model_name="Cantilever", node_count=len(mesh.nodes))
    """

    def __init__(self) -> None:
        self._stage_times: dict[str, float] = {}
        self._context_start: float | None = None
        self._context_total: float | None = None

    def __enter__(self) -> Profiler:
        self._context_start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._context_start is not None:
            self._context_total = time.perf_counter() - self._context_start

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time one block of code as stage ``name``.

        Calling this more than once for the same ``name`` accumulates the
        durations (e.g. a Newton-Raphson loop's per-iteration solve time
        all counting toward one ``"solve"`` total).

        Args:
            name: The stage name -- one of :data:`MESH`, :data:`ELEMENT`,
                :data:`ASSEMBLY`, :data:`BOUNDARY_CONDITION`,
                :data:`SOLVE`, :data:`POST_PROCESSING`, or any other
                label; only the six canonical names populate a matching
                :class:`PerformanceReport` field.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._stage_times[name] = self._stage_times.get(name, 0.0) + elapsed

    def stage_time(self, name: str) -> float | None:
        """Return the accumulated duration for stage ``name``, or ``None`` if never measured."""
        return self._stage_times.get(name)

    def report(self, **metadata: Any) -> PerformanceReport:
        """Build a :class:`PerformanceReport` from the stages measured so far.

        Args:
            **metadata: Extra :class:`PerformanceReport` fields to set
                (``solver_result``, ``execution_mode``, ``workers``,
                ``model_name``, ``node_count``, ``element_count``).

        Returns:
            A :class:`PerformanceReport` with ``total_time`` set from the
            enclosing ``with`` block if this profiler was used as a
            context manager, otherwise the sum of the measured stage
            times.
        """
        total = (
            self._context_total
            if self._context_total is not None
            else sum(self._stage_times.values())
        )
        return PerformanceReport(
            total_time=total,
            mesh_time=self._stage_times.get(MESH),
            element_time=self._stage_times.get(ELEMENT),
            assembly_time=self._stage_times.get(ASSEMBLY),
            boundary_condition_time=self._stage_times.get(BOUNDARY_CONDITION),
            solve_time=self._stage_times.get(SOLVE),
            post_processing_time=self._stage_times.get(POST_PROCESSING),
            **metadata,
        )


__all__ = [
    "ASSEMBLY",
    "BOUNDARY_CONDITION",
    "ELEMENT",
    "MESH",
    "POST_PROCESSING",
    "SOLVE",
    "PerformanceReport",
    "Profiler",
]
