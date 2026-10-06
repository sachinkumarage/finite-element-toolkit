"""Nested-parallelism protection (Version 34).

**The risk.** Parallel optimization evaluates candidates in worker
processes; a robust (uncertainty-aware) objective evaluated inside one
of those workers runs its own inner Monte Carlo study; if *that* study
also tried to parallelize across its own worker pool, each outer worker
would spawn a further pool of its own -- `workers x workers` processes
for what the caller asked to run on `workers` processes, with no bound
on how many levels deep this could nest (parallel optimization -> parallel
Monte Carlo -> parallel FEA, as the spec puts it).

**The guard.** :func:`is_inside_worker_process` uses
:func:`multiprocessing.parent_process` -- a standard-library check that
reliably answers "was the *current* process spawned by a
:class:`~concurrent.futures.ProcessPoolExecutor`/:mod:`multiprocessing`
pool, at any depth" -- regardless of the platform's start method (fork
or spawn) and without any toolkit-specific bookkeeping (environment
variables, global flags) that could itself race or leak across
processes. :func:`resolve_safe_execution_mode` is the single place every
:class:`~femtoolkit.orchestration.manager.ExecutionManager` consults
before honoring a ``"parallel"`` request: if the calling code is already
running inside a worker process, the request is silently downgraded to
``"serial"`` rather than rejected -- the safest available behavior (the
inner work still completes correctly, just without a nested pool), and
one that requires no cooperation from the many different call sites that
might end up nested this way.
"""

from __future__ import annotations

import logging
import multiprocessing
from typing import TYPE_CHECKING

from femtoolkit.orchestration.config import ExecutionMode

if TYPE_CHECKING:
    from femtoolkit.application.project import Project

_logger = logging.getLogger(__name__)


def is_inside_worker_process() -> bool:
    """Whether the current process was itself spawned by a multiprocessing pool.

    Returns:
        ``True`` if :func:`multiprocessing.parent_process` is not
        ``None`` (i.e. this process has a multiprocessing parent --
        it is a pool worker, at any nesting depth); ``False`` in the
        main process.
    """
    return multiprocessing.parent_process() is not None


def resolve_safe_execution_mode(requested_mode: ExecutionMode) -> ExecutionMode:
    """Downgrade a ``"parallel"`` request to ``"serial"`` when already inside a worker.

    Args:
        requested_mode: The execution mode the caller asked for.

    Returns:
        ``requested_mode`` unchanged, unless it is ``"parallel"`` and
        the current process is already a pool worker (see
        :func:`is_inside_worker_process`), in which case ``"serial"`` is
        returned and a warning is logged -- spawning a nested worker
        pool from inside a worker process is never attempted.
    """
    if requested_mode == "parallel" and is_inside_worker_process():
        _logger.warning(
            "Parallel execution requested from inside an existing worker process; "
            "downgrading to serial for this nested batch to avoid an uncontrolled "
            "number of nested worker pools."
        )
        return "serial"
    return requested_mode


def resolve_safe_project_execution(project: Project) -> Project:
    """Downgrade a project's own Version 27 element-level parallelism when nested.

    A :class:`~femtoolkit.application.project.Project` may itself
    request ``project.execution.mode == "parallel"`` (Version 27's
    per-element stiffness-matrix parallelism, a completely different
    layer from this package -- see :mod:`femtoolkit.orchestration`'s
    module docstring). If an orchestration worker process runs such a
    project (parallel optimization or parallel Monte Carlo -> parallel
    FEA), that element-level pool would itself nest inside this
    package's already-parallel worker, multiplying process count with
    no bound. This function is called from inside the worker, on the
    project instance that just crossed the process boundary (and is
    therefore this worker's own private copy, safe to mutate directly
    without affecting the calling process or any other task) -- it
    mutates and returns the same project, downgraded to serial element
    execution, only when both:

    - this process is itself a pool worker
      (:func:`is_inside_worker_process`), and
    - the project actually requested ``"parallel"`` element execution.

    Outside a worker process (ordinary serial orchestration, or the
    main process), ``project`` is returned unchanged -- there is no
    nesting risk to guard against, and mutating a caller-owned project
    object in the main process would be unsafe.

    Args:
        project: The project a task is about to execute.

    Returns:
        ``project``, with ``project.execution.mode`` forced to
        ``"serial"`` if a downgrade was needed.
    """
    if not is_inside_worker_process():
        return project
    if project.execution.mode != "parallel":
        return project
    _logger.warning(
        "A project requested parallel element-level execution (Version 27) while "
        "already running inside an orchestration worker process; downgrading this "
        "project's execution.mode to serial for this task to avoid nested worker pools."
    )
    project.execution.mode = "serial"
    return project


__all__ = [
    "is_inside_worker_process",
    "resolve_safe_execution_mode",
    "resolve_safe_project_execution",
]
