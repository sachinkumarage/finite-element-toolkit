"""Execution configuration for independent-task orchestration (Version 34).

:class:`OrchestrationConfig` is a small, validated description of *how*
a batch of independent simulation/evaluation tasks should run -- serially
in the calling process (the default, and every prior version's exact
behavior) or spread across several local worker processes. It carries no
behavior of its own; :func:`~femtoolkit.orchestration.manager.build_execution_manager`
turns one into a concrete
:class:`~femtoolkit.orchestration.manager.ExecutionManager`.

This is a distinct configuration object from
:class:`~femtoolkit.execution.config.ExecutionConfig` (Version 27),
which governs a completely different kind of parallelism -- computing
*independent elements' stiffness matrices within one FEA assembly*.
:class:`OrchestrationConfig` instead governs *independent whole
simulation tasks* (a parameter study's scenarios, a Monte Carlo study's
samples, an optimization's candidate designs). See
:mod:`femtoolkit.orchestration`'s module docstring for the full
architectural distinction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from femtoolkit.exceptions import InvalidOrchestrationConfigurationError

ExecutionMode = Literal["serial", "parallel"]

SUPPORTED_EXECUTION_MODES = ("serial", "parallel")

_MAX_AUTOMATIC_WORKERS = 8
"""Cap on the automatically-chosen worker count (``max_workers=None``).

Using every available CPU core by default can starve the rest of the
user's machine; this mirrors the exact same conservative-default
reasoning and ceiling as Version 27's
:data:`femtoolkit.execution.config._MAX_AUTOMATIC_WORKERS` -- a caller
who genuinely wants more can always pass ``max_workers`` explicitly.
``os.cpu_count()`` is never used directly as the default (spec section
19's explicit "never assume max_workers = os.cpu_count() is always
optimal").
"""


@dataclass(frozen=True)
class OrchestrationConfig:
    """How a batch of independent simulation tasks should be executed.

    Attributes:
        execution_mode: ``"serial"`` (the default -- runs every task in
            the calling process, in submission order, reproducing every
            prior version's exact behavior) or ``"parallel"`` (spreads
            tasks across :attr:`max_workers` local worker processes).
        max_workers: Number of worker processes to use when
            ``execution_mode == "parallel"``. ``None`` (the default)
            resolves automatically to ``min(os.cpu_count() or 1, 8)``
            via :func:`resolve_max_workers`. Must be a positive integer
            if given explicitly. Ignored when ``execution_mode ==
            "serial"``.
        fail_fast: If ``True``, a failed task stops the batch: no further
            tasks are submitted, and already-pending (not yet running)
            tasks are marked ``CANCELLED``/``SKIPPED`` rather than
            started. Already-*running* tasks are allowed to finish where
            the underlying worker pool cannot safely interrupt them
            mid-flight (see
            :mod:`femtoolkit.orchestration.backends.local_process`). If
            ``False`` (the default), every task runs regardless of
            earlier failures, matching every prior version's "run
            everything, report failures separately" behavior.
        timeout: Maximum wall-clock seconds allowed for any single task.
            ``None`` (the default) disables the timeout. If exceeded,
            the offending task is marked ``FAILED`` and -- because a
            standard-library process pool cannot reliably interrupt one
            specific already-running worker without risking an orphaned
            process -- the *entire* worker pool for that batch is shut
            down (spec section 18/20: never leave an orphaned worker
            process running); remaining tasks are marked ``CANCELLED``.
            This is a deliberately conservative, clearly documented
            trade-off, not a partial-interruption guarantee.
        chunk_size: How many tasks are handed to one worker per
            dispatch round-trip, for the ``"parallel"`` backend.
            ``"auto"`` (the default) picks a reasonable size from the
            workload and worker count. Must be a positive integer if
            given explicitly.
        preserve_order: If ``True`` (the default), the returned results
            list is in the same order as the submitted tasks, regardless
            of which worker finished first -- required for parameter
            studies and Monte Carlo statistics, where result identity
            must never silently depend on execution timing. If
            ``False``, results may be returned in completion order
            instead (only meaningful for callers that explicitly want to
            process results as they arrive and track identity via each
            result's own ``task_id``).
        random_seed: An optional base seed this batch's tasks may derive
            their own deterministic per-task seeds from (see
            :mod:`femtoolkit.orchestration.random_state`). Does **not**
            affect Version 30/31 parameter sweep or Monte Carlo sampling,
            which is always fully determined before any task is built
            (see the module docstring of
            :mod:`femtoolkit.orchestration.random_state` for why).

    Raises:
        InvalidOrchestrationConfigurationError: If any field is invalid.

    Example:
        >>> OrchestrationConfig()  # serial, matches every prior version
        OrchestrationConfig(execution_mode='serial', max_workers=None, ...)
        >>> OrchestrationConfig(execution_mode="parallel", max_workers=4)
        OrchestrationConfig(execution_mode='parallel', max_workers=4, ...)
    """

    execution_mode: ExecutionMode = "serial"
    max_workers: int | None = None
    fail_fast: bool = False
    timeout: float | None = None
    chunk_size: int | Literal["auto"] = "auto"
    preserve_order: bool = True
    random_seed: int | None = None

    def __post_init__(self) -> None:
        if self.execution_mode not in SUPPORTED_EXECUTION_MODES:
            raise InvalidOrchestrationConfigurationError(
                f"OrchestrationConfig.execution_mode must be one of "
                f"{SUPPORTED_EXECUTION_MODES}, got {self.execution_mode!r}."
            )
        if self.max_workers is not None and self.max_workers < 1:
            raise InvalidOrchestrationConfigurationError(
                f"OrchestrationConfig.max_workers must be a positive integer or None "
                f"(automatic), got {self.max_workers}."
            )
        if self.timeout is not None and self.timeout <= 0:
            raise InvalidOrchestrationConfigurationError(
                f"OrchestrationConfig.timeout must be positive or None (disabled), "
                f"got {self.timeout}."
            )
        if self.chunk_size != "auto" and (
            not isinstance(self.chunk_size, int) or self.chunk_size < 1
        ):
            raise InvalidOrchestrationConfigurationError(
                f"OrchestrationConfig.chunk_size must be 'auto' or a positive integer, "
                f"got {self.chunk_size!r}."
            )


def resolve_max_workers(config: OrchestrationConfig) -> int:
    """Resolve ``config.max_workers`` to a concrete positive worker count.

    Args:
        config: The orchestration configuration to resolve.

    Returns:
        ``config.max_workers`` if explicitly set, otherwise
        ``min(os.cpu_count() or 1, 8)``.
    """
    if config.max_workers is not None:
        return config.max_workers
    return min(os.cpu_count() or 1, _MAX_AUTOMATIC_WORKERS)


def resolve_chunk_size(config: OrchestrationConfig, task_count: int, worker_count: int) -> int:
    """Resolve ``config.chunk_size`` to a concrete positive chunk size.

    Args:
        config: The orchestration configuration (``chunk_size`` may be
            ``"auto"`` or an explicit positive integer).
        task_count: Number of tasks about to be processed.
        worker_count: Number of workers that will process them.

    Returns:
        ``config.chunk_size`` if explicitly set; otherwise a chunk size
        aiming for a handful of chunks per worker, never smaller than 1
        -- the same balance-vs-overhead reasoning as Version 27's
        :func:`femtoolkit.execution.executor.resolve_chunk_size`.
    """
    if config.chunk_size != "auto":
        return config.chunk_size
    if task_count <= 0 or worker_count <= 0:
        return 1
    target_chunks = worker_count * 4
    return max(1, task_count // target_chunks)


__all__ = [
    "SUPPORTED_EXECUTION_MODES",
    "ExecutionMode",
    "OrchestrationConfig",
    "resolve_chunk_size",
    "resolve_max_workers",
]
