"""Execution configuration for serial/parallel element-task processing (Version 27).

:class:`ExecutionConfig` is a small, immutable, validated description of
*how* independent per-element work (see the module docstring of
:mod:`femtoolkit.execution.executor`) should run -- serially in the
calling process (the toolkit's behavior through Version 26, and still the
default here) or spread across several worker processes. It carries no
behavior of its own; :func:`femtoolkit.execution.executor.create_executor`
turns one into a concrete
:class:`~femtoolkit.execution.executor.ElementExecutor`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from femtoolkit.exceptions import InvalidExecutionConfigurationError

ExecutionMode = Literal["serial", "parallel"]
ExecutionBackend = Literal["process", "thread"]

_MAX_AUTOMATIC_WORKERS = 8
"""Cap on the automatically-chosen worker count (``workers=None``).

Using every available CPU core by default can starve the rest of the
user's machine and, for small FEA models, adds process-startup overhead
that outweighs any benefit. Eight is a reasonable ceiling for the kind of
desktop/workstation hardware this toolkit targets; a caller who genuinely
wants more can always pass ``workers`` explicitly.
"""


def _validate_workers(workers: int | None) -> None:
    if workers is not None and workers < 1:
        raise InvalidExecutionConfigurationError(
            f"ExecutionConfig.workers must be a positive integer or None (automatic), "
            f"got {workers}."
        )


def _validate_chunk_size(chunk_size: int | Literal["auto"]) -> None:
    if chunk_size != "auto" and (not isinstance(chunk_size, int) or chunk_size < 1):
        raise InvalidExecutionConfigurationError(
            f"ExecutionConfig.chunk_size must be 'auto' or a positive integer, "
            f"got {chunk_size!r}."
        )


def _validate_mode(mode: str) -> None:
    if mode not in ("serial", "parallel"):
        raise InvalidExecutionConfigurationError(
            f"ExecutionConfig.mode must be 'serial' or 'parallel', got {mode!r}."
        )


def _validate_backend(backend: str) -> None:
    if backend not in ("process", "thread"):
        raise InvalidExecutionConfigurationError(
            f"ExecutionConfig.backend must be 'process' or 'thread', got {backend!r}."
        )


@dataclass(frozen=True)
class ExecutionConfig:
    """How element-level work should be executed.

    Attributes:
        mode: ``"serial"`` (the default -- runs in the calling process,
            identical to every prior version's behavior) or
            ``"parallel"`` (spreads work across :attr:`workers` worker
            processes/threads).
        workers: Number of workers to use when ``mode == "parallel"``.
            ``None`` (the default) resolves automatically to
            ``min(os.cpu_count() or 1, 8)`` via :func:`resolve_workers`.
            Must be a positive integer if given explicitly. Ignored when
            ``mode == "serial"``.
        chunk_size: Number of elements handed to a worker per task batch.
            ``"auto"`` (the default) picks a reasonable size from the
            workload and worker count (see
            :func:`~femtoolkit.execution.executor.resolve_chunk_size`).
            Must be a positive integer if given explicitly.
        backend: ``"process"`` (the default -- true parallelism via
            :class:`~concurrent.futures.ProcessPoolExecutor`, bypassing
            the GIL, at the cost of pickling every task's arguments and
            result) or ``"thread"`` (:class:`~concurrent.futures.ThreadPoolExecutor`,
            no pickling cost, but limited by the GIL except during the
            C-level NumPy/SciPy calls an element's own matrix computation
            releases it for).

    Raises:
        InvalidExecutionConfigurationError: If ``mode``, ``workers``,
            ``chunk_size``, or ``backend`` is invalid.

    Example:
        >>> ExecutionConfig()  # serial, matches every prior version
        ExecutionConfig(mode='serial', workers=None, chunk_size='auto', backend='process')
        >>> ExecutionConfig(mode="parallel", workers=4)
        ExecutionConfig(mode='parallel', workers=4, chunk_size='auto', backend='process')
    """

    mode: ExecutionMode = "serial"
    workers: int | None = None
    chunk_size: int | Literal["auto"] = "auto"
    backend: ExecutionBackend = "process"

    def __post_init__(self) -> None:
        _validate_mode(self.mode)
        _validate_workers(self.workers)
        _validate_chunk_size(self.chunk_size)
        _validate_backend(self.backend)


def resolve_workers(config: ExecutionConfig) -> int:
    """Resolve ``config.workers`` to a concrete positive worker count.

    ``config.workers`` may be ``None`` (automatic); this always returns
    the number of workers a :class:`~femtoolkit.execution.executor.ParallelExecutor`
    built from ``config`` will actually use.

    Args:
        config: The execution configuration to resolve.

    Returns:
        ``config.workers`` if explicitly set, otherwise
        ``min(os.cpu_count() or 1, 8)``.
    """
    if config.workers is not None:
        return config.workers
    return min(os.cpu_count() or 1, _MAX_AUTOMATIC_WORKERS)


__all__ = ["ExecutionBackend", "ExecutionConfig", "ExecutionMode", "resolve_workers"]
