"""Multi-process/multi-thread execution strategy (Version 27).

:class:`ParallelExecutor` spreads independent per-element tasks across
several workers using the Python standard library's
:mod:`concurrent.futures` -- no third-party dependency was added for
this (see the Version 27 release notes for why that was a deliberate
choice, not an oversight).

A fresh worker pool is created for each :meth:`~ParallelExecutor.map`
call and shut down (via a ``with`` block) before it returns. Version 27
deliberately does **not** keep a persistent background pool alive between
calls: a stray unclosed pool leaks OS processes, and this toolkit's
analyses call ``map`` at most a few times per solve, so pool-startup
overhead is not worth the bookkeeping a persistent pool would add (spec
section 14: "do not create persistent background workers unless
genuinely necessary").
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TypeVar

from femtoolkit.exceptions import TaskSerializationError, WorkerExecutionError
from femtoolkit.execution.config import ExecutionConfig, resolve_workers
from femtoolkit.execution.executor import ElementExecutor, resolve_chunk_size

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ParallelExecutor(ElementExecutor):
    """Runs tasks across several worker processes or threads.

    Attributes:
        config: The :class:`~femtoolkit.execution.config.ExecutionConfig`
            this executor was built from (``config.mode`` is expected to
            be ``"parallel"``; nothing prevents constructing one with a
            serial config, but :func:`~femtoolkit.execution.executor.create_executor`
            never does).

    With the default ``backend="process"``, ``func`` (and every item in
    ``items``, and every returned result) must be picklable -- in
    practice, this means ``func`` must be a plain function defined at
    module scope (importable by its ``module.qualname``), not a lambda,
    a nested function, or a bound method of an object that cannot itself
    be pickled. NumPy arrays, and the plain dataclasses this toolkit's
    elements and materials are built from, all pickle without special
    handling. ``backend="thread"`` has no such restriction (threads share
    the parent process's memory) but only achieves real parallelism
    during the portions of ``func`` that release the GIL, which for
    NumPy/SciPy-heavy element calculations is significant but not total.
    """

    config: ExecutionConfig
    _worker_count: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._worker_count = resolve_workers(self.config)

    def map(self, func: Callable[[T], R], items: Sequence[T]) -> list[R]:
        """Apply ``func`` to every item in ``items``, across worker processes/threads.

        Args:
            func: A callable of one argument (see the class docstring for
                the picklability requirement under ``backend="process"``).
            items: The inputs to process.

        Returns:
            ``[func(item) for item in items]``, in the same order as
            ``items`` regardless of which worker finishes first
            (:class:`concurrent.futures.Executor.map` guarantees
            input-order results internally).

        Raises:
            TaskSerializationError: If ``func`` or an item cannot be
                pickled to send to a worker process.
            WorkerExecutionError: If a worker raises while executing a
                task.
        """
        if not items:
            return []

        chunk_size = resolve_chunk_size(self.config, len(items), self._worker_count)
        pool_class = ProcessPoolExecutor if self.config.backend == "process" else ThreadPoolExecutor

        try:
            with pool_class(max_workers=self._worker_count) as pool:
                return list(pool.map(func, items, chunksize=chunk_size))
        except (pickle.PicklingError, AttributeError, TypeError) as error:
            raise TaskSerializationError(
                f"Could not send task {func!r} (or one of its {len(items)} item(s)) to a "
                f"worker process: {error}. With backend='process', the function and every "
                "item must be picklable -- typically a plain function defined at module "
                "scope, not a lambda or closure. Consider backend='thread' if the function "
                "genuinely cannot be made picklable."
            ) from error
        except TaskSerializationError:
            raise
        except Exception as error:
            # Deliberately broad: a worker-side exception can surface here as
            # almost any exception type (the original, re-raised by
            # concurrent.futures, or a BrokenProcessPool/BrokenThreadPool if a
            # worker crashed outright) -- every one of them must be converted
            # to a toolkit-specific exception rather than leaking a bare
            # concurrent.futures/multiprocessing failure to the caller.
            raise WorkerExecutionError(
                f"A worker failed while executing {func!r}: {error}"
            ) from error


__all__ = ["ParallelExecutor"]
