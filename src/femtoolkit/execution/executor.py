"""The element-task execution abstraction (Version 27).

Many of this toolkit's per-element operations -- computing an element's
stiffness matrix, its conductivity matrix, or recovering its stress from
a solved displacement field -- depend only on that element's own state
(geometry, material, and, for post-processing, the small slice of the
global solution touching that element's own nodes). Nothing about one
element's calculation reads or writes another element's data, so these
tasks are **embarrassingly parallel**: independent units of work with no
shared mutable state and no ordering dependency between them (see
:mod:`femtoolkit.analysis.assembly`'s module docstring for how the
*results* are later combined, which is a separate, inherently sequential
step -- see :mod:`femtoolkit.analysis.parallel_assembly`).

.. code-block:: text

    ElementExecutor (ABC)
    |-- SerialExecutor    (femtoolkit.execution.serial)   -- default
    `-- ParallelExecutor  (femtoolkit.execution.parallel) -- opt-in

Both implement one method, ``map(func, items) -> list``, matching the
built-in :func:`map`'s argument order but -- unlike a lazy
:class:`concurrent.futures.Executor.map` iterator -- always returning a
materialized ``list`` in the same order as ``items``, regardless of which
worker happens to finish first (spec section 19: "never rely on
completion order"). An empty ``items`` returns ``[]`` immediately without
starting any workers.

``func`` must be a plain, importable, module-level callable (not a
lambda, a closure, or a bound method of an object that itself cannot be
pickled) whenever a :class:`~femtoolkit.execution.parallel.ParallelExecutor`
using the ``"process"`` backend is in play -- see
:class:`~femtoolkit.execution.parallel.ParallelExecutor`'s docstring for
why, and :exc:`~femtoolkit.exceptions.TaskSerializationError` for what
happens if it is not.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TypeVar

from femtoolkit.execution.config import ExecutionConfig

T = TypeVar("T")
R = TypeVar("R")

_DEFAULT_CHUNKS_PER_WORKER = 4
"""Target number of chunks per worker when ``chunk_size == "auto"``.

A handful of chunks per worker lets :mod:`concurrent.futures` balance
uneven per-element cost across workers (spec section 23's "load
balancing") without dispatching one task per element, which would waste
most of the parallel speedup on inter-process communication overhead for
a typical FEA element (a cheap calculation relative to one IPC round
trip).
"""


class ElementExecutor(ABC):
    """Runs independent per-element tasks, serially or in parallel.

    Concrete executors are interchangeable: any code written against this
    interface works unchanged whether it was handed a
    :class:`~femtoolkit.execution.serial.SerialExecutor` or a
    :class:`~femtoolkit.execution.parallel.ParallelExecutor` -- exactly
    the same "one interface, swappable strategy" pattern
    :class:`~femtoolkit.solvers.base.LinearSolver` established in
    Version 26.
    """

    @abstractmethod
    def map(self, func: Callable[[T], R], items: Sequence[T]) -> list[R]:
        """Apply ``func`` to every item in ``items`` and return the results, in order.

        Args:
            func: A callable of one argument. Must be a picklable,
                module-level function when this executor runs tasks in
                separate processes (see the module docstring).
            items: The independent inputs to process. An empty sequence
                returns ``[]`` immediately.

        Returns:
            ``[func(item) for item in items]``, computed serially or in
            parallel depending on the concrete executor -- always in the
            same order as ``items``, regardless of completion order.

        Raises:
            TaskSerializationError: If ``func`` or an item cannot be sent
                to a worker process.
            WorkerExecutionError: If a worker raises while executing a
                task.
        """
        raise NotImplementedError


def resolve_chunk_size(config: ExecutionConfig, item_count: int, worker_count: int) -> int:
    """Resolve ``config.chunk_size`` to a concrete positive chunk size.

    Args:
        config: The execution configuration (``chunk_size`` may be
            ``"auto"`` or an explicit positive integer).
        item_count: Number of items about to be processed.
        worker_count: Number of workers that will process them.

    Returns:
        ``config.chunk_size`` if explicitly set; otherwise a chunk size
        aiming for roughly :data:`_DEFAULT_CHUNKS_PER_WORKER` chunks per
        worker, never smaller than 1.
    """
    if config.chunk_size != "auto":
        return config.chunk_size
    if item_count <= 0 or worker_count <= 0:
        return 1
    target_chunks = worker_count * _DEFAULT_CHUNKS_PER_WORKER
    return max(1, item_count // target_chunks)


def create_executor(config: ExecutionConfig | None = None) -> ElementExecutor:
    """Build the :class:`ElementExecutor` described by ``config``.

    Args:
        config: The execution configuration. ``None`` (the default)
            resolves to ``ExecutionConfig()`` -- serial execution,
            reproducing every prior version's exact behavior.

    Returns:
        A :class:`~femtoolkit.execution.serial.SerialExecutor` if
        ``config.mode == "serial"``, otherwise a
        :class:`~femtoolkit.execution.parallel.ParallelExecutor`
        configured per ``config``.
    """
    from femtoolkit.execution.parallel import ParallelExecutor
    from femtoolkit.execution.serial import SerialExecutor

    resolved_config = config if config is not None else ExecutionConfig()
    if resolved_config.mode == "serial":
        return SerialExecutor()
    return ParallelExecutor(config=resolved_config)


__all__ = ["ElementExecutor", "create_executor", "resolve_chunk_size"]
