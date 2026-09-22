"""The default, zero-overhead execution strategy (Version 27).

:class:`SerialExecutor` runs every task in the calling process, in a
plain Python loop -- exactly what every analysis in this toolkit already
did through Version 26. It exists so that code written against
:class:`~femtoolkit.execution.executor.ElementExecutor` has a real
implementation to use by default, without needing an ``if parallel: ...
else: ...`` branch at every call site.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

from femtoolkit.execution.executor import ElementExecutor

T = TypeVar("T")
R = TypeVar("R")


class SerialExecutor(ElementExecutor):
    """Runs every task in the calling process, one at a time, in order."""

    def map(self, func: Callable[[T], R], items: Sequence[T]) -> list[R]:
        """Apply ``func`` to every item in ``items``, serially.

        Args:
            func: A callable of one argument. Unlike
                :class:`~femtoolkit.execution.parallel.ParallelExecutor`,
                this places no picklability requirement on ``func``.
            items: The inputs to process.

        Returns:
            ``[func(item) for item in items]``.
        """
        return [func(item) for item in items]


__all__ = ["SerialExecutor"]
