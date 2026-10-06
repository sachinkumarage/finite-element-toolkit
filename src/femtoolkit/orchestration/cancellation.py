"""Cooperative cancellation for a running task batch (Version 34).

:class:`CancellationToken` is a small, thread-safe flag a caller can set
from outside a running :meth:`~femtoolkit.orchestration.manager.ExecutionManager.run`
call (e.g. from a GUI "Cancel" button's callback, running on the main
thread while the batch executes) to ask it to stop submitting further
tasks. It is **cooperative**: an already-*running* task is never forcibly
killed mid-computation by this token alone (see
:mod:`femtoolkit.orchestration.backends.local_process`'s module
docstring for how a parallel batch's worker pool is actually torn down
on cancellation/timeout). Checked between task submissions, not inside
one task's own computation, which this package has no way to interrupt
safely.
"""

from __future__ import annotations

import threading


class CancellationToken:
    """A thread-safe, one-way "please stop" flag for one execution batch.

    Once set, a token can never be un-set -- cancellation is a one-way
    trip for a given batch, matching the spec's "stop submitting new
    tasks" semantics rather than a pausable/resumable control.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation. Safe to call from any thread, any number of times."""
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        """Whether :meth:`cancel` has been called."""
        return self._event.is_set()


__all__ = ["CancellationToken"]
