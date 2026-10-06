"""Execution backends: swappable task-execution strategies (Version 34).

See :mod:`femtoolkit.orchestration.backends.base` for the shared
interface and the architectural rationale for keeping it generic.
"""

from __future__ import annotations

from femtoolkit.orchestration.backends.base import ExecutionBackend
from femtoolkit.orchestration.backends.local_process import LocalProcessBackend
from femtoolkit.orchestration.backends.serial import SerialBackend

__all__ = ["ExecutionBackend", "LocalProcessBackend", "SerialBackend"]
