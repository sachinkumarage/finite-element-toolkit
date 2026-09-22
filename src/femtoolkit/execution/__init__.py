"""Serial/parallel execution infrastructure for independent element-level tasks (Version 27).

See :mod:`femtoolkit.execution.executor` for the architectural
rationale (which FEA operations are safe to parallelize, and why) and
:mod:`femtoolkit.execution.parallel` for the concrete parallel strategy.
"""

from __future__ import annotations

from femtoolkit.execution.config import ExecutionBackend, ExecutionConfig, ExecutionMode
from femtoolkit.execution.executor import ElementExecutor, create_executor, resolve_chunk_size
from femtoolkit.execution.parallel import ParallelExecutor
from femtoolkit.execution.serial import SerialExecutor

__all__ = [
    "ElementExecutor",
    "ExecutionBackend",
    "ExecutionConfig",
    "ExecutionMode",
    "ParallelExecutor",
    "SerialExecutor",
    "create_executor",
    "resolve_chunk_size",
]
