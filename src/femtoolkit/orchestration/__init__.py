"""Parallel simulation execution and study orchestration (Version 34).

This package parallelizes **independent whole simulation tasks** --
entire scenarios, Monte Carlo samples, or optimization candidate
designs, each its own complete FEA solve -- across local worker
processes. It is deliberately generic: a parameter study (Version 30),
a Monte Carlo uncertainty analysis (Version 31), and a population-based
optimization algorithm (Versions 32/33) all submit batches of
independent tasks through the exact same
:class:`~femtoolkit.orchestration.manager.ExecutionManager`, rather than
each implementing its own parallel-execution logic.

.. code-block:: text

    Simulation Study / Monte Carlo Study / Optimization Batch
                            |
                     Task Generation
                            |
                    ExecutionManager            (this package: manager.py)
                            |
                    ExecutionBackend             (backends/: serial, local_process)
                            |
                      Worker Pool                (local_process only)
                            |
                    SimulationRunner / Design Evaluator   (reused unchanged)
                            |
                      TaskOutcome / SimulationResult

**Why this is a separate package from** :mod:`femtoolkit.execution`
**(Version 27).** Version 27's ``femtoolkit.execution`` already
parallelizes *independent elements' stiffness-matrix computation within
one FEA assembly* -- fine-grained, intra-solve parallelism, wired into
:mod:`femtoolkit.analysis.parallel_assembly` and exposed via
``Project.execution``. ``femtoolkit.orchestration`` is a completely
different, coarse-grained layer one level up: it parallelizes *entire,
independent simulation runs* (an entire scenario, an entire Monte Carlo
sample, an entire candidate-design evaluation), each of which may
*internally* still use Version 27's element-level parallelism (or not --
see :mod:`femtoolkit.orchestration.nesting` for how the two are kept
from nesting into an uncontrolled number of worker pools). The two
packages' configuration objects, exception types, and module names are
kept entirely distinct (``OrchestrationConfig`` vs. ``ExecutionConfig``,
``ExecutionBackend`` here vs. ``ElementExecutor`` there) so neither is
ever confused with the other; ``femtoolkit.execution`` is not modified
by this version.

**What this package does not do.** No distributed computing, MPI,
cluster scheduling, cloud execution, GPU/CUDA, or shared-memory FEA
solving -- see :mod:`femtoolkit.orchestration.backends.base`'s module
docstring for the backends this version implements and the ones
explicitly reserved for later. This version is focused on safe, local,
process-based parallel execution of independent simulation tasks.

Typical usage, through the domain-specific adapters built on top of
this package's generic layer:

- :mod:`femtoolkit.orchestration.simulation` -- parameter studies and
  Monte Carlo studies (Versions 30/31).
- :mod:`femtoolkit.optimization.batch` -- optimization candidate
  evaluation (Versions 32/33).

A caller that only needs the generic layer directly uses
:func:`~femtoolkit.orchestration.manager.build_execution_manager`.
"""

from __future__ import annotations

from femtoolkit.orchestration.config import ExecutionMode, OrchestrationConfig
from femtoolkit.orchestration.manager import (
    ExecutionManager,
    ParallelExecutionManager,
    SerialExecutionManager,
    build_execution_manager,
)
from femtoolkit.orchestration.models import TaskOutcome, TaskStatus
from femtoolkit.orchestration.progress import ExecutionProgress, format_console_progress
from femtoolkit.orchestration.summary import (
    ExecutionSummary,
    build_execution_summary,
    execution_summary_to_dict,
)

__all__ = [
    "ExecutionManager",
    "ExecutionMode",
    "ExecutionProgress",
    "ExecutionSummary",
    "OrchestrationConfig",
    "ParallelExecutionManager",
    "SerialExecutionManager",
    "TaskOutcome",
    "TaskStatus",
    "build_execution_manager",
    "build_execution_summary",
    "execution_summary_to_dict",
    "format_console_progress",
]
