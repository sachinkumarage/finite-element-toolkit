"""Deterministic per-task random seed derivation (Version 34).

**Why this exists, and why it changes nothing about Version 30/31
sampling.** Version 30's parameter sweeps and Version 31's Monte Carlo
sampling both already draw their *entire* sample/scenario set up front,
in the calling process, from one ``numpy.random.default_rng(seed)``
instance, *before* any task is built or executed (see
:func:`~femtoolkit.studies.parameter_sweep.generate_scenarios` and
:func:`~femtoolkit.uncertainty.sampling.generate_samples`). By the time
an individual simulation task reaches a worker, no further randomness
remains to be drawn for it -- its inputs are already a fixed dict of
values. This means **the requirement "changing the number of workers
must not change the generated samples" already holds structurally**, for
free, with zero orchestration-layer involvement: execution order and
worker count can never affect a sample set that was fully determined
before execution began.

This module exists for the narrower, genuinely execution-order-sensitive
case: a *task's own internal computation* that needs reproducible
randomness of its own (for example, a future stochastic solver, or an
objective function that -- unlike
:func:`~femtoolkit.optimization.robust.robust_objective_statistic`'s
current single-shared-seed design -- chooses to vary its inner sampling
per design). :func:`derive_task_seed` turns one base seed plus a task's
own stable identifier into a deterministic child seed, so that:

- the same ``(base_seed, task_id)`` pair always derives the same seed,
  regardless of execution order, worker count, or serial vs. parallel
  execution;
- different tasks derive different (not correlated) seeds from the same
  base seed, avoiding the correctness trap of handing every worker a
  copy of the *same* seeded generator (which would make "independent"
  tasks silently reuse the exact same random stream).

Every :class:`~femtoolkit.orchestration.simulation.SimulationTask` built
by this package's study integrations carries a seed derived this way on
its ``random_seed`` field, available for any task that wants it --
whether or not the current task's own computation happens to consume it.
"""

from __future__ import annotations

import hashlib

_SEED_MODULUS = 2**32
"""`numpy.random.default_rng` accepts any non-negative integer seed;
this keeps derived seeds within the conventional 32-bit range most
tooling expects, without weakening the hash's distribution for this
module's purpose (seed derivation, not cryptography)."""


def derive_task_seed(base_seed: int | None, task_id: str) -> int | None:
    """Deterministically derive a per-task seed from a base seed and task ID.

    Args:
        base_seed: The study/batch's own base seed. ``None`` means "no
            reproducibility was requested for this batch" and is passed
            through unchanged -- there is nothing meaningful to derive a
            child seed from.
        task_id: The task's own stable, unique identifier (e.g. a
            scenario ID or sample index string). Must be the same string
            every time the same logical task is built, so re-running a
            batch (serially, in parallel, with a different worker count)
            derives the identical seed for the identical task.

    Returns:
        ``None`` if ``base_seed`` is ``None``; otherwise a deterministic
        non-negative integer seed, stable across processes and Python
        versions (built from :func:`hashlib.sha256`, not Python's
        salted built-in :func:`hash`).
    """
    if base_seed is None:
        return None
    digest = hashlib.sha256(f"{base_seed}:{task_id}".encode()).digest()
    return int.from_bytes(digest[:4], byteorder="big") % _SEED_MODULUS


__all__ = ["derive_task_seed"]
