"""Session-exit safety net against a known CPython multiprocessing teardown hang.

This test suite exercises many independent `concurrent.futures.ProcessPoolExecutor`
instances (Version 27 element-level parallelism, Version 34 orchestration, Version 35
training-data generation). Under this project's process sandboxing, a worker or
`multiprocessing.resource_tracker` helper process can occasionally die outside the
normal pool-shutdown sequence (observed as "resource_tracker: process died
unexpectedly, relaunching" on an otherwise fully passing run). When that happens,
CPython's own `concurrent.futures.process._python_exit` atexit handler -- which joins
every pool's manager thread ever created during the process's lifetime -- can hang
indefinitely during interpreter shutdown, well after every test has already completed
and been reported. This has no effect on test correctness: every result is already
finalized and printed by the time this hook runs.

The actual exit happens in `pytest_unconfigure`, not `pytest_sessionfinish`: the
terminal reporter's own summary line ("N passed in Ys") is itself printed from a
`pytest_sessionfinish` hookimpl, and hook call order between same-named hookimpls
across plugins is not something this file should depend on. `pytest_unconfigure`
runs strictly after every `pytest_sessionfinish` hookimpl has already completed, so
the summary is guaranteed to already be on the terminal before `os._exit` -- which
skips Python's normal interpreter shutdown/atexit sequence entirely, so a stuck
multiprocessing teardown can never block process exit.
"""

from __future__ import annotations

import os
import sys

import pytest

_exit_status = 0


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    global _exit_status
    _exit_status = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_exit_status)
