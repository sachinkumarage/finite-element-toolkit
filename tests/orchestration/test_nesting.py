"""Tests for femtoolkit.orchestration.nesting (Version 34)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

from femtoolkit.application.project import Project
from femtoolkit.orchestration.nesting import (
    is_inside_worker_process,
    resolve_safe_execution_mode,
    resolve_safe_project_execution,
)


def _check_inside_worker() -> bool:
    return is_inside_worker_process()


def _resolve_inside_worker(requested_mode: str) -> str:
    return resolve_safe_execution_mode(requested_mode)


def _resolve_project_execution_inside_worker(mode: str) -> str:
    project = Project(name="nested", analysis_type="linear_static")
    project.execution.mode = mode
    resolved = resolve_safe_project_execution(project)
    return resolved.execution.mode


def test_main_process_is_not_a_worker() -> None:
    assert not is_inside_worker_process()


def test_worker_process_is_detected_as_a_worker() -> None:
    with ProcessPoolExecutor(max_workers=1) as pool:
        result = pool.submit(_check_inside_worker).result()
    assert result is True


def test_serial_request_is_never_downgraded() -> None:
    assert resolve_safe_execution_mode("serial") == "serial"


def test_parallel_request_unchanged_outside_a_worker() -> None:
    assert resolve_safe_execution_mode("parallel") == "parallel"


def test_parallel_request_downgraded_inside_a_worker() -> None:
    with ProcessPoolExecutor(max_workers=1) as pool:
        result = pool.submit(_resolve_inside_worker, "parallel").result()
    assert result == "serial"


def test_project_execution_unchanged_outside_a_worker() -> None:
    project = Project(name="p", analysis_type="linear_static")
    project.execution.mode = "parallel"
    resolved = resolve_safe_project_execution(project)
    assert resolved.execution.mode == "parallel"


def test_project_execution_serial_unaffected_outside_a_worker() -> None:
    project = Project(name="p", analysis_type="linear_static")
    assert project.execution.mode == "serial"
    resolved = resolve_safe_project_execution(project)
    assert resolved.execution.mode == "serial"


def test_project_execution_downgraded_inside_a_worker() -> None:
    with ProcessPoolExecutor(max_workers=1) as pool:
        result = pool.submit(_resolve_project_execution_inside_worker, "parallel").result()
    assert result == "serial"


def test_project_execution_serial_stays_serial_inside_a_worker() -> None:
    with ProcessPoolExecutor(max_workers=1) as pool:
        result = pool.submit(_resolve_project_execution_inside_worker, "serial").result()
    assert result == "serial"
