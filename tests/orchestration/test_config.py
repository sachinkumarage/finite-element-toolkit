"""Tests for femtoolkit.orchestration.config (Version 34)."""

from __future__ import annotations

import pytest

from femtoolkit.exceptions import InvalidOrchestrationConfigurationError
from femtoolkit.orchestration.config import (
    OrchestrationConfig,
    resolve_chunk_size,
    resolve_max_workers,
)


def test_defaults_are_serial() -> None:
    config = OrchestrationConfig()
    assert config.execution_mode == "serial"
    assert config.max_workers is None
    assert config.fail_fast is False
    assert config.timeout is None
    assert config.chunk_size == "auto"
    assert config.preserve_order is True
    assert config.random_seed is None


def test_rejects_invalid_execution_mode() -> None:
    with pytest.raises(InvalidOrchestrationConfigurationError):
        OrchestrationConfig(execution_mode="turbo")


def test_rejects_non_positive_max_workers() -> None:
    with pytest.raises(InvalidOrchestrationConfigurationError):
        OrchestrationConfig(max_workers=0)
    with pytest.raises(InvalidOrchestrationConfigurationError):
        OrchestrationConfig(max_workers=-1)


def test_accepts_none_max_workers() -> None:
    OrchestrationConfig(max_workers=None)


def test_rejects_non_positive_timeout() -> None:
    with pytest.raises(InvalidOrchestrationConfigurationError):
        OrchestrationConfig(timeout=0.0)
    with pytest.raises(InvalidOrchestrationConfigurationError):
        OrchestrationConfig(timeout=-5.0)


def test_rejects_invalid_chunk_size() -> None:
    with pytest.raises(InvalidOrchestrationConfigurationError):
        OrchestrationConfig(chunk_size=0)
    with pytest.raises(InvalidOrchestrationConfigurationError):
        OrchestrationConfig(chunk_size="bogus")


def test_accepts_explicit_positive_chunk_size() -> None:
    OrchestrationConfig(chunk_size=4)


def test_resolve_max_workers_explicit() -> None:
    config = OrchestrationConfig(max_workers=3)
    assert resolve_max_workers(config) == 3


def test_resolve_max_workers_automatic_is_capped_and_positive() -> None:
    config = OrchestrationConfig()
    workers = resolve_max_workers(config)
    assert 1 <= workers <= 8


def test_resolve_chunk_size_explicit() -> None:
    config = OrchestrationConfig(chunk_size=5)
    assert resolve_chunk_size(config, task_count=100, worker_count=4) == 5


def test_resolve_chunk_size_auto_is_positive() -> None:
    config = OrchestrationConfig()
    assert resolve_chunk_size(config, task_count=100, worker_count=4) >= 1


def test_resolve_chunk_size_auto_handles_empty_workload() -> None:
    config = OrchestrationConfig()
    assert resolve_chunk_size(config, task_count=0, worker_count=4) == 1
    assert resolve_chunk_size(config, task_count=10, worker_count=0) == 1
