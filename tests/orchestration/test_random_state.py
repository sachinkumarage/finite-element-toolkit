"""Tests for femtoolkit.orchestration.random_state (Version 34)."""

from __future__ import annotations

from femtoolkit.orchestration.random_state import derive_task_seed


def test_none_base_seed_passes_through() -> None:
    assert derive_task_seed(None, "task-1") is None


def test_deterministic_for_same_inputs() -> None:
    first = derive_task_seed(42, "task-7")
    second = derive_task_seed(42, "task-7")
    assert first == second


def test_different_task_ids_derive_different_seeds() -> None:
    seeds = {derive_task_seed(42, f"task-{i}") for i in range(20)}
    assert len(seeds) == 20


def test_different_base_seeds_derive_different_seeds() -> None:
    assert derive_task_seed(1, "task-1") != derive_task_seed(2, "task-1")


def test_seed_is_non_negative_32_bit_integer() -> None:
    seed = derive_task_seed(123, "task-x")
    assert isinstance(seed, int)
    assert 0 <= seed < 2**32
