"""Tests for femtoolkit.surrogate.rom.snapshots."""

from __future__ import annotations

import numpy as np
import pytest

from femtoolkit.exceptions import InconsistentSnapshotError, ValidationError
from femtoolkit.runs.models import RunStatus
from femtoolkit.surrogate.rom.snapshots import (
    FieldSnapshot,
    build_snapshot_matrix,
    collect_field_snapshots_from_runs,
)


class _FakeRun:
    def __init__(self, run_id: str, status: RunStatus, field: np.ndarray | None) -> None:
        self.run_id = run_id
        self.status = status
        self._field = field


def test_build_snapshot_matrix_stacks_columns() -> None:
    snapshots = [
        FieldSnapshot(snapshot_id="a", field=np.array([1.0, 2.0, 3.0])),
        FieldSnapshot(snapshot_id="b", field=np.array([4.0, 5.0, 6.0])),
    ]
    matrix = build_snapshot_matrix(snapshots)
    assert matrix.shape == (3, 2)
    np.testing.assert_array_equal(matrix[:, 0], [1.0, 2.0, 3.0])


def test_build_snapshot_matrix_rejects_empty_list() -> None:
    with pytest.raises(ValidationError):
        build_snapshot_matrix([])


def test_build_snapshot_matrix_rejects_mismatched_dimensions() -> None:
    snapshots = [
        FieldSnapshot(snapshot_id="a", field=np.array([1.0, 2.0])),
        FieldSnapshot(snapshot_id="b", field=np.array([1.0, 2.0, 3.0])),
    ]
    with pytest.raises(InconsistentSnapshotError):
        build_snapshot_matrix(snapshots)


def test_collect_field_snapshots_from_runs_skips_failed_runs() -> None:
    runs = [
        _FakeRun("r1", RunStatus.COMPLETED, np.array([1.0, 2.0])),
        _FakeRun("r2", RunStatus.FAILED, None),
        _FakeRun("r3", RunStatus.COMPLETED, np.array([3.0, 4.0])),
    ]
    design_points = [{"t": 1.0}, {"t": 2.0}, {"t": 3.0}]
    snapshots = collect_field_snapshots_from_runs(runs, design_points, lambda run: run._field)
    assert len(snapshots) == 2
    assert snapshots[0].source_simulation_id == "r1"
    assert snapshots[1].design_point == {"t": 3.0}


def test_collect_field_snapshots_from_runs_requires_matching_lengths() -> None:
    with pytest.raises(ValidationError):
        collect_field_snapshots_from_runs([], [{"t": 1.0}], lambda run: None)
