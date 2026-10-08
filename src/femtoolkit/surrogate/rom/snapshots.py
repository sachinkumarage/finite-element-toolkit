"""Full-field snapshot collection for POD (Version 35).

Distinct from :mod:`femtoolkit.surrogate.datasets` (which collects
*scalar* input/output pairs for a surrogate model): a POD
:class:`FieldSnapshot` instead carries an entire full-field solution
vector (e.g. every nodal displacement DOF), since Proper Orthogonal
Decomposition reduces a *field*, not a scalar response.

This module performs no FEA computation itself --
:func:`collect_field_snapshots_from_runs` extracts an already-solved
field from :class:`~femtoolkit.runs.models.SimulationRun` objects
produced by the existing simulation infrastructure (Version 30/34),
exactly like :mod:`femtoolkit.surrogate.datasets` does for scalar
responses.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from femtoolkit.exceptions import InconsistentSnapshotError, ValidationError


@dataclass
class FieldSnapshot:
    """One full-field high-fidelity solution, with the design point that produced it.

    Attributes:
        snapshot_id: A unique identifier (typically the originating run's ID).
        field: The full-field solution vector (e.g. nodal displacements,
            in global DOF order).
        design_point: The input/design-variable values that produced
            this field, keyed by name.
        source_simulation_id: The originating
            :class:`~femtoolkit.runs.models.SimulationRun`'s ID, or
            ``None``.
        metadata: Free-form additional information.
        created_at: ISO-8601 UTC timestamp when this snapshot was recorded.
    """

    snapshot_id: str
    field: np.ndarray
    design_point: dict[str, float] = field(default_factory=dict)
    source_simulation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def build_snapshot_matrix(snapshots: list[FieldSnapshot]) -> np.ndarray:
    """Stack a list of :class:`FieldSnapshot` fields into one POD snapshot matrix.

    Args:
        snapshots: The field snapshots to stack, in the order they
            should appear as columns.

    Returns:
        A matrix of shape ``(n_dofs, len(snapshots))``.

    Raises:
        ValidationError: If ``snapshots`` is empty.
        InconsistentSnapshotError: If the snapshots do not all have the
            same field dimension (e.g. a different mesh/DOF count) --
            POD requires every column of the snapshot matrix to
            represent the same physical degrees of freedom.
    """
    if not snapshots:
        raise ValidationError("build_snapshot_matrix() requires at least one snapshot.")
    expected_size = snapshots[0].field.size
    for snapshot in snapshots:
        if snapshot.field.size != expected_size:
            raise InconsistentSnapshotError(
                f"Snapshot {snapshot.snapshot_id!r} has field dimension {snapshot.field.size}, "
                f"expected {expected_size} (every snapshot must share the same mesh/DOF count)."
            )
    return np.column_stack([snapshot.field.reshape(-1) for snapshot in snapshots])


def collect_field_snapshots_from_runs(
    runs: list,
    design_points: list[dict[str, float]],
    field_extractor: Callable[[Any], np.ndarray | None],
) -> list[FieldSnapshot]:
    """Build :class:`FieldSnapshot` objects from already-executed simulation runs.

    Args:
        runs: The executed :class:`~femtoolkit.runs.models.SimulationRun`
            objects, one per design point.
        design_points: Each run's design-variable values, same order
            and length as ``runs``.
        field_extractor: Given one completed run, returns its full-field
            solution vector (e.g.
            ``lambda run: run.result.solver_diagnostics.solution``), or
            ``None`` if the field is unavailable for that run.

    Returns:
        One :class:`FieldSnapshot` per run that both completed
        successfully and yielded a usable field. A failed run
        contributes no snapshot.

    Raises:
        ValidationError: If ``runs`` and ``design_points`` differ in length.
    """
    from femtoolkit.runs.models import RunStatus

    if len(runs) != len(design_points):
        raise ValidationError(
            f"runs ({len(runs)}) and design_points ({len(design_points)}) must have the "
            "same length."
        )

    snapshots: list[FieldSnapshot] = []
    for run, point in zip(runs, design_points, strict=True):
        if run.status != RunStatus.COMPLETED:
            continue
        field_values = field_extractor(run)
        if field_values is None:
            continue
        snapshots.append(
            FieldSnapshot(
                snapshot_id=run.run_id,
                field=np.asarray(field_values, dtype=float),
                design_point=dict(point),
                source_simulation_id=run.run_id,
            )
        )
    return snapshots


__all__ = ["FieldSnapshot", "build_snapshot_matrix", "collect_field_snapshots_from_runs"]
