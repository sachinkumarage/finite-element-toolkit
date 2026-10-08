"""Snapshot collection and the `SnapshotDataset` abstraction (Version 35).

**Engineering concept.** The high-fidelity FEA model is the function

.. code-block:: text

    y = f(x)

where ``x`` is a vector of input/design variables (Young's modulus,
thickness, load, ...) and ``y`` is a vector of scalar engineering
responses (maximum displacement, maximum stress, reaction force, a
natural frequency, ...). A *snapshot* is one evaluation of ``f``: one
input vector paired with the responses it produced. A *snapshot
dataset* is simply a collection of such pairs, structured so a
surrogate model (:mod:`femtoolkit.surrogate.models`) can be trained on
it without ever risking the input and output of one evaluation
becoming misaligned with another's.

.. code-block:: text

    High-Fidelity FEA
            |
            v
    Snapshot Data  (this module)
            |
            v
    Reduced Basis / Surrogate Model
            |
            v
    Fast Prediction

This module performs no FEA computation itself --
:func:`collect_snapshots_from_runs` only extracts scalar responses from
:class:`~femtoolkit.runs.models.SimulationRun` objects that were
already produced by the existing simulation infrastructure (Version 30
:class:`~femtoolkit.runs.manager.SimulationRunManager`, optionally run
in parallel through Version 34
:func:`~femtoolkit.orchestration.simulation.evaluate_simulation_batch`).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from femtoolkit.exceptions import (
    InconsistentSnapshotError,
    InsufficientSnapshotsError,
    ValidationError,
)
from femtoolkit.studies.extractors import Extractor


def _validate_finite_mapping(values: dict[str, float], names: list[str], role: str) -> None:
    missing = [name for name in names if name not in values]
    if missing:
        raise InconsistentSnapshotError(f"Snapshot is missing {role} value(s) for: {missing}.")
    extra = [name for name in values if name not in names]
    if extra:
        raise InconsistentSnapshotError(f"Snapshot carries unexpected {role} value(s): {extra}.")
    for name in names:
        value = float(values[name])
        if not np.isfinite(value):
            raise InconsistentSnapshotError(
                f"Snapshot {role} {name!r} has a non-finite value ({value!r})."
            )


@dataclass
class Snapshot:
    """One high-fidelity evaluation: a design point and the responses it produced.

    Attributes:
        snapshot_id: A unique identifier for this snapshot (e.g. the
            originating :class:`~femtoolkit.runs.models.SimulationRun`'s
            ``run_id``).
        inputs: This evaluation's input/design-variable values, keyed
            by name.
        outputs: This evaluation's response values, keyed by name.
        source_simulation_id: The originating simulation run's
            identifier, or ``None`` if this snapshot was not produced
            from a tracked :class:`~femtoolkit.runs.models.SimulationRun`
            (e.g. a synthetic benchmark point).
        units: A units string per input/output name, purely descriptive.
        metadata: Free-form additional information (e.g. which scenario
            or Monte Carlo sample produced this point).
        created_at: ISO-8601 UTC timestamp when this snapshot was recorded.
    """

    snapshot_id: str
    inputs: dict[str, float]
    outputs: dict[str, float]
    source_simulation_id: str | None = None
    units: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable representation of this snapshot."""
        return {
            "snapshot_id": self.snapshot_id,
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
            "source_simulation_id": self.source_simulation_id,
            "units": dict(self.units),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Snapshot:
        """Reconstruct a :class:`Snapshot` from :meth:`to_dict`'s output."""
        return cls(
            snapshot_id=data["snapshot_id"],
            inputs=dict(data["inputs"]),
            outputs=dict(data["outputs"]),
            source_simulation_id=data.get("source_simulation_id"),
            units=dict(data.get("units", {})),
            metadata=dict(data.get("metadata", {})),
            created_at=data.get("created_at", ""),
        )


@dataclass
class DatasetSplit:
    """A reproducible train/validation/test partition of a :class:`SnapshotDataset`.

    Attributes:
        train: The training subset.
        validation: The validation subset (may be empty for a very
            small dataset -- see :func:`split_dataset`).
        test: The held-out test subset.
        seed: The random seed used to shuffle snapshots before splitting.
    """

    train: SnapshotDataset
    validation: SnapshotDataset
    test: SnapshotDataset
    seed: int | None


@dataclass
class SnapshotDataset:
    """A structured, order-preserving collection of :class:`Snapshot` objects.

    .. code-block:: text

        SnapshotDataset
        |-- inputs
        |-- outputs
        |-- metadata
        |-- feature_names
        `-- response_names

    Every snapshot is validated against :attr:`feature_names`/
    :attr:`response_names` as it is added, so the input-output
    correspondence within the dataset can never silently become
    misaligned -- a snapshot missing a declared name, carrying an
    unexpected one, or carrying a non-finite value is rejected
    immediately (see :exc:`~femtoolkit.exceptions.InconsistentSnapshotError`).

    Attributes:
        dataset_id: A unique, stable identifier for this dataset's lineage.
        feature_names: The input variable names, in a fixed order.
        response_names: The output/response names, in a fixed order.
        snapshots: Every snapshot in this dataset, in insertion order.
        dataset_version: This dataset's version number -- ``1`` for a
            freshly created dataset, incremented by
            :meth:`with_additional_snapshots` (never by mutating
            ``snapshots`` directly on an existing instance in place; see
            that method's docstring for why the previous version is
            never silently overwritten).
        source_simulations: The identifiers of every simulation that
            contributed a snapshot to this dataset.
        created_at: ISO-8601 UTC timestamp when this dataset version was built.
        description: A short, human-readable description.
    """

    feature_names: list[str]
    response_names: list[str]
    snapshots: list[Snapshot] = field(default_factory=list)
    dataset_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dataset_version: int = 1
    source_simulations: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    description: str = ""

    def __post_init__(self) -> None:
        if not self.feature_names:
            raise ValidationError("SnapshotDataset requires at least one feature name.")
        if not self.response_names:
            raise ValidationError("SnapshotDataset requires at least one response name.")
        overlap = set(self.feature_names) & set(self.response_names)
        if overlap:
            raise ValidationError(
                f"Feature and response names must be disjoint; shared: {overlap}."
            )
        validated = list(self.snapshots)
        self.snapshots = []
        for snapshot in validated:
            self.add_snapshot(snapshot)

    def add_snapshot(self, snapshot: Snapshot) -> None:
        """Validate and append one snapshot to this dataset, in place.

        Args:
            snapshot: The snapshot to add.

        Raises:
            InconsistentSnapshotError: If ``snapshot``'s inputs/outputs
                do not exactly match :attr:`feature_names`/
                :attr:`response_names`, or contain a non-finite value.
        """
        _validate_finite_mapping(snapshot.inputs, self.feature_names, "input")
        _validate_finite_mapping(snapshot.outputs, self.response_names, "output")
        self.snapshots.append(snapshot)
        source_id = snapshot.source_simulation_id
        if source_id and source_id not in self.source_simulations:
            self.source_simulations.append(source_id)

    def with_additional_snapshots(self, new_snapshots: list[Snapshot]) -> SnapshotDataset:
        """Return a *new* dataset with ``new_snapshots`` appended, bumping the version.

        This is the only supported way to grow a dataset once it may
        already be referenced elsewhere (a trained surrogate's
        persisted metadata, a report): the returned object is a
        distinct instance with ``dataset_version`` incremented
        (``v1 -> v2``); ``self`` and everything that already holds a
        reference to it are left completely unchanged. A dataset
        version therefore never silently loses the identity of the
        snapshot set a prior surrogate/ROM model was trained against.

        Args:
            new_snapshots: The snapshots to add.

        Returns:
            A new :class:`SnapshotDataset` sharing ``dataset_id`` but
            with ``dataset_version = self.dataset_version + 1``.
        """
        combined = list(self.snapshots) + list(new_snapshots)
        updated = SnapshotDataset(
            feature_names=list(self.feature_names),
            response_names=list(self.response_names),
            snapshots=combined,
            dataset_id=self.dataset_id,
            dataset_version=self.dataset_version + 1,
            source_simulations=list(self.source_simulations),
            description=self.description,
        )
        return updated

    @property
    def n_samples(self) -> int:
        """How many snapshots this dataset holds."""
        return len(self.snapshots)

    @property
    def n_features(self) -> int:
        """How many input/feature dimensions this dataset has."""
        return len(self.feature_names)

    @property
    def n_responses(self) -> int:
        """How many output/response dimensions this dataset has."""
        return len(self.response_names)

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(X, Y)`` arrays in :attr:`feature_names`/:attr:`response_names` order.

        Returns:
            ``X`` has shape ``(n_samples, n_features)``; ``Y`` has shape
            ``(n_samples, n_responses)``. Row ``i`` of both always comes
            from :attr:`snapshots`'s ``i``-th entry -- the ordering that
            keeps every downstream model's input-output correspondence
            correct.
        """
        n = self.n_samples
        x = np.zeros((n, self.n_features), dtype=float)
        y = np.zeros((n, self.n_responses), dtype=float)
        for row, snapshot in enumerate(self.snapshots):
            x[row, :] = [snapshot.inputs[name] for name in self.feature_names]
            y[row, :] = [snapshot.outputs[name] for name in self.response_names]
        return x, y

    def split(
        self,
        train_fraction: float = 0.7,
        validation_fraction: float = 0.15,
        seed: int | None = None,
    ) -> DatasetSplit:
        """Reproducibly partition this dataset into train/validation/test subsets.

        Args:
            train_fraction: Fraction of snapshots assigned to training.
            validation_fraction: Fraction assigned to validation (the
                remainder goes to test). For a small dataset
                (fewer than 10 snapshots), the validation subset is
                allowed to be empty rather than forcing an unreasonably
                aggressive split -- see the module-level guidance on
                sensible defaults for small engineering datasets.
            seed: A random seed; the same seed reproduces the same split.

        Returns:
            A :class:`DatasetSplit`.

        Raises:
            ValidationError: If the fractions are not both in ``(0, 1)``
                and summing to at most ``1``, or if the dataset has
                fewer than 3 snapshots (too few to form any meaningful
                train/test separation).
            InsufficientSnapshotsError: If there are not enough
                snapshots to split at all.
        """
        if not (0.0 < train_fraction < 1.0) or not (0.0 <= validation_fraction < 1.0):
            raise ValidationError(
                f"train_fraction must lie in (0, 1) and validation_fraction in [0, 1); got "
                f"train_fraction={train_fraction}, validation_fraction={validation_fraction}."
            )
        if train_fraction + validation_fraction > 1.0:
            raise ValidationError("train_fraction + validation_fraction must not exceed 1.0.")
        if self.n_samples < 3:
            raise InsufficientSnapshotsError(
                f"Dataset {self.dataset_id!r} has only {self.n_samples} snapshot(s); at least "
                "3 are required to form a train/test split."
            )

        rng = np.random.default_rng(seed)
        order = rng.permutation(self.n_samples)

        n_train = max(1, round(self.n_samples * train_fraction))
        n_val = round(self.n_samples * validation_fraction) if self.n_samples >= 10 else 0
        n_train = min(n_train, self.n_samples - 1)
        n_val = min(n_val, self.n_samples - n_train - 1)
        n_val = max(n_val, 0)

        train_idx = order[:n_train]
        val_idx = order[n_train : n_train + n_val]
        test_idx = order[n_train + n_val :]
        if test_idx.size == 0:
            # Guarantee at least one held-out test snapshot: a surrogate must never be
            # evaluated only on its own training data.
            test_idx = train_idx[-1:]
            train_idx = train_idx[:-1]

        def _subset(indices: np.ndarray) -> SnapshotDataset:
            return SnapshotDataset(
                feature_names=list(self.feature_names),
                response_names=list(self.response_names),
                snapshots=[self.snapshots[i] for i in indices],
                dataset_id=self.dataset_id,
                dataset_version=self.dataset_version,
                source_simulations=list(self.source_simulations),
                description=self.description,
            )

        return DatasetSplit(
            train=_subset(train_idx), validation=_subset(val_idx), test=_subset(test_idx), seed=seed
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable representation of this dataset."""
        return {
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "feature_names": list(self.feature_names),
            "response_names": list(self.response_names),
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
            "source_simulations": list(self.source_simulations),
            "created_at": self.created_at,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SnapshotDataset:
        """Reconstruct a :class:`SnapshotDataset` from :meth:`to_dict`'s output."""
        return cls(
            feature_names=list(data["feature_names"]),
            response_names=list(data["response_names"]),
            snapshots=[Snapshot.from_dict(item) for item in data.get("snapshots", [])],
            dataset_id=data.get("dataset_id", str(uuid.uuid4())),
            dataset_version=data.get("dataset_version", 1),
            source_simulations=list(data.get("source_simulations", [])),
            created_at=data.get("created_at", ""),
            description=data.get("description", ""),
        )


def collect_snapshots_from_runs(
    runs: list,
    design_points: list[dict[str, float]],
    response_extractors: dict[str, Extractor],
    *,
    feature_units: dict[str, str] | None = None,
    response_units: dict[str, str] | None = None,
) -> SnapshotDataset:
    """Build a :class:`SnapshotDataset` from already-executed simulation runs.

    This is the bridge between the existing simulation-execution
    infrastructure (Version 30
    :class:`~femtoolkit.runs.manager.SimulationRunManager`, Version 34
    :func:`~femtoolkit.orchestration.simulation.evaluate_simulation_batch`,
    a Version 30 parameter study, or a Version 31 Monte Carlo study) and
    the surrogate-modeling framework -- it performs no simulation of its
    own, only extracts named scalar responses
    (:mod:`femtoolkit.studies.extractors`) from runs that already
    finished.

    Args:
        runs: The executed :class:`~femtoolkit.runs.models.SimulationRun`
            objects, one per design point, in the same order as
            ``design_points``.
        design_points: Each run's input/design-variable values, keyed
            by feature name, same order and length as ``runs``.
        response_extractors: The named result-quantity extractors (see
            :data:`~femtoolkit.studies.extractors.EXTRACTORS`) to apply
            to every completed run, keyed by response name.
        feature_units: An optional units string per feature name.
        response_units: An optional units string per response name.

    Returns:
        A :class:`SnapshotDataset` holding one snapshot per run that
        completed successfully *and* for which every extractor returned
        a usable value. A run that failed, or from which a response
        could not be extracted, contributes no snapshot -- it is never
        silently fabricated.

    Raises:
        ValidationError: If ``runs`` and ``design_points`` differ in
            length, or no usable snapshot could be built at all.
    """
    from femtoolkit.runs.models import RunStatus

    if len(runs) != len(design_points):
        raise ValidationError(
            f"runs ({len(runs)}) and design_points ({len(design_points)}) must have the "
            "same length."
        )

    feature_names = sorted({name for point in design_points for name in point})
    response_names = sorted(response_extractors)
    dataset = SnapshotDataset(feature_names=feature_names, response_names=response_names)

    units: dict[str, str] = {}
    units.update(feature_units or {})
    units.update(response_units or {})

    for run, point in zip(runs, design_points, strict=True):
        if run.status != RunStatus.COMPLETED:
            continue
        outputs: dict[str, float] = {}
        for name, extractor in response_extractors.items():
            value = extractor(run)
            if value is None:
                outputs = {}
                break
            outputs[name] = value
        if not outputs:
            continue
        known_names = feature_names + response_names
        dataset.add_snapshot(
            Snapshot(
                snapshot_id=run.run_id,
                inputs=dict(point),
                outputs=outputs,
                source_simulation_id=run.run_id,
                units={name: value for name, value in units.items() if name in known_names},
            )
        )

    if dataset.n_samples == 0:
        raise ValidationError(
            "No usable snapshot could be built: every run either failed or was missing a "
            "requested response quantity."
        )
    return dataset


__all__ = [
    "DatasetSplit",
    "Snapshot",
    "SnapshotDataset",
    "collect_snapshots_from_runs",
]
