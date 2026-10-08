"""A lightweight paired low-/high-fidelity dataset (Version 37).

.. code-block:: text

    x1 -> Low Result -> High Result
    x2 -> Low Result -> High Result
    x3 -> Low Result -> High Result

A sample is "paired" once both a low-fidelity and a high-fidelity result exist for
the same design point -- only paired samples carry a meaningful discrepancy
``delta(x) = y_H(x) - y_L(x)``. This module stores samples and computes
discrepancies; it is not a database, and converting a paired sample set into a
trainable dataset reuses :class:`~femtoolkit.surrogate.datasets.Snapshot`/
:class:`~femtoolkit.surrogate.datasets.SnapshotDataset` (Version 35) directly
rather than inventing a second dataset abstraction.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from femtoolkit.exceptions import IncompatibleFidelityResultError, ValidationError
from femtoolkit.surrogate.datasets import Snapshot, SnapshotDataset


def compute_discrepancy(
    low_result: dict[str, float], high_result: dict[str, float]
) -> dict[str, float]:
    """Compute ``delta(x) = y_H(x) - y_L(x)`` for one paired sample.

    Args:
        low_result: The low-fidelity response value(s), keyed by response name.
        high_result: The high-fidelity response value(s), keyed by response name.

    Returns:
        The discrepancy, keyed by response name.

    Raises:
        IncompatibleFidelityResultError: If ``low_result`` and ``high_result`` do
            not carry exactly the same response names, or either contains a
            non-finite value.
    """
    if set(low_result) != set(high_result):
        raise IncompatibleFidelityResultError(
            f"Low- and high-fidelity results must cover the same responses; got "
            f"low={sorted(low_result)}, high={sorted(high_result)}."
        )
    discrepancy: dict[str, float] = {}
    for name in low_result:
        low_value = float(low_result[name])
        high_value = float(high_result[name])
        if not (np.isfinite(low_value) and np.isfinite(high_value)):
            raise IncompatibleFidelityResultError(
                f"Response {name!r} has a non-finite low/high value: "
                f"low={low_value!r}, high={high_value!r}."
            )
        discrepancy[name] = high_value - low_value
    return discrepancy


@dataclass
class MultiFidelitySample:
    """One design point and its low-/high-fidelity result(s).

    Attributes:
        sample_id: A unique identifier for this sample.
        inputs: The design point's input values, keyed by name.
        low_result: The low-fidelity response value(s), keyed by response name.
        high_result: The high-fidelity response value(s), keyed by response name,
            or ``None`` if no high-fidelity evaluation has been run for this point
            yet (an unpaired sample).
        created_at: ISO-8601 UTC timestamp when this sample was recorded.
    """

    sample_id: str
    inputs: dict[str, float]
    low_result: dict[str, float]
    high_result: dict[str, float] | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def is_paired(self) -> bool:
        """Whether this sample has both a low- and a high-fidelity result."""
        return self.high_result is not None

    def discrepancy(self) -> dict[str, float]:
        """The discrepancy ``delta(x) = y_H(x) - y_L(x)`` for this sample.

        Raises:
            ValidationError: If this sample is not paired.
            IncompatibleFidelityResultError: See :func:`compute_discrepancy`.
        """
        if self.high_result is None:
            raise ValidationError(
                f"Sample {self.sample_id!r} has no high-fidelity result; it is not paired."
            )
        return compute_discrepancy(self.low_result, self.high_result)


@dataclass
class MultiFidelityDataset:
    """A collection of :class:`MultiFidelitySample` objects, paired and unpaired.

    Attributes:
        feature_names: The input variable names, in a fixed order.
        response_names: The response names every low-fidelity (and, once paired,
            high-fidelity) result must cover.
        samples: Every sample in this dataset, in insertion order.
        dataset_id: A unique, stable identifier for this dataset.
    """

    feature_names: list[str]
    response_names: list[str]
    samples: list[MultiFidelitySample] = field(default_factory=list)
    dataset_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if not self.feature_names:
            raise ValidationError("MultiFidelityDataset requires at least one feature name.")
        if not self.response_names:
            raise ValidationError("MultiFidelityDataset requires at least one response name.")
        validated = list(self.samples)
        self.samples = []
        for sample in validated:
            self.add_sample(sample)

    def add_sample(self, sample: MultiFidelitySample) -> None:
        """Validate and append one sample to this dataset, in place.

        Args:
            sample: The sample to add.

        Raises:
            ValidationError: If ``sample``'s inputs or low-fidelity result do not
                exactly match :attr:`feature_names`/:attr:`response_names`.
            IncompatibleFidelityResultError: If ``sample`` is paired but its
                high-fidelity result does not match :attr:`response_names`.
        """
        _validate_keys(sample.inputs, self.feature_names, "input")
        _validate_keys(sample.low_result, self.response_names, "low-fidelity response")
        if sample.high_result is not None:
            try:
                _validate_keys(sample.high_result, self.response_names, "high-fidelity response")
            except ValidationError as error:
                raise IncompatibleFidelityResultError(str(error)) from error
        self.samples.append(sample)

    @property
    def n_samples(self) -> int:
        """How many samples this dataset holds (paired and unpaired)."""
        return len(self.samples)

    def paired_samples(self) -> list[MultiFidelitySample]:
        """Every sample that has both a low- and a high-fidelity result."""
        return [sample for sample in self.samples if sample.is_paired]

    def unpaired_samples(self) -> list[MultiFidelitySample]:
        """Every sample that still only has a low-fidelity result."""
        return [sample for sample in self.samples if not sample.is_paired]

    def discrepancies(self) -> dict[str, np.ndarray]:
        """The discrepancy for every paired sample, per response.

        Returns:
            Each response name mapped to a NumPy array of discrepancy values, in
            the same order as :meth:`paired_samples`.
        """
        paired = self.paired_samples()
        result: dict[str, np.ndarray] = {
            name: np.zeros(len(paired)) for name in self.response_names
        }
        for row, sample in enumerate(paired):
            delta = sample.discrepancy()
            for name in self.response_names:
                result[name][row] = delta[name]
        return result

    def to_discrepancy_dataset(self) -> SnapshotDataset:
        """Convert every paired sample into a Version 35 ``x -> delta(x)`` dataset.

        Returns:
            A :class:`~femtoolkit.surrogate.datasets.SnapshotDataset` whose
            response values are discrepancies, ready to train a discrepancy
            surrogate with :func:`~femtoolkit.surrogate.workflows.training.train_surrogate`
            directly.

        Raises:
            ValidationError: If no sample in this dataset is paired yet.
        """
        paired = self.paired_samples()
        if not paired:
            raise ValidationError(
                "to_discrepancy_dataset() requires at least one paired sample "
                "(a sample with both a low- and a high-fidelity result)."
            )
        snapshots = [
            Snapshot(
                snapshot_id=sample.sample_id,
                inputs=dict(sample.inputs),
                outputs=sample.discrepancy(),
            )
            for sample in paired
        ]
        return SnapshotDataset(
            feature_names=list(self.feature_names),
            response_names=list(self.response_names),
            snapshots=snapshots,
            dataset_id=f"{self.dataset_id}-discrepancy",
        )


def _validate_keys(values: dict[str, float], expected: list[str], role: str) -> None:
    missing = [name for name in expected if name not in values]
    if missing:
        raise ValidationError(f"Sample is missing {role} value(s) for: {missing}.")
    extra = [name for name in values if name not in expected]
    if extra:
        raise ValidationError(f"Sample carries unexpected {role} value(s): {extra}.")


__all__ = [
    "MultiFidelityDataset",
    "MultiFidelitySample",
    "compute_discrepancy",
]
