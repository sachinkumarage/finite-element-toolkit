"""`ReducedOrderModel`: the shared abstraction behind reduced-order techniques (Version 35).

.. code-block:: text

    ReducedOrderModel
    |-- fit()
    |-- reduce()
    |-- reconstruct()
    |-- predict()
    |-- error()
    `-- validate()

A reduced-order model approximates a high-dimensional full-field
solution ``u`` (e.g. every nodal displacement DOF) by a much smaller
set of reduced coordinates ``q``:

.. code-block:: text

    u ~ V q

where ``V`` is a reduced basis (:mod:`femtoolkit.surrogate.rom.pod`
builds one via Proper Orthogonal Decomposition). :meth:`reduce` is the
projection ``u -> q``; :meth:`reconstruct` is the approximation
``q -> V q``; :meth:`predict` is the full round trip
``u -> reduce -> reconstruct -> u_hat``, used to measure how much
information the reduced basis actually retains.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from femtoolkit.exceptions import ValidationError

DEFAULT_RECONSTRUCTION_EPSILON = 1e-12
"""The small constant added to a reconstruction relative-error denominator."""


@dataclass(frozen=True)
class ReconstructionError:
    """The error between a full-field solution and its reduced-basis reconstruction.

    Attributes:
        absolute_error: The L2 norm of ``u - u_hat``.
        relative_error: ``absolute_error / (||u|| + epsilon)``.
        l2_norm_actual: The L2 norm of ``u`` itself.
        l2_norm_reconstructed: The L2 norm of ``u_hat``.
    """

    absolute_error: float
    relative_error: float
    l2_norm_actual: float
    l2_norm_reconstructed: float


def compute_reconstruction_error(
    actual: np.ndarray, reconstructed: np.ndarray, epsilon: float = DEFAULT_RECONSTRUCTION_EPSILON
) -> ReconstructionError:
    """Compute the :class:`ReconstructionError` between one full-field vector and its approximation.

    Args:
        actual: The true full-field vector ``u``.
        reconstructed: The reduced-basis approximation ``u_hat``.
        epsilon: A small constant preventing division by zero for a
            (degenerate) all-zero field.

    Returns:
        A :class:`ReconstructionError`.
    """
    actual = np.asarray(actual, dtype=float)
    reconstructed = np.asarray(reconstructed, dtype=float)
    absolute = float(np.linalg.norm(actual - reconstructed))
    norm_actual = float(np.linalg.norm(actual))
    return ReconstructionError(
        absolute_error=absolute,
        relative_error=absolute / (norm_actual + epsilon),
        l2_norm_actual=norm_actual,
        l2_norm_reconstructed=float(np.linalg.norm(reconstructed)),
    )


@dataclass(frozen=True)
class RomValidationReport:
    """The outcome of validating a reduced-order model against held-out full-field snapshots.

    Attributes:
        n_snapshots: How many held-out snapshots were checked.
        rank: The reduced basis size used.
        captured_energy: The fraction of training-snapshot energy the
            basis captures (see :mod:`femtoolkit.surrogate.rom.pod`).
        per_snapshot_errors: Each held-out snapshot's :class:`ReconstructionError`.
        max_relative_error: The largest observed relative error.
        mean_relative_error: The average observed relative error.
        worst_snapshot_index: The index (into ``per_snapshot_errors``) of
            the worst-represented held-out snapshot.
    """

    n_snapshots: int
    rank: int
    captured_energy: float
    per_snapshot_errors: list[ReconstructionError]
    max_relative_error: float
    mean_relative_error: float
    worst_snapshot_index: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable representation of this report."""
        return {
            "n_snapshots": self.n_snapshots,
            "rank": self.rank,
            "captured_energy": self.captured_energy,
            "max_relative_error": self.max_relative_error,
            "mean_relative_error": self.mean_relative_error,
            "worst_snapshot_index": self.worst_snapshot_index,
            "per_snapshot_relative_errors": [e.relative_error for e in self.per_snapshot_errors],
        }


class ReducedOrderModel(ABC):
    """The shared abstraction behind every reduced-order technique in this package."""

    name: str = "rom"

    @abstractmethod
    def fit(self, snapshot_matrix: np.ndarray) -> ReducedOrderModel:
        """Build the reduced basis from a snapshot matrix (columns = full-field snapshots)."""
        raise NotImplementedError

    @abstractmethod
    def reduce(self, u: np.ndarray) -> np.ndarray:
        """Project one or more full-field vectors onto the reduced coordinates ``q``."""
        raise NotImplementedError

    @abstractmethod
    def reconstruct(self, q: np.ndarray) -> np.ndarray:
        """Expand reduced coordinates ``q`` back into an approximate full-field vector."""
        raise NotImplementedError

    def predict(self, u: np.ndarray) -> np.ndarray:
        """Project and immediately reconstruct ``u`` -- the full reduce/reconstruct round trip."""
        return self.reconstruct(self.reduce(u))

    def error(self, u: np.ndarray) -> ReconstructionError:
        """Compute the :class:`ReconstructionError` for one full-field vector ``u``."""
        return compute_reconstruction_error(u, self.predict(u))

    def validate(self, held_out_snapshots: np.ndarray) -> RomValidationReport:
        """Validate this fitted model against held-out full-field snapshots (columns).

        Training reconstruction accuracy does not imply generalization
        -- this method exists specifically so a caller checks held-out
        snapshots the basis never saw during :meth:`fit`.

        Args:
            held_out_snapshots: Held-out full-field snapshots, one per
                column, same row dimension as the training data.

        Returns:
            A :class:`RomValidationReport`.

        Raises:
            ValidationError: If ``held_out_snapshots`` is empty.
        """
        held_out_snapshots = np.asarray(held_out_snapshots, dtype=float)
        if held_out_snapshots.ndim == 1:
            held_out_snapshots = held_out_snapshots.reshape(-1, 1)
        if held_out_snapshots.shape[1] == 0:
            raise ValidationError("validate() requires at least one held-out snapshot.")

        errors = [self.error(held_out_snapshots[:, i]) for i in range(held_out_snapshots.shape[1])]
        relative = [e.relative_error for e in errors]
        return RomValidationReport(
            n_snapshots=held_out_snapshots.shape[1],
            rank=self.rank,
            captured_energy=self.captured_energy,
            per_snapshot_errors=errors,
            max_relative_error=max(relative),
            mean_relative_error=float(np.mean(relative)),
            worst_snapshot_index=int(np.argmax(relative)),
        )

    @property
    @abstractmethod
    def rank(self) -> int:
        """The number of modes retained in the reduced basis."""
        raise NotImplementedError

    @property
    @abstractmethod
    def captured_energy(self) -> float:
        """The fraction of training-snapshot energy the reduced basis captures."""
        raise NotImplementedError


__all__ = [
    "DEFAULT_RECONSTRUCTION_EPSILON",
    "ReconstructionError",
    "ReducedOrderModel",
    "RomValidationReport",
    "compute_reconstruction_error",
]
