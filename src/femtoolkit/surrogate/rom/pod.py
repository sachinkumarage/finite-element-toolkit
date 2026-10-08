"""Proper Orthogonal Decomposition (POD) (Version 35).

**The snapshot matrix.**

.. math::

    X = \\begin{bmatrix} | & | & & | \\\\ u_1 & u_2 & \\cdots & u_m \\\\
    | & | & & | \\end{bmatrix}

Each column ``u_k`` is one full-field high-fidelity solution (e.g.
every nodal displacement DOF from one FEA run). Its singular value
decomposition,

.. math::

    X = U \\Sigma V^T,

gives an orthonormal basis (the left singular vectors, columns of
``U``) ordered by how much of the snapshot set's variance each
direction explains. Keeping only the first ``r`` columns of ``U`` as
the reduced basis ``V_r`` (this module's :attr:`PODModel.basis`) is the
standard POD truncation: :math:`u \\approx V_r q`, with the reduced
coordinates :math:`q = V_r^T u`.

**Energy captured.** The energy captured by keeping ``r`` modes is

.. math::

    E_r = \\frac{\\sum_{i=1}^{r} \\sigma_i^2}{\\sum_{i=1}^{n} \\sigma_i^2}

-- the fraction of the snapshot set's total "variance" (in the sense
of the Frobenius norm of ``X``) the truncated basis reproduces exactly
on the training snapshots themselves. A basis can be selected either by
a **fixed rank** (``rank=r``) or by the **minimum rank that reaches an
energy threshold** (``energy_threshold=0.999``), never both at once.
"""

from __future__ import annotations

import numpy as np

from femtoolkit.exceptions import InvalidReducedBasisConfigurationError, ValidationError
from femtoolkit.surrogate.rom.base import ReducedOrderModel


class PODModel(ReducedOrderModel):
    """A Proper Orthogonal Decomposition reduced-order model.

    Attributes:
        singular_values: The snapshot matrix's singular values, largest
            first, once fitted.
        basis: The retained reduced basis ``V_r``, shape
            ``(n_dofs, selected_rank)``, once fitted.
        total_modes: The number of modes available before truncation
            (``min(n_dofs, n_snapshots)``).
        selected_modes: The number of modes actually retained.
    """

    name = "pod"

    def __init__(self) -> None:
        self.singular_values: np.ndarray | None = None
        self.basis: np.ndarray | None = None
        self.total_modes: int = 0
        self.selected_modes: int = 0
        self._total_energy: float = 0.0

    def fit(
        self,
        snapshot_matrix: np.ndarray,
        rank: int | None = None,
        energy_threshold: float | None = None,
    ) -> PODModel:
        """Compute the SVD of ``snapshot_matrix`` and select a reduced basis.

        Args:
            snapshot_matrix: The snapshot matrix ``X``, one full-field
                snapshot per column.
            rank: A fixed number of modes to retain. Mutually exclusive
                with ``energy_threshold``. ``None`` with
                ``energy_threshold`` also ``None`` retains every
                available mode (no truncation).
            energy_threshold: Retain the minimum number of modes whose
                cumulative captured energy (see the module docstring)
                reaches this threshold, in ``(0, 1]``. Mutually
                exclusive with ``rank``.

        Returns:
            ``self``, for chaining.

        Raises:
            ValidationError: If ``snapshot_matrix`` is empty, or both
                ``rank`` and ``energy_threshold`` are given.
            InvalidReducedBasisConfigurationError: If ``rank`` is not a
                positive integer within the available mode count, or
                ``energy_threshold`` does not lie in ``(0, 1]``.
        """
        snapshot_matrix = np.asarray(snapshot_matrix, dtype=float)
        if snapshot_matrix.ndim != 2 or snapshot_matrix.size == 0:
            raise ValidationError("fit() requires a non-empty 2D snapshot matrix.")
        if rank is not None and energy_threshold is not None:
            raise ValidationError("Specify at most one of rank/energy_threshold, not both.")

        u, singular_values, _vt = np.linalg.svd(snapshot_matrix, full_matrices=False)
        total_modes = singular_values.size
        total_energy = float(np.sum(singular_values**2))

        if rank is not None:
            if not (isinstance(rank, int) and 1 <= rank <= total_modes):
                raise InvalidReducedBasisConfigurationError(
                    f"rank must be an integer in [1, {total_modes}], got {rank!r}."
                )
            selected = rank
        elif energy_threshold is not None:
            if not (0.0 < energy_threshold <= 1.0):
                raise InvalidReducedBasisConfigurationError(
                    f"energy_threshold must lie in (0, 1], got {energy_threshold!r}."
                )
            if total_energy <= 0.0:
                selected = total_modes
            else:
                cumulative_energy = np.cumsum(singular_values**2) / total_energy
                selected = int(np.searchsorted(cumulative_energy, energy_threshold) + 1)
                selected = min(selected, total_modes)
        else:
            selected = total_modes

        self.singular_values = singular_values
        self.basis = u[:, :selected]
        self.total_modes = total_modes
        self.selected_modes = selected
        self._total_energy = total_energy
        return self

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called successfully."""
        return self.basis is not None

    def _require_fitted(self) -> None:
        if self.basis is None:
            raise ValidationError("PODModel must be fit() before use.")

    def reduce(self, u: np.ndarray) -> np.ndarray:
        """Project full-field vector(s) onto reduced coordinates: ``q = V_r^T u``."""
        self._require_fitted()
        u = np.asarray(u, dtype=float)
        return self.basis.T @ u

    def reconstruct(self, q: np.ndarray) -> np.ndarray:
        """Expand reduced coordinates back to the full field: ``u_hat = V_r q``."""
        self._require_fitted()
        q = np.asarray(q, dtype=float)
        return self.basis @ q

    @property
    def rank(self) -> int:
        """The number of modes retained in the reduced basis."""
        self._require_fitted()
        return self.selected_modes

    @property
    def captured_energy(self) -> float:
        """The fraction of training-snapshot energy the reduced basis captures (:math:`E_r`)."""
        self._require_fitted()
        if self._total_energy <= 0.0:
            return 1.0
        return float(np.sum(self.singular_values[: self.selected_modes] ** 2) / self._total_energy)

    @property
    def discarded_energy(self) -> float:
        """``1.0 - captured_energy`` -- the energy fraction lost by truncation."""
        return 1.0 - self.captured_energy

    def energy_spectrum(self) -> np.ndarray:
        """The cumulative captured-energy fraction for every possible rank, ``1..total_modes``.

        Useful for a POD energy-convergence plot
        (:mod:`femtoolkit.surrogate.workflows` reporting).
        """
        self._require_fitted()
        if self._total_energy <= 0.0:
            return np.ones(self.total_modes)
        return np.cumsum(self.singular_values**2) / self._total_energy


__all__ = ["PODModel"]
