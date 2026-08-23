"""Rayleigh (proportional) damping.

Unlike mass and stiffness, damping in a real structure has no single
agreed-upon physical model -- energy dissipation comes from many
different mechanisms (material friction, joint slip, air resistance)
that are individually difficult to characterize. **Rayleigh damping**
sidesteps this by constructing a damping matrix as a linear combination
of the mass and stiffness matrices already available:

.. code-block:: text

    C = alpha * M + beta * K

``alpha`` (the mass-proportional coefficient) damps low-frequency
(long-wavelength, rigid-body-like) motion more strongly; ``beta`` (the
stiffness-proportional coefficient) damps high-frequency (short-
wavelength) motion more strongly. Neither coefficient has a direct
physical unit interpretation on its own -- they are calibrated (e.g. to
match a target damping ratio at one or two frequencies of interest),
not measured directly. This is a mathematically convenient, widely used
engineering approximation, not a first-principles damping model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from femtoolkit.exceptions import ValidationError


@dataclass(frozen=True)
class RayleighDamping:
    """Mass- and stiffness-proportional (Rayleigh) damping.

    Attributes:
        alpha: Mass-proportional damping coefficient, in 1/s. Must be
            non-negative and finite.
        beta: Stiffness-proportional damping coefficient, in s. Must be
            non-negative and finite.

    Raises:
        ValidationError: If ``alpha`` or ``beta`` is negative or not
            finite.

    Example:
        >>> damping = RayleighDamping(alpha=0.01, beta=0.0001)
        >>> c = damping.damping_matrix(mass, stiffness)
    """

    alpha: float = 0.0
    beta: float = 0.0

    def __post_init__(self) -> None:
        """Validate the damping coefficients immediately after construction.

        Raises:
            ValidationError: If ``alpha`` or ``beta`` is negative or not
                finite.
        """
        if not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValidationError(f"RayleighDamping alpha must be non-negative, got {self.alpha}.")
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValidationError(f"RayleighDamping beta must be non-negative, got {self.beta}.")

    def damping_matrix(self, mass: np.ndarray, stiffness: np.ndarray) -> np.ndarray:
        """Build the damping matrix, ``C = alpha * M + beta * K``.

        Args:
            mass: The global (or element) mass matrix.
            stiffness: The global (or element) stiffness matrix, the same
                shape as ``mass``.

        Returns:
            A NumPy array the same shape as ``mass``/``stiffness``.

        Raises:
            ValidationError: If ``mass`` and ``stiffness`` do not have
                the same shape.
        """
        if mass.shape != stiffness.shape:
            raise ValidationError(
                f"mass and stiffness must have the same shape, got {mass.shape} and "
                f"{stiffness.shape}."
            )
        return self.alpha * mass + self.beta * stiffness
