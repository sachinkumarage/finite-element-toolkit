"""Damping models: Rayleigh (Version 11) and modal damping ratios (Version 12).

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

**Modal damping** (:class:`ModalDamping`) is the more direct alternative
used by modal superposition
(:func:`~femtoolkit.analysis.modal_superposition.modal_superposition`):
rather than building a full ``C`` matrix and hoping it happens to be
(at least approximately) proportional, the analyst specifies a damping
*ratio* ``zeta_i`` directly for each mode -- the standard practice in
modal-based dynamic analysis, since physically meaningful damping ratios
(typically a few percent of critical damping) are usually known or
assumed per mode, not derived from a matrix.
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


@dataclass(frozen=True)
class ModalDamping:
    """A modal damping ratio, constant across all modes or specified per mode.

    For mode ``i``, the classical damping ratio is

    .. code-block:: text

        zeta_i = c_i / (2 * sqrt(k_i * m_i))

    For a *mass-normalized* mode (``m_i = 1``, ``k_i = omega_i^2`` --
    see :func:`~femtoolkit.analysis.modal.mass_normalize_mode_shapes`),
    this simplifies to ``c_i = 2 * zeta_i * omega_i``, the modal damping
    coefficient :meth:`modal_damping_coefficients` returns.

    Attributes:
        damping_ratios: Either a single non-negative float applied to
            every mode, or a tuple of per-mode ratios (validated against
            the actual mode count when used, via :meth:`ratios_for`).
            A damping ratio of ``0.0`` is undamped; ``1.0`` is critically
            damped; values are typically well below ``1.0`` for real
            structures (a few percent is common), but larger values are
            not rejected outright since overdamped modal responses are
            still mathematically well defined.

    Raises:
        ValidationError: If any ratio is negative or not finite.

    Example:
        >>> constant = ModalDamping(damping_ratios=0.02)  # 2% for every mode
        >>> per_mode = ModalDamping(damping_ratios=(0.02, 0.03, 0.05))
    """

    damping_ratios: float | tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate every damping ratio immediately after construction.

        Raises:
            ValidationError: If any ratio is negative or not finite.
        """
        if isinstance(self.damping_ratios, (int, float)) and not isinstance(
            self.damping_ratios, bool
        ):
            self._validate_ratio(self.damping_ratios, index=None)
        else:
            for index, ratio in enumerate(self.damping_ratios):
                self._validate_ratio(ratio, index=index)

    @staticmethod
    def _validate_ratio(ratio: float, index: int | None) -> None:
        if not math.isfinite(ratio) or ratio < 0:
            location = "damping_ratios" if index is None else f"damping_ratios[{index}]"
            raise ValidationError(f"ModalDamping {location} must be non-negative, got {ratio}.")

    def ratios_for(self, num_modes: int) -> np.ndarray:
        """Return one damping ratio per mode, broadcasting a constant value if needed.

        Args:
            num_modes: Number of modes to produce a ratio for.

        Returns:
            A length-``num_modes`` array of damping ratios.

        Raises:
            ValidationError: If per-mode ratios were supplied and their
                count does not match ``num_modes``.
        """
        if isinstance(self.damping_ratios, (int, float)) and not isinstance(
            self.damping_ratios, bool
        ):
            return np.full(num_modes, float(self.damping_ratios))

        ratios = np.asarray(self.damping_ratios, dtype=float)
        if len(ratios) != num_modes:
            raise ValidationError(
                f"ModalDamping has {len(ratios)} per-mode ratios but {num_modes} modes "
                "were requested."
            )
        return ratios

    def modal_damping_coefficients(self, angular_frequencies: np.ndarray) -> np.ndarray:
        """Compute mass-normalized modal damping coefficients, ``c_i = 2 * zeta_i * omega_i``.

        Args:
            angular_frequencies: Natural circular frequency of each
                mode, in rad/s (mass-normalized modal mass is assumed to
                be exactly ``1.0`` per mode).

        Returns:
            A length-``len(angular_frequencies)`` array of modal damping
            coefficients.
        """
        ratios = self.ratios_for(len(angular_frequencies))
        return 2.0 * ratios * np.asarray(angular_frequencies, dtype=float)


def modal_damping_ratios_from_matrix(
    mass_normalized_mode_shapes: np.ndarray,
    damping_matrix: np.ndarray,
    angular_frequencies: np.ndarray,
) -> np.ndarray:
    """Derive an equivalent modal damping ratio from an existing damping matrix.

    .. code-block:: text

        zeta_i = (phi_i^T * C * phi_i) / (2 * omega_i)

    valid for mass-normalized mode shapes (``m_i = 1``). This only gives
    the *exact* damping ratio if ``damping_matrix`` is proportional to
    ``M`` and/or ``K`` (e.g. :class:`RayleighDamping`, which is exactly
    diagonalized by the mode shapes); for a general, non-proportional
    damping matrix, off-diagonal modal coupling terms are silently
    discarded and this is an approximation -- consistent with Version
    12's scope (see the module docstring: advanced non-proportional
    damping is out of scope).

    Args:
        mass_normalized_mode_shapes: Mode shape matrix, mass-normalized
            (see :func:`~femtoolkit.analysis.modal.mass_normalize_mode_shapes`),
            one mode per column.
        damping_matrix: The damping matrix ``C`` to project (e.g. from
            :meth:`RayleighDamping.damping_matrix`).
        angular_frequencies: Natural circular frequency of each mode, in
            rad/s. Must be strictly positive (rigid-body modes have no
            well-defined damping ratio).

    Returns:
        A length-``n_modes`` array of damping ratios.

    Raises:
        ValidationError: If any ``angular_frequencies`` entry is not
            strictly positive.
    """
    angular_frequencies = np.asarray(angular_frequencies, dtype=float)
    if np.any(angular_frequencies <= 0) or not np.all(np.isfinite(angular_frequencies)):
        raise ValidationError(
            "modal_damping_ratios_from_matrix requires strictly positive angular "
            "frequencies; rigid-body modes (omega <= 0) have no well-defined damping ratio."
        )

    n_modes = mass_normalized_mode_shapes.shape[1]
    ratios = np.empty(n_modes)
    for i in range(n_modes):
        phi = mass_normalized_mode_shapes[:, i]
        modal_damping_coefficient = float(phi @ damping_matrix @ phi)
        ratios[i] = modal_damping_coefficient / (2.0 * angular_frequencies[i])
    return ratios
