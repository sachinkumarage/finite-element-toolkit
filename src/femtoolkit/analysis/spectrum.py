"""Response spectrum foundation: spectral acceleration vs. period, and modal spectral response.

A **response spectrum** tabulates the maximum response (here, spectral
*pseudo-acceleration* ``Sa``) of a single-DOF oscillator as a function
of its own natural period ``T`` (or, equivalently, natural frequency)
-- a compact summary of "how hard would a structure with this period
get shaken by this particular excitation." :class:`ResponseSpectrum`
is a lightweight, linearly interpolated representation of that
function; it carries no assumption about *where* the tabulated values
came from (an earthquake record, a code-specified design spectrum, a
synthetic curve) and implements no building-code-specific logic --
that is deliberately out of scope for this version (see the module's
own limitations).

:func:`modal_spectral_response` provides the next step: given a modal
analysis (with participation factors, see
:mod:`femtoolkit.analysis.modal`) and a response spectrum, it computes
each mode's **peak modal response** to that spectrum. Combining those
per-mode peaks into a single estimated peak *physical* response (e.g.
via SRSS or CQC) is an *additional* modeling choice with its own
assumptions about modal correlation -- also out of scope here; this
module stops at the mathematically well-defined per-mode quantities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from femtoolkit.exceptions import ValidationError

if TYPE_CHECKING:
    from femtoolkit.analysis.modal import ModalResult


@dataclass(frozen=True)
class ResponseSpectrum:
    """A tabulated response spectrum: spectral acceleration as a function of period.

    Attributes:
        periods: Natural periods, in seconds, strictly ascending. Must
            have at least two points (interpolation needs a bracketing
            interval).
        accelerations: Spectral (pseudo-)acceleration at each period, in
            the same units the caller intends to use consistently (e.g.
            m/s^2 or ``g``) -- this class performs no unit conversion.

    Raises:
        ValidationError: If ``periods``/``accelerations`` are not 1D
            arrays of matching length >= 2, if any value is not finite,
            if any period is negative, if periods are not strictly
            ascending, or if any acceleration is negative.

    Example:
        >>> spectrum = ResponseSpectrum(
        ...     periods=[0.0, 0.2, 0.5, 1.0, 2.0],
        ...     accelerations=[0.4, 1.0, 0.8, 0.4, 0.2],
        ... )
        >>> spectrum.evaluate(0.35)
        0.9...
    """

    periods: np.ndarray
    accelerations: np.ndarray

    def __post_init__(self) -> None:
        """Validate the spectrum's data immediately after construction.

        Raises:
            ValidationError: See the class docstring.
        """
        periods = np.asarray(self.periods, dtype=float)
        accelerations = np.asarray(self.accelerations, dtype=float)
        object.__setattr__(self, "periods", periods)
        object.__setattr__(self, "accelerations", accelerations)

        if periods.ndim != 1 or accelerations.ndim != 1:
            raise ValidationError("periods and accelerations must be one-dimensional.")
        if periods.shape != accelerations.shape:
            raise ValidationError(
                f"periods and accelerations must have the same length, got "
                f"{periods.shape[0]} and {accelerations.shape[0]}."
            )
        if periods.shape[0] < 2:
            raise ValidationError("A ResponseSpectrum requires at least two points.")
        if not np.all(np.isfinite(periods)) or not np.all(np.isfinite(accelerations)):
            raise ValidationError("periods and accelerations must all be finite.")
        if np.any(periods < 0):
            raise ValidationError("periods must be non-negative.")
        if np.any(accelerations < 0):
            raise ValidationError("accelerations must be non-negative.")
        if np.any(np.diff(periods) <= 0):
            raise ValidationError("periods must be sorted in strictly ascending order.")

    def evaluate(self, period: float) -> float:
        """Return the spectral acceleration at ``period``, linearly interpolated.

        Args:
            period: The period to evaluate, in seconds. Must lie within
                ``[periods[0], periods[-1]]`` -- extrapolation beyond the
                tabulated range is not supported (a response spectrum
                has no defined meaning outside the range it was built
                for).

        Returns:
            The interpolated spectral acceleration, ``Sa(period)``.

        Raises:
            ValidationError: If ``period`` is not finite, is negative,
                or falls outside ``[periods[0], periods[-1]]``.
        """
        if not math.isfinite(period) or period < 0:
            raise ValidationError(f"period must be non-negative and finite, got {period}.")
        if period < self.periods[0] or period > self.periods[-1]:
            raise ValidationError(
                f"period {period} is outside the spectrum's defined range "
                f"[{self.periods[0]}, {self.periods[-1]}]; extrapolation is not supported."
            )
        return float(np.interp(period, self.periods, self.accelerations))


@dataclass(frozen=True)
class ModalSpectralResponseResult:
    """Per-mode peak response to a :class:`ResponseSpectrum`.

    Attributes:
        frequencies: Natural frequency of each included mode, in Hz.
        periods: Natural period of each included mode, in seconds.
        participation_factors: Modal participation factor of each mode
            (see :func:`~femtoolkit.analysis.modal.modal_participation_factors`).
        spectral_accelerations: ``Sa(T_i)``, the spectrum evaluated at
            each mode's own period.
        modal_displacements: Peak modal coordinate response,
            ``q_i,max = Gamma_i * Sa(T_i) / omega_i^2`` (see the module
            docstring's derivation).
        equivalent_static_forces: Peak modal "equivalent static force,"
            ``F_i,max = M_eff,i * Sa(T_i)`` -- the standard modal
            static-equivalent-load quantity, still per mode (combining
            modes into one estimated total is out of scope).
    """

    frequencies: np.ndarray
    periods: np.ndarray
    participation_factors: np.ndarray
    spectral_accelerations: np.ndarray
    modal_displacements: np.ndarray
    equivalent_static_forces: np.ndarray


def modal_spectral_response(
    modal_result: ModalResult, spectrum: ResponseSpectrum
) -> ModalSpectralResponseResult:
    """Compute each physical mode's peak response to a response spectrum.

    .. code-block:: text

        q_i,max = Gamma_i * Sa(T_i) / omega_i^2       (peak modal coordinate)
        F_i,max = M_eff,i * Sa(T_i)                    (peak equivalent static force)

    Rigid-body modes (undefined period) are skipped automatically.

    Args:
        modal_result: A modal analysis result computed *with* a
            ``direction`` (see
            :func:`~femtoolkit.analysis.modal.modal_analysis_of_system`/
            :func:`~femtoolkit.analysis.modal.modal_analysis`), so
            participation factors and effective modal mass are already
            populated.
        spectrum: The response spectrum to evaluate at each mode's period.

    Returns:
        A :class:`ModalSpectralResponseResult`, one entry per physical
        (non-rigid-body) mode in ``modal_result``.

    Raises:
        ValidationError: If ``modal_result`` was not computed with a
            ``direction`` (no participation factors available), if it
            contains no physical modes, or if any mode's period falls
            outside ``spectrum``'s defined range.
    """
    if modal_result.participation_factors is None:
        raise ValidationError(
            "modal_result has no participation factors; compute it with a `direction` "
            "(e.g. modal_analysis_of_system(system, direction='x')) first."
        )

    physical_mask = ~modal_result.is_rigid_body_mode
    if not np.any(physical_mask):
        raise ValidationError("modal_result contains no physical (non-rigid-body) modes.")

    frequencies = modal_result.frequencies[physical_mask]
    periods = modal_result.periods[physical_mask]
    participation = modal_result.participation_factors[physical_mask]
    angular_frequencies = modal_result.angular_frequencies[physical_mask]
    effective_mass = modal_result.effective_modal_mass[physical_mask]

    spectral_accelerations = np.array([spectrum.evaluate(float(t)) for t in periods])
    modal_displacements = participation * spectral_accelerations / angular_frequencies**2
    equivalent_static_forces = effective_mass * spectral_accelerations

    return ModalSpectralResponseResult(
        frequencies=frequencies,
        periods=periods,
        participation_factors=participation,
        spectral_accelerations=spectral_accelerations,
        modal_displacements=modal_displacements,
        equivalent_static_forces=equivalent_static_forces,
    )
