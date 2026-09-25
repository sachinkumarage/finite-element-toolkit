"""A reliability foundation: empirical limit-exceedance frequency (Version 31).

**Engineering concept.** A limit state separates "acceptable" from
"unacceptable" outcomes: ``g(X) = R(X) - S(X)``, where ``R`` is
resistance/capacity and ``S`` is demand; failure is conventionally
``g(X) < 0``. This module supports the common simplified case of one
scalar output quantity compared against a fixed threshold (e.g.
"maximum displacement exceeds 5 mm"), estimating

.. math::

    P(Y > L)

directly from the Monte Carlo sample's *empirical* frequency -- the
fraction of successful runs whose output exceeded (or fell below) the
threshold.

**This is not a reliability index.** A rigorous reliability method
(FORM/SORM, importance sampling, subset simulation, ...) is
deliberately out of scope for this version -- see the module docstring
of :mod:`femtoolkit.uncertainty` for the full list of exclusions. An
empirical frequency from a few hundred or thousand samples carries
substantial sampling uncertainty of its own, especially for a rare
event (a small exceedance count): :class:`ExceedanceResult` always
reports the underlying sample count precisely so this limitation stays
visible, and never claims a "reliability index" or "probability of
failure" beyond what a plain frequency count actually supports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from femtoolkit.exceptions import ValidationError

_DIRECTIONS = ("above", "below")


@dataclass(frozen=True)
class ExceedanceResult:
    """An empirical limit-exceedance frequency estimated from a Monte Carlo sample.

    Attributes:
        quantity_label: The evaluated output quantity's name.
        threshold: The limit value compared against.
        direction: ``"above"`` (exceedance means ``value > threshold``)
            or ``"below"`` (``value < threshold``).
        n_samples: How many successful samples the estimate is based on.
        n_exceeding: How many of those samples exceeded the threshold.
        exceedance_frequency: ``n_exceeding / n_samples`` -- an
            empirical frequency, not a rigorous probability-of-failure
            estimate.
    """

    quantity_label: str
    threshold: float
    direction: str
    n_samples: int
    n_exceeding: int
    exceedance_frequency: float


def exceedance_probability(
    values: np.ndarray, threshold: float, quantity_label: str = "quantity", direction: str = "above"
) -> ExceedanceResult:
    """Estimate the empirical frequency of a Monte Carlo output exceeding a threshold.

    Args:
        values: The successful runs' output values.
        threshold: The limit value to compare against.
        quantity_label: A human-readable name for the quantity.
        direction: ``"above"`` to count ``value > threshold``, or
            ``"below"`` to count ``value < threshold``.

    Returns:
        An :class:`ExceedanceResult`.

    Raises:
        ValidationError: If ``values`` is empty, or ``direction`` is
            not ``"above"``/``"below"``.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValidationError("exceedance_probability requires at least one successful sample.")
    if direction not in _DIRECTIONS:
        raise ValidationError(f"direction must be one of {_DIRECTIONS}, got {direction!r}.")

    mask = values > threshold if direction == "above" else values < threshold
    n_exceeding = int(np.sum(mask))
    n_samples = int(values.size)

    return ExceedanceResult(
        quantity_label=quantity_label,
        threshold=threshold,
        direction=direction,
        n_samples=n_samples,
        n_exceeding=n_exceeding,
        exceedance_frequency=n_exceeding / n_samples,
    )


__all__ = ["ExceedanceResult", "exceedance_probability"]
