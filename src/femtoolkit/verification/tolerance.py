"""Combined absolute/relative tolerance handling (Version 29).

Comparing a floating-point numerical result against a reference value
with exact equality is never correct -- round-off alone guarantees a
mismatch. A single fixed tolerance is not correct either: an absolute
tolerance appropriate for a millimeter-scale displacement is far too
loose for a near-zero reaction force, and a relative tolerance alone is
undefined (or numerically unstable) when the reference value is zero or
very small. This module implements the standard combined criterion,
matching (for example) :func:`numpy.isclose`'s own formula:

.. code-block:: text

    |numerical - reference| <= absolute_tolerance + relative_tolerance * |reference|

so a comparison against a near-zero reference still has a meaningful
(the absolute term's) tolerance, and a comparison against a
large-magnitude reference scales sensibly (the relative term dominates).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from femtoolkit.exceptions import ValidationError


@dataclass(frozen=True)
class Tolerance:
    """A combined absolute/relative tolerance for comparing two floating-point values.

    Attributes:
        absolute: The absolute tolerance component. Must be non-negative.
        relative: The relative tolerance component (dimensionless
            fraction of ``|reference|``). Must be non-negative.

    Raises:
        ValidationError: If ``absolute`` or ``relative`` is negative,
            not finite, or both are exactly zero (a tolerance of zero
            everywhere reduces to floating-point exact equality, which
            this toolkit's verification framework never uses -- see the
            module docstring).

    Example:
        >>> tolerance = Tolerance(absolute=1e-9, relative=1e-3)
        >>> tolerance.is_satisfied(numerical=1.0009, reference=1.0)
        True
    """

    absolute: float = 1e-9
    relative: float = 1e-6

    def __post_init__(self) -> None:
        for name, value in (("absolute", self.absolute), ("relative", self.relative)):
            if not _is_finite_non_negative(value):
                raise ValidationError(
                    f"Tolerance.{name} must be a non-negative, finite number, got {value}."
                )
        if self.absolute == 0.0 and self.relative == 0.0:
            raise ValidationError(
                "Tolerance.absolute and Tolerance.relative cannot both be zero -- that "
                "reduces to floating-point exact equality, which is never a reliable "
                "verification criterion."
            )

    def allowed_error(self, reference: float) -> float:
        """Return the maximum error this tolerance allows for a given ``reference`` value.

        Args:
            reference: The reference value the comparison is made against.

        Returns:
            ``absolute + relative * |reference|``.
        """
        return self.absolute + self.relative * abs(reference)

    def is_satisfied(self, numerical: float, reference: float) -> bool:
        """Check whether ``numerical`` is within tolerance of ``reference``.

        Args:
            numerical: The value to check (e.g. an FEA result).
            reference: The reference value being compared against.

        Returns:
            ``True`` if ``|numerical - reference| <= allowed_error(reference)``.
        """
        return abs(numerical - reference) <= self.allowed_error(reference)


def _is_finite_non_negative(value: float) -> bool:
    return math.isfinite(value) and value >= 0.0


__all__ = ["Tolerance"]
