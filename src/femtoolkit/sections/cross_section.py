"""Cross-sectional property data model.

For a 1D axial bar or 2D truss element, the only cross-sectional property
that affects behavior is the cross-sectional area. Version 5 adds the two
further properties needed by a 2D Euler-Bernoulli frame element:

* ``second_moment_of_area`` (``I``) -- the section's resistance to
  bending, used by the flexural stiffness terms ``EI``.
* ``extreme_fiber_distance`` (``c``) -- the distance from the neutral
  axis to the outermost fiber, used only by the optional bending-stress
  utility ``sigma = M * c / I``.

Both are optional so that :class:`CrossSection` still supports the
axial-only ``BarElement`` and ``TrussElement2D`` with just an area.
Torsional constant, plastic section modulus, radius of gyration, shear
area, and arbitrary polygon profiles are out of scope for this version.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from femtoolkit.exceptions import ValidationError


@dataclass
class CrossSection:
    """The cross-sectional properties of a structural element.

    Attributes:
        area: Cross-sectional area, in square meters (SI units). Must be
            positive. Required by every element type.
        second_moment_of_area: Second moment of area (moment of inertia)
            about the bending axis, in meters to the fourth power. Must be
            positive when given. Required by :class:`~femtoolkit.mesh.frame_element.FrameElement2D`;
            not used by axial-only elements.
        extreme_fiber_distance: Distance from the neutral axis to the
            outermost fiber, in meters. Must be positive when given. Only
            used by the extreme-fiber bending-stress utility
            (``sigma = M * extreme_fiber_distance / second_moment_of_area``).

    Raises:
        ValidationError: If ``area`` is not a positive, finite number, or
            if ``second_moment_of_area`` or ``extreme_fiber_distance`` is
            given but is not a positive, finite number.

    Example:
        >>> section = CrossSection(area=0.01)
        >>> beam_section = CrossSection(
        ...     area=0.01, second_moment_of_area=8.333e-6, extreme_fiber_distance=0.05
        ... )
    """

    area: float
    second_moment_of_area: float | None = None
    extreme_fiber_distance: float | None = None

    def __post_init__(self) -> None:
        """Validate the cross-sectional properties immediately after construction.

        Raises:
            ValidationError: If ``area`` is not a positive, finite number,
                or if ``second_moment_of_area`` or ``extreme_fiber_distance``
                is given but is not a positive, finite number.
        """
        if not math.isfinite(self.area) or self.area <= 0:
            raise ValidationError(f"CrossSection area must be positive, got {self.area}.")

        if self.second_moment_of_area is not None and (
            not math.isfinite(self.second_moment_of_area) or self.second_moment_of_area <= 0
        ):
            raise ValidationError(
                "CrossSection second_moment_of_area must be positive, got "
                f"{self.second_moment_of_area}."
            )

        if self.extreme_fiber_distance is not None and (
            not math.isfinite(self.extreme_fiber_distance) or self.extreme_fiber_distance <= 0
        ):
            raise ValidationError(
                "CrossSection extreme_fiber_distance must be positive, got "
                f"{self.extreme_fiber_distance}."
            )
