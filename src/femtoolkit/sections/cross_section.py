"""Cross-sectional property data model.

For a 1D axial bar element, the only cross-sectional property that
affects its behavior is the cross-sectional area. Section properties
relevant to bending or torsion (moment of inertia, torsional constant,
section modulus) belong to future beam-element functionality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from femtoolkit.exceptions import ValidationError


@dataclass
class CrossSection:
    """The cross-sectional area of a 1D axial bar element.

    Attributes:
        area: Cross-sectional area, in square meters (SI units). Must be
            positive.

    Raises:
        ValidationError: If ``area`` is not a positive, finite number.

    Example:
        >>> section = CrossSection(area=0.01)
    """

    area: float

    def __post_init__(self) -> None:
        """Validate the cross-sectional area immediately after construction.

        Raises:
            ValidationError: If ``area`` is not a positive, finite number.
        """
        if not math.isfinite(self.area) or self.area <= 0:
            raise ValidationError(f"CrossSection area must be positive, got {self.area}.")
