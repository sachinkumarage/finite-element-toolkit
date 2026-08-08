"""Material data model.

This module defines :class:`Material`, a plain data representation of an
isotropic engineering material. Version 1 stores properties only; it does
not implement stress-strain relationships, plasticity, or any other
constitutive behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from femtoolkit.exceptions import ValidationError

# Physically valid range for the Poisson's ratio of an isotropic, linear
# elastic material. See e.g. Gercek, H. (2007), "Poisson's ratio values for
# rocks", for the theoretical bounds of -1.0 to 0.5 (exclusive).
_MIN_POISSONS_RATIO = -1.0
_MAX_POISSONS_RATIO = 0.5


@dataclass
class Material:
    """An isotropic engineering material, described by its basic properties.

    Version 1 treats a material purely as a data record. It does not model
    constitutive behavior such as plasticity, thermal effects, or composite
    layups; those belong to future versions of the toolkit.

    Attributes:
        name: Human-readable material name (e.g. ``"Steel"``). Must not be
            empty or whitespace-only.
        density: Mass density in kg/m^3 (SI units). Must be positive.
        youngs_modulus: Young's modulus (modulus of elasticity) in pascals.
            Must be positive.
        poissons_ratio: Poisson's ratio (dimensionless). Must lie within the
            physically valid range for an isotropic elastic material,
            ``-1.0 < poissons_ratio < 0.5``.

    Raises:
        ValidationError: If any attribute value is physically invalid.

    Example:
        >>> steel = Material(
        ...     name="Steel",
        ...     density=7850.0,
        ...     youngs_modulus=200e9,
        ...     poissons_ratio=0.3,
        ... )
    """

    name: str
    density: float
    youngs_modulus: float
    poissons_ratio: float

    def __post_init__(self) -> None:
        """Validate material properties immediately after construction.

        Raises:
            ValidationError: If ``name`` is empty, ``density`` is not
                positive, ``youngs_modulus`` is not positive, or
                ``poissons_ratio`` is outside the physically valid range.
        """
        if not self.name or not self.name.strip():
            raise ValidationError("Material name must not be empty.")

        if not math.isfinite(self.density) or self.density <= 0:
            raise ValidationError(f"Material density must be positive, got {self.density}.")

        if not math.isfinite(self.youngs_modulus) or self.youngs_modulus <= 0:
            raise ValidationError(
                f"Material Young's modulus must be positive, got {self.youngs_modulus}."
            )

        if (
            not math.isfinite(self.poissons_ratio)
            or not _MIN_POISSONS_RATIO < self.poissons_ratio < _MAX_POISSONS_RATIO
        ):
            raise ValidationError(
                "Material Poisson's ratio must be within "
                f"({_MIN_POISSONS_RATIO}, {_MAX_POISSONS_RATIO}), "
                f"got {self.poissons_ratio}."
            )
