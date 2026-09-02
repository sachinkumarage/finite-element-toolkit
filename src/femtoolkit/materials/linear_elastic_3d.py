"""3D linear elastic constitutive model for TET4/HEX8 solid elements.

:class:`~femtoolkit.materials.linear_elastic_2d.LinearElastic2D` reduces a
3D stress state to a 2D one via a plane-stress or plane-strain assumption.
A genuine 3D solid element (:class:`~femtoolkit.mesh.tet4_element.Tet4Element3D`,
:class:`~femtoolkit.mesh.hex8_element.Hex8Element3D`) needs no such
reduction -- it already carries all six independent Voigt stress/strain
components -- so :class:`LinearElastic3D` is a small, dedicated
constitutive model, structurally the direct 3D analogue of
``LinearElastic2D``, built on :func:`~femtoolkit.continuum.constitutive.isotropic_3d_matrix`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from femtoolkit.continuum.constitutive import isotropic_3d_matrix
from femtoolkit.exceptions import ValidationError

_MIN_POISSONS_RATIO = -1.0
_MAX_POISSONS_RATIO = 0.5


@dataclass(frozen=True)
class LinearElastic3D:
    """An isotropic linear elastic material for 3D solid (TET4/HEX8) analysis.

    Attributes:
        youngs_modulus: Young's modulus, in pascals. Must be positive.
        poisson_ratio: Poisson's ratio (dimensionless). Must lie within
            the physically valid range for an isotropic elastic
            material, ``-1.0 < poisson_ratio < 0.5``.
        density: Mass density, in kg/m^3. Required (unlike
            :class:`~femtoolkit.materials.linear_elastic_2d.LinearElastic2D`,
            where density is optional): a 3D solid element's mass matrix
            (:mod:`femtoolkit.analysis.mass`) always needs it, and there
            is no thickness-carrying 2D use case here to make it
            optional for.

    Raises:
        ValidationError: If ``youngs_modulus`` or ``poisson_ratio`` is
            not a physically valid, finite value, or if ``density`` is
            not positive and finite.

    Example:
        >>> steel = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
        >>> steel.constitutive_matrix.shape
        (6, 6)
    """

    youngs_modulus: float
    poisson_ratio: float
    density: float

    def __post_init__(self) -> None:
        """Validate the elastic constants and density immediately after construction.

        Raises:
            ValidationError: If ``youngs_modulus`` or ``poisson_ratio``
                is not a physically valid, finite value, or ``density``
                is not positive and finite.
        """
        if not math.isfinite(self.youngs_modulus) or self.youngs_modulus <= 0:
            raise ValidationError(
                f"LinearElastic3D youngs_modulus must be positive, got {self.youngs_modulus}."
            )
        if not math.isfinite(self.poisson_ratio) or not (
            _MIN_POISSONS_RATIO < self.poisson_ratio < _MAX_POISSONS_RATIO
        ):
            raise ValidationError(
                "LinearElastic3D poisson_ratio must be within "
                f"({_MIN_POISSONS_RATIO}, {_MAX_POISSONS_RATIO}), got {self.poisson_ratio}."
            )
        if not math.isfinite(self.density) or self.density <= 0:
            raise ValidationError(
                f"LinearElastic3D density must be positive, got {self.density}."
            )

    @property
    def constitutive_matrix(self) -> np.ndarray:
        """The material's 6x6 constitutive matrix ``D``, for ``sigma = D @ epsilon``.

        Computed by :func:`~femtoolkit.continuum.constitutive.isotropic_3d_matrix`.
        """
        return isotropic_3d_matrix(self.youngs_modulus, self.poisson_ratio)

    @property
    def shear_modulus(self) -> float:
        """The material's shear modulus, ``mu = E / (2 * (1 + v))``.

        Exposed directly since :class:`~femtoolkit.materials.j2_plasticity.J2Plasticity3D`'s
        return-mapping formulas are expressed in terms of it.
        """
        return self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def bulk_modulus(self) -> float:
        """The material's bulk modulus, ``K = E / (3 * (1 - 2*v))``."""
        return self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poisson_ratio))
