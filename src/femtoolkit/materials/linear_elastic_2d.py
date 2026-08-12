"""2D linear elastic constitutive model for continuum elements.

:class:`~femtoolkit.materials.material.Material` is a plain data record
of isotropic properties (density, Young's modulus, Poisson's ratio) used
by every structural element (bar, truss, frame). A continuum element
needs one more thing a structural element does not: a **plane-stress or
plane-strain assumption**, which determines its constitutive matrix.
Rather than overloading `Material` with a field that only makes sense for
2D continuum elements, :class:`LinearElastic2D` is a small, dedicated
constitutive model that a :class:`~femtoolkit.mesh.cst_element.CSTElement2D`
uses instead of `Material`.

This is deliberately a minimal, focused abstraction -- not a general
material-model framework. Future constitutive models (orthotropic
elasticity, plasticity, nonlinear elasticity) would be separate classes
following the same shape (elastic constants in, a constitutive matrix
out), not new responsibilities bolted onto this one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from femtoolkit.continuum.constitutive import plane_strain_matrix, plane_stress_matrix
from femtoolkit.exceptions import ValidationError

_MIN_POISSONS_RATIO = -1.0
_MAX_POISSONS_RATIO = 0.5

PlaneFormulation = Literal["plane_stress", "plane_strain"]


@dataclass(frozen=True)
class LinearElastic2D:
    """An isotropic linear elastic material for 2D continuum analysis.

    Attributes:
        youngs_modulus: Young's modulus, in pascals. Must be positive.
        poisson_ratio: Poisson's ratio (dimensionless). Must lie within
            the physically valid range for an isotropic elastic
            material, ``-1.0 < poisson_ratio < 0.5``.
        formulation: Either ``"plane_stress"`` (thin, flat bodies loaded
            in-plane, ``sigma_z = 0``) or ``"plane_strain"`` (bodies long
            in the out-of-plane direction and restrained from extending
            along it, ``epsilon_z = 0``). See
            :mod:`femtoolkit.continuum.constitutive` for the underlying
            physics and formulas.

    Raises:
        ValidationError: If ``youngs_modulus`` or ``poisson_ratio`` is
            not a physically valid, finite value, or if ``formulation``
            is not one of the two supported values.

    Example:
        >>> steel = LinearElastic2D(
        ...     youngs_modulus=210e9, poisson_ratio=0.3, formulation="plane_stress"
        ... )
        >>> steel.constitutive_matrix.shape
        (3, 3)
    """

    youngs_modulus: float
    poisson_ratio: float
    formulation: PlaneFormulation

    def __post_init__(self) -> None:
        """Validate the elastic constants and formulation immediately after construction.

        Raises:
            ValidationError: If ``youngs_modulus`` or ``poisson_ratio``
                is not a physically valid, finite value, or if
                ``formulation`` is not ``"plane_stress"`` or
                ``"plane_strain"``.
        """
        if not math.isfinite(self.youngs_modulus) or self.youngs_modulus <= 0:
            raise ValidationError(
                f"LinearElastic2D youngs_modulus must be positive, got {self.youngs_modulus}."
            )
        if not math.isfinite(self.poisson_ratio) or not (
            _MIN_POISSONS_RATIO < self.poisson_ratio < _MAX_POISSONS_RATIO
        ):
            raise ValidationError(
                "LinearElastic2D poisson_ratio must be within "
                f"({_MIN_POISSONS_RATIO}, {_MAX_POISSONS_RATIO}), got {self.poisson_ratio}."
            )
        if self.formulation not in ("plane_stress", "plane_strain"):
            raise ValidationError(
                'LinearElastic2D formulation must be "plane_stress" or "plane_strain", '
                f"got {self.formulation!r}."
            )

    @property
    def constitutive_matrix(self) -> np.ndarray:
        """The material's 3x3 constitutive matrix ``D``, for ``sigma = D @ epsilon``.

        Computed by :func:`~femtoolkit.continuum.constitutive.plane_stress_matrix`
        or :func:`~femtoolkit.continuum.constitutive.plane_strain_matrix`,
        depending on :attr:`formulation`.
        """
        if self.formulation == "plane_stress":
            return plane_stress_matrix(self.youngs_modulus, self.poisson_ratio)
        return plane_strain_matrix(self.youngs_modulus, self.poisson_ratio)
