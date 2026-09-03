"""St. Venant-Kirchhoff finite-strain elastic materials (Version 16).

Every material through Version 15 is a **small-strain** model: it relates
engineering (infinitesimal) strain to stress. Geometric nonlinearity
(:mod:`femtoolkit.analysis.geometric_nonlinear`) needs a material that
relates **Green-Lagrange strain** ``E`` to the **second Piola-Kirchhoff
stress** ``S`` (see :mod:`femtoolkit.continuum.deformation` and
:mod:`femtoolkit.continuum.stress` for both) instead.

**St. Venant-Kirchhoff elasticity** is the simplest possible choice: a
*linear* relationship between ``S`` and ``E``,

.. code-block:: text

    S = C : E

using the *exact same* fourth-order isotropic elasticity tensor
(:func:`~femtoolkit.continuum.constitutive.isotropic_3d_matrix`) already
used for small-strain linear elasticity
(:class:`~femtoolkit.materials.linear_elastic_3d.LinearElastic3D`) --
only now applied to the geometrically exact Green-Lagrange strain rather
than the small-strain tensor. This is deliberately the least-novel choice
mathematically (reusing an already-validated constitutive matrix), so the
new behavior this version introduces is isolated entirely to the
*kinematics* (how ``E`` itself is computed from displacement, a nonlinear,
objective relationship -- see :mod:`femtoolkit.continuum.deformation`),
not to a new constitutive law.

**Limitations.** St. Venant-Kirchhoff elasticity handles arbitrarily large
*rotations* and *displacements* correctly (that is exactly what makes it
useful here -- see the objectivity discussion in
:mod:`femtoolkit.continuum.deformation`), but is **not** a general-purpose
hyperelastic model: because ``S`` grows *linearly* without bound as ``E``
grows, it produces non-physical stress-strain behavior at large
*compressive* strain (it does not stiffen the way real materials, and
proper hyperelastic models such as Neo-Hookean, do, as compression
approaches full material collapse) and has no strain-energy-based
guarantee of stability at extreme strain. It is a standard, well-
established "foundation" choice for validating a Total Lagrangian
implementation at moderate strain -- advanced hyperelasticity is
explicitly out of scope for this version (see the Version 17 preview in
the project README).

Both materials implement the *exact same*
:class:`~femtoolkit.materials.nonlinear.NonlinearMaterial` interface
(``initial_state``/``trial_state``/``tangent_modulus``) as every other
material in this package -- no new material protocol needed. Both are
**path-independent** (elastic, no history): ``trial_state`` ignores
``committed_state`` entirely, exactly like
:class:`~femtoolkit.materials.nonlinear.ElasticMaterialAdapter`, and
``tangent_modulus`` returns the same constant matrix regardless of state,
since ``S = C : E`` is linear in ``E``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from femtoolkit.continuum.constitutive import isotropic_3d_matrix
from femtoolkit.exceptions import ValidationError
from femtoolkit.materials.nonlinear import MaterialState, NonlinearMaterial

_MIN_POISSONS_RATIO = -1.0
_MAX_POISSONS_RATIO = 0.5


@dataclass(frozen=True)
class SaintVenantKirchhoff3D(NonlinearMaterial):
    """A 3D St. Venant-Kirchhoff finite-strain elastic material.

    Attributes:
        youngs_modulus: Young's modulus ``E``, in pascals. Must be positive.
        poisson_ratio: Poisson's ratio ``v`` (dimensionless). Must lie in
            ``(-1.0, 0.5)``, the physically valid range for an isotropic
            elastic material.

    Raises:
        ValidationError: If ``youngs_modulus`` or ``poisson_ratio`` is not
            a physically valid, finite value.

    Example:
        >>> material = SaintVenantKirchhoff3D(youngs_modulus=200e9, poisson_ratio=0.3)
        >>> state = material.trial_state(
        ...     strain=np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ...     committed_state=material.initial_state(),
        ... )
    """

    youngs_modulus: float
    poisson_ratio: float

    def __post_init__(self) -> None:
        """Validate the elastic constants immediately after construction.

        Raises:
            ValidationError: If ``youngs_modulus`` or ``poisson_ratio`` is
                not a physically valid, finite value.
        """
        if not math.isfinite(self.youngs_modulus) or self.youngs_modulus <= 0:
            raise ValidationError(
                f"SaintVenantKirchhoff3D youngs_modulus must be positive, got "
                f"{self.youngs_modulus}."
            )
        if not math.isfinite(self.poisson_ratio) or not (
            _MIN_POISSONS_RATIO < self.poisson_ratio < _MAX_POISSONS_RATIO
        ):
            raise ValidationError(
                "SaintVenantKirchhoff3D poisson_ratio must be within "
                f"({_MIN_POISSONS_RATIO}, {_MAX_POISSONS_RATIO}), got {self.poisson_ratio}."
            )

    @property
    def constitutive_matrix(self) -> np.ndarray:
        """The material's constant 6x6 constitutive matrix ``C``, for ``S = C @ E``."""
        return isotropic_3d_matrix(self.youngs_modulus, self.poisson_ratio)

    def initial_state(self) -> MaterialState:
        """Return the zero-strain, zero-stress initial state."""
        return MaterialState(
            strain=np.zeros(6), stress=np.zeros(6), plastic_strain=np.zeros(6), yielded=False
        )

    def trial_state(self, strain: np.ndarray, committed_state: MaterialState) -> MaterialState:
        """Compute ``S = C @ E``; path-independent, so ``committed_state`` is unused.

        Args:
            strain: Green-Lagrange strain, Voigt
                ``[E_xx, E_yy, E_zz, 2*E_xy, 2*E_yz, 2*E_xz]``.
            committed_state: Unused (see the class docstring).

        Returns:
            A new :class:`~femtoolkit.materials.nonlinear.MaterialState`
            with ``stress`` set to the second Piola-Kirchhoff stress ``S``.
        """
        del committed_state
        strain_array = np.asarray(strain, dtype=float)
        stress = self.constitutive_matrix @ strain_array
        return MaterialState(
            strain=strain_array, stress=stress, plastic_strain=np.zeros(6), yielded=False
        )

    def tangent_modulus(self, state: MaterialState) -> np.ndarray:
        """Return the constant constitutive matrix ``C``, independent of ``state``."""
        del state
        return self.constitutive_matrix


@dataclass(frozen=True)
class SaintVenantKirchhoff1D(NonlinearMaterial):
    """A uniaxial (1D) St. Venant-Kirchhoff finite-strain elastic material.

    The 1D reduction of :class:`SaintVenantKirchhoff3D`, used by the
    large-displacement truss (:mod:`femtoolkit.analysis.geometric_nonlinear`),
    exactly analogous to how the existing 1D materials
    (:class:`~femtoolkit.materials.nonlinear.ElasticPerfectlyPlasticMaterial1D`,
    :class:`~femtoolkit.materials.hardening.BilinearIsotropicHardeningMaterial1D`)
    give a truss/bar a scalar constitutive model:

    .. code-block:: text

        S = youngs_modulus * E

    where ``E`` is the truss's own (scalar) Green-Lagrange axial strain,
    ``E = (L^2 - L0^2) / (2 * L0^2)``.

    Attributes:
        youngs_modulus: Young's modulus, in pascals. Must be positive.

    Raises:
        ValidationError: If ``youngs_modulus`` is not positive and finite.
    """

    youngs_modulus: float

    def __post_init__(self) -> None:
        """Validate ``youngs_modulus`` immediately after construction.

        Raises:
            ValidationError: If ``youngs_modulus`` is not positive and finite.
        """
        if not math.isfinite(self.youngs_modulus) or self.youngs_modulus <= 0:
            raise ValidationError(
                f"SaintVenantKirchhoff1D youngs_modulus must be positive, got "
                f"{self.youngs_modulus}."
            )

    def initial_state(self) -> MaterialState:
        """Return the zero-strain, zero-stress initial state."""
        return MaterialState.zero(0.0)

    def trial_state(self, strain: float, committed_state: MaterialState) -> MaterialState:
        """Compute ``S = youngs_modulus * E``; path-independent, so ``committed_state`` is unused.

        Args:
            strain: Scalar Green-Lagrange axial strain.
            committed_state: Unused (see the class docstring).

        Returns:
            A new :class:`~femtoolkit.materials.nonlinear.MaterialState`.
        """
        del committed_state
        stress = self.youngs_modulus * strain
        return MaterialState(strain=strain, stress=stress, plastic_strain=0.0, yielded=False)

    def tangent_modulus(self, state: MaterialState) -> float:
        """Return the constant ``youngs_modulus``, independent of ``state``."""
        del state
        return self.youngs_modulus
