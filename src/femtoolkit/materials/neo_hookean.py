"""Compressible Neo-Hookean hyperelastic material (Version 17).

The simplest widely-used genuinely nonlinear hyperelastic model, and the
standard first extension beyond St. Venant-Kirchhoff
(:class:`~femtoolkit.materials.finite_strain.SaintVenantKirchhoff3D`) for
materials -- rubber, elastomers -- that must remain physically sensible
under large compressive as well as tensile strain, which St. Venant-
Kirchhoff does not guarantee.

**Formulation.** Using the (non-isochoric-decoupled) compressible form,
expressed directly in the raw first invariant ``I1 = tr(C)`` and the
volume ratio ``J = sqrt(det(C))``:

.. code-block:: text

    W = mu/2 * (I1 - 3) - mu * ln(J) + lambda/2 * (ln J)^2

``mu`` (shear modulus) and ``lambda`` (first Lame parameter) are computed
from ``youngs_modulus``/``poisson_ratio`` via the same standard
conversion already used throughout this project's other isotropic
materials (:class:`~femtoolkit.materials.linear_elastic_3d.LinearElastic3D`,
:class:`~femtoolkit.materials.finite_strain.SaintVenantKirchhoff3D`).

**Why this formula is stress-free at the reference configuration without
needing the isochoric/volumetric decomposition.** At ``F = I``
(``C = I``, ``I1 = 3``, ``J = 1``), every term vanishes: ``I1 - 3 = 0``
and ``ln(J) = 0``. This is a property specific to this particular
(single-``mu``-coupled) formula -- contrast
:mod:`femtoolkit.materials.mooney_rivlin`'s module docstring, which
explains why its two-parameter model does *not* have this property
without the decoupled form.

**Stress -- derived directly (analytical override).** Using
``d(I1)/dC = I`` and ``d(J)/dC = (J/2) C^-1`` (both standard, verified
identities):

.. code-block:: text

    S = 2*dW/dC = mu*(I - C^-1) + lambda*ln(J)*C^-1

This matches the standard, widely-published compressible Neo-Hookean
second Piola-Kirchhoff stress formula, and is cross-validated in the test
suite against :class:`~femtoolkit.materials.hyperelastic.HyperelasticMaterial`'s
numerical-differentiation-of-energy default (they must agree to within
finite-difference tolerance). The material tangent (``dS/dE``) uses that
base class's numerical default -- differentiating this already-analytical
``S`` -- rather than hand-deriving the fourth-order tensor
``d(C^-1)/dC``, a substantially more error-prone derivation reserved for
Version 18+ if ever needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from femtoolkit.continuum.invariants import first_invariant, jacobian_from_right_cauchy_green
from femtoolkit.exceptions import ValidationError
from femtoolkit.materials.hyperelastic import HyperelasticMaterial
from femtoolkit.materials.nonlinear import MaterialState, NonlinearMaterial

_MIN_POISSONS_RATIO = -1.0
_MAX_POISSONS_RATIO = 0.5


@dataclass(frozen=True)
class NeoHookean3D(HyperelasticMaterial, NonlinearMaterial):
    """A compressible Neo-Hookean hyperelastic material.

    Attributes:
        youngs_modulus: Young's modulus, in pascals. Must be positive.
        poisson_ratio: Poisson's ratio (dimensionless). Must lie in
            ``(-1.0, 0.5)``, the physically valid range for an isotropic
            elastic material (used here only to define the small-strain
            limit of ``mu``/``lambda`` -- the model itself is valid for
            arbitrarily large strain).

    Raises:
        ValidationError: If ``youngs_modulus`` or ``poisson_ratio`` is
            not a physically valid, finite value.

    Example:
        >>> rubber = NeoHookean3D(youngs_modulus=5e6, poisson_ratio=0.45)
        >>> rubber.strain_energy_density(np.eye(3))
        0.0
    """

    youngs_modulus: float
    poisson_ratio: float

    def __post_init__(self) -> None:
        """Validate the elastic constants immediately after construction.

        Raises:
            ValidationError: If ``youngs_modulus`` or ``poisson_ratio``
                is not a physically valid, finite value.
        """
        if not math.isfinite(self.youngs_modulus) or self.youngs_modulus <= 0:
            raise ValidationError(
                f"NeoHookean3D youngs_modulus must be positive, got {self.youngs_modulus}."
            )
        if not math.isfinite(self.poisson_ratio) or not (
            _MIN_POISSONS_RATIO < self.poisson_ratio < _MAX_POISSONS_RATIO
        ):
            raise ValidationError(
                "NeoHookean3D poisson_ratio must be within "
                f"({_MIN_POISSONS_RATIO}, {_MAX_POISSONS_RATIO}), got {self.poisson_ratio}."
            )

    @property
    def shear_modulus(self) -> float:
        """The material's shear modulus, ``mu = E / (2 * (1 + v))``."""
        return self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def lame_lambda(self) -> float:
        """The material's first Lame parameter, ``lambda = E*v / ((1+v)(1-2v))``."""
        return (
            self.youngs_modulus
            * self.poisson_ratio
            / ((1.0 + self.poisson_ratio) * (1.0 - 2.0 * self.poisson_ratio))
        )

    def _energy_from_right_cauchy_green(self, right_cauchy_green_tensor: np.ndarray) -> float:
        """``W = mu/2*(I1-3) - mu*ln(J) + lambda/2*(ln J)^2``."""
        i1 = first_invariant(right_cauchy_green_tensor)
        jacobian = jacobian_from_right_cauchy_green(right_cauchy_green_tensor)
        mu, lame_lambda = self.shear_modulus, self.lame_lambda
        log_j = math.log(jacobian)
        return 0.5 * mu * (i1 - 3.0) - mu * log_j + 0.5 * lame_lambda * log_j**2

    def _stress_from_strain_voigt(self, strain_voigt: np.ndarray) -> np.ndarray:
        """Analytical override: ``S = mu*(I - C^-1) + lambda*ln(J)*C^-1``."""
        from femtoolkit.continuum.tensor import tensor_to_voigt_stress, voigt_strain_to_tensor

        strain_tensor = voigt_strain_to_tensor(np.asarray(strain_voigt, dtype=float))
        c = 2.0 * strain_tensor + np.eye(3)
        jacobian = jacobian_from_right_cauchy_green(c)
        c_inverse = np.linalg.inv(c)
        mu, lame_lambda = self.shear_modulus, self.lame_lambda

        s_tensor = mu * (np.eye(3) - c_inverse) + lame_lambda * math.log(jacobian) * c_inverse
        return tensor_to_voigt_stress(s_tensor)

    # --- NonlinearMaterial interface (plugs directly into the existing ---
    # --- Total Lagrangian TET4/HEX8 dispatch, femtoolkit.analysis.     ---
    # --- geometric_nonlinear, with no changes needed there).           ---

    def initial_state(self) -> MaterialState:
        """Return the zero-strain, zero-stress initial state."""
        return MaterialState(
            strain=np.zeros(6), stress=np.zeros(6), plastic_strain=np.zeros(6), yielded=False
        )

    def trial_state(self, strain: np.ndarray, committed_state: MaterialState) -> MaterialState:
        """Compute stress at ``strain`` (path-independent; ``committed_state`` unused).

        Args:
            strain: Green-Lagrange strain, Voigt
                ``[E_xx, E_yy, E_zz, 2E_xy, 2E_yz, 2E_xz]``.
            committed_state: Unused (hyperelastic materials have no
                history -- see the module docstring for
                :mod:`femtoolkit.materials.hyperelastic`).

        Returns:
            A new :class:`~femtoolkit.materials.nonlinear.MaterialState`
            with ``stress`` set to the second Piola-Kirchhoff stress.
        """
        del committed_state
        strain_array = np.asarray(strain, dtype=float)
        stress = self._stress_from_strain_voigt(strain_array)
        return MaterialState(
            strain=strain_array, stress=stress, plastic_strain=np.zeros(6), yielded=False
        )

    def tangent_modulus(self, state: MaterialState) -> np.ndarray:
        """Return the material tangent ``D = dS/dE`` (Voigt 6x6) at ``state.strain``."""
        return self._tangent_from_strain_voigt(np.asarray(state.strain, dtype=float))
