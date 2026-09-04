"""Compressible 2-parameter Mooney-Rivlin hyperelastic material (Version 17).

A richer isotropic hyperelastic model than Neo-Hookean, adding a second
invariant-dependent term that lets it fit a wider range of rubber-like
stress-strain curves (in particular, Neo-Hookean alone tends to
underpredict stiffening at moderate-to-large stretch in simple tension;
a nonzero ``C01`` term corrects this). Only the basic 2-parameter form is
implemented here -- higher-order Mooney-Rivlin variants (3-, 5-, 9-
parameter) are out of scope for this version.

**Formulation -- and why it must use the isochoric/volumetric
decomposition.** A naive compressible extension of the classic
incompressible 2-parameter form,

.. code-block:: text

    W_naive = C10*(I1 - 3) + C01*(I2 - 3) + K/2*(J-1)^2   (DO NOT USE)

is **not** stress-free at the reference configuration (``F = I``) in
general: at ``C = I``, ``I1 = I2 = 3`` so the first two terms vanish and
``J=1`` so the volumetric term vanishes too, which looks fine -- but the
*stress* (the gradient of ``W``, not ``W`` itself) does not vanish there
unless ``C10`` and ``C01`` happen to satisfy a special relationship,
because ``dI2/dC = I1*I - C`` evaluated at ``C=I`` gives ``dI2/dC|_{C=I}
= 3I - I = 2I``, not zero -- so the ``C01`` term alone contributes a
spurious residual stress at the supposedly undeformed state. This is a
well-known pitfall of the "raw invariant" compressible extension, caught
directly (via the numerical differentiation this class relies on, and
confirmed against the reference-state test) while developing this module.

The standard fix is the **decoupled isochoric/volumetric** form, using
the modified (isochoric) invariants
(:func:`~femtoolkit.continuum.invariants.isochoric_first_invariant`,
:func:`~femtoolkit.continuum.invariants.isochoric_second_invariant`),
which are *both exactly 3* whenever ``C`` represents a pure volume change
(:math:`\\bar I_1(\\alpha^2 I) = \\bar I_2(\\alpha^2 I) = 3` for any
``alpha > 0`` -- verified in the invariants module's own tests) and
otherwise depend only on the shape-changing part of the deformation:

.. code-block:: text

    W = C10*(I1_bar - 3) + C01*(I2_bar - 3) + K/2*(J-1)^2

This is exactly zero, with exactly zero gradient, at ``C = I`` -- no
special relationship between ``C10``/``C01`` required.

**Stress: analytical, derived and cross-validated -- revised from an
earlier numerical-only design.** An earlier version of this class
deliberately relied on
:class:`~femtoolkit.materials.hyperelastic.HyperelasticMaterial`'s default
numerical-differentiation-of-energy stress, to avoid hand-deriving the
isochoric invariants' more involved ``C``-gradient. That design was
reversed after direct testing surfaced a genuine numerical-robustness
failure specific to *composing* two independent finite differences: the
Total Lagrangian element dispatch
(:mod:`femtoolkit.analysis.geometric_nonlinear`, Version 16) computes its
material tangent ``K_t`` via its *own* central difference of the internal
force with respect to nodal displacement, with a step floored at
``~1e-6`` meters -- which, for a unit-scale element, perturbs strain by
only ``~1e-12``. That is far *below* this class's own (former) inner
strain-based finite-difference step floor (``~1e-6`` strain, see
:mod:`femtoolkit.materials.hyperelastic`), so the outer difference was
differencing stress values computed from strain perturbations too small
to move the inner finite difference's result at all -- producing an
effectively-noise ``K_t``, confirmed directly to have large *negative*
eigenvalues at the reference configuration (should be positive, matching
the small-strain shear/bulk stiffness) and to break Newton-Raphson
convergence immediately, even at a small applied load, in
``tests/validation/test_hyperelastic_block_test.py``'s simple-shear case.
Eliminating the inner finite difference (via this analytical formula)
removes the problem at its root, exactly the way
:class:`~femtoolkit.materials.neo_hookean.NeoHookean3D` was already built.

Using ``dI1/dC = I``, ``dI2/dC = I1*I - C``, ``dJ/dC = (J/2) C^-1``
(the same identities used by ``neo_hookean.py``, plus the product/chain
rule applied through ``I1_bar = J^(-2/3) I1`` and ``I2_bar = J^(-4/3)
I2``):

.. code-block:: text

    S = 2*dW/dC
      = 2*C10*J^(-2/3)*(I - (I1/3)*C^-1)
      + 2*C01*J^(-4/3)*(I1*I - C - (2/3)*I2*C^-1)
      + K*(J-1)*J*C^-1

Every term vanishes at ``C = I`` (``I1=I2=3``, ``J=1``) by direct
substitution -- the same reference-state property the base class's
numerical default already gave, now exact rather than approximate.
Cross-validated against that numerical default (which remains available,
inherited, and is still what the material tangent's default
differentiates) to within finite-difference tolerance across the
reference state, shear, and general triaxial deformation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from femtoolkit.continuum.invariants import (
    isochoric_first_invariant,
    isochoric_second_invariant,
    jacobian_from_right_cauchy_green,
)
from femtoolkit.exceptions import ValidationError
from femtoolkit.materials.hyperelastic import HyperelasticMaterial
from femtoolkit.materials.nonlinear import MaterialState, NonlinearMaterial


@dataclass(frozen=True)
class MooneyRivlin3D(HyperelasticMaterial, NonlinearMaterial):
    """A compressible 2-parameter Mooney-Rivlin hyperelastic material.

    Attributes:
        c10: First Mooney-Rivlin parameter, in pascals. Must be positive
            (a positive initial shear-like response requires
            ``c10 + c01 > 0``; requiring ``c10 > 0`` alone already
            guarantees this since ``c01`` is required non-negative).
        c01: Second Mooney-Rivlin parameter, in pascals. Must be
            non-negative. Setting ``c01 = 0`` reduces this model to
            (isochoric-decoupled) Neo-Hookean behavior in its deviatoric
            response.
        bulk_modulus: Volumetric penalty parameter, in pascals. Must be
            positive; larger values more strongly resist volume change
            (see the nearly-incompressible discussion in
            ``tests/validation/test_nearly_incompressible.py``).

    Raises:
        ValidationError: If any parameter is not physically valid.

    Example:
        >>> rubber = MooneyRivlin3D(c10=0.4e6, c01=0.1e6, bulk_modulus=200e6)
        >>> rubber.strain_energy_density(np.eye(3))
        0.0
    """

    c10: float
    c01: float
    bulk_modulus: float

    def __post_init__(self) -> None:
        """Validate the material parameters immediately after construction.

        Raises:
            ValidationError: If ``c10`` is not positive, ``c01`` is
                negative, or ``bulk_modulus`` is not positive.
        """
        if not math.isfinite(self.c10) or self.c10 <= 0:
            raise ValidationError(f"MooneyRivlin3D c10 must be positive, got {self.c10}.")
        if not math.isfinite(self.c01) or self.c01 < 0:
            raise ValidationError(f"MooneyRivlin3D c01 must be non-negative, got {self.c01}.")
        if not math.isfinite(self.bulk_modulus) or self.bulk_modulus <= 0:
            raise ValidationError(
                f"MooneyRivlin3D bulk_modulus must be positive, got {self.bulk_modulus}."
            )

    @property
    def initial_shear_modulus(self) -> float:
        """The material's effective shear modulus at the reference configuration, ``2*(C10+C01)``.

        The standard relation between Mooney-Rivlin parameters and the
        small-strain shear modulus -- useful for comparing this model's
        small-strain behavior against a linear-elastic or Neo-Hookean
        material with a known shear modulus.
        """
        return 2.0 * (self.c10 + self.c01)

    def _energy_from_right_cauchy_green(self, right_cauchy_green_tensor: np.ndarray) -> float:
        """``W = C10*(I1_bar-3) + C01*(I2_bar-3) + K/2*(J-1)^2`` (decoupled form)."""
        i1_bar = isochoric_first_invariant(right_cauchy_green_tensor)
        i2_bar = isochoric_second_invariant(right_cauchy_green_tensor)
        jacobian = jacobian_from_right_cauchy_green(right_cauchy_green_tensor)
        return (
            self.c10 * (i1_bar - 3.0)
            + self.c01 * (i2_bar - 3.0)
            + 0.5 * self.bulk_modulus * (jacobian - 1.0) ** 2
        )

    def _stress_from_strain_voigt(self, strain_voigt: np.ndarray) -> np.ndarray:
        """Analytical override -- see the module docstring for the derivation and why it exists."""
        from femtoolkit.continuum.invariants import first_invariant, second_invariant
        from femtoolkit.continuum.tensor import tensor_to_voigt_stress, voigt_strain_to_tensor

        strain_tensor = voigt_strain_to_tensor(np.asarray(strain_voigt, dtype=float))
        c = 2.0 * strain_tensor + np.eye(3)
        i1 = first_invariant(c)
        i2 = second_invariant(c)
        jacobian = jacobian_from_right_cauchy_green(c)
        c_inverse = np.linalg.inv(c)
        identity = np.eye(3)

        deviatoric_first = (
            2.0 * self.c10 * jacobian ** (-2.0 / 3.0) * (identity - (i1 / 3.0) * c_inverse)
        )
        deviatoric_second = (
            2.0
            * self.c01
            * jacobian ** (-4.0 / 3.0)
            * (i1 * identity - c - (2.0 / 3.0) * i2 * c_inverse)
        )
        volumetric = self.bulk_modulus * (jacobian - 1.0) * jacobian * c_inverse

        s_tensor = deviatoric_first + deviatoric_second + volumetric
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
                history -- see :mod:`femtoolkit.materials.hyperelastic`).

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
