"""Hyperelastic material interface (Version 17).

**What is hyperelasticity?** A hyperelastic material is defined not by a
direct stress-strain *law* but by a scalar **strain-energy density**
function, ``W``, such that stress is *derived* from it (``S = 2 dW/dC``,
see below) rather than independently postulated. This has a profound
consequence: the stress response is automatically **conservative**
(path-independent, thermodynamically consistent) -- loading and unloading
along *any* path return exactly the same energy, with none dissipated --
because stress is, by construction, a gradient of a scalar potential.

This is fundamentally different from the two other constitutive
categories in this toolkit:

* **Linear elasticity** (:class:`~femtoolkit.materials.linear_elastic_3d.LinearElastic3D`):
  a *linear* stress-strain law, ``sigma = D @ epsilon``, valid only for
  small strain.
* **Plasticity** (:class:`~femtoolkit.materials.j2_plasticity.J2Plasticity3D`):
  explicitly **not** conservative -- energy is genuinely dissipated
  (converted to heat) once the material yields, and the stress at a given
  strain depends on the strain *history*, not just the current strain.
  This is why plastic materials need a committed/trial state machine;
  hyperelastic materials, having no history at all, do not.
* **Hyperelasticity** (this module): a potentially *strongly nonlinear*
  but still fully conservative stress-strain relationship, appropriate
  for materials that return to their exact original shape after any load
  is removed -- rubber, elastomers, soft polymers, seals, gaskets, and
  other flexible components that undergo large, recoverable deformation
  far beyond the small-strain regime where linear elasticity applies.
  :class:`~femtoolkit.materials.finite_strain.SaintVenantKirchhoff3D`
  (Version 16) is technically already hyperelastic (``W = 1/2 E:C:E`` for
  a constant ``C``), but is a special case simple enough not to need this
  module's machinery; Neo-Hookean and Mooney-Rivlin
  (:mod:`femtoolkit.materials.neo_hookean`,
  :mod:`femtoolkit.materials.mooney_rivlin`) are genuinely nonlinear in
  ``W``, and are what this module is built to support cleanly.

**Objectivity constrains the interface.** A rigid-body rotation
superposed on any deformation must not change ``W`` or create stress (see
:mod:`femtoolkit.continuum.deformation`'s discussion of Green-Lagrange
strain's objectivity, which applies here identically). This is only
possible if ``W`` depends on the deformation gradient ``F`` *only*
through the right Cauchy-Green tensor ``C = F^T F`` -- so this class's one
abstract method, :meth:`HyperelasticMaterial._energy_from_right_cauchy_green`,
takes ``C`` directly, not ``F``. Every public, ``F``-based method
(:meth:`strain_energy_density`, :meth:`second_piola_kirchhoff_stress`,
etc.) is a generic, non-abstract wrapper around it.

**Total strain energy.** The *density* ``W`` returned by this interface
is energy per unit **reference** volume; the total stored energy of a
body is ``U = integral_V0 W dV0`` -- always integrated over the reference
configuration, consistent with the Total Lagrangian formulation used
throughout :mod:`femtoolkit.analysis.geometric_nonlinear`.

**Stress, without hand-deriving a tensor gradient per material.** The
safest way to guarantee "stress is the derivative of energy" -- the
defining property of hyperelasticity, and this project's top mathematical-
correctness priority for this version -- is to *compute* it that way. This
class's default :meth:`_stress_from_strain_voigt` differentiates
``_energy_from_right_cauchy_green`` numerically, directly with respect to
the six independent components of the **Voigt Green-Lagrange strain
vector** (not the C tensor's nine, symmetry-constrained entries -- this
sidesteps the well-known subtlety of differentiating with respect to a
symmetric tensor argument entirely). This works because, in the
engineering-shear Voigt convention used throughout this project (see
:mod:`femtoolkit.continuum.tensor`), the identity ``S_voigt = dW/dE_voigt``
holds with **no extra factors** -- the same convention that already makes
``sigma_voigt = D @ epsilon_voigt`` correct without correction in the
small-strain code. A concrete material may **override**
:meth:`_stress_from_strain_voigt` with a validated closed-form analytical
formula (:class:`~femtoolkit.materials.neo_hookean.NeoHookean3D` does);
:class:`~femtoolkit.materials.mooney_rivlin.MooneyRivlin3D` relies on this
safe default instead, deliberately, after a hand-derived closed form for
its (more complex, isochoric-decoupled) energy gradient was judged too
easy to get subtly wrong -- exactly the same engineering judgment already
used for :class:`~femtoolkit.materials.j2_plasticity.J2Plasticity3D`'s
tangent (Version 15) and the Total Lagrangian ``K_material`` split
(Version 16).

**Material tangent**, likewise, defaults to central-difference numerical
differentiation of the (analytical-or-numerical) stress with respect to
strain, symmetrized -- the exact same pattern already validated for
:class:`~femtoolkit.materials.j2_plasticity.J2Plasticity3D`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from femtoolkit.continuum.deformation import right_cauchy_green, validate_deformation_gradient
from femtoolkit.continuum.stress import (
    cauchy_stress_from_second_piola_kirchhoff,
    first_piola_kirchhoff_from_second,
)
from femtoolkit.continuum.tensor import tensor_to_voigt_strain, voigt_stress_to_tensor

_FINITE_DIFFERENCE_RELATIVE_STEP: float = 1e-6
"""Relative finite-difference step size for numerical differentiation
(both stress-from-energy and tangent-from-stress), scaled by the
*overall* strain state's magnitude -- see :func:`_finite_difference_step`
for why a single component's own value is the wrong thing to scale by.

**A note on what this step size does and does not fix, from direct
testing during development.** An early version of this module used a
step scaled by each *individual* Voigt component's own magnitude; at a
component that started at exactly zero (e.g. no shear yet), that step
collapsed to a floating-point-noise-dominated value, producing a visibly
wrong stress (confirmed by comparing against
:class:`~femtoolkit.materials.neo_hookean.NeoHookean3D`'s analytical
formula, which the numerical default should reproduce). Scaling by the
whole vector's norm instead (this function) fixes that. Separately, a
large, *physically extreme* deformation state (independently stretching
each axis enough to change the reference volume by 10%+ while using a
bulk modulus hundreds of times stiffer than the shear-like parameters --
a combination equilibrium loading would essentially never reach) was
found, during testing, to produce a locally non-positive-definite
material tangent for :class:`~femtoolkit.materials.mooney_rivlin.MooneyRivlin3D`.
Direct linearization checks (comparing ``stress(E + dE)`` against
``stress(E) + tangent @ dE`` for small ``dE``) confirmed the tangent was
an *accurate* derivative of the implemented energy at that point -- this
is genuine coupled volumetric/shear tangent behavior at an extreme,
non-equilibrium kinematic state, not a numerical-differentiation defect,
and is *not* fixed by any step-size choice. See
``tests/validation/test_nearly_incompressible.py`` for where this is
tested and documented properly (at deformation states an actual
Newton-Raphson equilibrium solve would reach), rather than papered over
here.
"""

_FINITE_DIFFERENCE_ABSOLUTE_FLOOR: float = 1e-6
"""Absolute **minimum step size** (strain units) for the finite difference.

Applied directly as a floor on the step itself -- *not* multiplied again
by :data:`_FINITE_DIFFERENCE_RELATIVE_STEP` -- specifically so it still
gives a well-conditioned, non-collapsing step at the reference
configuration (``strain_voigt`` exactly zero, the most common case this
guards): with the naive alternative of treating this constant as a
"reference strain scale" fed back through the relative step, the
resulting step at zero strain would be
``_FINITE_DIFFERENCE_RELATIVE_STEP * _FINITE_DIFFERENCE_ABSOLUTE_FLOOR ~
1e-12`` -- exactly the too-small-step, floating-point-round-off-noise bug
this project's development caught once already (see
:mod:`femtoolkit.materials.neo_hookean`'s module docstring for the
concrete failure mode: differentiating *energy* -- typically 1e6-1e9 Pa
in magnitude for these models -- with a 1e-12 step leaves a numerator
`W(x+h)-W(x-h)` too small to distinguish from machine-epsilon round-off
on `W` itself, no matter how large `W`'s true derivative is).
"""


def _finite_difference_step(strain_voigt: np.ndarray, component: int) -> float:
    """Return a finite-difference step size for one component of a Voigt strain vector.

    Scaled by the **norm of the whole strain vector**, not the single
    component being perturbed: a component that happens to be exactly
    zero (e.g. no shear yet, or the reference configuration) must not
    collapse the step to (near-)zero when *other* components already
    carry a meaningful deformation scale -- doing so starves the
    resulting central difference of any real signal, leaving pure
    floating-point round-off noise. The absolute floor
    (:data:`_FINITE_DIFFERENCE_ABSOLUTE_FLOOR`) is applied to the
    **final step**, not to the strain scale being multiplied by the
    relative step -- see that constant's docstring for why the
    distinction matters.
    """
    del component  # scale is the same for every component; kept for call-site clarity
    reference_scale = float(np.linalg.norm(strain_voigt))
    step = _FINITE_DIFFERENCE_RELATIVE_STEP * reference_scale
    return max(step, _FINITE_DIFFERENCE_ABSOLUTE_FLOOR)


class HyperelasticMaterial(ABC):
    """Abstract interface for an isotropic, objective hyperelastic constitutive model.

    Concrete materials implement exactly one method,
    :meth:`_energy_from_right_cauchy_green` -- a simple scalar formula, no
    tensor calculus required. Every other method (stress in any of the
    three standard measures, and the material tangent) has a safe,
    reusable default built on it, described in the module docstring.
    """

    @abstractmethod
    def _energy_from_right_cauchy_green(self, right_cauchy_green_tensor: np.ndarray) -> float:
        """Return the strain-energy density ``W(C)``, per unit reference volume.

        Args:
            right_cauchy_green_tensor: The symmetric 3x3 right
                Cauchy-Green tensor ``C = F^T F``.

        Returns:
            ``W``, in joules per cubic meter.

        Raises:
            InvalidDeformationGradientError: If ``C`` is not a physically
                valid (finite, positive-determinant) tensor.
        """
        raise NotImplementedError

    def _energy_from_strain_voigt(self, strain_voigt: np.ndarray) -> float:
        """Return ``W`` from the Voigt Green-Lagrange strain vector, ``C = 2E + I``."""
        right_cauchy_green_tensor = _right_cauchy_green_from_strain_voigt(strain_voigt)
        return self._energy_from_right_cauchy_green(right_cauchy_green_tensor)

    def strain_energy_density(self, deformation_gradient_tensor: np.ndarray) -> float:
        """Return the strain-energy density ``W(F)``, per unit reference volume.

        Args:
            deformation_gradient_tensor: The 3x3 deformation gradient ``F``.

        Returns:
            ``W``, in joules per cubic meter.

        Raises:
            InvalidDeformationGradientError: If ``F`` is invalid.
        """
        validate_deformation_gradient(deformation_gradient_tensor)
        c = right_cauchy_green(deformation_gradient_tensor)
        return self._energy_from_right_cauchy_green(c)

    def _stress_from_strain_voigt(self, strain_voigt: np.ndarray) -> np.ndarray:
        """Return the second Piola-Kirchhoff stress, Voigt form, from Voigt Green-Lagrange strain.

        Default implementation: central-difference numerical
        differentiation of :meth:`_energy_from_strain_voigt` with respect
        to each of the 6 Voigt strain components -- see the module
        docstring for why this is thermodynamically consistent by
        construction. Override for a validated analytical formula.

        Args:
            strain_voigt: Green-Lagrange strain, Voigt
                ``[E_xx, E_yy, E_zz, 2E_xy, 2E_yz, 2E_xz]``.

        Returns:
            A length-6 NumPy array, Voigt
            ``[S_xx, S_yy, S_zz, S_xy, S_yz, S_xz]``.
        """
        strain_array = np.asarray(strain_voigt, dtype=float)
        stress = np.zeros(6)
        for component in range(6):
            step = _finite_difference_step(strain_array, component)
            perturbation = np.zeros(6)
            perturbation[component] = step

            energy_plus = self._energy_from_strain_voigt(strain_array + perturbation)
            energy_minus = self._energy_from_strain_voigt(strain_array - perturbation)
            stress[component] = (energy_plus - energy_minus) / (2.0 * step)
        return stress

    def second_piola_kirchhoff_stress(self, deformation_gradient_tensor: np.ndarray) -> np.ndarray:
        """Return the second Piola-Kirchhoff stress ``S``, Voigt form, from ``F``.

        Args:
            deformation_gradient_tensor: The 3x3 deformation gradient ``F``.

        Returns:
            A length-6 NumPy array, Voigt
            ``[S_xx, S_yy, S_zz, S_xy, S_yz, S_xz]``.

        Raises:
            InvalidDeformationGradientError: If ``F`` is invalid.
        """
        validate_deformation_gradient(deformation_gradient_tensor)
        c = right_cauchy_green(deformation_gradient_tensor)
        strain_voigt = tensor_to_voigt_strain(0.5 * (c - np.eye(3)))
        return self._stress_from_strain_voigt(strain_voigt)

    def first_piola_kirchhoff_stress(self, deformation_gradient_tensor: np.ndarray) -> np.ndarray:
        """Return the first Piola-Kirchhoff stress ``P = F @ S``, from ``F``.

        Generic, reusing :func:`~femtoolkit.continuum.stress.first_piola_kirchhoff_from_second`.

        Args:
            deformation_gradient_tensor: The 3x3 deformation gradient ``F``.

        Returns:
            A (generally non-symmetric) 3x3 NumPy array.
        """
        s_voigt = self.second_piola_kirchhoff_stress(deformation_gradient_tensor)
        s_tensor = voigt_stress_to_tensor(s_voigt)
        return first_piola_kirchhoff_from_second(deformation_gradient_tensor, s_tensor)

    def cauchy_stress(self, deformation_gradient_tensor: np.ndarray) -> np.ndarray:
        """Return the Cauchy (true) stress, ``sigma = (1/J) F S F^T``, from ``F``.

        Generic, reusing
        :func:`~femtoolkit.continuum.stress.cauchy_stress_from_second_piola_kirchhoff`.

        Args:
            deformation_gradient_tensor: The 3x3 deformation gradient ``F``.

        Returns:
            A symmetric 3x3 NumPy array.
        """
        s_voigt = self.second_piola_kirchhoff_stress(deformation_gradient_tensor)
        s_tensor = voigt_stress_to_tensor(s_voigt)
        return cauchy_stress_from_second_piola_kirchhoff(deformation_gradient_tensor, s_tensor)

    def _tangent_from_strain_voigt(self, strain_voigt: np.ndarray) -> np.ndarray:
        """Return the material tangent ``D = dS/dE``, Voigt 6x6, from Voigt Green-Lagrange strain.

        Default implementation: central-difference numerical
        differentiation of :meth:`_stress_from_strain_voigt` (whichever
        implementation is active -- analytical or numerical) with respect
        to each of the 6 Voigt strain components, symmetrized -- the same
        pattern used by
        :meth:`~femtoolkit.materials.j2_plasticity.J2Plasticity3D.tangent_modulus`.

        Args:
            strain_voigt: Green-Lagrange strain, Voigt form.

        Returns:
            A 6x6 NumPy array.
        """
        strain_array = np.asarray(strain_voigt, dtype=float)
        jacobian = np.zeros((6, 6))
        for component in range(6):
            step = _finite_difference_step(strain_array, component)
            perturbation = np.zeros(6)
            perturbation[component] = step

            stress_plus = self._stress_from_strain_voigt(strain_array + perturbation)
            stress_minus = self._stress_from_strain_voigt(strain_array - perturbation)
            jacobian[:, component] = (stress_plus - stress_minus) / (2.0 * step)

        # The exact tangent is symmetric for a hyperelastic (conservative)
        # material; central differencing only approximates that, so
        # symmetrizing removes purely numerical asymmetry.
        return 0.5 * (jacobian + jacobian.T)

    def material_tangent(self, deformation_gradient_tensor: np.ndarray) -> np.ndarray:
        """Return the material tangent ``D = dS/dE``, Voigt 6x6, from ``F``.

        Args:
            deformation_gradient_tensor: The 3x3 deformation gradient ``F``.

        Returns:
            A 6x6 NumPy array.

        Raises:
            InvalidDeformationGradientError: If ``F`` is invalid.
        """
        validate_deformation_gradient(deformation_gradient_tensor)
        c = right_cauchy_green(deformation_gradient_tensor)
        strain_voigt = tensor_to_voigt_strain(0.5 * (c - np.eye(3)))
        return self._tangent_from_strain_voigt(strain_voigt)


def _right_cauchy_green_from_strain_voigt(strain_voigt: np.ndarray) -> np.ndarray:
    """Reconstruct ``C = 2E + I`` from Voigt Green-Lagrange strain.

    Never requires reconstructing a deformation gradient ``F`` -- for an
    objective material, ``W``/``S`` depend on ``F`` only through ``C``, so
    working directly in ``C`` (equivalently ``E``) space is both simpler
    and unambiguous (unlike ``F``, which is not uniquely determined by
    ``C`` alone).
    """
    from femtoolkit.continuum.tensor import voigt_strain_to_tensor

    strain_tensor = voigt_strain_to_tensor(np.asarray(strain_voigt, dtype=float))
    return 2.0 * strain_tensor + np.eye(3)
