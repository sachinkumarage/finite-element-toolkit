"""J2 (von Mises) small-strain plasticity with linear isotropic hardening, in 3D.

Every plasticity model through Version 14 is either genuinely scalar (1D
bar materials, :mod:`femtoolkit.materials.nonlinear`,
:mod:`femtoolkit.materials.hardening`) or a *decoupled* per-component
adapter (:class:`~femtoolkit.materials.hardening.DecoupledIsotropicHardeningAdapter2D`)
that applies the scalar return map independently to each Voigt strain
component, with no coupling between them -- honestly documented there as
*not* a physically accurate multiaxial yield surface. :class:`J2Plasticity3D`
is the first genuine multiaxial plasticity model in this toolkit: a real
von Mises yield surface in 6-component stress space, with an associative
flow rule and a proper tensor return-mapping algorithm, implemented on
the *exact same* :class:`~femtoolkit.materials.nonlinear.NonlinearMaterial`
interface (``initial_state``/``trial_state``/``tangent_modulus``) and
trial/committed :class:`~femtoolkit.materials.nonlinear.MaterialState`
pattern every prior material uses -- no new material protocol needed.

**Why the deviatoric stress, not the full stress.** Physically, yielding
in metals is caused by dislocation slip along crystallographic planes,
driven by *shear* (shape-distorting) stress -- not by a uniform
hydrostatic pressure or tension, which changes a material's volume but
not its shape. This is an experimentally verified fact (see e.g.
Bridgman's high-pressure experiments): ordinary metals' yield behavior is
essentially unaffected by superimposed hydrostatic stress, even at
pressures large enough to matter. The von Mises yield criterion encodes
this directly by building the yield function entirely from the
**deviatoric** stress ``s = sigma - sigma_m * I`` (see
:mod:`femtoolkit.continuum.tensor`), which has the hydrostatic part
subtracted out by construction:

.. code-block:: text

    f(sigma, sigma_y) = sqrt(3/2 * s:s) - sigma_y = sigma_vm - sigma_y

**Associative flow and linear isotropic hardening.** The plastic strain
increment direction follows the yield surface's outward normal
(associative flow, the simplest and most common choice for metals):

.. code-block:: text

    Delta epsilon_p = Delta gamma * df/dsigma = Delta gamma * (3/2) * s / sigma_vm

and the yield stress grows linearly with the accumulated equivalent
plastic strain ``alpha`` (Version 15's scope, per the assignment brief --
nonlinear hardening laws are out of scope):

.. code-block:: text

    sigma_y = sigma_y0 + H * alpha

**Radial return mapping.** For isotropic elasticity plus associative J2
flow with *linear* hardening, the classical radial-return algorithm gives
a closed-form (non-iterative) solution -- no local Newton loop is needed,
unlike general (nonlinear-hardening) J2 plasticity:

.. code-block:: text

    sigma_trial = D @ (epsilon - epsilon_p_old)          (elastic predictor, full Hooke's law)
    s_trial     = deviatoric(sigma_trial)
    sigma_vm_trial = sqrt(3/2 * s_trial:s_trial)
    f_trial     = sigma_vm_trial - (sigma_y0 + H*alpha_old)

    if f_trial <= tolerance: elastic step
    else:
        Delta gamma = f_trial / (3*mu + H)                (mu = shear modulus; closed form,
                                                             the exact 3D analogue of the 1D
                                                             `f_trial / (E + H)` formula in
                                                             femtoolkit.materials.hardening)
        n           = (3/2) * s_trial / sigma_vm_trial     (flow direction; preserved
                                                             through the return, by the
                                                             classical radial-return theorem)
        s           = s_trial * (1 - 3*mu*Delta gamma / sigma_vm_trial)
        sigma       = s + hydrostatic(sigma_trial)         (hydrostatic part is untouched --
                                                             see the module docstring's note
                                                             on why)
        epsilon_p   = epsilon_p_old + Delta gamma * n
        alpha       = alpha_old + Delta gamma

The name "radial return" describes exactly this geometric picture: the
trial deviatoric stress is scaled straight back ("returned") toward the
origin, along the same radial direction ``n`` it already points in --
never rotated -- until it lands back on the (grown) yield surface.

**Algorithmic tangent.** :meth:`J2Plasticity3D.tangent_modulus` receives
only the trial ``state`` -- the :class:`~femtoolkit.materials.nonlinear.NonlinearMaterial`
interface deliberately does not pass ``committed_state`` to it (see that
module's docstring). A fully consistent closed-form tangent for radial
return is a well-known but easy-to-mis-derive tensor formula (a wrong
sign or factor there would not fail a basic correctness test -- Newton-
Raphson would still converge, just sub-optimally -- making it exactly the
kind of silent, hard-to-catch bug this project's "mathematical
correctness first" priority warns against). Instead, this class uses the
numerical-differentiation fallback the assignment brief explicitly
sanctions for this situation: it reconstructs its own committed reference
(``epsilon_p_old``, ``alpha_old``) from the single extra scalar
:attr:`~femtoolkit.materials.nonlinear.MaterialState.plastic_multiplier`
now carried on ``state`` (see that field's docstring), then evaluates its
*own* return-mapping formula above at ``state.strain +/- h`` for each of
the 6 Voigt components via central differences, and symmetrizes the
result (the exact tangent is provably symmetric for associative
plasticity; central differencing only approximates that, so the
symmetrization removes purely numerical asymmetry). This is
self-consistent with the actual return map *by construction* -- it
cannot drift from the real algorithm the way a separately-derived
closed-form tensor formula could -- at the cost of 12 extra material
evaluations per tangent request, negligible next to a linear solve.
Elastic steps skip all of this and return the exact elastic ``D``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from femtoolkit.continuum.constitutive import isotropic_3d_matrix
from femtoolkit.continuum.tensor import (
    deviatoric_stress,
    hydrostatic_stress,
    tensor_to_voigt_strain,
    tensor_to_voigt_stress,
    voigt_strain_to_tensor,
    voigt_stress_to_tensor,
    von_mises_stress_from_tensor,
)
from femtoolkit.exceptions import (
    ConstitutiveUpdateError,
    InvalidMaterialStateError,
    ValidationError,
)
from femtoolkit.materials.nonlinear import MaterialState, NonlinearMaterial

_MIN_POISSONS_RATIO = -1.0
_MAX_POISSONS_RATIO = 0.5

YIELD_FUNCTION_TOLERANCE: float = 1e-6
"""Absolute stress tolerance (Pa) a trial yield-function value must exceed
before a plastic correction is triggered -- the same value and rationale
as :data:`femtoolkit.materials.hardening.YIELD_FUNCTION_TOLERANCE`
(kept as an independent module-level constant here rather than importing
it, since the two modules are otherwise independent and this avoids a
cross-import purely for a shared numerical constant).
"""

_TANGENT_RELATIVE_STEP: float = 1e-6
"""Relative finite-difference step size used by :meth:`J2Plasticity3D.tangent_modulus`
for each Voigt strain component, scaled by that component's own
magnitude (see :data:`_TANGENT_ABSOLUTE_FLOOR` for the floor applied when
that magnitude is ~zero)."""

_TANGENT_ABSOLUTE_FLOOR: float = 1e-9
"""Absolute floor (strain units) for the per-component finite-difference
step in :meth:`J2Plasticity3D.tangent_modulus`, preventing a zero step
when a Voigt strain component is (numerically) exactly zero."""


def _require_j2_state(committed_state: MaterialState) -> None:
    """Reject a committed state that is not a length-6 (3D Voigt) state.

    Raises:
        InvalidMaterialStateError: If ``committed_state.plastic_strain``
            is not a length-6 array.
    """
    plastic_strain = np.asarray(committed_state.plastic_strain, dtype=float)
    if plastic_strain.shape != (6,):
        raise InvalidMaterialStateError(
            "J2Plasticity3D requires a 3D, 6-component (Voigt) committed MaterialState, "
            f"got plastic_strain of shape {plastic_strain.shape}. Use a 3D-capable "
            "material for TET4/HEX8 solid elements instead."
        )


def _require_finite_strain(strain: np.ndarray) -> None:
    """Reject a non-finite strain before it can corrupt a return-mapping computation.

    Raises:
        ConstitutiveUpdateError: If any component of ``strain`` is not finite.
    """
    if not np.all(np.isfinite(strain)):
        raise ConstitutiveUpdateError(
            f"J2Plasticity3D received a non-finite strain ({strain!r}); this usually "
            "indicates a diverging Newton-Raphson iteration upstream, not a problem "
            "with the material model itself."
        )


@dataclass(frozen=True)
class J2Plasticity3D(NonlinearMaterial):
    """A 3D von Mises (J2) elastoplastic material with linear isotropic hardening.

    See the module docstring for the full radial-return derivation. State
    (total strain, plastic strain tensor, equivalent plastic strain,
    stress) is carried entirely in the
    :class:`~femtoolkit.materials.nonlinear.MaterialState` the caller
    passes in and gets back -- this class itself holds no mutable state,
    so a single instance is safe to share across many independent
    integration points (see :mod:`femtoolkit.analysis.nonlinear_elements`
    for how TET4/HEX8 give each Gauss point its own state).

    Attributes:
        youngs_modulus: Elastic (Young's) modulus ``E``, in pascals.
            Must be positive.
        poisson_ratio: Poisson's ratio ``v`` (dimensionless). Must lie in
            ``(-1.0, 0.5)``, the physically valid range for an isotropic
            elastic material.
        yield_stress: Initial (virgin) yield stress ``sigma_y0``, in
            pascals. Must be positive.
        hardening_modulus: Isotropic hardening modulus ``H``, in
            pascals. Must be non-negative (``0`` reproduces perfectly
            plastic von Mises behavior).

    Raises:
        ValidationError: If any parameter is invalid.

    Example:
        >>> steel = J2Plasticity3D(
        ...     youngs_modulus=200e9, poisson_ratio=0.3, yield_stress=250e6,
        ...     hardening_modulus=10e9,
        ... )
        >>> state = steel.trial_state(
        ...     strain=np.array([0.003, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ...     committed_state=steel.initial_state(),
        ... )
        >>> state.yielded
        True
    """

    youngs_modulus: float
    poisson_ratio: float
    yield_stress: float
    hardening_modulus: float

    def __post_init__(self) -> None:
        """Validate every parameter immediately after construction.

        Raises:
            ValidationError: If ``youngs_modulus`` or ``poisson_ratio``
                is not a physically valid, finite value, or
                ``yield_stress`` is not positive, or ``hardening_modulus``
                is negative.
        """
        if not math.isfinite(self.youngs_modulus) or self.youngs_modulus <= 0:
            raise ValidationError(
                f"J2Plasticity3D youngs_modulus must be positive, got {self.youngs_modulus}."
            )
        if not math.isfinite(self.poisson_ratio) or not (
            _MIN_POISSONS_RATIO < self.poisson_ratio < _MAX_POISSONS_RATIO
        ):
            raise ValidationError(
                "J2Plasticity3D poisson_ratio must be within "
                f"({_MIN_POISSONS_RATIO}, {_MAX_POISSONS_RATIO}), got {self.poisson_ratio}."
            )
        if not math.isfinite(self.yield_stress) or self.yield_stress <= 0:
            raise ValidationError(
                f"J2Plasticity3D yield_stress must be positive, got {self.yield_stress}."
            )
        if not math.isfinite(self.hardening_modulus) or self.hardening_modulus < 0:
            raise ValidationError(
                "J2Plasticity3D hardening_modulus must be non-negative, got "
                f"{self.hardening_modulus}."
            )

    @property
    def constitutive_matrix(self) -> np.ndarray:
        """The material's elastic 6x6 constitutive matrix ``D``."""
        return isotropic_3d_matrix(self.youngs_modulus, self.poisson_ratio)

    @property
    def shear_modulus(self) -> float:
        """The material's shear modulus, ``mu = E / (2 * (1 + v))``."""
        return self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))

    def initial_state(self) -> MaterialState:
        """Return the zero-strain, zero-stress, unyielded initial state."""
        return MaterialState(
            strain=np.zeros(6),
            stress=np.zeros(6),
            plastic_strain=np.zeros(6),
            yielded=False,
            hardening_variable=0.0,
            back_stress=0.0,
            plastic_multiplier=0.0,
        )

    def _return_map(
        self, strain: np.ndarray, plastic_strain_old: np.ndarray, alpha_old: float
    ) -> tuple[np.ndarray, np.ndarray, float, float, bool]:
        """Evaluate the closed-form elastic-predictor/radial-return algorithm.

        Pure function of its three arguments -- no reference to
        ``self``'s mutable state (there is none) or to any committed
        :class:`~femtoolkit.materials.nonlinear.MaterialState`. Shared by
        :meth:`trial_state` (called once, with the true committed
        reference) and :meth:`tangent_modulus` (called several times,
        with a reconstructed reference held fixed, for numerical
        differentiation -- see the module docstring).

        Args:
            strain: Total (trial) strain, Voigt ``[xx, yy, zz, xy, yz, xz]``.
            plastic_strain_old: Plastic strain at the start of this
                evaluation, same Voigt convention.
            alpha_old: Equivalent plastic strain at the start of this
                evaluation.

        Returns:
            ``(stress, plastic_strain_new, alpha_new, delta_gamma, yielded)``.
        """
        shear_modulus = self.shear_modulus
        elastic_trial_strain = strain - plastic_strain_old
        trial_stress_voigt = self.constitutive_matrix @ elastic_trial_strain
        trial_stress_tensor = voigt_stress_to_tensor(trial_stress_voigt)
        s_trial = deviatoric_stress(trial_stress_tensor)
        sigma_vm_trial = von_mises_stress_from_tensor(trial_stress_tensor)
        current_yield_stress = self.yield_stress + self.hardening_modulus * alpha_old
        f_trial = sigma_vm_trial - current_yield_stress

        if f_trial <= YIELD_FUNCTION_TOLERANCE:
            return trial_stress_voigt, plastic_strain_old, alpha_old, 0.0, False

        delta_gamma = f_trial / (3.0 * shear_modulus + self.hardening_modulus)
        flow_direction = 1.5 * s_trial / sigma_vm_trial

        s = s_trial * (1.0 - 3.0 * shear_modulus * delta_gamma / sigma_vm_trial)
        stress_tensor = s + hydrostatic_stress(trial_stress_tensor)
        stress_voigt = tensor_to_voigt_stress(stress_tensor)

        plastic_strain_old_tensor = voigt_strain_to_tensor(plastic_strain_old)
        plastic_strain_new_tensor = plastic_strain_old_tensor + delta_gamma * flow_direction
        plastic_strain_new = tensor_to_voigt_strain(plastic_strain_new_tensor)

        alpha_new = alpha_old + delta_gamma
        return stress_voigt, plastic_strain_new, alpha_new, delta_gamma, True

    def trial_state(self, strain: np.ndarray, committed_state: MaterialState) -> MaterialState:
        """Return-map a trial elastic stress onto the (possibly expanded) yield surface.

        Args:
            strain: The proposed total strain, Voigt
                ``[epsilon_xx, epsilon_yy, epsilon_zz, gamma_xy, gamma_yz, gamma_xz]``.
            committed_state: The material's last converged state,
                supplying the starting plastic strain and equivalent
                plastic strain.

        Returns:
            A new :class:`~femtoolkit.materials.nonlinear.MaterialState` at ``strain``.

        Raises:
            InvalidMaterialStateError: If ``committed_state`` is not a
                3D (6-component) state.
            ConstitutiveUpdateError: If ``strain`` is not finite.
        """
        _require_j2_state(committed_state)
        strain_array = np.asarray(strain, dtype=float)
        _require_finite_strain(strain_array)

        plastic_strain_old = np.asarray(committed_state.plastic_strain, dtype=float)
        alpha_old = float(committed_state.hardening_variable)

        stress, plastic_strain_new, alpha_new, delta_gamma, yielded = self._return_map(
            strain_array, plastic_strain_old, alpha_old
        )
        return MaterialState(
            strain=strain_array,
            stress=stress,
            plastic_strain=plastic_strain_new,
            yielded=yielded,
            hardening_variable=alpha_new,
            back_stress=0.0,
            plastic_multiplier=delta_gamma,
        )

    def tangent_modulus(self, state: MaterialState) -> np.ndarray:
        """Return the elastic ``D`` on an elastic step, or a numerical tangent on a plastic one.

        See the module docstring for the full method and its rationale.

        Args:
            state: The (typically trial) state to evaluate the tangent at.

        Returns:
            A 6x6 NumPy array.
        """
        if not state.yielded or state.plastic_multiplier <= 0.0:
            return self.constitutive_matrix

        delta_gamma = state.plastic_multiplier
        alpha_old = float(state.hardening_variable) - delta_gamma

        stress_tensor = voigt_stress_to_tensor(np.asarray(state.stress, dtype=float))
        s_final = deviatoric_stress(stress_tensor)
        sigma_vm_final = von_mises_stress_from_tensor(stress_tensor)
        flow_direction = 1.5 * s_final / sigma_vm_final

        plastic_strain_new_tensor = voigt_strain_to_tensor(
            np.asarray(state.plastic_strain, dtype=float)
        )
        plastic_strain_old_tensor = plastic_strain_new_tensor - delta_gamma * flow_direction
        plastic_strain_old = tensor_to_voigt_strain(plastic_strain_old_tensor)

        strain0 = np.asarray(state.strain, dtype=float)
        jacobian = np.zeros((6, 6))
        for component in range(6):
            step = _TANGENT_RELATIVE_STEP * max(abs(strain0[component]), _TANGENT_ABSOLUTE_FLOOR)
            perturbation = np.zeros(6)
            perturbation[component] = step

            stress_plus, *_ = self._return_map(
                strain0 + perturbation, plastic_strain_old, alpha_old
            )
            stress_minus, *_ = self._return_map(
                strain0 - perturbation, plastic_strain_old, alpha_old
            )
            jacobian[:, component] = (stress_plus - stress_minus) / (2.0 * step)

        # The exact consistent tangent is symmetric for associative J2
        # plasticity; central differencing only approximates that, so
        # symmetrizing removes purely numerical asymmetry.
        return 0.5 * (jacobian + jacobian.T)
