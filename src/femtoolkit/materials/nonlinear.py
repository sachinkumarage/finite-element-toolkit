"""Nonlinear constitutive material interface, an elastic adapter, and a 1D plasticity model.

Every material model through Version 12 (:class:`~femtoolkit.materials.material.Material`,
:class:`~femtoolkit.materials.linear_elastic_2d.LinearElastic2D`) is
**linear**: stress is a fixed matrix (or scalar) times strain,
``sigma = D @ epsilon``, with no history and no state. Nonlinear
analysis (:mod:`femtoolkit.analysis.nonlinear_analysis`) needs a
constitutive model that can report a *different* stress for the same
strain depending on what strain history the material has already been
through (e.g. whether it has yielded) -- :class:`NonlinearMaterial` is
that interface, and this module keeps it completely separate from
(and does not modify) the existing linear material classes.

**Trial vs. committed state.** A Newton-Raphson iteration proposes a
displacement update that may or may not lead to a converged load step.
Evaluating a material at that proposed (*trial*) strain must never
overwrite the material's last known-good (*committed*) state -- an
un-converged iteration must be fully discardable. :meth:`NonlinearMaterial.trial_state`
enforces this by construction: it is a **pure function** of
``(strain, committed_state) -> new_state``, never mutating
``committed_state`` (which is itself an immutable
:class:`MaterialState`). The caller
(:mod:`femtoolkit.analysis.nonlinear_analysis`) decides when a trial
state becomes the new committed state -- only once its load step
converges.

Two concrete materials are provided:

* :class:`ElasticMaterialAdapter` -- wraps an existing (linear) modulus
  (a scalar ``E`` or a 2D constitutive matrix ``D``) as a
  :class:`NonlinearMaterial` with no history at all. This exists
  specifically to *validate* the nonlinear solver: solving a linear
  problem through the nonlinear Newton-Raphson machinery must converge,
  in a single iteration per load step, to the same displacement the
  existing linear solver produces (see
  ``tests/validation/test_linear_regression.py``) -- proof the
  nonlinear architecture (internal force, tangent stiffness, residual,
  assembly, boundary conditions) is correct *before* any genuine
  nonlinearity is introduced.
* :class:`ElasticPerfectlyPlasticMaterial1D` -- a real (not
  approximated or faked) uniaxial elastic-perfectly-plastic model, used
  for the 1D nonlinear bar validation. Multiaxial (2D/3D) plasticity --
  which needs a yield *surface* and a return-mapping algorithm, not just
  a stress bound -- is deliberately out of scope for this version (see
  :mod:`femtoolkit.analysis.nonlinear_elements`'s docstring for how CST
  and Q4 nonlinear support is validated instead, with
  :class:`ElasticMaterialAdapter`).

Version 14 (:mod:`femtoolkit.materials.hardening`) builds hardening
plasticity models directly on top of the interface defined here, without
any change to it: :class:`NonlinearMaterial` already had exactly the
three methods (``initial_state``/``trial_state``/``tangent_modulus``) a
hardening return-mapping algorithm needs, and the trial/committed state
pattern already gives hardening materials the same safe-to-discard
Newton iteration guarantee for free. The only change made *here* for
Version 14 is additive: two new, zero-defaulted :class:`MaterialState`
fields (``hardening_variable`` for isotropic hardening's expanding yield
surface, ``back_stress`` for kinematic hardening's translating one) that
every Version 13 material silently ignores.

Version 15 (:mod:`femtoolkit.materials.j2_plasticity`) adds one more
zero-defaulted field the same way: ``plastic_multiplier`` (``Delta
gamma``, the incremental plastic multiplier produced by *this*
``trial_state`` call). Unlike ``hardening_variable``/``back_stress``
(the accumulated, committed-and-carried-forward state), this is a
per-call scalar, always ``0.0`` for an elastic step and for every
material that does not set it. :class:`~femtoolkit.materials.j2_plasticity.J2Plasticity3D`
uses it to reconstruct its own committed reference state (the plastic
strain and hardening variable *before* this step's correction) inside
``tangent_modulus`` -- which the interface deliberately does not pass
``committed_state`` to -- so it can differentiate its own return map
numerically without mutating any shared instance state (see that
module's docstring for why this matters for Gauss-point independence).

Version 18 (:mod:`femtoolkit.materials.finite_strain_plasticity`) adds
one final field, ``plastic_deformation_gradient``: small-strain
plasticity's *additive* split ``epsilon = epsilon_e + epsilon_p`` has no
meaning once strains are large enough that rotation must be tracked
exactly, so finite-strain plasticity instead carries the **plastic
deformation gradient** ``Fp`` -- the multiplicative decomposition
``F = Fe @ Fp`` splits total deformation into an (irrecoverable) plastic
part and a (recoverable, stress-producing) elastic part. Defaulted to
``None`` (not an identity matrix -- see the field's own docstring for
why), so every material through Version 17 is completely unaffected by
this field's addition; a finite-strain plastic material always sets it.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from femtoolkit.exceptions import ValidationError

if TYPE_CHECKING:
    from femtoolkit.materials.linear_elastic_2d import LinearElastic2D
    from femtoolkit.materials.linear_elastic_3d import LinearElastic3D
    from femtoolkit.materials.material import Material

_PLASTIC_TANGENT_RATIO = 1e-6
"""Fraction of the elastic modulus used as the tangent modulus once a
:class:`ElasticPerfectlyPlasticMaterial1D` point has yielded, instead of
a mathematically "exact" zero. A truly zero tangent at every yielded
point in a model would make the (reduced) tangent stiffness matrix
exactly singular the moment every element has yielded -- an avoidable,
purely numerical failure mode, not a physical one (real materials also
have at least a small residual stiffness in this regime, if only from
strain-rate or geometric effects this toolkit does not model). This
small residual stiffness keeps ``K_t`` solvable while remaining
negligible relative to the elastic modulus.
"""


@dataclass(frozen=True)
class MaterialState:
    """An immutable snapshot of a nonlinear material's state at one strain point.

    Attributes:
        strain: Total strain, a scalar (1D materials) or a length-3
            NumPy array (``[epsilon_x, epsilon_y, gamma_xy]``, 2D
            continuum materials).
        stress: Stress conjugate to ``strain``, same shape.
        plastic_strain: Accumulated plastic (permanent) strain, same
            shape as ``strain``. Zero for a purely elastic material.
        yielded: Whether this state is on (or beyond) the material's
            yield surface. A plain ``bool`` for every material introduced
            through Version 13, or a boolean array with one entry per
            independently-evaluated component for a decoupled
            multi-component material (see
            :class:`~femtoolkit.materials.hardening.DecoupledIsotropicHardeningAdapter2D`).
        hardening_variable: The accumulated (isotropic) hardening
            variable, conventionally written ``alpha`` -- the total
            accumulated plastic strain magnitude, which drives how far
            an isotropic yield surface has expanded
            (:class:`~femtoolkit.materials.hardening.BilinearIsotropicHardeningMaterial1D`).
            Zero for materials with no isotropic hardening (the default,
            so every Version 13 material is unaffected by this field's
            addition).
        back_stress: The kinematic hardening back-stress, conventionally
            written ``X`` -- the center of a *translated* (rather than
            expanded) yield surface
            (:class:`~femtoolkit.materials.hardening.BilinearKinematicHardeningMaterial1D`).
            Zero for materials with no kinematic hardening (the default).
        plastic_multiplier: The incremental plastic multiplier
            (``Delta gamma``) produced by the ``trial_state`` call that
            returned this state -- always a plain scalar (unlike
            ``hardening_variable``, which may be a per-component array
            for a decoupled multi-component material), since a single
            integration point has one consistency parameter regardless
            of how many strain components it carries. Zero for an
            elastic step and for every material that does not set it
            (the default, so every prior material is unaffected by this
            field's addition). See
            :class:`~femtoolkit.materials.j2_plasticity.J2Plasticity3D`
            for the one material that uses it.
        plastic_deformation_gradient: The 3x3 plastic deformation
            gradient ``Fp`` (Version 18), for materials using the
            multiplicative decomposition ``F = Fe @ Fp``. ``None`` (the
            default, not an identity matrix) for every material that
            does not use this field -- a real ``np.eye(3)`` default
            would be indistinguishable from "this material genuinely
            computed an identity ``Fp``" and cannot be used as a dataclass
            field default besides (NumPy arrays are unhashable). See
            :class:`~femtoolkit.materials.finite_strain_plasticity.FiniteStrainPlasticMaterial`
            for the one family of materials that sets it.
    """

    strain: float | np.ndarray
    stress: float | np.ndarray
    plastic_strain: float | np.ndarray
    yielded: bool | np.ndarray
    hardening_variable: float | np.ndarray = 0.0
    back_stress: float | np.ndarray = 0.0
    plastic_multiplier: float = 0.0
    plastic_deformation_gradient: np.ndarray | None = None

    @property
    def elastic_strain(self) -> float | np.ndarray:
        """The elastic strain, ``epsilon_e = epsilon - epsilon_p``.

        Not stored directly (there is nothing to keep consistent by
        storing it -- it is always exactly this difference), but exposed
        as a convenience since "elastic strain" is a state variable
        engineers reason about directly.
        """
        return self.strain - self.plastic_strain

    @staticmethod
    def zero(like: float | np.ndarray = 0.0) -> MaterialState:
        """Build the zero-strain, zero-stress, unyielded initial state.

        Args:
            like: A scalar ``0.0`` (default, for 1D materials) or an
                array whose shape the zero state should match (2D
                continuum materials, e.g. ``np.zeros(3)``).

        Returns:
            A :class:`MaterialState` with every field at zero.
        """
        zero_value = 0.0 if np.isscalar(like) else np.zeros_like(np.asarray(like, dtype=float))
        return MaterialState(
            strain=zero_value,
            stress=zero_value,
            plastic_strain=zero_value,
            yielded=False,
            hardening_variable=zero_value,
            back_stress=zero_value,
        )


class NonlinearMaterial(ABC):
    """Abstract interface for a (possibly path-dependent) nonlinear constitutive model.

    Concrete materials implement three pure functions: an initial
    (zero-strain) state, a trial-state update from a committed state and
    a new total strain, and a tangent modulus consistent with a given
    state. Nothing in this interface mutates a :class:`MaterialState` --
    every method returns a new one, keeping "trial" and "committed"
    strictly separate (see the module docstring).
    """

    @abstractmethod
    def initial_state(self) -> MaterialState:
        """Return this material's zero-strain initial state."""
        raise NotImplementedError

    @abstractmethod
    def trial_state(
        self, strain: float | np.ndarray, committed_state: MaterialState
    ) -> MaterialState:
        """Compute a new trial state for a given total strain.

        Must be a pure function: ``committed_state`` (and any state
        object previously returned) is never modified.

        Args:
            strain: The proposed (trial) total strain -- a scalar or a
                length-3 array, matching ``committed_state.strain``'s shape.
            committed_state: The material's last converged state (the
                start-of-step state every trial evaluation within that
                step is measured from).

        Returns:
            A new :class:`MaterialState` reflecting ``strain``.
        """
        raise NotImplementedError

    @abstractmethod
    def tangent_modulus(self, state: MaterialState) -> float | np.ndarray:
        """Return ``D_t = d(stress)/d(strain)`` consistent with ``state``.

        Args:
            state: The state (typically a trial state) to evaluate the
                tangent at.

        Returns:
            A scalar (1D materials) or a 3x3 NumPy array (2D continuum
            materials).
        """
        raise NotImplementedError


@dataclass(frozen=True)
class ElasticMaterialAdapter(NonlinearMaterial):
    """A linear-elastic material exposed through the :class:`NonlinearMaterial` interface.

    Has no history at all: every trial state depends only on the
    current strain (``stress = modulus @ strain`` or ``modulus * strain``),
    never on ``committed_state``, and the tangent modulus is always the
    same constant ``modulus``. Solving through the nonlinear
    Newton-Raphson machinery with this material must converge in
    exactly one iteration per load step (the residual is already zero
    after the first, exact linear solve) and reproduce the existing
    linear solver's displacement -- the critical validation this
    material exists for (see the module docstring).

    Attributes:
        modulus: Young's modulus ``E`` (a positive scalar, for 1D use),
            a constitutive matrix ``D`` (a 3x3 NumPy array, for 2D
            continuum use), or a 6x6 NumPy array (for 3D solid use,
            Version 15 -- see :meth:`from_linear_elastic_3d`).

    Raises:
        ValidationError: If ``modulus`` is a scalar that is not positive
            and finite, or an array that is not 3x3 or 6x6.

    Example:
        >>> adapter = ElasticMaterialAdapter.from_material(steel)
        >>> adapter_2d = ElasticMaterialAdapter.from_linear_elastic_2d(plane_stress_steel)
        >>> adapter_3d = ElasticMaterialAdapter.from_linear_elastic_3d(steel_3d)
    """

    modulus: float | np.ndarray

    def __post_init__(self) -> None:
        """Validate the modulus immediately after construction.

        Raises:
            ValidationError: If ``modulus`` is a scalar that is not
                positive and finite, or an array that is not 3x3 or 6x6.
        """
        if np.isscalar(self.modulus):
            if not math.isfinite(self.modulus) or self.modulus <= 0:
                raise ValidationError(
                    f"ElasticMaterialAdapter modulus must be positive, got {self.modulus}."
                )
        else:
            modulus_array = np.asarray(self.modulus)
            if modulus_array.shape not in ((3, 3), (6, 6)):
                raise ValidationError(
                    "ElasticMaterialAdapter modulus array must have shape (3, 3) or (6, 6), "
                    f"got {modulus_array.shape}."
                )

    @classmethod
    def from_material(cls, material: Material) -> ElasticMaterialAdapter:
        """Build an adapter from an existing 1D :class:`~femtoolkit.materials.material.Material`.

        Args:
            material: The linear material to wrap (only its
                ``youngs_modulus`` is used).

        Returns:
            A scalar-modulus :class:`ElasticMaterialAdapter`.
        """
        return cls(modulus=material.youngs_modulus)

    @classmethod
    def from_linear_elastic_2d(cls, material: LinearElastic2D) -> ElasticMaterialAdapter:
        """Build an adapter from an existing 2D linear-elastic material.

        Args:
            material: The linear material to wrap (its
                ``constitutive_matrix`` is used).

        Returns:
            A 3x3-matrix-modulus :class:`ElasticMaterialAdapter`.
        """
        return cls(modulus=material.constitutive_matrix)

    @classmethod
    def from_linear_elastic_3d(cls, material: LinearElastic3D) -> ElasticMaterialAdapter:
        """Build an adapter from an existing 3D linear-elastic material (Version 15).

        Exists for exactly the reason :meth:`from_linear_elastic_2d` does:
        validating that solving a linear 3D (TET4/HEX8) problem through
        the nonlinear Newton-Raphson machinery converges in one iteration
        per load step and reproduces
        :class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis`'s
        displacement exactly, before genuine 3D plasticity
        (:class:`~femtoolkit.materials.j2_plasticity.J2Plasticity3D`) is
        introduced.

        Args:
            material: The linear material to wrap (its
                ``constitutive_matrix`` is used).

        Returns:
            A 6x6-matrix-modulus :class:`ElasticMaterialAdapter`.
        """
        return cls(modulus=material.constitutive_matrix)

    def initial_state(self) -> MaterialState:
        """Return the zero-strain state, shaped to match ``modulus``."""
        like = 0.0 if np.isscalar(self.modulus) else np.zeros(np.asarray(self.modulus).shape[0])
        return MaterialState.zero(like)

    def trial_state(
        self, strain: float | np.ndarray, committed_state: MaterialState
    ) -> MaterialState:
        """Compute ``stress = modulus @ strain`` (or ``modulus * strain``); never yields."""
        if np.isscalar(self.modulus):
            stress = self.modulus * strain
            plastic_strain = 0.0
        else:
            stress = self.modulus @ np.asarray(strain, dtype=float)
            plastic_strain = np.zeros(np.asarray(self.modulus).shape[0])
        return MaterialState(
            strain=strain, stress=stress, plastic_strain=plastic_strain, yielded=False
        )

    def tangent_modulus(self, state: MaterialState) -> float | np.ndarray:
        """Return the constant ``modulus``, independent of ``state``."""
        return self.modulus


@dataclass(frozen=True)
class ElasticPerfectlyPlasticMaterial1D(NonlinearMaterial):
    """A uniaxial elastic-perfectly-plastic material: ``sigma = E*epsilon`` until yield.

    .. code-block:: text

        Elastic:   sigma = E * epsilon                    while |sigma| <= sigma_y
        Yielded:   sigma = sign(epsilon_elastic) * sigma_y  once |sigma| would exceed sigma_y

    No hardening: once yielded, stress stays exactly at
    ``+/- sigma_y`` regardless of how much further strain accumulates
    in the same direction (all additional strain becomes plastic
    strain). Because a scalar stress can only ever be inside, on, or
    trying to cross a single interval ``[-sigma_y, sigma_y]``, the
    "return mapping" from a trial elastic stress back onto the yield
    surface is exact and non-iterative in 1D -- unlike a true multiaxial
    yield *surface*, there is no ambiguity about *which* direction to
    return in.

    Attributes:
        youngs_modulus: Elastic (Young's) modulus ``E``, in pascals.
            Must be positive.
        yield_stress: Yield stress ``sigma_y``, in pascals. Must be
            positive.

    Raises:
        ValidationError: If ``youngs_modulus`` or ``yield_stress`` is
            not positive and finite.

    Example:
        >>> steel = ElasticPerfectlyPlasticMaterial1D(youngs_modulus=200e9, yield_stress=250e6)
        >>> state = steel.trial_state(strain=0.002, committed_state=steel.initial_state())
        >>> state.yielded
        True
    """

    youngs_modulus: float
    yield_stress: float

    def __post_init__(self) -> None:
        """Validate the elastic-plastic parameters immediately after construction.

        Raises:
            ValidationError: If ``youngs_modulus`` or ``yield_stress``
                is not positive and finite.
        """
        if not math.isfinite(self.youngs_modulus) or self.youngs_modulus <= 0:
            raise ValidationError(
                "ElasticPerfectlyPlasticMaterial1D youngs_modulus must be positive, got "
                f"{self.youngs_modulus}."
            )
        if not math.isfinite(self.yield_stress) or self.yield_stress <= 0:
            raise ValidationError(
                "ElasticPerfectlyPlasticMaterial1D yield_stress must be positive, got "
                f"{self.yield_stress}."
            )

    def initial_state(self) -> MaterialState:
        """Return the zero-strain, zero-stress, unyielded initial state."""
        return MaterialState.zero(0.0)

    def trial_state(self, strain: float, committed_state: MaterialState) -> MaterialState:
        """Return-map a trial elastic stress back onto ``[-sigma_y, sigma_y]`` if needed.

        Args:
            strain: The proposed total (scalar) strain.
            committed_state: The material's last converged state,
                supplying the starting accumulated plastic strain.

        Returns:
            A new :class:`MaterialState` at ``strain``.
        """
        elastic_strain = strain - committed_state.plastic_strain
        trial_stress = self.youngs_modulus * elastic_strain

        if abs(trial_stress) <= self.yield_stress:
            return MaterialState(
                strain=strain,
                stress=trial_stress,
                plastic_strain=committed_state.plastic_strain,
                yielded=False,
            )

        sign = math.copysign(1.0, trial_stress)
        stress = sign * self.yield_stress
        plastic_strain = strain - stress / self.youngs_modulus
        return MaterialState(
            strain=strain, stress=stress, plastic_strain=plastic_strain, yielded=True
        )

    def tangent_modulus(self, state: MaterialState) -> float:
        """Return ``E`` if elastic, or a small regularized value if yielded.

        See :data:`_PLASTIC_TANGENT_RATIO` for why the plastic tangent
        is a small positive number rather than exactly zero.
        """
        if state.yielded:
            return self.youngs_modulus * _PLASTIC_TANGENT_RATIO
        return self.youngs_modulus
