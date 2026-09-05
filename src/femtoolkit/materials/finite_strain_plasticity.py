"""Finite-strain (multiplicative) J2 plasticity in 3D (Version 18).

Every plastic material through Version 15 (:mod:`femtoolkit.materials.j2_plasticity`)
uses the **additive** small-strain decomposition ``epsilon = epsilon_e +
epsilon_p``: valid only while strains and rotations stay small enough
that a linear strain measure is meaningful. Version 16/17 introduced
genuinely large deformation (the deformation gradient ``F``, Green-
Lagrange strain, hyperelasticity) but no plasticity yet -- this module is
where the two combine.

**Multiplicative decomposition.** For large deformation, the physically
correct generalization of the additive split is *multiplicative*:

.. code-block:: text

    F = Fe @ Fp

``Fp`` (the **plastic deformation gradient**) maps the reference
configuration to a fictitious, stress-free **intermediate configuration**
-- the shape the material would relax to if suddenly unloaded, capturing
permanent (irrecoverable) deformation. ``Fe`` (the **elastic deformation
gradient**) maps that intermediate configuration to the actual current
one, and is what determines stress -- exactly analogous to how, in the
small-strain theory, only ``epsilon_e = epsilon - epsilon_p`` (not the
total strain) enters the elastic law.

**Why this module only ever needs ``C``, never ``F`` itself.** Exactly
like :class:`~femtoolkit.materials.hyperelastic.HyperelasticMaterial`
(Version 17), :meth:`NonlinearMaterial.trial_state` only ever receives
Green-Lagrange strain -- equivalently, the right Cauchy-Green tensor
``C = F^T F`` -- never ``F`` itself. This turns out to be exactly enough:
the **trial elastic right Cauchy-Green tensor**, defined on the
intermediate configuration,

.. code-block:: text

    Ce_trial = Fe_trial^T @ Fe_trial = Fp_old^-T @ C @ Fp_old^-1

depends only on ``C`` (known) and ``Fp_old`` (the committed plastic
deformation gradient from the previous step) -- never on ``F`` itself.
One direct consequence: this formulation is **objective by construction,
unconditionally** -- not just for a stress-free initial state, but for
*any* state -- because the material function literally cannot see
whether a rigid rotation was superposed on ``F`` (``C`` is identical
either way). Section 11's mandated rigid-rotation test is still included
explicitly, as a concrete regression check, even though the deeper
guarantee is architectural.

**Principal logarithmic (Hencky) strain and the exponential return map.**
Eigendecomposing the symmetric ``Ce_trial = sum_i lambda_i^2 * m_i (x) m_i``
gives three principal elastic stretches ``lambda_i`` and orthonormal
directions ``m_i`` (on the intermediate configuration). Their logarithms,
``epsilon_e_i = ln(lambda_i)``, are the **principal elastic (Hencky) log
strains** -- and a well-established result in finite-strain plasticity
(Weber & Anand 1990; Simo 1992) is that, for an **isotropic** elastic law
expressed in this principal log-strain space using the *same* linear
(Lame) relation as ordinary small-strain elasticity,

.. code-block:: text

    tau_i = 2*mu*epsilon_e_i + lambda*(epsilon_e_1 + epsilon_e_2 + epsilon_e_3)

(``tau_i``, principal **Kirchhoff-like stresses** conjugate to
``epsilon_e_i``), the associative-J2-flow radial-return algebra is
**identical in form** to :class:`~femtoolkit.materials.j2_plasticity.J2Plasticity3D`'s
small-strain closed-form return map -- just applied to 3 principal scalars
instead of a 6-component Voigt tensor. This is not a coincidence or an
approximation: it is the precise reason this family of algorithms is
standard practice, and it is *why* this module reduces exactly to
:class:`~femtoolkit.materials.j2_plasticity.J2Plasticity3D` in the
small-strain limit (``ln(1+x) ~= x`` for small ``x`` -- see
``tests/validation/test_finite_strain_plasticity_small_strain_limit.py``).

Because the radial return never rotates the deviatoric flow direction,
the plastic update has an **exact, closed-form (non-iterative) solution**
via the matrix exponential -- coaxial with ``Ce_trial``, so no local
Newton loop is ever needed, exactly like Version 15:

.. code-block:: text

    Fp_new = expm(Delta_gamma * N) @ Fp_old
           = [sum_i exp(Delta_gamma * N_i) * m_i (x) m_i] @ Fp_old

``N_i = 1.5 * s_trial_i / tau_vm_trial`` (the principal flow direction,
the exact 3-component analogue of :class:`J2Plasticity3D`'s ``flow_direction``).

**Pulling stress back to the reference configuration.** The updated
Kirchhoff-like principal stresses convert to principal values of the
**second Piola-Kirchhoff-like stress on the intermediate configuration**,
``Se_i = tau_i / Ce_new_i`` (``Ce_new_i = exp(2*epsilon_e_new_i)``, the
standard PK2-Kirchhoff relation applied per-principal-direction), then
pull back to the reference-configuration PK2 stress this module's callers
expect via the chain rule applied to ``Ce = Fp^-T @ C @ Fp^-1`` (linear in
``C`` at fixed ``Fp``):

.. code-block:: text

    S = Fp_old^-1 @ Se @ Fp_old^-T

**Consistent tangent.** Deriving a closed-form algorithmic tangent for
this eigendecomposition-based return map is a genuinely hard tensor
calculus exercise (spectral derivatives of a matrix logarithm/exponential
with repeated-eigenvalue edge cases) -- exactly the kind of derivation
this project's established policy (:class:`J2Plasticity3D`, Version 15;
:class:`~femtoolkit.materials.mooney_rivlin.MooneyRivlin3D`, Version 17)
says to avoid in favor of a validated numerical tangent. Following that
same policy, :meth:`FiniteStrainPlasticMaterial.tangent_modulus`
reconstructs the committed reference (``Fp_old``, ``alpha_old``) from the
*trial* state alone -- exploiting the same "radial return preserves
direction" fact used to update ``Fp`` -- then central-differences the
full return map over each of the 6 Voigt strain components, symmetrized.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from femtoolkit.continuum.stress import (
    cauchy_stress_from_second_piola_kirchhoff,
    first_piola_kirchhoff_from_second,
)
from femtoolkit.continuum.tensor import (
    tensor_to_voigt_strain,
    tensor_to_voigt_stress,
    voigt_strain_to_tensor,
    voigt_stress_to_tensor,
)
from femtoolkit.exceptions import (
    ConstitutiveUpdateError,
    InvalidDeformationGradientError,
    ValidationError,
)
from femtoolkit.materials.nonlinear import MaterialState, NonlinearMaterial

YIELD_FUNCTION_TOLERANCE: float = 1e-6
"""Absolute stress tolerance (Pa) a trial yield-function value must exceed
before a plastic correction is triggered -- same value and rationale as
:data:`femtoolkit.materials.j2_plasticity.YIELD_FUNCTION_TOLERANCE`."""

MIN_ELASTIC_EIGENVALUE: float = 1e-12
"""Minimum acceptable eigenvalue of the trial elastic right Cauchy-Green
tensor ``Ce_trial``. Mirrors
:data:`femtoolkit.continuum.invariants.MIN_RIGHT_CAUCHY_GREEN_DETERMINANT`'s
role: guards every ``log`` in this module against a non-positive argument
(an inverted or degenerate elastic deformation) before it can happen."""

_TANGENT_RELATIVE_STEP: float = 1e-6
"""Relative finite-difference step size for :meth:`FiniteStrainPlasticMaterial.tangent_modulus`,
scaled by the *overall* strain vector's norm -- see :func:`_tangent_finite_difference_step`
for why a single component's own magnitude is the wrong thing to scale by.

Unlike :data:`femtoolkit.materials.j2_plasticity._TANGENT_RELATIVE_STEP`
(which scales by each Voigt component's *own* magnitude), this module
cannot safely use that convention: this material's return map
eigendecomposes the trial elastic right Cauchy-Green tensor, and for a
loading state with an axisymmetric (repeated-eigenvalue) elastic
stretch -- e.g. any uniaxial strain, ``Ce = diag(1+2E_xx, 1, 1)`` -- the
eigenvectors spanning the degenerate subspace are numerically sensitive
to how that degeneracy is broken. A step scaled by a single strain
component that happens to be exactly zero (as at least 3 of the 6
Voigt components are for any uniaxial state) collapses to a
floating-point-noise-dominated value (confirmed directly during
development: ``1e-6 * max(0, 1e-9) = 1e-15``), which is fine for
:class:`~femtoolkit.materials.j2_plasticity.J2Plasticity3D`'s *linear*
elastic law (immune to step size by construction) but produces a
visibly wrong tangent entry here -- caught by comparing against an
independently-implemented finite difference in
``tests/validation/test_finite_strain_plasticity_tangent.py``. Scaling
by the whole strain vector's norm instead (this function) fixes it, the
same fix already validated for
:mod:`femtoolkit.materials.hyperelastic` in Version 17."""

_TANGENT_ABSOLUTE_FLOOR: float = 1e-9
"""Absolute **minimum step size** (strain units), applied directly as a
floor on the *final* step -- not multiplied again by
:data:`_TANGENT_RELATIVE_STEP` -- so it still gives a well-conditioned
step at the reference configuration (strain exactly zero). See
:data:`_TANGENT_RELATIVE_STEP`'s docstring and
:mod:`femtoolkit.materials.hyperelastic`'s identical pattern."""

_MIN_POISSONS_RATIO = -1.0
_MAX_POISSONS_RATIO = 0.5


def _tangent_finite_difference_step(strain_voigt: np.ndarray) -> float:
    """Return a single finite-difference step size, shared by all 6 Voigt strain components.

    See :data:`_TANGENT_RELATIVE_STEP`'s docstring for why scaling by the
    whole strain vector's norm (rather than each component's own value)
    is required here.
    """
    reference_scale = float(np.linalg.norm(strain_voigt))
    step = _TANGENT_RELATIVE_STEP * reference_scale
    return max(step, _TANGENT_ABSOLUTE_FLOOR)


def _require_finite_strain(strain: np.ndarray) -> None:
    """Reject a non-finite strain before it can corrupt a return-mapping computation.

    Raises:
        ConstitutiveUpdateError: If any component of ``strain`` is not finite.
    """
    if not np.all(np.isfinite(strain)):
        raise ConstitutiveUpdateError(
            f"FiniteStrainPlasticMaterial received a non-finite strain ({strain!r}); "
            "this usually indicates a diverging Newton-Raphson iteration upstream, "
            "not a problem with the material model itself."
        )


def _committed_plastic_deformation_gradient(committed_state: MaterialState) -> np.ndarray:
    """Return ``committed_state.plastic_deformation_gradient``, defaulting to identity.

    A committed state produced by a material that does not know about
    this field (or the very first, hand-built initial state) has
    ``plastic_deformation_gradient = None`` -- physically equivalent to
    "no plastic deformation has occurred yet."
    """
    fp = committed_state.plastic_deformation_gradient
    return np.eye(3) if fp is None else np.asarray(fp, dtype=float)


class FiniteStrainPlasticMaterial(NonlinearMaterial, ABC):
    """ABC for isotropic, multiplicative (``F = Fe @ Fp``) finite-strain plasticity models.

    See the module docstring for the full derivation. Concrete materials
    implement the elastic constants (:attr:`shear_modulus`,
    :attr:`lame_lambda`) and the scalar, principal-space radial-return
    update (:meth:`_principal_return_map`) -- a simple 3-component
    calculation, no tensor calculus required. Every other piece -- forming
    the trial elastic right Cauchy-Green tensor, its eigendecomposition,
    the exact exponential-map plastic update, and pulling the resulting
    stress back to a reference-configuration second Piola-Kirchhoff
    stress -- is shared, generic machinery implemented here once.
    """

    @property
    @abstractmethod
    def shear_modulus(self) -> float:
        """The material's (small-strain-equivalent) shear modulus ``mu``, in pascals."""
        raise NotImplementedError

    @property
    @abstractmethod
    def lame_lambda(self) -> float:
        """The material's (small-strain-equivalent) first Lame parameter, in pascals."""
        raise NotImplementedError

    @abstractmethod
    def _principal_return_map(
        self, principal_elastic_log_strain_trial: np.ndarray, alpha_old: float
    ) -> tuple[np.ndarray, np.ndarray, float, float, bool]:
        """Evaluate the yield-criterion-specific scalar return map, in principal space.

        Args:
            principal_elastic_log_strain_trial: The 3 trial principal
                elastic (Hencky) log strains, ``ln(lambda_i)``.
            alpha_old: The equivalent plastic strain (or other scalar
                isotropic hardening variable) at the start of this
                evaluation.

        Returns:
            ``(principal_elastic_log_strain_new, flow_direction, alpha_new, delta_gamma, yielded)``:
            the corrected principal elastic log strains; the (deviatoric)
            flow direction (all zero for an elastic step); the updated
            hardening variable; the incremental plastic multiplier
            (``0.0`` if elastic); and whether this step yielded.
        """
        raise NotImplementedError

    def initial_state(self) -> MaterialState:
        """Return the zero-strain, zero-stress, unyielded initial state (``Fp = I``)."""
        return MaterialState(
            strain=np.zeros(6),
            stress=np.zeros(6),
            plastic_strain=np.zeros(6),
            yielded=False,
            hardening_variable=0.0,
            back_stress=0.0,
            plastic_multiplier=0.0,
            plastic_deformation_gradient=np.eye(3),
        )

    def _elastic_trial_principal_log_strain(
        self, strain_voigt: np.ndarray, plastic_deformation_gradient_old: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(principal_elastic_log_strain_trial, eigenvectors)`` from ``Ce_trial``.

        Raises:
            InvalidDeformationGradientError: If ``Ce_trial`` is not a
                valid (finite, positive-eigenvalue) tensor -- an
                inverted or degenerate elastic deformation.
        """
        strain_tensor = voigt_strain_to_tensor(strain_voigt)
        right_cauchy_green = 2.0 * strain_tensor + np.eye(3)
        fp_old_inverse = np.linalg.inv(plastic_deformation_gradient_old)
        ce_trial = fp_old_inverse.T @ right_cauchy_green @ fp_old_inverse
        ce_trial = 0.5 * (ce_trial + ce_trial.T)

        if not np.all(np.isfinite(ce_trial)):
            raise InvalidDeformationGradientError(
                f"Trial elastic right Cauchy-Green tensor is non-finite: {ce_trial!r}."
            )
        eigenvalues, eigenvectors = np.linalg.eigh(ce_trial)
        if np.any(eigenvalues < MIN_ELASTIC_EIGENVALUE):
            raise InvalidDeformationGradientError(
                f"Trial elastic right Cauchy-Green tensor has a non-positive eigenvalue "
                f"{eigenvalues!r}; the elastic deformation has inverted or collapsed."
            )
        return 0.5 * np.log(eigenvalues), eigenvectors

    def _principal_kirchhoff_stress(self, principal_elastic_log_strain: np.ndarray) -> np.ndarray:
        """Return the 3 principal Kirchhoff-like stresses conjugate to ``epsilon_e``."""
        return (
            2.0 * self.shear_modulus * principal_elastic_log_strain
            + self.lame_lambda * np.sum(principal_elastic_log_strain)
        )

    def _return_map(
        self,
        strain_voigt: np.ndarray,
        plastic_deformation_gradient_old: np.ndarray,
        alpha_old: float,
    ) -> tuple[np.ndarray, np.ndarray, float, float, bool]:
        """Evaluate the full elastic-predictor/plastic-corrector update.

        Pure function of its three arguments. Shared by :meth:`trial_state`
        (called once, with the true committed reference) and
        :meth:`tangent_modulus` (called several times, with a
        reconstructed reference held fixed, for numerical
        differentiation -- see the module docstring).

        Returns:
            ``(stress_voigt, plastic_deformation_gradient_new, alpha_new, delta_gamma, yielded)``.
        """
        eps_e_trial, eigenvectors = self._elastic_trial_principal_log_strain(
            strain_voigt, plastic_deformation_gradient_old
        )
        eps_e_new, flow_direction, alpha_new, delta_gamma, yielded = self._principal_return_map(
            eps_e_trial, alpha_old
        )

        if yielded:
            plastic_stretch = eigenvectors @ np.diag(np.exp(delta_gamma * flow_direction))
            plastic_stretch = plastic_stretch @ eigenvectors.T
            plastic_deformation_gradient_new = plastic_stretch @ plastic_deformation_gradient_old
        else:
            plastic_deformation_gradient_new = plastic_deformation_gradient_old

        # Stress must be evaluated at the *converged* elastic state (conjugate
        # to Ce_new, which is defined via Fp_new) -- not pulled back through
        # Fp_old, which is only the frozen reference the *trial* predictor used.
        # Using Fp_old here was a real bug caught during development: it left
        # every plastic step's stress silently wrong by O(accumulated plastic
        # strain), invisible in a single isolated plastic step from a pristine
        # Fp=I reference, but obvious as a path-dependence artifact once a
        # second plastic step was taken from a non-identity committed Fp -- see
        # tests/validation/test_finite_strain_plasticity_path_independence.py.
        tau_new = self._principal_kirchhoff_stress(eps_e_new)
        ce_new_eigenvalues = np.exp(2.0 * eps_e_new)
        se_principal = tau_new / ce_new_eigenvalues
        se_tensor = eigenvectors @ np.diag(se_principal) @ eigenvectors.T

        fp_new_inverse = np.linalg.inv(plastic_deformation_gradient_new)
        s_tensor = fp_new_inverse @ se_tensor @ fp_new_inverse.T
        s_tensor = 0.5 * (s_tensor + s_tensor.T)
        stress_voigt = tensor_to_voigt_stress(s_tensor)

        return stress_voigt, plastic_deformation_gradient_new, alpha_new, delta_gamma, yielded

    def trial_state(self, strain: np.ndarray, committed_state: MaterialState) -> MaterialState:
        """Return-map a trial elastic state onto the (possibly expanded) yield surface.

        Args:
            strain: The proposed total Green-Lagrange strain, Voigt
                ``[E_xx, E_yy, E_zz, 2E_xy, 2E_yz, 2E_xz]``.
            committed_state: The material's last converged state,
                supplying the starting plastic deformation gradient and
                hardening variable.

        Returns:
            A new :class:`~femtoolkit.materials.nonlinear.MaterialState` at ``strain``.

        Raises:
            ConstitutiveUpdateError: If ``strain`` is not finite.
            InvalidDeformationGradientError: If the resulting trial
                elastic deformation is inverted or degenerate.
        """
        strain_array = np.asarray(strain, dtype=float)
        _require_finite_strain(strain_array)

        fp_old = _committed_plastic_deformation_gradient(committed_state)
        alpha_old = float(committed_state.hardening_variable)

        stress, fp_new, alpha_new, delta_gamma, yielded = self._return_map(
            strain_array, fp_old, alpha_old
        )

        plastic_right_cauchy_green = fp_new.T @ fp_new
        plastic_strain_tensor = 0.5 * (plastic_right_cauchy_green - np.eye(3))
        plastic_strain_voigt = tensor_to_voigt_strain(plastic_strain_tensor)

        return MaterialState(
            strain=strain_array,
            stress=stress,
            plastic_strain=plastic_strain_voigt,
            yielded=yielded,
            hardening_variable=alpha_new,
            back_stress=0.0,
            plastic_multiplier=delta_gamma,
            plastic_deformation_gradient=fp_new,
        )

    def first_piola_kirchhoff_stress(
        self, state: MaterialState, deformation_gradient_tensor: np.ndarray
    ) -> np.ndarray:
        """Return the first Piola-Kirchhoff stress ``P = F @ S``, from a state and its ``F``.

        ``state.stress`` holds the second Piola-Kirchhoff stress ``S``
        (the measure the Total Lagrangian element dispatch consumes
        directly); this and :meth:`cauchy_stress` are reporting
        convenience methods only, generic wrappers reusing
        :func:`~femtoolkit.continuum.stress.first_piola_kirchhoff_from_second`
        -- exactly mirroring
        :class:`~femtoolkit.materials.hyperelastic.HyperelasticMaterial`'s
        same-named method.
        ``deformation_gradient_tensor`` must be the same ``F`` the strain
        ``state.strain`` (equivalently ``C = F^T F``) was computed from --
        e.g. via :func:`~femtoolkit.analysis.geometric_nonlinear.tet4_deformation_gradient`.

        Args:
            state: A state this material produced (trial or committed).
            deformation_gradient_tensor: The 3x3 deformation gradient
                ``F`` consistent with ``state.strain``.

        Returns:
            A (generally non-symmetric) 3x3 NumPy array.
        """
        s_tensor = voigt_stress_to_tensor(np.asarray(state.stress, dtype=float))
        return first_piola_kirchhoff_from_second(deformation_gradient_tensor, s_tensor)

    def cauchy_stress(
        self, state: MaterialState, deformation_gradient_tensor: np.ndarray
    ) -> np.ndarray:
        """Return the Cauchy (true) stress ``sigma = (1/J) F S F^T``, from a state and its ``F``.

        See :meth:`first_piola_kirchhoff_stress` for the shared rationale.

        Args:
            state: A state this material produced (trial or committed).
            deformation_gradient_tensor: The 3x3 deformation gradient
                ``F`` consistent with ``state.strain``.

        Returns:
            A symmetric 3x3 NumPy array.
        """
        s_tensor = voigt_stress_to_tensor(np.asarray(state.stress, dtype=float))
        return cauchy_stress_from_second_piola_kirchhoff(deformation_gradient_tensor, s_tensor)

    def _reconstruct_committed_reference(self, state: MaterialState) -> tuple[np.ndarray, float]:
        """Recover ``(Fp_old, alpha_old)`` from a trial ``state`` alone.

        See the module docstring's tangent section: exploits that radial
        return never rotates the (deviatoric) flow direction, so it can
        be recovered from the *final* elastic log strain (itself
        recoverable from ``state.strain`` and ``state.plastic_deformation_gradient``,
        both already known) exactly as validly as from the trial one.
        """
        fp_new = np.asarray(state.plastic_deformation_gradient, dtype=float)
        delta_gamma = float(state.plastic_multiplier)
        alpha_old = float(state.hardening_variable) - delta_gamma

        if not state.yielded or delta_gamma <= 0.0:
            return fp_new, alpha_old

        strain_array = np.asarray(state.strain, dtype=float)
        strain_tensor = voigt_strain_to_tensor(strain_array)
        right_cauchy_green = 2.0 * strain_tensor + np.eye(3)
        fp_new_inverse = np.linalg.inv(fp_new)
        ce_new = fp_new_inverse.T @ right_cauchy_green @ fp_new_inverse
        ce_new = 0.5 * (ce_new + ce_new.T)
        ce_new_eigenvalues, eigenvectors = np.linalg.eigh(ce_new)
        eps_e_new = 0.5 * np.log(ce_new_eigenvalues)

        tau_new = self._principal_kirchhoff_stress(eps_e_new)
        mean_tau = float(np.mean(tau_new))
        deviatoric_tau = tau_new - mean_tau
        von_mises_tau = float(np.sqrt(1.5 * np.sum(deviatoric_tau**2)))
        flow_direction = 1.5 * deviatoric_tau / von_mises_tau

        plastic_stretch_inverse = eigenvectors @ np.diag(np.exp(-delta_gamma * flow_direction))
        plastic_stretch_inverse = plastic_stretch_inverse @ eigenvectors.T
        fp_old = plastic_stretch_inverse @ fp_new
        return fp_old, alpha_old

    def tangent_modulus(self, state: MaterialState) -> np.ndarray:
        """Return a central-difference material tangent, consistent with ``state``.

        See the module docstring for why this is numerical rather than a
        hand-derived closed form, and how the committed reference is
        reconstructed from ``state`` alone.

        Args:
            state: The (typically trial) state to evaluate the tangent at.

        Returns:
            A 6x6 NumPy array.
        """
        fp_old, alpha_old = self._reconstruct_committed_reference(state)
        strain0 = np.asarray(state.strain, dtype=float)
        step = _tangent_finite_difference_step(strain0)

        jacobian = np.zeros((6, 6))
        for component in range(6):
            perturbation = np.zeros(6)
            perturbation[component] = step

            stress_plus, *_ = self._return_map(strain0 + perturbation, fp_old, alpha_old)
            stress_minus, *_ = self._return_map(strain0 - perturbation, fp_old, alpha_old)
            jacobian[:, component] = (stress_plus - stress_minus) / (2.0 * step)

        # The exact consistent tangent is symmetric for associative J2 flow;
        # central differencing only approximates that, so symmetrizing
        # removes purely numerical asymmetry.
        return 0.5 * (jacobian + jacobian.T)


@dataclass(frozen=True)
class J2FiniteStrainPlasticity3D(FiniteStrainPlasticMaterial):
    """A finite-strain, multiplicative von Mises (J2) elastoplastic material.

    The finite-strain generalization of
    :class:`~femtoolkit.materials.j2_plasticity.J2Plasticity3D` -- same
    elastic constants, same yield stress/hardening parameters, same
    linear isotropic hardening law -- extended to large deformation via
    the multiplicative decomposition (see the module docstring).

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
        >>> steel = J2FiniteStrainPlasticity3D(
        ...     youngs_modulus=200e9, poisson_ratio=0.3, yield_stress=250e6,
        ...     hardening_modulus=10e9,
        ... )
        >>> state = steel.trial_state(
        ...     strain=np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
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
        if not np.isfinite(self.youngs_modulus) or self.youngs_modulus <= 0:
            raise ValidationError(
                "J2FiniteStrainPlasticity3D youngs_modulus must be positive, got "
                f"{self.youngs_modulus}."
            )
        if not np.isfinite(self.poisson_ratio) or not (
            _MIN_POISSONS_RATIO < self.poisson_ratio < _MAX_POISSONS_RATIO
        ):
            raise ValidationError(
                "J2FiniteStrainPlasticity3D poisson_ratio must be within "
                f"({_MIN_POISSONS_RATIO}, {_MAX_POISSONS_RATIO}), got "
                f"{self.poisson_ratio}."
            )
        if not np.isfinite(self.yield_stress) or self.yield_stress <= 0:
            raise ValidationError(
                "J2FiniteStrainPlasticity3D yield_stress must be positive, got "
                f"{self.yield_stress}."
            )
        if not np.isfinite(self.hardening_modulus) or self.hardening_modulus < 0:
            raise ValidationError(
                "J2FiniteStrainPlasticity3D hardening_modulus must be non-negative, got "
                f"{self.hardening_modulus}."
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

    def _principal_return_map(
        self, principal_elastic_log_strain_trial: np.ndarray, alpha_old: float
    ) -> tuple[np.ndarray, np.ndarray, float, float, bool]:
        """The exact analogue of :meth:`J2Plasticity3D._return_map`, in principal space."""
        shear_modulus = self.shear_modulus
        tau_trial = self._principal_kirchhoff_stress(principal_elastic_log_strain_trial)
        mean_tau = float(np.mean(tau_trial))
        deviatoric_tau_trial = tau_trial - mean_tau
        von_mises_trial = float(np.sqrt(1.5 * np.sum(deviatoric_tau_trial**2)))
        current_yield_stress = self.yield_stress + self.hardening_modulus * alpha_old
        f_trial = von_mises_trial - current_yield_stress

        if f_trial <= YIELD_FUNCTION_TOLERANCE:
            return (
                principal_elastic_log_strain_trial,
                np.zeros(3),
                alpha_old,
                0.0,
                False,
            )

        delta_gamma = f_trial / (3.0 * shear_modulus + self.hardening_modulus)
        flow_direction = 1.5 * deviatoric_tau_trial / von_mises_trial
        eps_e_new = principal_elastic_log_strain_trial - delta_gamma * flow_direction
        alpha_new = alpha_old + delta_gamma
        return eps_e_new, flow_direction, alpha_new, delta_gamma, True
