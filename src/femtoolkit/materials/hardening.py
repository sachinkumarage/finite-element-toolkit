"""Hardening plasticity material models: bilinear isotropic, bilinear kinematic, and multilinear.

Version 13's :class:`~femtoolkit.materials.nonlinear.ElasticPerfectlyPlasticMaterial1D`
has a fixed yield stress: once yielded, stress can never exceed
``sigma_y`` no matter how much further the material strains. Real
materials usually **harden** instead -- the resistance to further
plastic flow grows (or at least shifts) as plastic deformation
accumulates. This module adds three hardening models, all built on the
*exact same* :class:`~femtoolkit.materials.nonlinear.NonlinearMaterial`
interface (``initial_state``/``trial_state``/``tangent_modulus``) and
the exact same trial/committed :class:`~femtoolkit.materials.nonlinear.MaterialState`
pattern introduced in Version 13 -- no new material protocol is needed,
because the existing one already expresses everything a return-mapping
algorithm requires. Version 13's CST/Q4 nonlinear element support
(:mod:`femtoolkit.analysis.nonlinear_elements`) and the Newton-Raphson
orchestrator (:mod:`femtoolkit.analysis.nonlinear_analysis`) work with
every material in this module completely unchanged.

**Strain decomposition and the elastic stress law** (shared by every
model here):

.. code-block:: text

    epsilon = epsilon_e + epsilon_p        (total = elastic + plastic strain)
    sigma   = E * epsilon_e = E * (epsilon - epsilon_p)

**Isotropic hardening** grows the yield surface symmetrically around
zero stress as plastic strain accumulates, without moving its center:

.. code-block:: text

    f = |sigma| - (sigma_y0 + H * alpha)      (alpha = accumulated plastic strain)

where ``sigma_y0`` is the initial yield stress and ``H`` is the
(isotropic) hardening modulus. Physically, this models a material that
becomes uniformly stronger (in both tension and compression) after
being plastically deformed -- a reasonable approximation for the first
loading of a virgin material, but it does *not* reproduce the
Bauschinger effect (see below), because the yield surface stays
centered on zero.

**Kinematic hardening** instead *translates* a fixed-size yield surface
by a back-stress ``X``, without growing it:

.. code-block:: text

    f = |sigma - X| - sigma_y0

Physically, this models a material whose *virgin* size never changes,
but whose center shifts in the direction of plastic flow -- so a
material that has just yielded in tension re-yields in compression
*earlier* than it would have virgin, an experimentally observed
phenomenon in metals called the **Bauschinger effect**
(:class:`BilinearKinematicHardeningMaterial1D`, validated in
``tests/validation/test_bauschinger_effect.py``). Only a simple linear
back-stress evolution law is implemented here (``dX = H * d(epsilon_p)``)
-- nonlinear (e.g. Armstrong-Frederick) back-stress models are out of
scope.

**Return mapping.** For both bilinear models, the trial (elastic-predictor)
stress is computed first, assuming no new plastic flow occurred; if that
trial stress violates the yield condition, a single closed-form plastic
correction (the *return map*) restores it to the (moved or grown) yield
surface:

.. code-block:: text

    sigma_trial = E * (epsilon - epsilon_p_old)     (elastic predictor)
    f_trial     = |eta_trial| - Y                    (eta/Y depend on the hardening type)
    if f_trial <= tolerance: elastic step
    else:
        d_gamma = f_trial / (E + H)                  (plastic multiplier)
        sigma   = sigma_trial - E * d_gamma * sign(eta_trial)
        epsilon_p = epsilon_p_old + d_gamma * sign(eta_trial)

As in Version 13's perfectly-plastic model, this is **exact and
non-iterative** in 1D: a scalar stress can only be inside, on, or
trying to cross a single yield interval, so there is never any
ambiguity about which direction to return in (unlike true multiaxial
plasticity, which needs an iterative return-mapping algorithm on a real
yield *surface* -- still out of scope here, see
:class:`DecoupledIsotropicHardeningAdapter2D` below for how CST/Q4
plasticity is validated instead).

**Consistent (algorithmic) tangent.** For both bilinear models, the
tangent consistent with the return map above is:

.. code-block:: text

    D_t = E                    (elastic step)
    D_t = E * H / (E + H)      (plastic step -- always between 0 and E)

Using this *exact* linearization of the return-mapped stress (rather
than, say, the elastic modulus ``E`` throughout) is what gives
Newton-Raphson its quadratic convergence rate once the load-stepping
predictor is close to the yielded solution; a poorly chosen tangent
still converges, just more slowly. As in Version 13, ``H = 0`` would
make the plastic tangent exactly zero -- :data:`_MIN_TANGENT_RATIO`
floors it to a small, physically negligible fraction of ``E`` instead,
the same numerical safeguard against an exactly singular tangent
stiffness matrix used throughout this project.

**Multilinear hardening**
(:class:`MultilinearIsotropicHardeningMaterial1D`) takes a different,
simpler approach for a *monotonically loaded* material with more than
one hardening slope: rather than tracking plastic strain through a
return map, it defines stress directly as a piecewise-linear function
of *total* strain (a full stress-strain curve, elastic segment
included), which is unambiguous and non-iterative as long as loading
never reverses. Reversal is explicitly out of scope (see that class's
docstring) and raises
:class:`~femtoolkit.exceptions.UnsupportedLoadingPathError` rather than
silently returning a physically wrong stress.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from femtoolkit.exceptions import (
    ConstitutiveUpdateError,
    InvalidMaterialStateError,
    UnsupportedLoadingPathError,
    ValidationError,
)
from femtoolkit.materials.nonlinear import MaterialState, NonlinearMaterial

YIELD_FUNCTION_TOLERANCE: float = 1e-6
"""Absolute stress tolerance (Pa) a trial yield-function value must
exceed before a plastic correction is triggered. Kept tiny relative to
any realistic yield stress (Pa-to-GPa range) so it never masks a
genuine, physically meaningful yield event -- it exists purely to avoid
triggering a numerically meaningless plastic correction for a trial
stress that is elastic to within floating-point noise.
"""

_MIN_TANGENT_RATIO: float = 1e-6
"""Fraction of the elastic modulus used as a floor for the plastic
tangent modulus, matching the numerical safeguard already documented in
:mod:`femtoolkit.materials.nonlinear` -- avoids an exactly singular
tangent stiffness on a zero-hardening-modulus branch (``H = 0``, which
should reproduce :class:`~femtoolkit.materials.nonlinear.ElasticPerfectlyPlasticMaterial1D`).
"""

_STRAIN_MONOTONICITY_TOLERANCE: float = 1e-9
"""Absolute strain tolerance used by :class:`MultilinearIsotropicHardeningMaterial1D`
to decide whether a trial strain still represents monotonic loading
relative to the committed strain (see its docstring)."""


def _require_scalar_state(material_name: str, committed_state: MaterialState) -> None:
    """Reject a non-scalar committed state passed to a 1D material.

    Raises:
        InvalidMaterialStateError: If ``committed_state.plastic_strain``
            is not a scalar.
    """
    if not np.isscalar(committed_state.plastic_strain):
        raise InvalidMaterialStateError(
            f"{material_name} requires a scalar (1D) committed MaterialState, got "
            f"plastic_strain of type {type(committed_state.plastic_strain).__name__}. "
            "Use a 2D-capable material for CST/Q4 continuum elements instead."
        )


def _require_finite(material_name: str, strain: float) -> None:
    """Reject a non-finite strain before it can corrupt a return-mapping computation.

    Raises:
        ConstitutiveUpdateError: If ``strain`` is not finite.
    """
    if not math.isfinite(strain):
        raise ConstitutiveUpdateError(
            f"{material_name} received a non-finite strain ({strain!r}); this usually "
            "indicates a diverging Newton-Raphson iteration upstream, not a problem "
            "with the material model itself."
        )


def _validate_hardening_parameters(
    material_name: str, youngs_modulus: float, yield_stress: float, hardening_modulus: float
) -> None:
    """Shared parameter validation for the two bilinear hardening models.

    Raises:
        ValidationError: If any parameter is invalid.
    """
    if not math.isfinite(youngs_modulus) or youngs_modulus <= 0:
        raise ValidationError(
            f"{material_name} youngs_modulus must be positive, got {youngs_modulus}."
        )
    if not math.isfinite(yield_stress) or yield_stress <= 0:
        raise ValidationError(f"{material_name} yield_stress must be positive, got {yield_stress}.")
    if not math.isfinite(hardening_modulus) or hardening_modulus < 0:
        raise ValidationError(
            f"{material_name} hardening_modulus must be non-negative, got {hardening_modulus}."
        )


@dataclass(frozen=True)
class BilinearIsotropicHardeningMaterial1D(NonlinearMaterial):
    """A uniaxial bilinear isotropic hardening elastoplastic material.

    The yield surface expands symmetrically about zero stress as plastic
    strain accumulates -- see the module docstring for the physical
    meaning of isotropic hardening and its limitation (no Bauschinger
    effect). Setting ``hardening_modulus=0`` reproduces
    :class:`~femtoolkit.materials.nonlinear.ElasticPerfectlyPlasticMaterial1D`'s
    stress-strain response exactly (verified in
    ``tests/test_hardening_materials.py``), aside from the small,
    physically negligible tangent-stiffness regularization both models
    share.

    Attributes:
        youngs_modulus: Elastic (Young's) modulus ``E``, in pascals.
            Must be positive.
        yield_stress: Initial yield stress ``sigma_y0``, in pascals.
            Must be positive.
        hardening_modulus: Isotropic hardening modulus ``H``, in
            pascals. Must be non-negative (``0`` reproduces perfectly
            plastic behavior).

    Raises:
        ValidationError: If any parameter is invalid.

    Example:
        >>> steel = BilinearIsotropicHardeningMaterial1D(
        ...     youngs_modulus=200e9, yield_stress=250e6, hardening_modulus=20e9
        ... )
        >>> state = steel.trial_state(strain=0.002, committed_state=steel.initial_state())
        >>> state.yielded
        True
    """

    youngs_modulus: float
    yield_stress: float
    hardening_modulus: float

    def __post_init__(self) -> None:
        """Validate parameters immediately after construction.

        Raises:
            ValidationError: If any parameter is invalid.
        """
        _validate_hardening_parameters(
            "BilinearIsotropicHardeningMaterial1D",
            self.youngs_modulus,
            self.yield_stress,
            self.hardening_modulus,
        )

    def initial_state(self) -> MaterialState:
        """Return the zero-strain, zero-stress, unyielded initial state."""
        return MaterialState.zero(0.0)

    def trial_state(self, strain: float, committed_state: MaterialState) -> MaterialState:
        """Return-map a trial elastic stress onto the (possibly expanded) yield surface.

        Args:
            strain: The proposed total (scalar) strain.
            committed_state: The material's last converged state,
                supplying the starting plastic strain and hardening
                variable ``alpha``.

        Returns:
            A new :class:`~femtoolkit.materials.nonlinear.MaterialState` at ``strain``.

        Raises:
            InvalidMaterialStateError: If ``committed_state`` is not scalar.
            ConstitutiveUpdateError: If ``strain`` is not finite.
        """
        _require_scalar_state("BilinearIsotropicHardeningMaterial1D", committed_state)
        _require_finite("BilinearIsotropicHardeningMaterial1D", strain)

        youngs_modulus = self.youngs_modulus
        hardening_modulus = self.hardening_modulus
        plastic_strain_old = committed_state.plastic_strain
        alpha_old = committed_state.hardening_variable

        elastic_trial_strain = strain - plastic_strain_old
        trial_stress = youngs_modulus * elastic_trial_strain
        current_yield_stress = self.yield_stress + hardening_modulus * alpha_old
        trial_yield_function = abs(trial_stress) - current_yield_stress

        if trial_yield_function <= YIELD_FUNCTION_TOLERANCE:
            return MaterialState(
                strain=strain,
                stress=trial_stress,
                plastic_strain=plastic_strain_old,
                yielded=False,
                hardening_variable=alpha_old,
                back_stress=0.0,
            )

        plastic_multiplier = trial_yield_function / (youngs_modulus + hardening_modulus)
        sign = math.copysign(1.0, trial_stress)
        stress = trial_stress - youngs_modulus * plastic_multiplier * sign
        plastic_strain = plastic_strain_old + plastic_multiplier * sign
        alpha = alpha_old + plastic_multiplier
        return MaterialState(
            strain=strain,
            stress=stress,
            plastic_strain=plastic_strain,
            yielded=True,
            hardening_variable=alpha,
            back_stress=0.0,
        )

    def tangent_modulus(self, state: MaterialState) -> float:
        """Return ``E`` if elastic, or the consistent plastic tangent ``EH/(E+H)`` if yielded.

        See :data:`_MIN_TANGENT_RATIO` for why the plastic tangent is
        floored rather than allowed to reach exactly zero when
        ``hardening_modulus == 0``.
        """
        if not state.yielded:
            return self.youngs_modulus
        youngs_modulus = self.youngs_modulus
        hardening_modulus = self.hardening_modulus
        tangent = youngs_modulus * hardening_modulus / (youngs_modulus + hardening_modulus)
        return max(tangent, youngs_modulus * _MIN_TANGENT_RATIO)


@dataclass(frozen=True)
class BilinearKinematicHardeningMaterial1D(NonlinearMaterial):
    """A uniaxial bilinear kinematic hardening elastoplastic material.

    The yield surface keeps a fixed size (``2 * yield_stress`` wide) but
    *translates* by a back-stress ``X`` as plastic strain accumulates --
    see the module docstring for the physical meaning of kinematic
    hardening and the Bauschinger effect it reproduces (validated in
    ``tests/validation/test_bauschinger_effect.py``).

    Attributes:
        youngs_modulus: Elastic (Young's) modulus ``E``, in pascals.
            Must be positive.
        yield_stress: Yield stress ``sigma_y0`` (the fixed half-width of
            the yield surface), in pascals. Must be positive.
        hardening_modulus: Kinematic hardening modulus ``H``, in
            pascals. Must be non-negative.

    Raises:
        ValidationError: If any parameter is invalid.

    Example:
        >>> steel = BilinearKinematicHardeningMaterial1D(
        ...     youngs_modulus=200e9, yield_stress=250e6, hardening_modulus=20e9
        ... )
    """

    youngs_modulus: float
    yield_stress: float
    hardening_modulus: float

    def __post_init__(self) -> None:
        """Validate parameters immediately after construction.

        Raises:
            ValidationError: If any parameter is invalid.
        """
        _validate_hardening_parameters(
            "BilinearKinematicHardeningMaterial1D",
            self.youngs_modulus,
            self.yield_stress,
            self.hardening_modulus,
        )

    def initial_state(self) -> MaterialState:
        """Return the zero-strain, zero-stress, unyielded, zero-back-stress initial state."""
        return MaterialState.zero(0.0)

    def trial_state(self, strain: float, committed_state: MaterialState) -> MaterialState:
        """Return-map a trial elastic stress onto the (translated) yield surface.

        Args:
            strain: The proposed total (scalar) strain.
            committed_state: The material's last converged state,
                supplying the starting plastic strain and back stress.

        Returns:
            A new :class:`~femtoolkit.materials.nonlinear.MaterialState` at ``strain``.

        Raises:
            InvalidMaterialStateError: If ``committed_state`` is not scalar.
            ConstitutiveUpdateError: If ``strain`` is not finite.
        """
        _require_scalar_state("BilinearKinematicHardeningMaterial1D", committed_state)
        _require_finite("BilinearKinematicHardeningMaterial1D", strain)

        youngs_modulus = self.youngs_modulus
        hardening_modulus = self.hardening_modulus
        plastic_strain_old = committed_state.plastic_strain
        back_stress_old = committed_state.back_stress

        elastic_trial_strain = strain - plastic_strain_old
        trial_stress = youngs_modulus * elastic_trial_strain
        shifted_trial_stress = trial_stress - back_stress_old
        trial_yield_function = abs(shifted_trial_stress) - self.yield_stress

        if trial_yield_function <= YIELD_FUNCTION_TOLERANCE:
            return MaterialState(
                strain=strain,
                stress=trial_stress,
                plastic_strain=plastic_strain_old,
                yielded=False,
                hardening_variable=0.0,
                back_stress=back_stress_old,
            )

        plastic_multiplier = trial_yield_function / (youngs_modulus + hardening_modulus)
        sign = math.copysign(1.0, shifted_trial_stress)
        stress = trial_stress - youngs_modulus * plastic_multiplier * sign
        plastic_strain = plastic_strain_old + plastic_multiplier * sign
        back_stress = back_stress_old + hardening_modulus * plastic_multiplier * sign
        return MaterialState(
            strain=strain,
            stress=stress,
            plastic_strain=plastic_strain,
            yielded=True,
            hardening_variable=0.0,
            back_stress=back_stress,
        )

    def tangent_modulus(self, state: MaterialState) -> float:
        """Return ``E`` if elastic, or the consistent plastic tangent ``EH/(E+H)`` if yielded.

        Algebraically identical to
        :meth:`BilinearIsotropicHardeningMaterial1D.tangent_modulus` --
        the 1D consistent tangent for linear bilinear hardening has the
        same form whether the yield surface expands or translates, since
        both reduce to "elastic modulus in series with hardening
        modulus" for a scalar stress state.
        """
        if not state.yielded:
            return self.youngs_modulus
        youngs_modulus = self.youngs_modulus
        hardening_modulus = self.hardening_modulus
        tangent = youngs_modulus * hardening_modulus / (youngs_modulus + hardening_modulus)
        return max(tangent, youngs_modulus * _MIN_TANGENT_RATIO)


def _check_monotonic_loading(material_name: str, strain: float, committed_strain: float) -> None:
    """Reject a strain that would unload or reverse direction relative to ``committed_strain``.

    Raises:
        UnsupportedLoadingPathError: If ``strain`` is not a monotonic
            continuation of ``committed_strain``.
    """
    if abs(committed_strain) < _STRAIN_MONOTONICITY_TOLERANCE:
        return  # Starting from (near) zero: either direction is a valid first step.
    if strain * committed_strain < 0:
        raise UnsupportedLoadingPathError(
            f"{material_name} only supports monotonic (single-direction) loading; "
            f"strain reversed sign from {committed_strain:.6e} to {strain:.6e}."
        )
    if abs(strain) < abs(committed_strain) - _STRAIN_MONOTONICITY_TOLERANCE:
        raise UnsupportedLoadingPathError(
            f"{material_name} only supports monotonic (non-decreasing magnitude) "
            f"loading; strain magnitude decreased from {committed_strain:.6e} to {strain:.6e}."
        )


@dataclass(frozen=True)
class MultilinearIsotropicHardeningMaterial1D(NonlinearMaterial):
    """A uniaxial multilinear hardening material defined by a full total-strain/stress curve.

    Unlike the two bilinear models above, this material is defined
    directly as a piecewise-linear **total** stress-strain curve
    (``strain_points``, ``stress_points``), starting at the origin --
    the first segment (from ``(0, 0)`` to ``(strain_points[1],
    stress_points[1])``) *is* the elastic response, so
    :attr:`youngs_modulus` is derived from it rather than supplied
    separately. Every later segment is a hardening branch with its own
    (generally lower) slope. Because stress is a direct, unambiguous
    function of total strain along this curve, no plastic-strain
    bookkeeping or return mapping is needed for **monotonic** loading:
    the current segment is simply looked up and evaluated.

    Isotropic symmetry is assumed: the given curve is mirrored for
    compressive strain (``sigma(-epsilon) = -sigma(epsilon)``), matching
    the physical picture of an isotropic (symmetric-about-zero) yield
    surface.

    **Monotonic loading only.** This direct total-strain evaluation
    cannot represent unloading (which must follow the *elastic* slope
    back from wherever the material currently sits, not retrace the
    original loading curve) -- attempting to decrease the strain
    magnitude, or reverse its sign, once past the elastic segment raises
    :class:`~femtoolkit.exceptions.UnsupportedLoadingPathError` rather
    than silently returning a wrong stress. A strain beyond the last
    tabulated point is extrapolated using the final segment's slope
    (matching standard multilinear-hardening practice), rather than
    clamping to a flat plateau.

    Attributes:
        strain_points: Strictly increasing total-strain points, starting
            at ``0.0``. Must have at least 2 entries.
        stress_points: Stress at each corresponding strain point,
            starting at ``0.0`` and non-decreasing (no softening
            segments).

    Raises:
        ValidationError: If the curve data is invalid (mismatched
            lengths, fewer than 2 points, non-finite values, a nonzero
            first point, non-strictly-increasing strain, or a
            decreasing stress segment).

    Example:
        >>> curve = MultilinearIsotropicHardeningMaterial1D(
        ...     strain_points=(0.0, 0.001, 0.005, 0.02),
        ...     stress_points=(0.0, 200e6, 250e6, 300e6),
        ... )
        >>> curve.youngs_modulus
        200000000000.0
    """

    strain_points: Sequence[float]
    stress_points: Sequence[float]

    def __post_init__(self) -> None:
        """Validate and normalize the curve data immediately after construction.

        Raises:
            ValidationError: If the curve data is invalid.
        """
        strain_points = tuple(float(value) for value in self.strain_points)
        stress_points = tuple(float(value) for value in self.stress_points)

        if len(strain_points) != len(stress_points):
            raise ValidationError(
                "MultilinearIsotropicHardeningMaterial1D strain_points and stress_points "
                f"must have the same length, got {len(strain_points)} and {len(stress_points)}."
            )
        if len(strain_points) < 2:
            raise ValidationError(
                "MultilinearIsotropicHardeningMaterial1D requires at least 2 curve points "
                f"(an elastic segment), got {len(strain_points)}."
            )
        if not all(math.isfinite(value) for value in strain_points + stress_points):
            raise ValidationError(
                "MultilinearIsotropicHardeningMaterial1D strain_points/stress_points must "
                "all be finite."
            )
        if strain_points[0] != 0.0 or stress_points[0] != 0.0:
            raise ValidationError(
                "MultilinearIsotropicHardeningMaterial1D curve must start at (0.0, 0.0), got "
                f"({strain_points[0]}, {stress_points[0]})."
            )
        if any(b <= a for a, b in zip(strain_points, strain_points[1:], strict=False)):
            raise ValidationError(
                "MultilinearIsotropicHardeningMaterial1D strain_points must be strictly "
                f"increasing, got {strain_points}."
            )
        if any(b < a for a, b in zip(stress_points, stress_points[1:], strict=False)):
            raise ValidationError(
                "MultilinearIsotropicHardeningMaterial1D stress_points must be "
                f"non-decreasing (no softening segments), got {stress_points}."
            )

        object.__setattr__(self, "strain_points", strain_points)
        object.__setattr__(self, "stress_points", stress_points)

    @property
    def youngs_modulus(self) -> float:
        """Elastic modulus, derived from the curve's first (elastic) segment."""
        return self.stress_points[1] / self.strain_points[1]

    def _segment_index(self, strain_magnitude: float) -> int:
        """Return the curve segment index ``strain_magnitude`` falls in (or extrapolates past)."""
        index = int(np.searchsorted(self.strain_points, strain_magnitude, side="right")) - 1
        return min(max(index, 0), len(self.strain_points) - 2)

    def _segment_slope(self, index: int) -> float:
        """Return the local slope (tangent modulus) of curve segment ``index``."""
        rise = self.stress_points[index + 1] - self.stress_points[index]
        run = self.strain_points[index + 1] - self.strain_points[index]
        return rise / run

    def initial_state(self) -> MaterialState:
        """Return the zero-strain, zero-stress, unyielded initial state."""
        return MaterialState.zero(0.0)

    def trial_state(self, strain: float, committed_state: MaterialState) -> MaterialState:
        """Evaluate the multilinear curve at ``strain`` (monotonic loading only).

        Args:
            strain: The proposed total (scalar) strain.
            committed_state: The material's last converged state --
                only its ``strain`` is used, to check monotonicity.

        Returns:
            A new :class:`~femtoolkit.materials.nonlinear.MaterialState` at ``strain``.

        Raises:
            InvalidMaterialStateError: If ``committed_state`` is not scalar.
            ConstitutiveUpdateError: If ``strain`` is not finite.
            UnsupportedLoadingPathError: If ``strain`` would unload or
                reverse direction relative to ``committed_state.strain``.
        """
        _require_scalar_state("MultilinearIsotropicHardeningMaterial1D", committed_state)
        _require_finite("MultilinearIsotropicHardeningMaterial1D", strain)
        _check_monotonic_loading(
            "MultilinearIsotropicHardeningMaterial1D", strain, committed_state.strain
        )

        sign = 1.0 if strain >= 0.0 else -1.0
        magnitude = abs(strain)
        index = self._segment_index(magnitude)
        slope = self._segment_slope(index)
        stress_magnitude = self.stress_points[index] + slope * (
            magnitude - self.strain_points[index]
        )
        stress = sign * stress_magnitude

        plastic_strain = strain - stress / self.youngs_modulus
        yielded = index > 0
        return MaterialState(
            strain=strain,
            stress=stress,
            plastic_strain=plastic_strain,
            yielded=yielded,
            hardening_variable=abs(plastic_strain),
            back_stress=0.0,
        )

    def tangent_modulus(self, state: MaterialState) -> float:
        """Return the current curve segment's slope, floored against a zero-slope plateau."""
        index = self._segment_index(abs(state.strain))
        slope = self._segment_slope(index)
        return max(slope, self.youngs_modulus * _MIN_TANGENT_RATIO)


def _require_vector_state(material_name: str, committed_state: MaterialState) -> None:
    """Reject a committed state that is not a length-3 Voigt vector.

    Raises:
        InvalidMaterialStateError: If ``committed_state.plastic_strain``
            does not have shape ``(3,)``.
    """
    plastic_strain = np.asarray(committed_state.plastic_strain)
    if plastic_strain.shape != (3,):
        raise InvalidMaterialStateError(
            f"{material_name} requires a 2D, 3-component committed MaterialState "
            f"(plastic_strain shape (3,)), got shape {plastic_strain.shape}."
        )


@dataclass(frozen=True)
class DecoupledIsotropicHardeningAdapter2D(NonlinearMaterial):
    """A per-component bilinear isotropic hardening material for CST/Q4 architecture validation.

    **This is not a physically accurate multiaxial plasticity model.**
    Genuine multiaxial plasticity needs a real yield *surface* expressed
    in stress invariants (e.g. von Mises/J2) and a corresponding
    multi-dimensional return-mapping algorithm -- explicitly out of
    scope for Version 14 (see the Version 15 preview in the project
    README). This adapter instead applies
    :class:`BilinearIsotropicHardeningMaterial1D`'s exact scalar return
    map **independently to each of the 3 Voigt strain components**
    (``epsilon_x``, ``epsilon_y``, ``gamma_xy``), with no coupling
    between them at all.

    That is a deliberate, clearly documented simplification -- not a
    disguised claim of real 2D plasticity -- matching this project's
    established pattern (see :mod:`femtoolkit.analysis.nonlinear_elements`,
    Version 13) of validating CST/Q4's *architecture* (internal force,
    tangent stiffness, independent per-Gauss-point state,
    Newton-Raphson convergence against a genuinely path-dependent,
    non-constant tangent) honestly, without faking behavior a real
    model would need a yield surface for. Each component tracks its own
    plastic strain and hardening variable independently, which is
    exactly why :attr:`~femtoolkit.materials.nonlinear.MaterialState.plastic_strain`,
    :attr:`~femtoolkit.materials.nonlinear.MaterialState.hardening_variable`,
    and :attr:`~femtoolkit.materials.nonlinear.MaterialState.yielded`
    are length-3 arrays here rather than scalars.

    Attributes:
        youngs_modulus: Elastic modulus ``E`` applied to every
            component, in pascals. Must be positive.
        yield_stress: Initial yield stress ``sigma_y0`` applied to every
            component, in pascals. Must be positive.
        hardening_modulus: Isotropic hardening modulus ``H`` applied to
            every component, in pascals. Must be non-negative.

    Raises:
        ValidationError: If any parameter is invalid.
    """

    youngs_modulus: float
    yield_stress: float
    hardening_modulus: float

    def __post_init__(self) -> None:
        """Validate parameters immediately after construction.

        Raises:
            ValidationError: If any parameter is invalid.
        """
        _validate_hardening_parameters(
            "DecoupledIsotropicHardeningAdapter2D",
            self.youngs_modulus,
            self.yield_stress,
            self.hardening_modulus,
        )

    def initial_state(self) -> MaterialState:
        """Return the zero-strain, zero-stress, unyielded initial state (length-3 arrays)."""
        return MaterialState.zero(np.zeros(3))

    def trial_state(self, strain: np.ndarray, committed_state: MaterialState) -> MaterialState:
        """Return-map each Voigt component independently onto its own expanded yield surface.

        Args:
            strain: The proposed total strain, a length-3 array.
            committed_state: The material's last converged state
                (length-3 ``plastic_strain``/``hardening_variable``).

        Returns:
            A new :class:`~femtoolkit.materials.nonlinear.MaterialState` at ``strain``.

        Raises:
            InvalidMaterialStateError: If ``committed_state`` is not length-3.
            ConstitutiveUpdateError: If ``strain`` contains a non-finite value.
        """
        _require_vector_state("DecoupledIsotropicHardeningAdapter2D", committed_state)
        strain = np.asarray(strain, dtype=float)
        if not np.all(np.isfinite(strain)):
            raise ConstitutiveUpdateError(
                f"DecoupledIsotropicHardeningAdapter2D received a non-finite strain "
                f"({strain!r}); this usually indicates a diverging Newton-Raphson "
                "iteration upstream, not a problem with the material model itself."
            )

        youngs_modulus = self.youngs_modulus
        hardening_modulus = self.hardening_modulus
        plastic_strain_old = np.asarray(committed_state.plastic_strain, dtype=float)
        alpha_old = np.asarray(committed_state.hardening_variable, dtype=float)

        elastic_trial_strain = strain - plastic_strain_old
        trial_stress = youngs_modulus * elastic_trial_strain
        current_yield_stress = self.yield_stress + hardening_modulus * alpha_old
        trial_yield_function = np.abs(trial_stress) - current_yield_stress

        yielded = trial_yield_function > YIELD_FUNCTION_TOLERANCE
        plastic_multiplier = np.where(
            yielded, trial_yield_function / (youngs_modulus + hardening_modulus), 0.0
        )
        sign = np.where(trial_stress >= 0.0, 1.0, -1.0)

        stress = trial_stress - youngs_modulus * plastic_multiplier * sign
        plastic_strain = plastic_strain_old + plastic_multiplier * sign
        alpha = alpha_old + plastic_multiplier

        return MaterialState(
            strain=strain,
            stress=stress,
            plastic_strain=plastic_strain,
            yielded=yielded,
            hardening_variable=alpha,
            back_stress=np.zeros(3),
        )

    def tangent_modulus(self, state: MaterialState) -> np.ndarray:
        """Return a diagonal 3x3 matrix of each component's own consistent tangent.

        Each diagonal entry is independently ``E`` (elastic) or
        ``EH/(E+H)`` (plastic, floored per :data:`_MIN_TANGENT_RATIO`)
        depending on that component's own ``yielded`` flag -- there is
        no off-diagonal coupling, consistent with the fully decoupled
        return map in :meth:`trial_state`.
        """
        youngs_modulus = self.youngs_modulus
        hardening_modulus = self.hardening_modulus
        plastic_tangent = max(
            youngs_modulus * hardening_modulus / (youngs_modulus + hardening_modulus),
            youngs_modulus * _MIN_TANGENT_RATIO,
        )
        yielded = np.asarray(state.yielded, dtype=bool)
        diagonal = np.where(yielded, plastic_tangent, youngs_modulus)
        return np.diag(diagonal)
