"""Natural frequency (modal) analysis: the generalized eigenvalue problem.

**Free, undamped vibration** of a structure follows ``M u'' + K u = 0``.
Seeking a harmonic solution ``u(t) = phi * sin(omega*t)`` and substituting
back gives the **generalized eigenvalue problem**:

.. code-block:: text

    K phi = lambda * M phi          (lambda = omega^2)

Each eigenvalue/eigenvector pair ``(lambda_i, phi_i)`` is a **natural
mode**: the structure can oscillate in the shape ``phi_i`` (its **mode
shape**) at the **natural circular frequency** ``omega_i = sqrt(lambda_i)``
(rad/s), converted to an ordinary frequency ``f_i = omega_i / (2*pi)``
(Hz, cycles/s). Both ``K`` and ``M`` are symmetric, and for a physically
valid, well-supported model ``M`` is symmetric positive definite -- this
module solves the problem with :func:`scipy.linalg.eigh`'s generalized
form, the standard, numerically robust LAPACK-backed routine for exactly
this class of problem (this is why SciPy is a dependency of this
package: NumPy alone has no *generalized* eigenvalue solver, and
hand-rolling one via, say, a manual Cholesky reduction to a standard
eigenvalue problem would itself be exactly the kind of "unreliable
custom eigenvalue algorithm" this module deliberately avoids).

**Rigid-body modes.** If a model is insufficiently constrained (or, via
:func:`natural_frequencies` directly, not constrained at all), some
eigenvalues come out at or very near zero -- these are **rigid-body
modes**: the structure can translate/rotate as a whole with no strain
energy, not physical vibration. This module never removes or hides
them: :attr:`ModalAnalysisResult.is_rigid_body_mode` flags every
eigenvalue within :attr:`ModalAnalysisResult.rigid_body_tolerance` of
zero, so callers can distinguish "this is an expected rigid-body mode"
from "this is the first real vibration mode" explicitly, rather than
guessing from the raw number. Tiny *negative* eigenvalues within the
same tolerance are numerical noise (floating-point roundoff around an
exact zero) and are clipped to zero before computing frequencies (a
negative number has no real square root); a negative eigenvalue
*larger* in magnitude than the tolerance is not noise -- it means ``K``
or ``M`` was not assembled correctly (not positive semi-definite), and
raises :class:`~femtoolkit.exceptions.EigenvalueComputationError`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.linalg

from femtoolkit.analysis.dof import TranslationDOF
from femtoolkit.analysis.dynamic_system import free_and_constrained_indices
from femtoolkit.exceptions import EigenvalueComputationError, ValidationError

if TYPE_CHECKING:
    from femtoolkit.analysis.dof import DOFMap
    from femtoolkit.analysis.dynamic_system import DynamicSystem

Direction = Literal["x", "y"]

DEFAULT_RIGID_BODY_TOLERANCE: float = 1e-6
"""Default tolerance, in (rad/s)^2, below which an eigenvalue's absolute
value is treated as a rigid-body mode rather than physical vibration or
numerical error. Compared against ``lambda = omega^2`` directly, not
frequency -- see :class:`ModalAnalysisResult`.
"""


@dataclass(frozen=True)
class ModalAnalysisResult:
    """The result of a natural-frequency (modal) analysis.

    Attributes:
        eigenvalues: ``lambda_i = omega_i^2``, in (rad/s)^2, ascending
            order, as returned by the eigenvalue solve (*before*
            clipping small negative numerical noise to zero -- see the
            module docstring).
        angular_frequencies: ``omega_i = sqrt(max(lambda_i, 0))``, in rad/s.
        frequencies: ``f_i = omega_i / (2*pi)``, in Hz.
        mode_shapes: Column ``i`` is the i-th mode shape ``phi_i``,
            normalized so its largest-magnitude component is exactly
            ``1.0`` (or ``-1.0`` if that component happened to be
            negative in the raw eigenvector -- sign is arbitrary, see
            below). Shape ``(n_dofs, n_modes)``.
        is_rigid_body_mode: ``True`` at index ``i`` if
            ``abs(eigenvalues[i]) < rigid_body_tolerance``.
        rigid_body_tolerance: The tolerance used to compute
            ``is_rigid_body_mode``.

    **Mode shape normalization convention.** Each mode shape is scaled so
    its maximum absolute component equals ``1.0`` -- a normalization
    chosen purely for comparability between modes and against hand
    calculations, *not* a physical amplitude. A mode shape's absolute
    scale carries no physical meaning on its own (only its *ratios*
    between DOFs do); actual vibration amplitude depends on the initial
    conditions or applied forcing, which modal analysis alone does not
    determine. The overall sign of each column is also arbitrary (an
    eigenvector and its negative describe the same mode).
    """

    eigenvalues: np.ndarray
    angular_frequencies: np.ndarray
    frequencies: np.ndarray
    mode_shapes: np.ndarray
    is_rigid_body_mode: np.ndarray
    rigid_body_tolerance: float


def _validate_square_matching(stiffness: np.ndarray, mass: np.ndarray) -> int:
    if stiffness.ndim != 2 or stiffness.shape[0] != stiffness.shape[1]:
        raise ValidationError(f"stiffness must be a square matrix, got shape {stiffness.shape}.")
    if mass.shape != stiffness.shape:
        raise ValidationError(
            f"mass must have the same shape as stiffness ({stiffness.shape}), got {mass.shape}."
        )
    return stiffness.shape[0]


def _normalize_mode_shapes(eigenvectors: np.ndarray) -> np.ndarray:
    max_abs = np.max(np.abs(eigenvectors), axis=0, keepdims=True)
    return eigenvectors / max_abs


def natural_frequencies(
    stiffness: np.ndarray,
    mass: np.ndarray,
    num_modes: int | None = None,
    rigid_body_tolerance: float = DEFAULT_RIGID_BODY_TOLERANCE,
) -> ModalAnalysisResult:
    """Solve the generalized eigenvalue problem ``K phi = lambda M phi``.

    Args:
        stiffness: Symmetric global (or reduced) stiffness matrix ``K``.
        mass: Symmetric global (or reduced) mass matrix ``M``, the same
            shape as ``stiffness``.
        num_modes: Number of lowest modes to return, or ``None`` (the
            default) to return all of them. Lowest-frequency modes are
            returned first (ascending eigenvalue order), which is the
            standard convention -- these are the modes of engineering
            interest (a real structure's response is dominated by its
            lowest few modes).
        rigid_body_tolerance: Tolerance, in (rad/s)^2, below which an
            eigenvalue's absolute value is treated as a rigid-body mode
            (see the module docstring). Must be non-negative.

    Returns:
        A :class:`ModalAnalysisResult`.

    Raises:
        ValidationError: If ``stiffness``/``mass`` are not square
            matrices of matching shape, ``num_modes`` is not a positive
            integer no larger than the system size, or
            ``rigid_body_tolerance`` is negative.
        EigenvalueComputationError: If the underlying LAPACK solve fails
            (e.g. ``mass`` is not positive definite), or an eigenvalue is
            negative by more than ``rigid_body_tolerance`` (indicating an
            invalid ``K``/``M`` pair, not a rigid-body mode).

    Example:
        >>> result = natural_frequencies(k_global, m_global, num_modes=5)
        >>> result.frequencies  # Hz, ascending
        >>> result.is_rigid_body_mode  # True for near-zero-frequency modes
    """
    n = _validate_square_matching(stiffness, mass)
    if num_modes is not None and (not isinstance(num_modes, int) or not (0 < num_modes <= n)):
        raise ValidationError(f"num_modes must be an integer in [1, {n}], got {num_modes!r}.")
    if not math.isfinite(rigid_body_tolerance) or rigid_body_tolerance < 0:
        raise ValidationError(
            f"rigid_body_tolerance must be non-negative, got {rigid_body_tolerance}."
        )

    try:
        eigenvalues, eigenvectors = scipy.linalg.eigh(stiffness, mass)
    except (scipy.linalg.LinAlgError, ValueError) as error:
        raise EigenvalueComputationError(
            "Generalized eigenvalue solve failed; the mass matrix is likely not "
            "positive definite (e.g. missing density, or a DOF with zero mass)."
        ) from error

    most_negative = float(eigenvalues.min())
    if most_negative < -rigid_body_tolerance:
        raise EigenvalueComputationError(
            f"Eigenvalue {most_negative} is negative beyond rigid_body_tolerance "
            f"({rigid_body_tolerance}); this indicates an invalid stiffness or mass "
            "matrix (not positive semi-definite), not a rigid-body mode."
        )

    is_rigid_body_mode = np.abs(eigenvalues) < rigid_body_tolerance
    angular_frequencies = np.sqrt(np.clip(eigenvalues, 0.0, None))
    frequencies = angular_frequencies / (2.0 * math.pi)
    mode_shapes = _normalize_mode_shapes(eigenvectors)

    if num_modes is not None:
        eigenvalues = eigenvalues[:num_modes]
        angular_frequencies = angular_frequencies[:num_modes]
        frequencies = frequencies[:num_modes]
        mode_shapes = mode_shapes[:, :num_modes]
        is_rigid_body_mode = is_rigid_body_mode[:num_modes]

    return ModalAnalysisResult(
        eigenvalues=eigenvalues,
        angular_frequencies=angular_frequencies,
        frequencies=frequencies,
        mode_shapes=mode_shapes,
        is_rigid_body_mode=is_rigid_body_mode,
        rigid_body_tolerance=rigid_body_tolerance,
    )


def natural_frequencies_of_system(
    system: DynamicSystem,
    num_modes: int | None = None,
    rigid_body_tolerance: float = DEFAULT_RIGID_BODY_TOLERANCE,
) -> ModalAnalysisResult:
    """Solve for a constrained dynamic system's natural frequencies.

    ``system`` is a :class:`~femtoolkit.analysis.dynamic_system.DynamicSystem`.
    Reduces ``system.stiffness``/``system.mass`` to their free-DOF
    submatrices (removing every DOF referenced by
    ``system.boundary_conditions`` -- the same free/constrained
    partition :func:`~femtoolkit.analysis.system.solve` uses for the
    static case), solves :func:`natural_frequencies` on the reduced
    matrices, then expands the mode shapes back to the full DOF space
    (each constrained DOF gets exactly ``0.0``, consistent with its
    prescribed zero motion).

    Args:
        system: The dynamic system to analyze.
        num_modes: Number of lowest modes to return, or ``None`` for all.
        rigid_body_tolerance: See :func:`natural_frequencies`.

    Returns:
        A :class:`ModalAnalysisResult` with ``mode_shapes`` in the full
        (unreduced) DOF space, ``system.dof_map.total_dofs`` rows.

    Raises:
        ValidationError: If every DOF is constrained (nothing to solve).
        EigenvalueComputationError: See :func:`natural_frequencies`.
    """
    total_dofs = system.dof_map.total_dofs
    free_indices, _, _ = free_and_constrained_indices(system)

    k_free = system.stiffness[np.ix_(free_indices, free_indices)]
    m_free = system.mass[np.ix_(free_indices, free_indices)]
    reduced = natural_frequencies(k_free, m_free, num_modes, rigid_body_tolerance)

    full_mode_shapes = np.zeros((total_dofs, reduced.mode_shapes.shape[1]))
    full_mode_shapes[free_indices, :] = reduced.mode_shapes

    return ModalAnalysisResult(
        eigenvalues=reduced.eigenvalues,
        angular_frequencies=reduced.angular_frequencies,
        frequencies=reduced.frequencies,
        mode_shapes=full_mode_shapes,
        is_rigid_body_mode=reduced.is_rigid_body_mode,
        rigid_body_tolerance=rigid_body_tolerance,
    )


# ---------------------------------------------------------------------------
# Version 12: mass-normalized mode shapes, participation factors, effective
# modal mass, and periods -- all built on the Version 11 functions above,
# without changing their behavior.
# ---------------------------------------------------------------------------


def mass_normalize_mode_shapes(mode_shapes: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Rescale each mode shape so ``phi^T * M * phi = 1`` (mass normalization).

    This is a *different* normalization convention from
    :class:`ModalAnalysisResult`'s default (largest component = 1.0) --
    it does not replace or change that default, which remains exactly as
    it was in Version 11. Mass normalization is the convention modal
    participation factors and effective modal mass are usually expressed
    against (see :func:`modal_participation_factors`), because it makes
    the modal mass matrix ``Phi^T * M * Phi`` exactly the identity.

    Works on mode shapes in *any* prior scale (including the default
    max-abs-1 convention, or the zero-padded full-DOF-space shapes from
    :func:`natural_frequencies_of_system`): rescaling by
    ``1 / sqrt(phi^T M phi)`` is scale-invariant -- it produces the same
    final mass-normalized mode shape (up to the same sign ambiguity every
    eigenvector has) regardless of the input scale.

    Args:
        mode_shapes: Mode shape matrix, one mode per column (e.g.
            ``ModalAnalysisResult.mode_shapes``).
        mass: The mass matrix the mode shapes were computed against
            (same DOF space as ``mode_shapes``' rows).

    Returns:
        A new mode shape matrix, the same shape as ``mode_shapes``, with
        each column independently mass-normalized.

    Raises:
        EigenvalueComputationError: If a mode's generalized mass
            ``phi^T M phi`` is not positive (a malformed or
            non-physical mode shape/mass pairing).
    """
    normalized = np.empty_like(mode_shapes)
    for i in range(mode_shapes.shape[1]):
        phi = mode_shapes[:, i]
        generalized_mass = float(phi @ mass @ phi)
        if not math.isfinite(generalized_mass) or generalized_mass <= 0:
            raise EigenvalueComputationError(
                f"Mode {i}'s generalized mass phi^T*M*phi = {generalized_mass} is not "
                "positive; cannot mass-normalize."
            )
        normalized[:, i] = phi / math.sqrt(generalized_mass)
    return normalized


def influence_vector(dof_map: DOFMap, direction: Direction) -> np.ndarray:
    """Build a rigid-body unit-displacement (influence) vector for one global direction.

    ``r[i] = 1.0`` if global DOF ``i`` is the requested translational
    direction at some node, else ``0.0`` -- the displacement pattern the
    whole structure would show under a unit rigid-body translation along
    that axis, used as the reference direction for
    :func:`modal_participation_factors`.

    Args:
        dof_map: DOF map defining the global DOF numbering.
        direction: ``"x"`` or ``"y"`` (case-insensitive).

    Returns:
        A NumPy array of shape ``(dof_map.total_dofs,)``.

    Raises:
        ValidationError: If ``direction`` is not ``"x"`` or ``"y"``, or
            that direction is not active for ``dof_map.dofs_per_node``
            (e.g. ``"y"`` for a 1-DOF-per-node axial bar model).
    """
    dof_by_name = {"x": TranslationDOF.X, "y": TranslationDOF.Y}
    key = direction.lower() if isinstance(direction, str) else direction
    if key not in dof_by_name:
        raise ValidationError(f'direction must be "x" or "y", got {direction!r}.')
    target_dof = dof_by_name[key]
    if target_dof >= dof_map.dofs_per_node:
        raise ValidationError(
            f'direction {direction!r} (dof index {target_dof}) is not active for a '
            f"DOF map with dofs_per_node={dof_map.dofs_per_node}."
        )

    r = np.zeros(dof_map.total_dofs)
    for node_id in dof_map.node_ids:
        r[dof_map.global_index(node_id, target_dof)] = 1.0
    return r


def modal_participation_factors(
    mode_shapes: np.ndarray, mass: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Compute the modal participation factor of every mode for one direction.

    .. code-block:: text

        Gamma_i = (phi_i^T * M * r) / (phi_i^T * M * phi_i)

    which reduces to ``Gamma_i = phi_i^T * M * r`` when ``mode_shapes``
    is already mass-normalized (see :func:`mass_normalize_mode_shapes`).

    **``Gamma_i`` alone is normalization-dependent.** Unlike
    :func:`effective_modal_mass`, the raw participation factor scales
    inversely with the mode shape's own scale (``phi -> c*phi`` gives
    ``Gamma_i -> Gamma_i / c``), so a bare ``Gamma_i`` value is only
    meaningful paired with the specific ``mode_shapes`` scale it was
    computed against -- the standard tabulated "participation factor"
    convention assumes mass-normalized mode shapes, which is what
    :func:`~femtoolkit.analysis.modal.modal_analysis`/
    :func:`~femtoolkit.analysis.modal.modal_analysis_of_system` report
    it against. What *is* invariant to mode shape scale is the physical
    contribution ``Gamma_i * phi_i`` (and therefore
    :func:`effective_modal_mass`, which depends only on that product) --
    this function uses the general formula directly so that invariant
    combination comes out correctly regardless of which normalization
    ``mode_shapes`` happens to be in.

    Args:
        mode_shapes: Mode shape matrix, one mode per column.
        mass: The mass matrix the mode shapes were computed against.
        direction: The reference influence vector ``r`` (see
            :func:`influence_vector`), the same length as
            ``mode_shapes``' rows.

    Returns:
        A length-``n_modes`` array, one participation factor per mode
        (per column of ``mode_shapes``).
    """
    n_modes = mode_shapes.shape[1]
    participation_factors = np.empty(n_modes)
    for i in range(n_modes):
        phi = mode_shapes[:, i]
        generalized_mass = float(phi @ mass @ phi)
        participation_factors[i] = (phi @ mass @ direction) / generalized_mass
    return participation_factors


def effective_modal_mass(
    participation_factors: np.ndarray, mode_shapes: np.ndarray, mass: np.ndarray
) -> np.ndarray:
    """Compute the effective modal mass of every mode.

    .. code-block:: text

        M_eff,i = Gamma_i^2 * (phi_i^T * M * phi_i)

    which reduces to ``M_eff,i = Gamma_i^2`` for mass-normalized mode
    shapes. Unlike a bare participation factor (see
    :func:`modal_participation_factors`), ``M_eff,i`` *is* invariant to
    the mode shapes' normalization -- the ``Gamma_i^2`` scaling and the
    ``phi_i^T*M*phi_i`` scaling exactly cancel (``phi -> c*phi`` gives
    ``Gamma_i -> Gamma_i/c`` and generalized mass ``-> c^2 *`` itself),
    so this function gives the same physically meaningful effective mass
    regardless of which normalization ``mode_shapes``/``participation_factors``
    were computed against.

    Args:
        participation_factors: One participation factor per mode (see
            :func:`modal_participation_factors`).
        mode_shapes: Mode shape matrix, one mode per column.
        mass: The mass matrix the mode shapes were computed against.

    Returns:
        A length-``n_modes`` array, the effective modal mass of each
        mode, in kilograms.
    """
    n_modes = mode_shapes.shape[1]
    generalized_masses = np.array(
        [float(mode_shapes[:, i] @ mass @ mode_shapes[:, i]) for i in range(n_modes)]
    )
    return participation_factors**2 * generalized_masses


def effective_modal_mass_ratio(
    effective_mass: np.ndarray, mass: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Compute each mode's effective modal mass as a fraction of the total participating mass.

    .. code-block:: text

        ratio_i = M_eff,i / (r^T * M * r)

    ``r^T * M * r`` is the total physical mass associated with
    ``direction`` -- summing every mode's ``ratio_i`` (over a complete
    set of modes) recovers exactly ``1.0``, an identity that comes from
    the completeness of the eigenbasis (verified numerically in
    ``tests/validation/test_effective_modal_mass.py``).

    Args:
        effective_mass: One effective modal mass per mode (see
            :func:`effective_modal_mass`).
        mass: The mass matrix ``direction`` and the mode shapes were
            computed against.
        direction: The reference influence vector ``r``.

    Returns:
        A length-``n_modes`` array of dimensionless ratios.

    Raises:
        ValidationError: If the total participating mass ``r^T*M*r`` is
            not positive (e.g. ``direction`` is all zero).
    """
    total_participating_mass = float(direction @ mass @ direction)
    if not math.isfinite(total_participating_mass) or total_participating_mass <= 0:
        raise ValidationError(
            f"Total participating mass r^T*M*r = {total_participating_mass} must be "
            "positive; check that `direction` is a non-zero influence vector."
        )
    return effective_mass / total_participating_mass


def compute_periods(angular_frequencies: np.ndarray, is_rigid_body_mode: np.ndarray) -> np.ndarray:
    """Compute the natural period ``T = 2*pi / omega`` of every mode.

    Args:
        angular_frequencies: Natural circular frequencies, in rad/s (see
            :attr:`ModalAnalysisResult.angular_frequencies`).
        is_rigid_body_mode: Boolean mask flagging rigid-body modes (see
            :attr:`ModalAnalysisResult.is_rigid_body_mode`).

    Returns:
        A length-``n_modes`` array of periods, in seconds.
        **Rigid-body modes** (``omega ~= 0``) have an undefined
        (infinite) period -- reported as ``float("inf")`` rather than
        raising or dividing by (near) zero, since a rigid-body mode is a
        legitimate, expected part of an analysis (e.g. an unconstrained
        model), not an error condition.
    """
    periods = np.full_like(angular_frequencies, np.inf)
    physical = ~is_rigid_body_mode
    periods[physical] = 2.0 * math.pi / angular_frequencies[physical]
    return periods


@dataclass(frozen=True)
class ModalResult:
    """The full Version 12 modal analysis result, built on :class:`ModalAnalysisResult`.

    Attributes:
        modal_analysis_result: The underlying Version 11
            :class:`ModalAnalysisResult` (eigenvalues, angular
            frequencies, frequencies, default-normalized mode shapes,
            rigid-body flags) -- exposed both directly and through the
            convenience properties below.
        periods: Natural period of each mode, in seconds (``inf`` for
            rigid-body modes). See :func:`compute_periods`.
        mass_normalized_mode_shapes: Mode shapes rescaled so
            ``phi^T*M*phi = 1`` (see :func:`mass_normalize_mode_shapes`).
        direction: The influence vector participation/effective-mass
            quantities were computed against, or ``None`` if no
            ``direction`` was requested.
        participation_factors: Modal participation factors, computed
            against ``mass_normalized_mode_shapes`` (the standard
            convention -- see
            :func:`modal_participation_factors`), or ``None``.
        effective_modal_mass: Effective modal mass per mode, in
            kilograms, or ``None``.
        effective_modal_mass_ratio: Effective modal mass as a fraction
            of the total participating mass, per mode, or ``None``.
        cumulative_mass_ratio: Running cumulative sum of
            ``effective_modal_mass_ratio`` (mode 1, modes 1-2, modes
            1-3, ...), or ``None``.
    """

    modal_analysis_result: ModalAnalysisResult
    periods: np.ndarray
    mass_normalized_mode_shapes: np.ndarray
    direction: np.ndarray | None = None
    participation_factors: np.ndarray | None = None
    effective_modal_mass: np.ndarray | None = None
    effective_modal_mass_ratio: np.ndarray | None = None
    cumulative_mass_ratio: np.ndarray | None = None

    @property
    def eigenvalues(self) -> np.ndarray:
        """See :attr:`ModalAnalysisResult.eigenvalues`."""
        return self.modal_analysis_result.eigenvalues

    @property
    def angular_frequencies(self) -> np.ndarray:
        """See :attr:`ModalAnalysisResult.angular_frequencies`."""
        return self.modal_analysis_result.angular_frequencies

    @property
    def frequencies(self) -> np.ndarray:
        """See :attr:`ModalAnalysisResult.frequencies`."""
        return self.modal_analysis_result.frequencies

    @property
    def mode_shapes(self) -> np.ndarray:
        """See :attr:`ModalAnalysisResult.mode_shapes` (default max-abs-1 normalization)."""
        return self.modal_analysis_result.mode_shapes

    @property
    def is_rigid_body_mode(self) -> np.ndarray:
        """See :attr:`ModalAnalysisResult.is_rigid_body_mode`."""
        return self.modal_analysis_result.is_rigid_body_mode


def _build_modal_result(
    base: ModalAnalysisResult,
    mass: np.ndarray,
    direction: np.ndarray | None,
) -> ModalResult:
    periods = compute_periods(base.angular_frequencies, base.is_rigid_body_mode)
    mass_normalized = mass_normalize_mode_shapes(base.mode_shapes, mass)

    participation = effective_mass = ratio = cumulative = None
    if direction is not None:
        # Computed against the mass-normalized mode shapes, not
        # `base.mode_shapes`' default (max-abs-1) normalization: this
        # matches the standard convention (see
        # `modal_participation_factors`'s docstring) and makes
        # `participation_factors` directly comparable to hand
        # calculations or other FEA tools, which universally report
        # participation factors against mass-normalized modes.
        # `effective_modal_mass`/ratio/cumulative are unaffected by this
        # choice -- they are provably invariant to mode shape scale.
        participation = modal_participation_factors(mass_normalized, mass, direction)
        effective_mass = effective_modal_mass(participation, mass_normalized, mass)
        ratio = effective_modal_mass_ratio(effective_mass, mass, direction)
        cumulative = np.cumsum(ratio)

    return ModalResult(
        modal_analysis_result=base,
        periods=periods,
        mass_normalized_mode_shapes=mass_normalized,
        direction=direction,
        participation_factors=participation,
        effective_modal_mass=effective_mass,
        effective_modal_mass_ratio=ratio,
        cumulative_mass_ratio=cumulative,
    )


def modal_analysis(
    stiffness_matrix: np.ndarray,
    mass_matrix: np.ndarray,
    num_modes: int | None = None,
    direction: np.ndarray | None = None,
    rigid_body_tolerance: float = DEFAULT_RIGID_BODY_TOLERANCE,
) -> ModalResult:
    """Full Version 12 modal analysis on raw stiffness/mass matrices.

    Thin wrapper over :func:`natural_frequencies` (Version 11, unchanged)
    that adds periods, mass-normalized mode shapes, and -- if
    ``direction`` is given -- modal participation factors and effective
    modal mass.

    Args:
        stiffness_matrix: Symmetric (global or reduced) stiffness matrix ``K``.
        mass_matrix: Symmetric mass matrix ``M``, the same shape as ``stiffness_matrix``.
        num_modes: Number of lowest modes to return, or ``None`` for all.
        direction: A raw influence vector (the same length as
            ``stiffness_matrix``), or ``None`` to skip participation
            factor/effective mass computation. Use
            :func:`modal_analysis_of_system` for named ``"x"``/``"y"``
            directions resolved from a DOF map.
        rigid_body_tolerance: See :func:`natural_frequencies`.

    Returns:
        A :class:`ModalResult`.

    Raises:
        ValidationError: See :func:`natural_frequencies`; also raised if
            ``direction``'s length does not match the system size.
        EigenvalueComputationError: See :func:`natural_frequencies`.

    Example:
        >>> modal_results = modal_analysis(stiffness_matrix=K, mass_matrix=M, num_modes=10)
        >>> modal_results.frequencies
        >>> modal_results.periods
    """
    base = natural_frequencies(stiffness_matrix, mass_matrix, num_modes, rigid_body_tolerance)

    if direction is not None:
        direction = np.asarray(direction, dtype=float)
        if direction.shape != (mass_matrix.shape[0],):
            raise ValidationError(
                f"direction must have shape ({mass_matrix.shape[0]},), got {direction.shape}."
            )

    return _build_modal_result(base, mass_matrix, direction)


def modal_analysis_of_system(
    system: DynamicSystem,
    num_modes: int | None = None,
    direction: Direction | np.ndarray | None = None,
    rigid_body_tolerance: float = DEFAULT_RIGID_BODY_TOLERANCE,
) -> ModalResult:
    """Full Version 12 modal analysis for a constrained dynamic system.

    Reuses :func:`natural_frequencies_of_system` (Version 11) for the
    free/constrained DOF reduction and eigenvalue solve, then computes
    periods, mass-normalized mode shapes, and -- if ``direction`` is
    given -- participation factors and effective modal mass, all
    evaluated with the *full* (unreduced) ``system.mass`` against the
    zero-padded (at constrained DOFs) full-space mode shapes.

    **Why the direction vector is masked at constrained DOFs.** A
    supported (fixed) DOF cannot move, so it contributes no *dynamic*
    (participating) mass -- only the mass associated with free DOFs
    should be counted. Zero-padding the mode shapes alone is not
    sufficient to enforce this: with a *consistent* mass matrix, a free
    DOF's row still has nonzero mass-coupling entries in a constrained
    DOF's column, so an unmasked ``r`` (1.0 at every DOF in the
    requested direction, free or constrained) would leak a spurious
    contribution through that coupling. Masking ``r`` to zero at every
    constrained DOF eliminates it, giving exactly the same participation
    factors and effective modal mass as computing directly on the
    reduced free-DOF matrices (verified in
    ``tests/validation/test_effective_modal_mass.py``), without this
    function needing to duplicate that reduction itself.

    Args:
        system: The dynamic system to analyze.
        num_modes: Number of lowest modes to return, or ``None`` for all.
        direction: ``"x"``/``"y"`` (resolved via
            :func:`influence_vector` against ``system.dof_map``), a raw
            influence vector, or ``None`` to skip participation
            factor/effective mass computation. Either way, the
            direction is masked to zero at constrained DOFs before use
            (see above).
        rigid_body_tolerance: See :func:`natural_frequencies`.

    Returns:
        A :class:`ModalResult` with mode shapes in the full (unreduced)
        DOF space.

    Raises:
        ValidationError: See :func:`natural_frequencies_of_system`; also
            raised if ``direction`` is an invalid direction name or an
            array of the wrong length.
        EigenvalueComputationError: See :func:`natural_frequencies`.
    """
    base = natural_frequencies_of_system(system, num_modes, rigid_body_tolerance)

    resolved_direction = None
    if direction is not None:
        if isinstance(direction, str):
            resolved_direction = influence_vector(system.dof_map, direction)
        else:
            resolved_direction = np.asarray(direction, dtype=float)
            expected_shape = (system.dof_map.total_dofs,)
            if resolved_direction.shape != expected_shape:
                raise ValidationError(
                    f"direction must have shape {expected_shape}, got "
                    f"{resolved_direction.shape}."
                )
        _, constrained_indices, _ = free_and_constrained_indices(system)
        resolved_direction = resolved_direction.copy()
        resolved_direction[constrained_indices] = 0.0

    return _build_modal_result(base, system.mass, resolved_direction)
