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
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg

from femtoolkit.exceptions import EigenvalueComputationError, ValidationError

if TYPE_CHECKING:
    from femtoolkit.analysis.dynamic_system import DynamicSystem

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
    constrained_indices = {
        system.dof_map.global_index(bc.node_id, bc.dof) for bc in system.boundary_conditions
    }
    free_indices = [i for i in range(total_dofs) if i not in constrained_indices]
    if not free_indices:
        raise ValidationError("Every DOF is constrained; there is nothing to analyze.")

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
