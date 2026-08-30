"""Custom exception types for the Finite Element Toolkit.

These exceptions give callers a way to distinguish domain-level errors
(invalid engineering data, broken model references) from generic Python
exceptions, without introducing a separate exception type for every
possible failure.
"""

from __future__ import annotations


class FiniteElementToolkitError(Exception):
    """Base class for all errors raised by the Finite Element Toolkit.

    Catching this exception is a convenient way for calling code to
    handle any toolkit-specific failure without needing to know about
    every individual exception subclass.
    """


class ValidationError(FiniteElementToolkitError):
    """Raised when engineering or model data fails a validation rule.

    Examples include a negative material density, a non-finite node
    coordinate, or an element that references an empty node list.
    """


class DegenerateElementError(ValidationError):
    """Raised when a continuum element's geometry has zero (or near-zero) area.

    A degenerate triangle -- collinear nodes, nearly collinear nodes, or
    duplicate node coordinates -- has no well-defined strain-displacement
    matrix (the CST formulation divides by the element's area). This is a
    :class:`ValidationError` specialization so existing code that catches
    the broader category still works, while callers that specifically
    care about element geometry can catch this exception precisely.
    """


class DuplicateIDError(FiniteElementToolkitError):
    """Raised when an entity is added to a container under an ID that is
    already in use.

    For example, adding two nodes with the same ID to a :class:`Mesh`.
    """


class DuplicateNodeCoordinatesError(ValidationError):
    """Raised when two distinct nodes in a mesh occupy the same physical location.

    Unlike :class:`DuplicateIDError` (two nodes sharing the same *ID*),
    this catches two nodes with *different* IDs placed at the same
    ``(x, y, z)`` coordinates -- a geometric modeling error that
    :meth:`~femtoolkit.mesh.mesh.Mesh.add_node` cannot detect on its own,
    since each node ID is unique by construction. See
    :func:`~femtoolkit.mesh.validation.validate_mesh`.
    """


class EntityNotFoundError(FiniteElementToolkitError):
    """Raised when a requested entity cannot be found in a container.

    For example, looking up a node ID that has not been added to the
    :class:`Mesh`.
    """


class InvalidAnalysisError(FiniteElementToolkitError):
    """Raised when a structural analysis cannot be set up as requested.

    Examples include attempting to solve an analysis whose mesh has no
    nodes or no elements.
    """


class InvalidElementError(FiniteElementToolkitError):
    """Raised when an analysis encounters an element type it cannot solve.

    For example, a :class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis`
    only supports :class:`~femtoolkit.mesh.bar_element.BarElement` instances.
    """


class InsufficientConstraintsError(FiniteElementToolkitError):
    """Raised when a structural analysis has no boundary conditions.

    Without at least one prescribed-displacement boundary condition, the
    global stiffness matrix is singular and the system cannot be solved.
    """


class SingularSystemError(FiniteElementToolkitError):
    """Raised when the reduced global stiffness matrix is singular.

    This typically indicates that the structure is a mechanism: even
    though boundary conditions were supplied, the free degrees of freedom
    are not fully restrained (for example, a substructure that is not
    connected to any support). Also raised when a Newmark-beta dynamic
    analysis's reduced *effective* stiffness matrix is singular, the
    dynamic analogue of the same failure mode.
    """


class EigenvalueComputationError(FiniteElementToolkitError):
    """Raised when a generalized eigenvalue (natural frequency) solve fails.

    Covers both an outright numerical failure of the underlying
    LAPACK-based solver (e.g. the mass matrix is not positive definite)
    and a solved eigenvalue that is negative by more than the caller's
    rigid-body tolerance -- physically impossible for a valid ``K``/``M``
    pair, and a strong signal that the stiffness or mass matrix supplied
    was not built correctly (e.g. mismatched DOF ordering).
    """


class NonlinearConvergenceError(FiniteElementToolkitError):
    """Raised when a Newton-Raphson load increment fails to converge.

    Raised by :class:`~femtoolkit.analysis.nonlinear_analysis.NonlinearAnalysis.solve`
    once a load step exhausts its iteration budget
    (``NonlinearSolverSettings.max_iterations``) without satisfying the
    configured convergence criterion. The load steps that *did* converge
    before the failure -- and the failed step's own final (uncommitted)
    residual/iteration information -- remain available via
    :attr:`step_results` on the exception itself, so a caller can inspect
    how far the analysis got without needing to catch a bare partial
    result from a non-raising API.

    Per Version 13's scope, no automatic step-size reduction is attempted
    on failure (see the module docstring for
    :mod:`femtoolkit.analysis.nonlinear_analysis`) -- this is the single,
    simple failure signal for this version, with automatic cutback left
    as a documented future extension point.
    """

    def __init__(self, message: str, step_results: object = ()) -> None:
        """Create the error, optionally attaching the partial step-result history.

        Args:
            message: Human-readable description of the failure.
            step_results: The sequence of
                :class:`~femtoolkit.results.nonlinear_result.LoadStepResult`
                objects completed (or attempted) before raising, exposed
                as :attr:`step_results` for callers that catch this
                exception.
        """
        super().__init__(message)
        self.step_results = step_results
