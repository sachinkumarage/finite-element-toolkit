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


class InvalidMaterialStateError(FiniteElementToolkitError):
    """Raised when a :class:`~femtoolkit.materials.nonlinear.MaterialState` is incompatible
    with the material being evaluated.

    For example, passing a scalar (1D) committed state's ``plastic_strain``
    to a 2D, 3-component material's
    :meth:`~femtoolkit.materials.nonlinear.NonlinearMaterial.trial_state`,
    or vice versa. This is a defensive check raised *before* the mismatch
    can propagate into a confusing NumPy broadcasting error deep inside a
    return-mapping computation.
    """


class ConstitutiveUpdateError(FiniteElementToolkitError):
    """Raised when a material's return-mapping (constitutive) update fails numerically.

    For example, a non-finite (``NaN``/``inf``) strain reaching a
    material's :meth:`~femtoolkit.materials.nonlinear.NonlinearMaterial.trial_state`
    -- typically the downstream symptom of a diverging Newton-Raphson
    iteration elsewhere in the analysis -- is caught here and reported as
    a single, clear domain error rather than an opaque NumPy warning or a
    silently propagated ``NaN`` stress.
    """


class InvalidDeformationGradientError(ValidationError):
    """Raised when a deformation gradient is invalid for finite-strain (Version 16) use.

    Covers a non-finite entry (typically the downstream symptom of a
    diverging Newton-Raphson iteration, the same failure mode
    :class:`ConstitutiveUpdateError` catches for small-strain materials)
    and a non-positive determinant -- physically, an element that has
    inverted or collapsed to zero volume under excessive deformation, which
    the Total Lagrangian formulation
    (:mod:`femtoolkit.analysis.geometric_nonlinear`) is not valid for. A
    :class:`ValidationError` specialization, mirroring
    :class:`DegenerateElementError`'s relationship to it.
    """


class UnsupportedLoadingPathError(FiniteElementToolkitError):
    """Raised when a material model is driven along a loading path it does not support.

    :class:`~femtoolkit.materials.hardening.MultilinearIsotropicHardeningMaterial1D`
    is defined directly as a stress-strain curve in *total strain* space
    and only supports monotonic (radial) loading -- see that class's
    docstring for why unloading/reversal cannot be represented correctly
    without additional (unimplemented) bookkeeping. Attempting to unload
    or reverse the loading direction raises this exception rather than
    silently returning a physically wrong stress.
    """


class InvalidElementConnectivityError(ValidationError):
    """Raised when an element's node connectivity is structurally invalid.

    Covers cases :class:`~femtoolkit.mesh.mesh.Mesh` itself cannot catch
    at construction time -- for example a mesh reconstructed from
    external data (see :mod:`femtoolkit.mesh.validation`) whose element
    references a node ID absent from that same data, or lists fewer
    node references than its element type requires. A
    :class:`ValidationError` specialization, mirroring
    :class:`DegenerateElementError`'s relationship to it.
    """


class InvalidMeshError(ValidationError):
    """Raised when a whole mesh fails a structural or geometric validity check.

    Used by :mod:`femtoolkit.mesh.validation` for mesh-wide problems that
    are not localized to a single element or node -- for example
    attempting an operation (refinement, quality evaluation) on a mesh
    whose validation report status is ``"ERROR"``.
    """


class UnsupportedRefinementError(ValidationError):
    """Raised when mesh refinement is requested for an element type with no
    reliable refinement rule implemented.

    :mod:`femtoolkit.mesh.refinement` only refines
    :class:`~femtoolkit.mesh.cst_element.CSTElement2D` and
    :class:`~femtoolkit.mesh.quad_element.QuadElement2D` elements, for
    which a mathematically well-defined uniform subdivision rule exists
    (edge-midpoint quadrisection). TET4/HEX8 and 1D elements are not yet
    supported -- see the module docstring for why.
    """


class UnsupportedQualityMetricError(ValidationError):
    """Raised when a shape-quality metric is requested for an element type
    with no mathematically meaningful definition for it.

    For example, a Jacobian-based quality metric has no meaning for a
    :class:`~femtoolkit.mesh.bar_element.BarElement` (a 1D line has no
    isoparametric area/volume mapping), and :mod:`femtoolkit.mesh.quality`
    raises this rather than fabricating a number.
    """


class SolverError(FiniteElementToolkitError):
    """Base class for errors raised by :mod:`femtoolkit.solvers` (Version 26).

    Catching this exception handles any problem specific to the linear
    solver infrastructure (dense, sparse direct, or iterative) without
    needing to know which concrete solver raised it. Does not replace
    :class:`SingularSystemError`, which stays the toolkit's existing,
    reused signal for a singular reduced system across every solver
    implementation.
    """


class SolverConvergenceError(SolverError):
    """Raised when an iterative linear solver (e.g. Conjugate Gradient) fails
    to converge within its configured maximum iteration count.

    Distinct from :class:`NonlinearConvergenceError` (a Newton-Raphson
    load-step failure): this is about one *linear* solve failing to
    reach its residual tolerance, not a nonlinear equilibrium iteration.
    """


class InvalidSolverConfigurationError(SolverError):
    """Raised when a solver is configured with invalid or incompatible settings.

    Examples: a non-positive tolerance or maximum-iteration count, or
    requesting a solver/matrix-representation combination the
    architecture does not support (e.g. an iterative solver paired with
    a dense matrix representation in this toolkit's solver-selection
    mechanism).
    """


class UnsupportedSolverError(SolverError):
    """Raised when an unknown or unavailable solver is requested by name.

    Used by the solver-selection mechanism
    (:func:`femtoolkit.solvers.create_solver`) when the requested
    ``matrix_type``/``solver_type`` combination does not name a
    supported solver.
    """


class ExecutionError(FiniteElementToolkitError):
    """Base class for errors raised by :mod:`femtoolkit.execution` (Version 27).

    Catching this exception handles any problem specific to serial/parallel
    element-task execution without needing to know whether the failure was
    a configuration mistake, a worker-side exception, or a serialization
    problem.
    """


class InvalidExecutionConfigurationError(ExecutionError):
    """Raised when an :class:`~femtoolkit.execution.config.ExecutionConfig` is invalid.

    Examples: a non-positive worker count, a non-positive explicit chunk
    size, or an unknown backend name.
    """


class TaskSerializationError(ExecutionError):
    """Raised when a task or its arguments cannot be sent to a worker process.

    :class:`~concurrent.futures.ProcessPoolExecutor` communicates with its
    worker processes by pickling the callable and every argument; a
    closure, lambda, or bound method of an unpicklable object cannot cross
    that boundary. This is a toolkit-specific wrapper around the
    underlying :class:`pickle.PicklingError`/:class:`AttributeError` so
    callers can catch one exception type regardless of which stdlib
    exception the pickling failure happened to surface as.
    """


class WorkerExecutionError(ExecutionError):
    """Raised when a worker process raises while executing a task.

    The original exception is chained via ``__cause__`` (``raise ... from
    error``) so its type and message remain inspectable; this wrapper
    exists because a worker-side exception is otherwise reported by
    :mod:`concurrent.futures` with a traceback that only makes sense in
    the worker process, not the caller's.
    """


class StudyError(FiniteElementToolkitError):
    """Base class for errors raised by :mod:`femtoolkit.studies` (Version 30).

    Catching this exception handles any problem specific to parameter
    studies (scenario generation, run orchestration, comparison) without
    needing to know the precise cause.
    """


class StudySizeExceededError(StudyError):
    """Raised when a parameter study would generate more scenarios than allowed.

    A multi-parameter study's scenario count grows combinatorially (the
    product of every parameter's value count); this guards against
    accidentally requesting an unreasonable number of simulation runs.
    Raised *before* any scenario is executed -- a study is never silently
    truncated.
    """


class DuplicateScenarioIdError(StudyError):
    """Raised when two scenarios in the same study share a scenario ID.

    Every scenario in a study must be uniquely identifiable so its run
    can be unambiguously traced back to it in run history and reports.
    """
