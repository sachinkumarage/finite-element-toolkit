"""Nonlinear (Newton-Raphson) analysis result representation.

:class:`NonlinearAnalysisResult` is the nonlinear counterpart of
:class:`~femtoolkit.results.analysis_result.AnalysisResult`: since a
nonlinear solve proceeds through a sequence of load increments rather
than a single solve, it holds one :class:`LoadStepResult` per increment
(load factor, displacement, reactions, residual norm, iteration count,
convergence status, and every element's material state at that step)
rather than a single snapshot. It is a deliberately separate class, not
a subclass or modification of ``AnalysisResult`` or
``DynamicResult`` -- neither of those is affected by this module's
existence, and every static/dynamic result from Versions 1-12 keeps
working exactly as before.

Version 14 adds no new classes here: hardening plasticity's extra state
(the isotropic hardening variable ``alpha`` and the kinematic back
stress ``X``) already travels through unchanged, since it lives on
:class:`~femtoolkit.materials.nonlinear.MaterialState` itself, which
this module already stores in full per element/Gauss-point. Only two
convenience accessors are new
(:meth:`NonlinearAnalysisResult.element_hardening_variable`,
:meth:`NonlinearAnalysisResult.element_back_stress`), mirroring the
existing :meth:`~NonlinearAnalysisResult.element_plastic_strain`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from femtoolkit.analysis.dof import DOFMap, TranslationDOF
from femtoolkit.analysis.nonlinear_elements import NonlinearElementState
from femtoolkit.exceptions import ValidationError
from femtoolkit.materials.nonlinear import MaterialState


@dataclass(frozen=True)
class LoadStepResult:
    """The outcome of one Newton-Raphson load increment.

    Attributes:
        load_factor: The fraction of the total applied load reached by
            this step, in ``(0, 1]``.
        displacement: Full global displacement vector at the end of
            this step (the converged value if ``converged`` is
            ``True``; the last attempted trial value otherwise).
        reactions: Full global reaction vector, ``F_int(u) - F_ext``,
            meaningful mainly at constrained DOFs -- the nonlinear
            analogue of the static solver's ``R = K@u - F``.
        residual_norm: The dimensionless residual norm ratio (see
            :func:`~femtoolkit.analysis.convergence.residual_norm_ratio`)
            at the end of this step's last iteration, regardless of
            which convergence criterion was actually used to stop.
        iterations: Number of Newton-Raphson iterations performed.
        converged: Whether this step satisfied the configured
            convergence criterion within the allowed iteration budget.
        element_states: Maps each element's ID to its
            :class:`~femtoolkit.materials.nonlinear.MaterialState`
            tuple (length 1 for CST, length 4 for Q4 -- one per Gauss
            point) at the end of this step.
    """

    load_factor: float
    displacement: np.ndarray
    reactions: np.ndarray
    residual_norm: float
    iterations: int
    converged: bool
    element_states: dict[int, NonlinearElementState]


@dataclass(frozen=True)
class NonlinearAnalysisResult:
    """Read-only results of a Newton-Raphson nonlinear analysis.

    Instances are produced by
    :meth:`~femtoolkit.analysis.nonlinear_analysis.NonlinearAnalysis.solve`
    and should not be constructed directly by application code.

    Attributes:
        dof_map: DOF map used for the analysis, defining the global DOF
            numbering every step's ``displacement``/``reactions``
            vectors are expressed in.
        step_results: One :class:`LoadStepResult` per load increment
            attempted, in order.
    """

    dof_map: DOFMap
    step_results: tuple[LoadStepResult, ...]

    @property
    def converged(self) -> bool:
        """Whether every load step converged."""
        return all(step.converged for step in self.step_results)

    def displacement(self, node_id: int, dof: int = TranslationDOF.X, step: int = -1) -> float:
        """Return the displacement at a node/DOF for one load step (default: the last).

        Args:
            node_id: ID of the node to query.
            dof: DOF direction to query.
            step: Index into :attr:`step_results` (default ``-1``, the
                final step).

        Returns:
            Displacement in meters.
        """
        index = self.dof_map.global_index(node_id, dof)
        return float(self.step_results[step].displacement[index])

    def displacement_history(self, node_id: int, dof: int = TranslationDOF.X) -> np.ndarray:
        """Return the displacement at a node/DOF across every load step.

        Args:
            node_id: ID of the node to query.
            dof: DOF direction to query.

        Returns:
            A NumPy array of shape ``(len(step_results),)``.
        """
        index = self.dof_map.global_index(node_id, dof)
        return np.array([step.displacement[index] for step in self.step_results])

    def node_displacement(self, node_id: int, step: int = -1) -> tuple[float, ...]:
        """Return every active displacement component of a node for one load step.

        Args:
            node_id: ID of the node to query.
            step: Index into :attr:`step_results` (default ``-1``).

        Returns:
            A tuple of length ``dof_map.dofs_per_node``, in meters.
        """
        return tuple(
            self.displacement(node_id, dof, step) for dof in range(self.dof_map.dofs_per_node)
        )

    def reaction(self, node_id: int, dof: int = TranslationDOF.X, step: int = -1) -> float:
        """Return the reaction force at a node/DOF for one load step (default: the last).

        Args:
            node_id: ID of the node to query.
            dof: DOF direction to query.
            step: Index into :attr:`step_results` (default ``-1``).

        Returns:
            Reaction force in newtons.
        """
        index = self.dof_map.global_index(node_id, dof)
        return float(self.step_results[step].reactions[index])

    def node_reaction(self, node_id: int, step: int = -1) -> tuple[float, ...]:
        """Return every active reaction component of a node for one load step.

        Args:
            node_id: ID of the node to query.
            step: Index into :attr:`step_results` (default ``-1``).

        Returns:
            A tuple of length ``dof_map.dofs_per_node``, in newtons.
        """
        return tuple(self.reaction(node_id, dof, step) for dof in range(self.dof_map.dofs_per_node))

    def element_state(self, element_id: int, step: int = -1, gauss_point: int = 0) -> MaterialState:
        """Return one element's material state for one load step.

        Args:
            element_id: ID of the element to query.
            step: Index into :attr:`step_results` (default ``-1``).
            gauss_point: Which Gauss point's state to return (``0`` for
                a CST element, which has exactly one; ``0``-``3`` for a
                Q4 element's four points).

        Returns:
            The element's :class:`~femtoolkit.materials.nonlinear.MaterialState`.

        Raises:
            ValidationError: If ``element_id`` was not part of this
                analysis.
        """
        states = self.step_results[step].element_states
        if element_id not in states:
            raise ValidationError(f"No element with id {element_id} in this nonlinear result.")
        return states[element_id].states[gauss_point]

    def element_strain(
        self, element_id: int, step: int = -1, gauss_point: int = 0
    ) -> float | np.ndarray:
        """Return one element's strain for one load step. See :meth:`element_state`."""
        return self.element_state(element_id, step, gauss_point).strain

    def element_stress(
        self, element_id: int, step: int = -1, gauss_point: int = 0
    ) -> float | np.ndarray:
        """Return one element's stress for one load step. See :meth:`element_state`."""
        return self.element_state(element_id, step, gauss_point).stress

    def element_elastic_strain(
        self, element_id: int, step: int = -1, gauss_point: int = 0
    ) -> float | np.ndarray:
        """Return one element's elastic strain for one load step. See :meth:`element_state`."""
        return self.element_state(element_id, step, gauss_point).elastic_strain

    def element_plastic_strain(
        self, element_id: int, step: int = -1, gauss_point: int = 0
    ) -> float | np.ndarray:
        """Return one element's plastic strain for one load step. See :meth:`element_state`."""
        return self.element_state(element_id, step, gauss_point).plastic_strain

    def element_hardening_variable(
        self, element_id: int, step: int = -1, gauss_point: int = 0
    ) -> float | np.ndarray:
        """Return one element's isotropic hardening variable (``alpha``) for one load step.

        See :meth:`element_state`. Zero for materials with no isotropic
        hardening (e.g. every Version 13 material).
        """
        return self.element_state(element_id, step, gauss_point).hardening_variable

    def element_back_stress(
        self, element_id: int, step: int = -1, gauss_point: int = 0
    ) -> float | np.ndarray:
        """Return one element's kinematic hardening back stress (``X``) for one load step.

        See :meth:`element_state`. Zero for materials with no kinematic
        hardening (e.g. every Version 13 material).
        """
        return self.element_state(element_id, step, gauss_point).back_stress

    def element_yielded(
        self, element_id: int, step: int = -1, gauss_point: int = 0
    ) -> bool | np.ndarray:
        """Return whether one element had yielded at one load step. See :meth:`element_state`."""
        return self.element_state(element_id, step, gauss_point).yielded

    def load_factors(self) -> np.ndarray:
        """Return the load factor of every load step, in order."""
        return np.array([step.load_factor for step in self.step_results])

    def iteration_counts(self) -> np.ndarray:
        """Return the Newton-Raphson iteration count of every load step, in order."""
        return np.array([step.iterations for step in self.step_results])

    def residual_norms(self) -> np.ndarray:
        """Return the final residual norm ratio of every load step, in order."""
        return np.array([step.residual_norm for step in self.step_results])
