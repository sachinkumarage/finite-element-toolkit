"""Dynamic (time-history) analysis result representation.

:class:`DynamicResult` is the dynamic counterpart of
:class:`~femtoolkit.results.analysis_result.AnalysisResult`: instead of
one displacement/reaction value per DOF, it holds a full **time
history** -- one row per time step -- for displacement, velocity,
acceleration, and reaction. It is a deliberately separate class, not a
subclass or modification of ``AnalysisResult``: a static result is a
single snapshot in time and every one of its existing methods
(``displacement()``, ``element_stress()``, etc.) returns a scalar or a
fixed-size vector; conflating that with a time-varying quantity would
either break its return-type contract or silently change its meaning
for every version that already depends on it. Static analysis
(Versions 1-10) is completely unaffected by this module's existence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from femtoolkit.analysis.dof import DOFMap, TranslationDOF
from femtoolkit.exceptions import ValidationError


@dataclass(frozen=True)
class DynamicResult:
    """Read-only time-history results of a solved dynamic (Newmark) analysis.

    Instances are produced by
    :meth:`~femtoolkit.analysis.dynamic_analysis.DynamicAnalysis.solve`
    and should not be constructed directly by application code.

    Attributes:
        dof_map: DOF map used for the analysis, defining the global DOF
            numbering that every ``*_history`` array's columns are
            expressed in.
        time: Time values, in seconds, of shape ``(n_steps + 1,)``
            (including ``t = 0``).
        displacement_history: Displacement at every time step, shape
            ``(n_steps + 1, dof_map.total_dofs)``.
        velocity_history: Velocity at every time step, same shape as
            ``displacement_history``.
        acceleration_history: Acceleration at every time step, same
            shape as ``displacement_history``.
        reaction_history: Reaction force at every time step, same shape
            as ``displacement_history``. Computed as
            ``R(t) = M*a(t) + C*v(t) + K*u(t) - F(t)``, the dynamic
            analogue of the static ``R = K*u - F`` -- meaningful mainly
            at constrained DOFs, approximately zero elsewhere.
    """

    dof_map: DOFMap
    time: np.ndarray
    displacement_history: np.ndarray
    velocity_history: np.ndarray
    acceleration_history: np.ndarray
    reaction_history: np.ndarray

    def __post_init__(self) -> None:
        """Validate that every history array's shape is consistent with ``time``/``dof_map``.

        Raises:
            ValidationError: If any ``*_history`` array's shape does not
                match ``(len(time), dof_map.total_dofs)``.
        """
        expected_shape = (len(self.time), self.dof_map.total_dofs)
        histories = {
            "displacement_history": self.displacement_history,
            "velocity_history": self.velocity_history,
            "acceleration_history": self.acceleration_history,
            "reaction_history": self.reaction_history,
        }
        for name, history in histories.items():
            if history.shape != expected_shape:
                raise ValidationError(
                    f"DynamicResult {name} must have shape {expected_shape}, "
                    f"got {history.shape}."
                )

    def displacement(self, node_id: int, dof: int = TranslationDOF.X) -> np.ndarray:
        """Return the displacement time history at a node for a given DOF direction.

        Args:
            node_id: ID of the node to query.
            dof: DOF direction to query.

        Returns:
            Displacement in meters, shape ``(n_steps + 1,)``.

        Raises:
            EntityNotFoundError: If ``node_id`` was not part of the analysis.
            ValidationError: If ``dof`` is not active for this analysis.
        """
        index = self.dof_map.global_index(node_id, dof)
        return self.displacement_history[:, index]

    def velocity(self, node_id: int, dof: int = TranslationDOF.X) -> np.ndarray:
        """Return the velocity time history at a node for a given DOF direction.

        Args:
            node_id: ID of the node to query.
            dof: DOF direction to query.

        Returns:
            Velocity in m/s, shape ``(n_steps + 1,)``.
        """
        index = self.dof_map.global_index(node_id, dof)
        return self.velocity_history[:, index]

    def acceleration(self, node_id: int, dof: int = TranslationDOF.X) -> np.ndarray:
        """Return the acceleration time history at a node for a given DOF direction.

        Args:
            node_id: ID of the node to query.
            dof: DOF direction to query.

        Returns:
            Acceleration in m/s^2, shape ``(n_steps + 1,)``.
        """
        index = self.dof_map.global_index(node_id, dof)
        return self.acceleration_history[:, index]

    def reaction(self, node_id: int, dof: int = TranslationDOF.X) -> np.ndarray:
        """Return the reaction force time history at a node for a given DOF direction.

        Args:
            node_id: ID of the node to query.
            dof: DOF direction to query.

        Returns:
            Reaction force in newtons, shape ``(n_steps + 1,)``.
        """
        index = self.dof_map.global_index(node_id, dof)
        return self.reaction_history[:, index]

    def node_displacement(self, node_id: int) -> np.ndarray:
        """Return every active displacement component's time history for a node.

        Args:
            node_id: ID of the node to query.

        Returns:
            Shape ``(n_steps + 1, dof_map.dofs_per_node)``.
        """
        return np.column_stack(
            [self.displacement(node_id, dof) for dof in range(self.dof_map.dofs_per_node)]
        )

    def node_reaction(self, node_id: int) -> np.ndarray:
        """Return every active reaction component's time history for a node.

        Args:
            node_id: ID of the node to query.

        Returns:
            Shape ``(n_steps + 1, dof_map.dofs_per_node)``.
        """
        return np.column_stack(
            [self.reaction(node_id, dof) for dof in range(self.dof_map.dofs_per_node)]
        )
