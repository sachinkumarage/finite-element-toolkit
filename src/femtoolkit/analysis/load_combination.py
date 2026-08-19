"""Load combinations: multiple load cases combined with load factors.

A :class:`LoadCombination` represents a factored sum of load cases -- the
standard structural engineering practice of combining, say, dead load and
live load with code-prescribed factors (e.g. ``1.2 * Dead + 1.6 * Live``
for an ultimate/strength combination):

.. code-block:: text

    ultimate = LoadCombination(
        name="Ultimate",
        factors={dead_load: 1.2, live_load: 1.6},
    )

**How factors are applied.** Each load case's *loads* (nodal,
distributed, gravity, thermal -- all resolved to plain
:class:`~femtoolkit.analysis.loads.NodalLoad` instances) are scaled by
that case's factor and summed. **Boundary conditions are not scaled**:
a support is a physical feature of the structure, not something that
scales with a load factor, so every load case's boundary conditions are
unioned instead. If two load cases in the same combination prescribe
different values for the same node/DOF, that is a genuine modeling
conflict and raises :class:`~femtoolkit.exceptions.ValidationError`
rather than silently picking one. Multi-point constraints are unioned
the same way (also unscaled -- a kinematic tie is likewise a structural
feature, not a load).

This introduces no new solver: :meth:`LoadCombination.solve` builds an
ordinary :class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis`
from the combined boundary conditions, loads, and constraints, exactly
like a single :class:`~femtoolkit.analysis.load_case.LoadCase` does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from femtoolkit.analysis.boundary_conditions import BoundaryCondition
from femtoolkit.analysis.load_case import LoadCase
from femtoolkit.analysis.loads import NodalLoad
from femtoolkit.analysis.multi_point_constraint import MultiPointConstraint
from femtoolkit.analysis.static_linear import StaticLinearAnalysis
from femtoolkit.exceptions import ValidationError
from femtoolkit.mesh.mesh import Mesh

if TYPE_CHECKING:
    from femtoolkit.results.analysis_result import AnalysisResult

_BOUNDARY_CONDITION_VALUE_TOLERANCE: float = 1e-9


@dataclass
class LoadCombination:
    """A named, factored combination of load cases.

    Attributes:
        name: Human-readable label for this combination (e.g. ``"Ultimate"``).
        factors: Maps each included :class:`~femtoolkit.analysis.load_case.LoadCase`
            to its load factor (dimensionless multiplier applied to that
            case's loads).

    Raises:
        ValidationError: If ``name`` is empty, ``factors`` is empty, a
            key is not a :class:`LoadCase`, or a factor is not a finite
            number.

    Example:
        >>> ultimate = LoadCombination(
        ...     name="Ultimate",
        ...     factors={dead_load: 1.2, live_load: 1.6},
        ... )
        >>> result = ultimate.solve(mesh)
    """

    name: str
    factors: dict[LoadCase, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the combination immediately after construction.

        Raises:
            ValidationError: If ``name`` is empty, ``factors`` is empty,
                a key is not a :class:`LoadCase`, or a factor is not a
                finite number.
        """
        if not self.name or not self.name.strip():
            raise ValidationError("LoadCombination name must not be empty.")
        if not self.factors:
            raise ValidationError(
                f"LoadCombination {self.name!r} requires at least one load case."
            )
        for load_case, factor in self.factors.items():
            if not isinstance(load_case, LoadCase):
                raise ValidationError(
                    f"LoadCombination {self.name!r} factors keys must be LoadCase "
                    f"instances, got {load_case!r}."
                )
            if (
                not isinstance(factor, (int, float))
                or isinstance(factor, bool)
                or not math.isfinite(factor)
            ):
                raise ValidationError(
                    f"LoadCombination {self.name!r} factor for {load_case.name!r} must "
                    f"be a finite number, got {factor!r}."
                )

    def resolved_boundary_conditions(self, mesh: Mesh | None = None) -> list[BoundaryCondition]:
        """The union of every included load case's boundary conditions (unscaled).

        Args:
            mesh: The mesh to resolve each load case against. Defaults to
                each load case's own bound mesh.

        Returns:
            One :class:`~femtoolkit.analysis.boundary_conditions.BoundaryCondition`
            per constrained ``(node_id, dof)``, deduplicated across load cases.

        Raises:
            ValidationError: If two load cases prescribe different values
                for the same ``(node_id, dof)``.
        """
        combined: dict[tuple[int, int], BoundaryCondition] = {}
        for load_case in self.factors:
            for condition in load_case.resolved_boundary_conditions(mesh):
                key = (condition.node_id, condition.dof)
                existing = combined.get(key)
                if existing is not None and not math.isclose(
                    existing.value, condition.value, abs_tol=_BOUNDARY_CONDITION_VALUE_TOLERANCE
                ):
                    raise ValidationError(
                        f"LoadCombination {self.name!r} has conflicting boundary "
                        f"conditions for node {condition.node_id}, dof {condition.dof}: "
                        f"{existing.value} vs {condition.value}."
                    )
                combined[key] = condition
        return list(combined.values())

    def resolved_nodal_loads(self, mesh: Mesh | None = None) -> list[NodalLoad]:
        """Every included load case's nodal loads, scaled by its load factor.

        Args:
            mesh: The mesh to resolve each load case against. Defaults to
                each load case's own bound mesh.

        Returns:
            One :class:`~femtoolkit.analysis.loads.NodalLoad` per
            contribution, with ``value`` scaled by the owning load case's
            factor. Loads on the same DOF from different load cases are
            *not* pre-summed here; :func:`~femtoolkit.analysis.system.build_force_vector`
            sums them during assembly, exactly like Version 9's
            distributed loads.
        """
        combined: list[NodalLoad] = []
        for load_case, factor in self.factors.items():
            for load in load_case.resolved_nodal_loads(mesh):
                combined.append(NodalLoad(load.node_id, load.dof, load.value * factor))
        return combined

    def resolved_multi_point_constraints(self) -> list[MultiPointConstraint]:
        """The union of every included load case's multi-point constraints (unscaled).

        Returns:
            Deduplicated :class:`~femtoolkit.analysis.multi_point_constraint.MultiPointConstraint`
            instances across load cases.
        """
        seen: dict[tuple[int, int, int], MultiPointConstraint] = {}
        for load_case in self.factors:
            for constraint in load_case.multi_point_constraints():
                key = (constraint.node_id_a, constraint.node_id_b, constraint.dof)
                seen[key] = constraint
        return list(seen.values())

    def _infer_mesh(self) -> Mesh:
        for load_case in self.factors:
            if load_case.mesh is not None:
                return load_case.mesh
        raise ValidationError(
            f"LoadCombination {self.name!r} has no mesh; pass mesh explicitly or "
            "bind a mesh to at least one of its load cases."
        )

    def solve(self, mesh: Mesh | None = None) -> AnalysisResult:
        """Build a :class:`StaticLinearAnalysis` for the combined loads and solve it.

        Args:
            mesh: The mesh to solve against. Defaults to the mesh bound
                to the first included load case that has one.

        Returns:
            The :class:`~femtoolkit.results.analysis_result.AnalysisResult`.

        Raises:
            ValidationError: If no mesh is available, or if included load
                cases have conflicting boundary conditions.
        """
        target_mesh = mesh or self._infer_mesh()
        analysis = StaticLinearAnalysis(target_mesh)
        for condition in self.resolved_boundary_conditions(target_mesh):
            analysis.add_boundary_condition(condition)
        for load in self.resolved_nodal_loads(target_mesh):
            analysis.add_load(load)
        for constraint in self.resolved_multi_point_constraints():
            analysis.add_multi_point_constraint(constraint)
        return analysis.solve()
