"""Load case: a named collection of boundary conditions and loads.

A :class:`LoadCase` ties together every load-carrying concept in the
toolkit -- boundary regions and conditions (Version 9), distributed
tractions (Version 9), concentrated nodal loads (Version 2), gravity/body
forces, uniform thermal loads, and multi-point constraints (Version 10)
-- into the single object a caller actually wants to build up and then
solve:

.. code-block:: text

    Geometry -> Mesh -> Load Cases -> Load Combinations
        -> Boundary Conditions / Loads -> Solver -> Results

It does not introduce a new solver or a new assembly path: :meth:`LoadCase.solve`
builds an ordinary :class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis`
and adds every collected boundary condition and load to it (expanding
distributed/gravity/thermal loads into their equivalent nodal loads
first), exactly as if the caller had done so by hand.

**Mesh binding.** A load case can be built two ways:

* **Bound to a mesh up front** (the Version 9 style):
  ``LoadCase(name="Tension", mesh=mesh)``. Every ``fix_boundary``/
  ``add_distributed_load``/``add_gravity_load``/``add_thermal_load`` call
  resolves against that mesh immediately, and ``solve()`` needs no
  arguments.
* **Built independently of any mesh** (the Version 10 style, needed for
  :class:`~femtoolkit.analysis.load_combination.LoadCombination` and
  :class:`~femtoolkit.analysis.load_manager.LoadManager`, where the same
  named load cases -- ``dead_load = LoadCase(name="Dead Load")`` -- are
  later solved or combined against a mesh supplied separately):
  region-based and mesh-dependent additions are stored unresolved and
  only converted to concrete :class:`~femtoolkit.analysis.boundary_conditions.BoundaryCondition`/
  :class:`~femtoolkit.analysis.loads.NodalLoad` objects when a mesh
  becomes available, via :meth:`resolved_boundary_conditions`/
  :meth:`resolved_nodal_loads`, or the ``mesh`` argument to
  :meth:`solve`/:meth:`apply_to`.

Load combinations (multiple load cases combined with load factors) are
handled by :class:`~femtoolkit.analysis.load_combination.LoadCombination`,
not by this class -- a :class:`LoadCase` always represents exactly one
set of conditions and loads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from femtoolkit.analysis.body_load import GravityLoad, gravity_load_to_nodal_loads
from femtoolkit.analysis.boundary_conditions import (
    BoundaryCondition,
    boundary_conditions_for_region,
)
from femtoolkit.analysis.distributed_load import DistributedLoad, distributed_load_to_nodal_loads
from femtoolkit.analysis.loads import NodalLoad
from femtoolkit.analysis.multi_point_constraint import MultiPointConstraint
from femtoolkit.analysis.static_linear import StaticLinearAnalysis
from femtoolkit.analysis.thermal_load import TemperatureLoad, thermal_load_to_nodal_loads
from femtoolkit.exceptions import ValidationError
from femtoolkit.geometry.boundary import BoundaryRegion
from femtoolkit.mesh.mesh import Mesh

if TYPE_CHECKING:
    from femtoolkit.results.analysis_result import AnalysisResult


class LoadCase:
    """A named collection of boundary conditions and loads.

    Example:
        >>> domain = Rectangle(width=2.0, height=1.0)
        >>> mesh = create_quad_mesh(
        ...     width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01
        ... )
        >>> load_case = LoadCase(name="Tension", mesh=mesh)
        >>> load_case.fix_boundary(domain.boundary("left"), ux=0.0, uy=0.0)
        >>> traction = DistributedLoad(domain.boundary("right"), magnitude=1000.0)
        >>> load_case.add_distributed_load(traction)
        >>> result = load_case.solve()

        >>> # Mesh-independent style, for use with LoadCombination/LoadManager:
        >>> dead_load = LoadCase(name="Dead Load")
        >>> dead_load.add_gravity_load(GravityLoad(g=9.81))
        >>> result = dead_load.solve(mesh)
    """

    def __init__(self, name: str, mesh: Mesh | None = None, tolerance: float = 1e-9) -> None:
        """Create an empty load case, optionally bound to a mesh.

        Args:
            name: Human-readable label for this load case (e.g.
                ``"Dead Load"``). Must not be empty.
            mesh: The mesh this load case applies to, or ``None`` to
                build the load case independently of any mesh (see the
                module docstring's "Mesh binding" section).
            tolerance: Default distance tolerance, in meters, used when
                resolving boundary regions to nodes/edges.
        """
        if not name or not name.strip():
            raise ValidationError("LoadCase name must not be empty.")

        self.name = name
        self.mesh = mesh
        self.tolerance = tolerance
        self._boundary_conditions: list[BoundaryCondition] = []
        self._nodal_loads: list[NodalLoad] = []
        self._multi_point_constraints: list[MultiPointConstraint] = []
        self._pending_region_fixes: list[tuple[BoundaryRegion, float | None, float | None]] = []
        self._pending_distributed_loads: list[DistributedLoad] = []
        self._pending_gravity_loads: list[GravityLoad] = []
        self._pending_thermal_loads: list[TemperatureLoad] = []

    def add_boundary_condition(self, boundary_condition: BoundaryCondition) -> None:
        """Add a single, already-built boundary condition.

        Args:
            boundary_condition: The boundary condition to add.
        """
        self._boundary_conditions.append(boundary_condition)

    def fix_boundary(
        self, boundary: BoundaryRegion, *, ux: float | None = None, uy: float | None = None
    ) -> None:
        """Constrain every node on a named boundary region.

        Resolved immediately if this load case is bound to a mesh;
        otherwise stored and resolved later (see the module docstring).

        Args:
            boundary: The boundary region to constrain.
            ux: Prescribed X displacement, in meters. ``None`` leaves X free.
            uy: Prescribed Y displacement, in meters. ``None`` leaves Y free.
        """
        if self.mesh is not None:
            self._boundary_conditions.extend(
                boundary_conditions_for_region(
                    self.mesh, boundary, ux=ux, uy=uy, tolerance=self.tolerance
                )
            )
        else:
            self._pending_region_fixes.append((boundary, ux, uy))

    def add_nodal_load(self, load: NodalLoad) -> None:
        """Add a single concentrated nodal load.

        Args:
            load: The nodal load to add.
        """
        self._nodal_loads.append(load)

    def add_distributed_load(self, load: DistributedLoad) -> None:
        """Add a distributed boundary traction, expanded into equivalent nodal loads.

        Resolved immediately if this load case is bound to a mesh;
        otherwise stored and resolved later (see the module docstring).

        Args:
            load: The distributed load to add.
        """
        if self.mesh is not None:
            self._nodal_loads.extend(
                distributed_load_to_nodal_loads(self.mesh, load, tolerance=self.tolerance)
            )
        else:
            self._pending_distributed_loads.append(load)

    def add_gravity_load(self, gravity: GravityLoad) -> None:
        """Add a gravity (body-force) load, expanded into equivalent nodal loads.

        Resolved immediately if this load case is bound to a mesh;
        otherwise stored and resolved later (see the module docstring).

        Args:
            gravity: The gravity load to add.
        """
        if self.mesh is not None:
            self._nodal_loads.extend(gravity_load_to_nodal_loads(self.mesh, gravity))
        else:
            self._pending_gravity_loads.append(gravity)

    def add_thermal_load(self, thermal_load: TemperatureLoad) -> None:
        """Add a uniform temperature-change load, expanded into equivalent nodal loads.

        Resolved immediately if this load case is bound to a mesh;
        otherwise stored and resolved later (see the module docstring).

        Args:
            thermal_load: The thermal load to add.
        """
        if self.mesh is not None:
            self._nodal_loads.extend(thermal_load_to_nodal_loads(self.mesh, thermal_load))
        else:
            self._pending_thermal_loads.append(thermal_load)

    def add_multi_point_constraint(self, constraint: MultiPointConstraint) -> None:
        """Add a multi-point (equal-displacement) constraint.

        Args:
            constraint: The multi-point constraint to add.
        """
        self._multi_point_constraints.append(constraint)

    def multi_point_constraints(self) -> list[MultiPointConstraint]:
        """Every multi-point constraint added to this load case, in insertion order."""
        return list(self._multi_point_constraints)

    def resolved_boundary_conditions(self, mesh: Mesh | None = None) -> list[BoundaryCondition]:
        """Every boundary condition this load case implies, resolved against a mesh.

        Args:
            mesh: The mesh to resolve pending boundary regions against.
                Defaults to this load case's own bound mesh.

        Returns:
            Directly added :class:`~femtoolkit.analysis.boundary_conditions.BoundaryCondition`
            instances, plus one per constrained DOF at each node matching
            a pending :meth:`fix_boundary` region.

        Raises:
            ValidationError: If pending boundary regions exist and no
                mesh is available (neither passed in nor bound at
                construction).
        """
        target_mesh = mesh or self.mesh
        conditions = list(self._boundary_conditions)
        if self._pending_region_fixes:
            if target_mesh is None:
                raise ValidationError(
                    f"LoadCase {self.name!r} has boundary regions pending resolution "
                    "but no mesh; pass mesh explicitly or construct with mesh=... ."
                )
            for boundary, ux, uy in self._pending_region_fixes:
                conditions.extend(
                    boundary_conditions_for_region(
                        target_mesh, boundary, ux=ux, uy=uy, tolerance=self.tolerance
                    )
                )
        return conditions

    def resolved_nodal_loads(self, mesh: Mesh | None = None) -> list[NodalLoad]:
        """Every nodal load this load case implies, resolved against a mesh.

        Args:
            mesh: The mesh to resolve pending distributed/gravity/thermal
                loads against. Defaults to this load case's own bound mesh.

        Returns:
            Directly added :class:`~femtoolkit.analysis.loads.NodalLoad`
            instances, plus the equivalent nodal loads of every pending
            distributed, gravity, and thermal load.

        Raises:
            ValidationError: If pending mesh-dependent loads exist and no
                mesh is available (neither passed in nor bound at
                construction).
        """
        target_mesh = mesh or self.mesh
        loads = list(self._nodal_loads)
        has_pending = (
            self._pending_distributed_loads
            or self._pending_gravity_loads
            or self._pending_thermal_loads
        )
        if has_pending and target_mesh is None:
            raise ValidationError(
                f"LoadCase {self.name!r} has loads pending resolution but no mesh; "
                "pass mesh explicitly or construct with mesh=... ."
            )
        if target_mesh is not None:
            for distributed_load in self._pending_distributed_loads:
                loads.extend(
                    distributed_load_to_nodal_loads(
                        target_mesh, distributed_load, tolerance=self.tolerance
                    )
                )
            for gravity_load in self._pending_gravity_loads:
                loads.extend(gravity_load_to_nodal_loads(target_mesh, gravity_load))
            for thermal_load in self._pending_thermal_loads:
                loads.extend(thermal_load_to_nodal_loads(target_mesh, thermal_load))
        return loads

    def apply_to(self, analysis: StaticLinearAnalysis, mesh: Mesh | None = None) -> None:
        """Add every collected boundary condition, load, and constraint to an analysis.

        Args:
            analysis: The analysis to populate. Must already be
                constructed against this load case's mesh (or an
                equivalent one).
            mesh: The mesh to resolve any pending region-based or
                mesh-dependent loads against. Defaults to this load
                case's own bound mesh.
        """
        for boundary_condition in self.resolved_boundary_conditions(mesh):
            analysis.add_boundary_condition(boundary_condition)
        for load in self.resolved_nodal_loads(mesh):
            analysis.add_load(load)
        for constraint in self._multi_point_constraints:
            analysis.add_multi_point_constraint(constraint)

    def solve(self, mesh: Mesh | None = None) -> AnalysisResult:
        """Build a :class:`StaticLinearAnalysis` against a mesh and solve it.

        Args:
            mesh: The mesh to solve against. Defaults to this load
                case's own bound mesh.

        Returns:
            The :class:`~femtoolkit.results.analysis_result.AnalysisResult`.

        Raises:
            ValidationError: If neither ``mesh`` nor a mesh bound at
                construction is available.
        """
        target_mesh = mesh or self.mesh
        if target_mesh is None:
            raise ValidationError(
                f"LoadCase {self.name!r} has no mesh; pass mesh explicitly or "
                "construct with mesh=... ."
            )
        analysis = StaticLinearAnalysis(target_mesh)
        self.apply_to(analysis, target_mesh)
        return analysis.solve()
