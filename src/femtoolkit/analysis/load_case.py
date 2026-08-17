"""Load case: a named, mesh-bound collection of boundary conditions and loads.

A :class:`LoadCase` is a thin convenience wrapper that ties together the
Version 9 workflow -- geometry, boundary regions, boundary conditions,
and distributed loads -- into the single object a caller actually wants
to build up and then solve:

.. code-block:: text

    Geometry -> Mesh -> Boundary Regions -> Boundary Conditions / Loads
        -> FEA Assembly -> Solver -> Results

It does not introduce a new solver or a new assembly path: :meth:`LoadCase.solve`
builds an ordinary :class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis`
and adds every collected :class:`~femtoolkit.analysis.boundary_conditions.BoundaryCondition`
and :class:`~femtoolkit.analysis.loads.NodalLoad` to it (expanding any
:class:`~femtoolkit.analysis.distributed_load.DistributedLoad` into its
equivalent nodal loads first), exactly as if the caller had done so by
hand. Load combinations (multiple load cases combined with factors) are
out of scope for this version -- a :class:`LoadCase` represents exactly
one set of conditions and loads, solved once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from femtoolkit.analysis.boundary_conditions import (
    BoundaryCondition,
    boundary_conditions_for_region,
)
from femtoolkit.analysis.distributed_load import DistributedLoad, distributed_load_to_nodal_loads
from femtoolkit.analysis.loads import NodalLoad
from femtoolkit.analysis.static_linear import StaticLinearAnalysis
from femtoolkit.geometry.boundary import BoundaryRegion
from femtoolkit.mesh.mesh import Mesh

if TYPE_CHECKING:
    from femtoolkit.results.analysis_result import AnalysisResult


class LoadCase:
    """A named collection of boundary conditions and loads for one mesh.

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
    """

    def __init__(self, name: str, mesh: Mesh, tolerance: float = 1e-9) -> None:
        """Create an empty load case for the given mesh.

        Args:
            name: Human-readable label for this load case (e.g. ``"Tension"``).
            mesh: The mesh this load case applies to.
            tolerance: Default distance tolerance, in meters, used when
                resolving boundary regions to nodes/edges.
        """
        self.name = name
        self.mesh = mesh
        self.tolerance = tolerance
        self._boundary_conditions: list[BoundaryCondition] = []
        self._nodal_loads: list[NodalLoad] = []

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

        Args:
            boundary: The boundary region to constrain.
            ux: Prescribed X displacement, in meters. ``None`` leaves X free.
            uy: Prescribed Y displacement, in meters. ``None`` leaves Y free.
        """
        self._boundary_conditions.extend(
            boundary_conditions_for_region(
                self.mesh, boundary, ux=ux, uy=uy, tolerance=self.tolerance
            )
        )

    def add_nodal_load(self, load: NodalLoad) -> None:
        """Add a single concentrated nodal load.

        Args:
            load: The nodal load to add.
        """
        self._nodal_loads.append(load)

    def add_distributed_load(self, load: DistributedLoad) -> None:
        """Add a distributed boundary traction, expanded into equivalent nodal loads.

        Args:
            load: The distributed load to add.
        """
        self._nodal_loads.extend(
            distributed_load_to_nodal_loads(self.mesh, load, tolerance=self.tolerance)
        )

    def apply_to(self, analysis: StaticLinearAnalysis) -> None:
        """Add every collected boundary condition and load to an existing analysis.

        Args:
            analysis: The analysis to populate. Must already be
                constructed against this load case's mesh (or an
                equivalent one).
        """
        for boundary_condition in self._boundary_conditions:
            analysis.add_boundary_condition(boundary_condition)
        for load in self._nodal_loads:
            analysis.add_load(load)

    def solve(self) -> AnalysisResult:
        """Build a :class:`StaticLinearAnalysis` for this load case's mesh and solve it.

        Returns:
            The :class:`~femtoolkit.results.analysis_result.AnalysisResult`.
        """
        analysis = StaticLinearAnalysis(self.mesh)
        self.apply_to(analysis)
        return analysis.solve()
