"""Turns a plain :class:`~femtoolkit.application.project.Project` into real toolkit objects.

This is the one place in the application layer that touches the actual
FEA domain model (:mod:`femtoolkit.materials`, :mod:`femtoolkit.mesh`,
:mod:`femtoolkit.analysis`, :mod:`femtoolkit.thermal`) -- it performs no
numerical work of its own, only orchestration: build a material, build
a mesh over that material (reusing the Version 8 generator), and turn
region-based boundary-condition/load configuration into the existing
:class:`~femtoolkit.analysis.boundary_conditions.BoundaryCondition`/
:class:`~femtoolkit.analysis.loads.NodalLoad`/
:class:`~femtoolkit.thermal.thermal_boundary_conditions.PrescribedTemperature`/
:class:`~femtoolkit.thermal.thermal_boundary_conditions.PrescribedHeatFlux`
objects those solvers already understand, by resolving each region to
its nodes via :meth:`~femtoolkit.mesh.mesh.Mesh.nodes_on_boundary`
(Version 9). Nothing here assembles a stiffness matrix, integrates a
shape function, or solves a system -- see the module docstring's
architecture rule: the GUI (and this service layer beneath it) must
never become the FEA engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from femtoolkit.analysis.boundary_conditions import BoundaryCondition
from femtoolkit.analysis.dof import TranslationDOF
from femtoolkit.analysis.loads import NodalLoad
from femtoolkit.application.project import Project
from femtoolkit.exceptions import ValidationError
from femtoolkit.geometry.point import Point2D
from femtoolkit.geometry.rectangle import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.generator import create_quad_mesh, create_triangular_mesh
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.mesh.quality import MeshQualitySummary, compute_mesh_quality_summary
from femtoolkit.mesh.refinement import refine_uniform
from femtoolkit.solvers import create_solver

if TYPE_CHECKING:
    from femtoolkit.solvers.base import LinearSolver
from femtoolkit.thermal import PrescribedHeatFlux, PrescribedTemperature, ThermalMaterial

_MECHANICAL_DOF = {"X": TranslationDOF.X, "Y": TranslationDOF.Y}


@dataclass(frozen=True)
class MeshSummary:
    """A human-readable summary of a generated mesh (spec section 8).

    Attributes:
        num_nodes: Total node count.
        num_elements: Total element count.
        dimension: Always ``"2D"`` for this version's structured generator.
        element_type: ``"CSTElement2D"`` or ``"QuadElement2D"``.
        quality: The whole-mesh shape-quality summary (Version 8).
    """

    num_nodes: int
    num_elements: int
    dimension: str
    element_type: str
    quality: MeshQualitySummary


class ModelService:
    """Builds mesh/material/boundary-condition/load objects from a :class:`Project`."""

    def build_rectangle(self, project: Project) -> Rectangle:
        """Build the rectangular domain a project's mesh is generated over."""
        return Rectangle(
            width=project.mesh.width, height=project.mesh.height, origin=Point2D(0.0, 0.0)
        )

    def build_mesh(self, project: Project) -> Mesh:
        """Generate the structured mesh described by ``project.mesh``.

        A mechanical analysis assigns the project's real
        :class:`~femtoolkit.materials.linear_elastic_2d.LinearElastic2D`
        material to every element, exactly as
        :func:`~femtoolkit.mesh.generator.create_quad_mesh` requires. A
        thermal analysis uses a placeholder elastic material for
        geometry only (the same pattern this toolkit's own thermal
        examples have used since Version 20) -- the real thermal
        material is attached separately by :meth:`build_thermal_materials`.

        If ``project.mesh.refinement_passes`` is positive (Version 25),
        the generated mesh is then uniformly refined that many times via
        :func:`~femtoolkit.mesh.refinement.refine_uniform` before being
        returned -- this is how a mesh accepted on the GUI's Mesh
        Preparation workflow actually reaches the solver. The default,
        ``0``, reproduces every Version 24 project's exact prior
        behavior unchanged.

        Args:
            project: The project to build a mesh for.

        Returns:
            The generated :class:`~femtoolkit.mesh.mesh.Mesh`.
        """
        geometry_material = self._mechanical_material(project) or LinearElastic2D(
            youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress"
        )
        mesh_config = project.mesh
        if mesh_config.element_type == "quad":
            mesh = create_quad_mesh(
                width=mesh_config.width,
                height=mesh_config.height,
                nx=mesh_config.nx,
                ny=mesh_config.ny,
                material=geometry_material,
                thickness=mesh_config.thickness,
            )
        elif mesh_config.element_type == "cst":
            mesh = create_triangular_mesh(
                width=mesh_config.width,
                height=mesh_config.height,
                nx=mesh_config.nx,
                ny=mesh_config.ny,
                material=geometry_material,
                thickness=mesh_config.thickness,
            )
        else:
            raise ValidationError(f"Unknown mesh element_type {mesh_config.element_type!r}.")

        for _ in range(mesh_config.refinement_passes):
            mesh = refine_uniform(mesh)
        return mesh

    def mesh_summary(self, mesh: Mesh) -> MeshSummary:
        """Summarize a generated mesh's size, type, and shape quality."""
        element_type = type(mesh.elements[0]).__name__ if mesh.elements else "(none)"
        return MeshSummary(
            num_nodes=len(mesh.nodes),
            num_elements=len(mesh.elements),
            dimension="2D",
            element_type=element_type,
            quality=compute_mesh_quality_summary(mesh),
        )

    def build_solver(self, project: Project) -> LinearSolver | None:
        """Build the Version 26 solver strategy described by ``project.solver``.

        Args:
            project: The project whose ``solver`` configuration to
                build a solver for.

        Returns:
            ``None`` for the default ``matrix_type="dense"``,
            ``solver_type="direct"`` combination -- meaning "use every
            prior version's exact dense-solve behavior, unchanged"
            (see :class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis`'s
            ``solver`` parameter) -- otherwise the matching
            :class:`~femtoolkit.solvers.base.LinearSolver` instance
            from :func:`femtoolkit.solvers.create_solver`.

        Raises:
            UnsupportedSolverError: If ``project.solver`` names an
                unsupported matrix-type/solver-type combination.
            InvalidSolverConfigurationError: If ``project.solver.tolerance``/
                ``max_iterations`` is invalid.
        """
        solver_config = project.solver
        if solver_config.matrix_type == "dense" and solver_config.solver_type == "direct":
            return None
        return create_solver(
            matrix_type=solver_config.matrix_type,
            solver_type=solver_config.solver_type,
            tolerance=solver_config.tolerance,
            max_iterations=solver_config.max_iterations,
        )

    def build_thermal_materials(self, project: Project, mesh: Mesh) -> dict[int, ThermalMaterial]:
        """Assign the project's thermal material to every element in ``mesh``."""
        material = project.material
        thermal_material = ThermalMaterial(
            thermal_conductivity=material.thermal_conductivity,
            density=material.density or 1.0,
            specific_heat=material.specific_heat,
        )
        return {element.id: thermal_material for element in mesh.elements}

    def build_boundary_conditions(
        self, project: Project, mesh: Mesh
    ) -> list[BoundaryCondition] | list[PrescribedTemperature]:
        """Resolve every configured boundary condition to per-node solver objects.

        Args:
            project: The project the boundary conditions belong to.
            mesh: The mesh those boundary conditions are applied over
                (must have been built from the same ``project``).

        Returns:
            One :class:`~femtoolkit.analysis.boundary_conditions.BoundaryCondition`
            per constrained node/DOF (mechanical analyses), or one
            :class:`~femtoolkit.thermal.thermal_boundary_conditions.PrescribedTemperature`
            per constrained node (thermal analyses).
        """
        rectangle = self.build_rectangle(project)
        is_thermal = project.analysis_type in ("thermal_steady_state", "thermomechanical")

        resolved: list = []
        for bc in project.boundary_conditions:
            nodes = mesh.nodes_on_boundary(rectangle.boundary(bc.region))
            for node in nodes:
                if is_thermal:
                    resolved.append(PrescribedTemperature(node_id=node.id, value=bc.value))
                else:
                    resolved.append(
                        BoundaryCondition(
                            node_id=node.id, dof=_MECHANICAL_DOF[bc.dof], value=bc.value
                        )
                    )
        return resolved

    def build_loads(
        self, project: Project, mesh: Mesh
    ) -> list[NodalLoad] | list[PrescribedHeatFlux]:
        """Resolve every configured load to per-node solver objects.

        Args:
            project: The project the loads belong to.
            mesh: The mesh those loads are applied over (must have been
                built from the same ``project``).

        Returns:
            One :class:`~femtoolkit.analysis.loads.NodalLoad` per loaded
            node/DOF (mechanical analyses), or one
            :class:`~femtoolkit.thermal.thermal_boundary_conditions.PrescribedHeatFlux`
            per loaded node (thermal analyses). The configured
            magnitude is applied to **every** node on the region, not
            split into a single resultant.
        """
        rectangle = self.build_rectangle(project)
        is_thermal = project.analysis_type in ("thermal_steady_state", "thermomechanical")

        resolved: list = []
        for load in project.loads:
            nodes = mesh.nodes_on_boundary(rectangle.boundary(load.region))
            for node in nodes:
                if is_thermal:
                    resolved.append(PrescribedHeatFlux(node_id=node.id, value=load.magnitude))
                else:
                    resolved.append(
                        NodalLoad(
                            node_id=node.id, dof=_MECHANICAL_DOF[load.dof], value=load.magnitude
                        )
                    )
        return resolved

    def _mechanical_material(self, project: Project) -> LinearElastic2D | None:
        material = project.material
        if material.youngs_modulus is None or material.poisson_ratio is None:
            return None
        return LinearElastic2D(
            youngs_modulus=material.youngs_modulus,
            poisson_ratio=material.poisson_ratio,
            formulation="plane_stress",
            density=material.density,
        )
