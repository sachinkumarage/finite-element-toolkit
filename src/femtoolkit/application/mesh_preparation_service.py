"""The Version 25 Model Preparation workflow's service layer (spec section 15).

Wraps the new Version 25 mesh subsystem
(:mod:`femtoolkit.mesh.sizing`, :mod:`femtoolkit.mesh.validation`,
:mod:`femtoolkit.mesh.quality`, :mod:`femtoolkit.mesh.statistics`,
:mod:`femtoolkit.mesh.refinement`) behind one small service, exactly
mirroring how :class:`~femtoolkit.application.simulation_service.SimulationService`
wraps the solver: the GUI's Mesh page calls this service, never the
mesh subsystem directly, so meshing algorithms stay entirely out of
:mod:`femtoolkit.gui` (spec section 24's architecture rule).

This service never re-implements mesh generation -- :meth:`preview`
delegates straight to :class:`~femtoolkit.application.model_service.ModelService.build_mesh`,
which already knows how to turn a
:class:`~femtoolkit.application.project.Project`'s ``mesh`` configuration
(including ``refinement_passes``) into a real
:class:`~femtoolkit.mesh.mesh.Mesh`. "Accept Mesh" (spec section 15's
final workflow step) needs no separate action here: the previewed mesh
*is* exactly the mesh :class:`~femtoolkit.application.simulation_service.SimulationService`
will build and solve, since both go through the same
``project.mesh`` configuration and the same ``ModelService.build_mesh``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from femtoolkit.application.model_service import ModelService
from femtoolkit.mesh.quality.evaluator import DEFAULT_POOR_QUALITY_THRESHOLD, QualityEvaluator
from femtoolkit.mesh.quality.quality_report import MeshQualityReport
from femtoolkit.mesh.sizing import MeshSizingParameters
from femtoolkit.mesh.statistics import MeshStatistics, compute_mesh_statistics
from femtoolkit.mesh.validation import MeshValidationReport, generate_validation_report

if TYPE_CHECKING:
    from femtoolkit.application.project import Project
    from femtoolkit.mesh.mesh import Mesh


@dataclass
class MeshPreparationService:
    """Generate, inspect, validate, evaluate, and refine a project's mesh.

    Attributes:
        model_service: The underlying mesh-building service. Shared
            with :class:`~femtoolkit.application.simulation_service.SimulationService`
            so "the previewed mesh" and "the mesh that gets solved" are
            always, structurally, the exact same object.
    """

    model_service: ModelService = field(default_factory=ModelService)

    def apply_sizing(self, project: Project, sizing: MeshSizingParameters) -> None:
        """Write a sizing specification's ``(nx, ny)`` back into ``project.mesh``.

        Args:
            project: The project to update in place.
            sizing: The requested global/min/max element sizing.

        Raises:
            ValidationError: If ``sizing`` is inconsistent with
                ``project.mesh``'s current width/height.
        """
        nx, ny = sizing.subdivisions(project.mesh.width, project.mesh.height)
        project.mesh.nx = nx
        project.mesh.ny = ny

    def preview(self, project: Project) -> Mesh:
        """Generate the mesh ``project`` currently describes (including any refinement).

        This is exactly the mesh
        :class:`~femtoolkit.application.simulation_service.SimulationService`
        will solve -- there is no separate "accept" step needed beyond
        this preview already reflecting the live project configuration.

        Args:
            project: The project to build a mesh for.

        Returns:
            The generated :class:`~femtoolkit.mesh.mesh.Mesh`.
        """
        return self.model_service.build_mesh(project)

    def validation_report(
        self, mesh: Mesh, poor_quality_threshold: float = DEFAULT_POOR_QUALITY_THRESHOLD
    ) -> MeshValidationReport:
        """Run the Version 25 structured validation sweep over a mesh."""
        return generate_validation_report(mesh, poor_quality_threshold)

    def quality_report(
        self, mesh: Mesh, poor_quality_threshold: float = DEFAULT_POOR_QUALITY_THRESHOLD
    ) -> MeshQualityReport:
        """Evaluate a mesh's shape quality."""
        return QualityEvaluator(poor_quality_threshold).evaluate(mesh)

    def statistics(self, mesh: Mesh) -> MeshStatistics:
        """Compute a mesh's descriptive statistics."""
        return compute_mesh_statistics(mesh)

    def refine(self, project: Project) -> None:
        """Increment ``project.mesh.refinement_passes`` by one.

        The actual refinement is applied lazily, the next time
        :meth:`preview` (or a real simulation run) builds the mesh --
        keeping this service, like every other application service,
        free of any meshing algorithm of its own.

        Args:
            project: The project to update in place.
        """
        project.mesh.refinement_passes += 1

    def reset_refinement(self, project: Project) -> None:
        """Reset ``project.mesh.refinement_passes`` to zero."""
        project.mesh.refinement_passes = 0


__all__ = ["MeshPreparationService"]
