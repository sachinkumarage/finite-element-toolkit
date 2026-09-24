"""Executes a project through the existing solver pipeline and tracks it
as a `SimulationRun` (Version 30).

:class:`SimulationRunManager` performs **no finite element computation of
its own**. It calls
:meth:`~femtoolkit.application.simulation_service.SimulationService.run`
-- the exact same Version 24 validate -> build -> solve -> wrap pipeline
every other part of this toolkit uses -- times it, and wraps the outcome
into a :class:`~femtoolkit.runs.models.SimulationRun` with an immutable
configuration snapshot, structured failure information, and (for a
completed run) the reproducibility metadata and verification status a
study needs downstream.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from femtoolkit.application.project import Project
from femtoolkit.application.simulation_service import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_INVALID,
    SimulationRunResult,
    SimulationService,
)
from femtoolkit.reporting.metadata import ReproducibilityMetadata, collect_reproducibility_metadata
from femtoolkit.runs.models import RunStatus, SimulationRun, new_run_id, snapshot_configuration


def build_reproducibility_metadata(
    project: Project, run_result: SimulationRunResult
) -> ReproducibilityMetadata:
    """Collect reproducibility metadata for a completed run.

    Reuses :func:`~femtoolkit.reporting.metadata.collect_reproducibility_metadata`
    directly from the project's own configuration -- the exact fields the
    Version 29 GUI Verification & Validation page already gathers this
    same way -- rather than requiring a live mesh to be rebuilt just to
    describe it.

    Args:
        project: The (snapshotted) project configuration that was run.
        run_result: The completed run's
            :class:`~femtoolkit.application.simulation_service.SimulationRunResult`.

    Returns:
        A :class:`~femtoolkit.reporting.metadata.ReproducibilityMetadata` record.
    """
    solver_name = "Dense Direct"
    if run_result.solver_diagnostics is not None:
        solver_name = run_result.solver_diagnostics.solver_name

    return collect_reproducibility_metadata(
        model_name=project.name,
        analysis_type=project.analysis_type,
        mesh_statistics={
            "width": project.mesh.width,
            "height": project.mesh.height,
            "nx": project.mesh.nx,
            "ny": project.mesh.ny,
        },
        element_types=[project.mesh.element_type],
        material_properties={
            "youngs_modulus": project.material.youngs_modulus or 0.0,
            "poisson_ratio": project.material.poisson_ratio or 0.0,
        },
        boundary_conditions_summary=f"{len(project.boundary_conditions)} boundary condition(s).",
        loads_summary=f"{len(project.loads)} load(s).",
        solver_name=solver_name,
        execution_mode=project.execution.mode,
    )


class SimulationRunManager:
    """Executes projects through
    :class:`~femtoolkit.application.simulation_service.SimulationService`
    and returns each outcome as a tracked
    :class:`~femtoolkit.runs.models.SimulationRun`.
    """

    def __init__(self, simulation_service: SimulationService | None = None) -> None:
        self._simulation_service = simulation_service or SimulationService()

    def execute(self, project: Project, scenario_id: str | None = None) -> SimulationRun:
        """Run ``project`` and return a fully populated
        :class:`~femtoolkit.runs.models.SimulationRun`.

        Args:
            project: The exact configuration to execute (a scenario
                override, if any, must already be applied by the
                caller -- see
                :func:`~femtoolkit.studies.scenarios.apply_scenario`).
            scenario_id: The originating scenario's ID, if this run was
                generated from a scenario, else ``None``.

        Returns:
            A :class:`~femtoolkit.runs.models.SimulationRun` with
            ``status`` set to ``COMPLETED`` or ``FAILED`` -- this method
            never raises for a failure that
            :meth:`~femtoolkit.application.simulation_service.SimulationService.run`
            itself reports in a structured way (invalid configuration or
            a solver error); those are recorded on the returned run via
            ``error_stage``/``error_message`` instead.
        """
        run = SimulationRun(
            run_id=new_run_id(),
            project_id=project.project_id,
            configuration_snapshot=snapshot_configuration(project),
            scenario_id=scenario_id,
            status=RunStatus.RUNNING,
            started_at=datetime.now(UTC).isoformat(),
        )

        start = time.perf_counter()
        result = self._simulation_service.run(project)
        run.execution_time_seconds = time.perf_counter() - start
        run.completed_at = datetime.now(UTC).isoformat()
        run.result = result

        if result.status == RUN_STATUS_COMPLETED:
            run.status = RunStatus.COMPLETED
            if result.equilibrium_check is not None:
                run.verification_status = result.equilibrium_check.status
            run.reproducibility_metadata = build_reproducibility_metadata(project, result)
        else:
            run.status = RunStatus.FAILED
            run.error_stage = "validation" if result.status == RUN_STATUS_INVALID else "solve"
            run.error_message = "; ".join(result.errors) or "The run did not complete."

        return run
