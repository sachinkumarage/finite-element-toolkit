"""Validates and runs a project's analysis (Version 24; solver diagnostics
Version 26; execution/performance diagnostics Version 27).

This is the only place in the application layer that calls a solver's
``.solve()``. It follows the exact workflow spec section 12 asks for:
validate first (and refuse to run at all if validation fails, per
:mod:`femtoolkit.application.validation`), then build the real toolkit
objects (:mod:`femtoolkit.application.model_service`), run the existing
solver unchanged, and wrap the result through the existing Version 22
post-processing pipeline (:mod:`femtoolkit.postprocessing.adapters`,
:func:`~femtoolkit.postprocessing.field_calculator.with_derived_fields`)
-- this module never computes a displacement, stress, or temperature
itself. Since Version 26, ``project.solver`` selects which
:class:`~femtoolkit.solvers.base.LinearSolver` actually runs (see
:meth:`~femtoolkit.application.model_service.ModelService.build_solver`);
the resulting :class:`~femtoolkit.solvers.results.SolverResult`
diagnostics are attached to :class:`SimulationRunResult` so the GUI can
display them without this service performing any solving itself. Since
Version 27, ``project.execution`` likewise selects serial or parallel
element computation (see
:meth:`~femtoolkit.application.model_service.ModelService.build_execution_config`),
and the resulting
:class:`~femtoolkit.performance.profiler.PerformanceReport` is attached
the same way, with no execution/profiling logic living in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from femtoolkit.analysis.static_linear import StaticLinearAnalysis
from femtoolkit.application.model_service import ModelService
from femtoolkit.application.project import Project
from femtoolkit.application.validation import validate_project
from femtoolkit.exceptions import FiniteElementToolkitError
from femtoolkit.postprocessing import (
    EngineeringSummary,
    SimulationResult,
    from_static_linear,
    from_thermal_steady_state,
    summarize,
    with_derived_fields,
)
from femtoolkit.thermal import SteadyStateThermalAnalysis

if TYPE_CHECKING:
    from femtoolkit.performance.profiler import PerformanceReport
    from femtoolkit.solvers.results import SolverResult

RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_INVALID = "invalid"
RUN_STATUS_FAILED = "failed"


@dataclass
class SimulationRunResult:
    """The outcome of one :meth:`SimulationService.run` call.

    Attributes:
        status: One of ``"completed"``, ``"invalid"`` (failed
            pre-solve validation, the solver never ran), or ``"failed"``
            (the solver itself raised).
        simulation: The solved, derived-field-enriched result, if
            ``status == "completed"``.
        summary: The engineering summary of ``simulation``, if
            ``status == "completed"``.
        errors: Human-readable problem descriptions. Populated for
            ``"invalid"`` (one entry per validation failure) and
            ``"failed"`` (one entry describing the solver error);
            empty for ``"completed"``.
        solver_diagnostics: The
            :class:`~femtoolkit.solvers.results.SolverResult` from the
            solver that actually ran, if ``status == "completed"`` and
            ``project.solver`` selected a non-default (Version 26)
            solver strategy; ``None`` for the default dense solve
            (which has no diagnostics object) or when the run did not
            complete.
        performance_report: The
            :class:`~femtoolkit.performance.profiler.PerformanceReport`
            (Version 27) from the run, if ``status == "completed"`` and
            the analysis took the linear (non-nonlinear) solve path;
            ``None`` otherwise.
    """

    status: str
    simulation: SimulationResult | None = None
    summary: EngineeringSummary | None = None
    errors: list[str] = field(default_factory=list)
    solver_diagnostics: SolverResult | None = None
    performance_report: PerformanceReport | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the simulation completed successfully."""
        return self.status == RUN_STATUS_COMPLETED


class SimulationService:
    """Validates and executes a :class:`~femtoolkit.application.project.Project`."""

    def __init__(self, model_service: ModelService | None = None) -> None:
        self._model_service = model_service or ModelService()

    def run(self, project: Project) -> SimulationRunResult:
        """Validate and, if valid, run a project's analysis.

        Args:
            project: The project configuration to run.

        Returns:
            A :class:`SimulationRunResult` describing what happened.
        """
        validation = validate_project(project)
        if not validation.is_valid:
            return SimulationRunResult(status=RUN_STATUS_INVALID, errors=list(validation.errors))

        try:
            simulation, solver_diagnostics, performance_report = self._solve(project)
        except FiniteElementToolkitError as exc:
            return SimulationRunResult(status=RUN_STATUS_FAILED, errors=[f"Solver error: {exc}"])
        except Exception as exc:  # noqa: BLE001 - never let a GUI crash on an unexpected solver error
            return SimulationRunResult(
                status=RUN_STATUS_FAILED, errors=[f"Unexpected error: {exc}"]
            )

        return SimulationRunResult(
            status=RUN_STATUS_COMPLETED,
            simulation=simulation,
            summary=summarize(simulation),
            solver_diagnostics=solver_diagnostics,
            performance_report=performance_report,
        )

    def _solve(
        self, project: Project
    ) -> tuple[SimulationResult, SolverResult | None, PerformanceReport | None]:
        mesh = self._model_service.build_mesh(project)
        boundary_conditions = self._model_service.build_boundary_conditions(project, mesh)
        loads = self._model_service.build_loads(project, mesh)
        solver = self._model_service.build_solver(project)
        execution = self._model_service.build_execution_config(project)

        if project.analysis_type == "linear_static":
            analysis = StaticLinearAnalysis(mesh, solver=solver, execution=execution)
            for boundary_condition in boundary_conditions:
                analysis.add_boundary_condition(boundary_condition)
            for load in loads:
                analysis.add_load(load)
            raw_result = analysis.solve()
            simulation = with_derived_fields(from_static_linear(raw_result))
            return simulation, analysis.last_solver_result, analysis.last_performance_report

        if project.analysis_type == "thermal_steady_state":
            materials = self._model_service.build_thermal_materials(project, mesh)
            analysis = SteadyStateThermalAnalysis(
                mesh, materials, solver=solver, execution=execution
            )
            for boundary_condition in boundary_conditions:
                analysis.add_boundary_condition(boundary_condition)
            for heat_flux in loads:
                analysis.add_heat_flux(heat_flux)
            raw_result = analysis.solve()
            simulation = with_derived_fields(from_thermal_steady_state(raw_result))
            return simulation, analysis.last_solver_result, analysis.last_performance_report

        raise FiniteElementToolkitError(
            f"Analysis type {project.analysis_type!r} has no execution path in the GUI service "
            "layer."
        )
