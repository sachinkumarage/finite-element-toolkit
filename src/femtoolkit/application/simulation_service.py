"""Validates and runs a project's analysis (Version 24).

This is the only place in the application layer that calls a solver's
``.solve()``. It follows the exact workflow spec section 12 asks for:
validate first (and refuse to run at all if validation fails, per
:mod:`femtoolkit.application.validation`), then build the real toolkit
objects (:mod:`femtoolkit.application.model_service`), run the existing
solver unchanged, and wrap the result through the existing Version 22
post-processing pipeline (:mod:`femtoolkit.postprocessing.adapters`,
:func:`~femtoolkit.postprocessing.field_calculator.with_derived_fields`)
-- this module never computes a displacement, stress, or temperature
itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    """

    status: str
    simulation: SimulationResult | None = None
    summary: EngineeringSummary | None = None
    errors: list[str] = field(default_factory=list)

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
            simulation = self._solve(project)
        except FiniteElementToolkitError as exc:
            return SimulationRunResult(status=RUN_STATUS_FAILED, errors=[f"Solver error: {exc}"])
        except Exception as exc:  # noqa: BLE001 - never let a GUI crash on an unexpected solver error
            return SimulationRunResult(
                status=RUN_STATUS_FAILED, errors=[f"Unexpected error: {exc}"]
            )

        return SimulationRunResult(
            status=RUN_STATUS_COMPLETED, simulation=simulation, summary=summarize(simulation)
        )

    def _solve(self, project: Project) -> SimulationResult:
        mesh = self._model_service.build_mesh(project)
        boundary_conditions = self._model_service.build_boundary_conditions(project, mesh)
        loads = self._model_service.build_loads(project, mesh)

        if project.analysis_type == "linear_static":
            analysis = StaticLinearAnalysis(mesh)
            for boundary_condition in boundary_conditions:
                analysis.add_boundary_condition(boundary_condition)
            for load in loads:
                analysis.add_load(load)
            raw_result = analysis.solve()
            return with_derived_fields(from_static_linear(raw_result))

        if project.analysis_type == "thermal_steady_state":
            materials = self._model_service.build_thermal_materials(project, mesh)
            analysis = SteadyStateThermalAnalysis(mesh, materials)
            for boundary_condition in boundary_conditions:
                analysis.add_boundary_condition(boundary_condition)
            for heat_flux in loads:
                analysis.add_heat_flux(heat_flux)
            raw_result = analysis.solve()
            return with_derived_fields(from_thermal_steady_state(raw_result))

        raise FiniteElementToolkitError(
            f"Analysis type {project.analysis_type!r} has no execution path in the GUI service "
            "layer."
        )
