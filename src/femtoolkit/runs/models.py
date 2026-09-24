"""The `SimulationRun` model: one tracked execution (Version 30).

A run is deliberately *not* a new simulation engine -- it wraps the
outcome of the existing Version 24
:class:`~femtoolkit.application.simulation_service.SimulationService`
with the bookkeeping a reproducible parameter study needs: a run
identity, timing, status, and an immutable snapshot of the exact
configuration that produced it.

**Important principle -- immutable configuration snapshots.** A run must
preserve its historical configuration even if the live project
configuration changes later, so a past result never becomes ambiguous
about what produced it. :class:`~femtoolkit.application.project.Project`
and its sub-configs (`MaterialConfig`, `MeshConfig`, ...) are plain,
mutable dataclasses -- the GUI assigns directly to fields like
``project.material.youngs_modulus`` -- so this toolkit does not (and, to
avoid a risky, out-of-scope architectural change, does not attempt to)
enforce immutability at the language level with frozen dataclasses.
Instead, "immutable" is a *policy*: :func:`snapshot_configuration` takes
a deep copy at the moment a run starts, and every piece of code in
:mod:`femtoolkit.runs` and :mod:`femtoolkit.studies` promises never to
mutate a :attr:`SimulationRun.configuration_snapshot` afterward. Treat a
snapshot as read-only.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from femtoolkit.verification.status import VerificationStatus

if TYPE_CHECKING:
    from femtoolkit.application.project import Project
    from femtoolkit.application.simulation_service import SimulationRunResult
    from femtoolkit.reporting.metadata import ReproducibilityMetadata
    from femtoolkit.validation.results import ValidationResult


class RunStatus(Enum):
    """The lifecycle status of one :class:`SimulationRun`.

    Attributes:
        PENDING: Created but not yet started.
        RUNNING: Currently executing.
        COMPLETED: Finished successfully (the underlying
            :class:`~femtoolkit.application.simulation_service.SimulationRunResult`
            has ``status == "completed"``).
        FAILED: Finished unsuccessfully -- either the configuration was
            invalid (never reached the solver) or the solver itself
            raised; see :attr:`SimulationRun.error_stage`.
        CANCELLED: Not executed by choice (e.g. a study stopped before
            this run's turn). No execution engine in this version
            actually cancels an in-flight run -- this status exists for
            completeness and for callers that pre-populate a study's
            run list.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def new_run_id() -> str:
    """Generate a fresh, unique run identifier."""
    return str(uuid.uuid4())


def snapshot_configuration(project: Project) -> Project:
    """Deep-copy a project so a run's configuration is frozen at execution time.

    See the module docstring's "immutable configuration snapshots"
    principle -- the returned :class:`~femtoolkit.application.project.Project`
    is independent of ``project``; later mutating ``project`` (e.g. the
    live GUI project) has no effect on the returned snapshot, or on any
    run that already holds it.

    Args:
        project: The project configuration to snapshot.

    Returns:
        An independent deep copy of ``project``.
    """
    return copy.deepcopy(project)


@dataclass
class SimulationRun:
    """One tracked execution of a project, optionally under a scenario override.

    Attributes:
        run_id: A unique identifier for this run.
        project_id: The originating project's
            :attr:`~femtoolkit.application.project.Project.project_id`.
        configuration_snapshot: An immutable-by-policy snapshot (see
            :func:`snapshot_configuration`) of the exact configuration
            this run executed -- the scenario override already applied,
            if any.
        scenario_id: The originating
            :class:`~femtoolkit.studies.scenarios.Scenario`'s ID, or
            ``None`` for a run of the base project with no scenario
            override.
        status: The run's current lifecycle status.
        started_at: ISO-8601 UTC timestamp when execution began, or
            ``None`` if not yet started.
        completed_at: ISO-8601 UTC timestamp when execution ended
            (successfully or not), or ``None`` if not yet finished.
        execution_time_seconds: Wall-clock solve time, or ``None`` if
            not yet finished.
        result: The full
            :class:`~femtoolkit.application.simulation_service.SimulationRunResult`,
            populated once execution finishes -- its own ``status``
            (``"completed"``/``"invalid"``/``"failed"``) is the
            authoritative source ``error_stage`` is derived from.
            ``None`` before execution.
        verification_status: The Version 29 equilibrium-check status
            carried over from ``result.equilibrium_check`` for a
            completed run, or
            :attr:`~femtoolkit.verification.status.VerificationStatus.NOT_RUN`
            otherwise. This is never fabricated -- see
            :mod:`femtoolkit.runs.manager`.
        validation_result: A caller-supplied
            :class:`~femtoolkit.validation.results.ValidationResult`
            comparing this run against real reference data, or ``None``
            if no such comparison was ever made. Never populated
            automatically.
        reproducibility_metadata: A
            :class:`~femtoolkit.reporting.metadata.ReproducibilityMetadata`
            record collected for a completed run, or ``None``.
        error_stage: Which stage failed -- ``"validation"`` (the
            configuration never reached the solver) or ``"solve"`` (the
            solver itself failed) -- or ``None`` for a run that has not
            failed.
        error_message: A human-readable description of the failure, or
            ``None`` for a run that has not failed.
    """

    run_id: str
    project_id: str
    configuration_snapshot: Project
    scenario_id: str | None = None
    status: RunStatus = RunStatus.PENDING
    started_at: str | None = None
    completed_at: str | None = None
    execution_time_seconds: float | None = None
    result: SimulationRunResult | None = None
    verification_status: VerificationStatus = VerificationStatus.NOT_RUN
    validation_result: ValidationResult | None = None
    reproducibility_metadata: ReproducibilityMetadata | None = None
    error_stage: str | None = None
    error_message: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether this run finished with :attr:`RunStatus.COMPLETED`."""
        return self.status == RunStatus.COMPLETED

    @property
    def failed(self) -> bool:
        """Whether this run finished with :attr:`RunStatus.FAILED`."""
        return self.status == RunStatus.FAILED
