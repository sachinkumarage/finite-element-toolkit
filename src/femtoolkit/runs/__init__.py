"""Simulation run tracking and local run history (Version 30).

A :class:`~femtoolkit.runs.models.SimulationRun` is one tracked execution
of a :class:`~femtoolkit.application.project.Project` -- optionally under
a :class:`~femtoolkit.studies.scenarios.Scenario` override -- built on top
of the existing Version 24
:class:`~femtoolkit.application.simulation_service.SimulationService`
with zero duplication of FEA algorithms. See ``docs/studies.md`` for the
full guide.
"""

from __future__ import annotations

from femtoolkit.runs.history import RunHistory, RunRecord, record_from_run
from femtoolkit.runs.manager import SimulationRunManager, build_reproducibility_metadata
from femtoolkit.runs.models import RunStatus, SimulationRun, new_run_id, snapshot_configuration

__all__ = [
    "RunHistory",
    "RunRecord",
    "RunStatus",
    "SimulationRun",
    "SimulationRunManager",
    "build_reproducibility_metadata",
    "new_run_id",
    "record_from_run",
    "snapshot_configuration",
]
