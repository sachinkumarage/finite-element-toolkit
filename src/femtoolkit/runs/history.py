"""Lightweight local run-history persistence (Version 30).

A full :class:`~femtoolkit.runs.models.SimulationRun` holds a live
:class:`~femtoolkit.application.simulation_service.SimulationRunResult`
-- numpy arrays, a full field-by-field simulation result -- which is
appropriate for immediate in-process use (e.g. generating a report right
after a run finishes) but is not something a lightweight local history
should serialize wholesale. :class:`RunRecord` is the small,
JSON-serializable summary that actually gets persisted: scalar key
results, status, timings, and the run's configuration snapshot as plain
data.

Persistence mirrors
:class:`~femtoolkit.application.project_service.ProjectService`'s exact
JSON save/load pattern (the simplest architecture that can support
reliable querying) -- no server database, no SQLite dependency.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from femtoolkit.exceptions import ValidationError
from femtoolkit.runs.models import RunStatus, SimulationRun


@dataclass
class RunRecord:
    """A lightweight, JSON-serializable summary of one
    :class:`~femtoolkit.runs.models.SimulationRun`.

    Attributes:
        run_id: The originating run's unique identifier.
        project_id: The originating project's ID.
        project_name: The originating project's display name, at the
            time this run was executed (from its configuration
            snapshot).
        scenario_id: The originating scenario's ID, or ``None``.
        status: The run's final :class:`~femtoolkit.runs.models.RunStatus`, as its string value.
        started_at: ISO-8601 UTC timestamp when execution began.
        completed_at: ISO-8601 UTC timestamp when execution ended.
        execution_time_seconds: Wall-clock solve time.
        verification_status: The run's
            :class:`~femtoolkit.verification.status.VerificationStatus`, as its string value.
        validation_status: The run's validation
            :class:`~femtoolkit.verification.status.VerificationStatus`, as its string
            value, or ``None`` if no validation comparison was made.
        key_results: A small set of scalar headline results (e.g.
            ``"maximum_displacement"``), by name.
        error_stage: Which stage failed, or ``None``.
        error_message: A human-readable failure description, or ``None``.
        configuration_snapshot: The run's
            :attr:`~femtoolkit.runs.models.SimulationRun.configuration_snapshot`,
            as plain JSON-serializable data
            (:meth:`~femtoolkit.application.project.Project.to_dict`'s output).
    """

    run_id: str
    project_id: str
    project_name: str
    scenario_id: str | None
    status: str
    started_at: str | None
    completed_at: str | None
    execution_time_seconds: float | None
    verification_status: str
    validation_status: str | None
    key_results: dict[str, float] = field(default_factory=dict)
    error_stage: str | None = None
    error_message: str | None = None
    configuration_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a plain, JSON-serializable representation of this record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> RunRecord:
        """Reconstruct a :class:`RunRecord` from :meth:`to_dict`'s output."""
        return cls(
            run_id=data["run_id"],
            project_id=data["project_id"],
            project_name=data["project_name"],
            scenario_id=data.get("scenario_id"),
            status=data["status"],
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            execution_time_seconds=data.get("execution_time_seconds"),
            verification_status=data.get("verification_status", "not_run"),
            validation_status=data.get("validation_status"),
            key_results=dict(data.get("key_results", {})),
            error_stage=data.get("error_stage"),
            error_message=data.get("error_message"),
            configuration_snapshot=dict(data.get("configuration_snapshot", {})),
        )


_KEY_RESULT_FIELDS = (
    "maximum_displacement",
    "maximum_von_mises_stress",
    "maximum_temperature",
    "maximum_heat_flux",
)


def record_from_run(run: SimulationRun) -> RunRecord:
    """Build a lightweight, persistable :class:`RunRecord` summary from a live run.

    Args:
        run: The run to summarize.

    Returns:
        A :class:`RunRecord` capturing ``run``'s scalar headline results
        and status -- never the full field-by-field simulation result.
    """
    key_results: dict[str, float] = {}
    if run.result is not None and run.result.summary is not None:
        for name in _KEY_RESULT_FIELDS:
            value = getattr(run.result.summary, name)
            if value is not None:
                key_results[name] = value

    return RunRecord(
        run_id=run.run_id,
        project_id=run.project_id,
        project_name=run.configuration_snapshot.name,
        scenario_id=run.scenario_id,
        status=run.status.value,
        started_at=run.started_at,
        completed_at=run.completed_at,
        execution_time_seconds=run.execution_time_seconds,
        verification_status=run.verification_status.value,
        validation_status=(
            run.validation_result.status.value if run.validation_result is not None else None
        ),
        key_results=key_results,
        error_stage=run.error_stage,
        error_message=run.error_message,
        configuration_snapshot=run.configuration_snapshot.to_dict(),
    )


class RunHistory:
    """An ordered collection of :class:`RunRecord` entries, queryable and JSON-persistable.

    Mirrors :class:`~femtoolkit.application.project_service.ProjectService`'s
    ``to_json``/``from_json``/``save``/``load`` pattern exactly, so
    persisting a run history follows the same, already-established
    convention as persisting a project.
    """

    def __init__(self, records: list[RunRecord] | None = None) -> None:
        self._records: list[RunRecord] = list(records) if records is not None else []

    def add(self, record: RunRecord) -> None:
        """Append one record to this history."""
        self._records.append(record)

    def all(self) -> list[RunRecord]:
        """Every record in this history, in insertion order."""
        return list(self._records)

    def by_run_id(self, run_id: str) -> RunRecord | None:
        """The record with the given ``run_id``, or ``None`` if not found."""
        for record in self._records:
            if record.run_id == run_id:
                return record
        return None

    def by_project_id(self, project_id: str) -> list[RunRecord]:
        """Every record belonging to the given project."""
        return [record for record in self._records if record.project_id == project_id]

    def by_scenario_id(self, scenario_id: str) -> list[RunRecord]:
        """Every record belonging to the given scenario."""
        return [record for record in self._records if record.scenario_id == scenario_id]

    def by_status(self, status: RunStatus | str) -> list[RunRecord]:
        """Every record with the given status."""
        value = status.value if isinstance(status, RunStatus) else status
        return [record for record in self._records if record.status == value]

    def to_json(self) -> str:
        """Serialize this history to a JSON string."""
        return json.dumps([record.to_dict() for record in self._records], indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> RunHistory:
        """Reconstruct a :class:`RunHistory` from a JSON string.

        Raises:
            ValidationError: If ``text`` is not valid JSON.
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Run history file is not valid JSON: {exc}") from exc
        return cls([RunRecord.from_dict(item) for item in data])

    def save(self, path: str | Path) -> Path:
        """Save this history to a JSON file.

        Args:
            path: Destination file path.

        Returns:
            The resolved path written to.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> RunHistory:
        """Load a run history from a JSON file.

        Raises:
            ValidationError: If the file does not exist or is not valid
                run-history JSON.
        """
        source = Path(path)
        if not source.exists():
            raise ValidationError(f"Run history file not found: {source}")
        return cls.from_json(source.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self._records)
