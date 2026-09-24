"""Scenarios: named, overridden variations of a base project (Version 30).

**Engineering concept.** A *simulation scenario* is one specific "what
if" variation of a base model -- "what if the load were 3 kN instead of
1 kN," "what if this were aluminum instead of steel." A scenario is
defined entirely by *what it changes*: the base project itself is never
modified, and any parameter a scenario does not mention keeps the base
project's value. This is what makes a set of scenarios comparable to
each other and back to the base case -- every difference in the results
can be attributed to a difference the scenario explicitly declared.

:func:`apply_scenario` is the single place this override is performed:
it deep-copies the base :class:`~femtoolkit.application.project.Project`
and writes each override onto the copy by a dotted attribute path (e.g.
``"material.youngs_modulus"``, ``"loads.0.magnitude"``), reusing the
existing project dataclasses directly rather than introducing a second,
parallel configuration model.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from femtoolkit.application.project import Project
from femtoolkit.exceptions import DuplicateScenarioIdError, ValidationError


@dataclass
class Scenario:
    """One named, overridden variation of a base project.

    Attributes:
        scenario_id: A unique identifier for this scenario within a
            study (see :func:`validate_unique_scenario_ids`).
        name: A short, human-readable name (e.g. ``"3 kN tip load"``).
        description: A longer, free-text description of what this
            scenario represents and why it was defined.
        parameter_overrides: Dotted attribute paths (relative to the
            base :class:`~femtoolkit.application.project.Project`) to
            override values, e.g. ``{"material.youngs_modulus": 70e9,
            "loads.0.magnitude": -3000.0}``. Only the named parameters
            differ from the base project.
        solver_overrides: Dotted attribute paths relative to the base
            project's ``solver`` configuration, e.g. ``{"tolerance":
            1e-8}``.
        metadata: Free-form additional information about this scenario
            (never interpreted by this toolkit).
        tags: Short labels for filtering/grouping scenarios (e.g.
            ``["material-study", "steel"]``).
    """

    scenario_id: str
    name: str
    description: str = ""
    parameter_overrides: dict[str, Any] = field(default_factory=dict)
    solver_overrides: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


def validate_unique_scenario_ids(scenarios: list[Scenario]) -> None:
    """Check that every scenario in ``scenarios`` has a unique ``scenario_id``.

    Args:
        scenarios: The scenarios to check.

    Raises:
        DuplicateScenarioIdError: If any ``scenario_id`` appears more
            than once.
    """
    seen: set[str] = set()
    for scenario in scenarios:
        if scenario.scenario_id in seen:
            raise DuplicateScenarioIdError(
                f"Duplicate scenario_id {scenario.scenario_id!r}; every scenario in a "
                "study must have a unique ID."
            )
        seen.add(scenario.scenario_id)


def get_by_path(root: Any, path: str) -> Any:
    """Read the value at a dotted attribute path (the read-only counterpart of an override).

    Args:
        root: The object to read from (typically a
            :class:`~femtoolkit.application.project.Project`).
        path: A dotted attribute path, in the same format
            :func:`apply_scenario` accepts as an override key.

    Returns:
        The value found at ``path``.

    Raises:
        ValidationError: If any segment of ``path`` does not exist.
    """
    target = root
    for segment in path.split("."):
        target = _resolve_segment(target, segment, path)
    return target


def apply_scenario(base_project: Project, scenario: Scenario) -> Project:
    """Return a new project with ``scenario``'s overrides applied on top of ``base_project``.

    ``base_project`` is never mutated -- a deep copy is made first, and
    every override is written onto the copy.

    Args:
        base_project: The unmodified base project configuration.
        scenario: The scenario whose ``parameter_overrides``/
            ``solver_overrides`` to apply.

    Returns:
        A new :class:`~femtoolkit.application.project.Project`,
        independent of ``base_project``, with every override applied.

    Raises:
        ValidationError: If an override path does not correspond to a
            real field on the project (or a sub-config/list element
            along the path).
    """
    project = copy.deepcopy(base_project)
    for path, value in scenario.parameter_overrides.items():
        _set_by_path(project, path, value)
    for path, value in scenario.solver_overrides.items():
        _set_by_path(project.solver, path, value)
    return project


def _resolve_segment(obj: Any, segment: str, path: str) -> Any:
    if isinstance(obj, list):
        if not segment.lstrip("-").isdigit():
            raise ValidationError(
                f"Invalid parameter override path {path!r}: {segment!r} is not a valid "
                "list index."
            )
        index = int(segment)
        if not (-len(obj) <= index < len(obj)):
            raise ValidationError(
                f"Invalid parameter override path {path!r}: index {index} is out of range "
                f"for a list of length {len(obj)}."
            )
        return obj[index]
    if not hasattr(obj, segment):
        raise ValidationError(
            f"Invalid parameter override path {path!r}: {type(obj).__name__!r} has no "
            f"field {segment!r}."
        )
    return getattr(obj, segment)


def _set_by_path(root: Any, path: str, value: Any) -> None:
    segments = path.split(".")
    target = root
    for segment in segments[:-1]:
        target = _resolve_segment(target, segment, path)

    last = segments[-1]
    if isinstance(target, list):
        if not last.lstrip("-").isdigit():
            raise ValidationError(
                f"Invalid parameter override path {path!r}: {last!r} is not a valid list index."
            )
        index = int(last)
        if not (-len(target) <= index < len(target)):
            raise ValidationError(
                f"Invalid parameter override path {path!r}: index {index} is out of range "
                f"for a list of length {len(target)}."
            )
        target[index] = value
        return
    if not hasattr(target, last):
        raise ValidationError(
            f"Invalid parameter override path {path!r}: {type(target).__name__!r} has no "
            f"field {last!r}."
        )
    setattr(target, last, value)
