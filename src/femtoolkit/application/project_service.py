"""Project lifecycle management: create, save, load, reset (Version 24).

Section 5 of this version's brief asks for a project/workspace concept
with create/name/save/load/reset operations, serialized as JSON. This
service is the only place that touches the filesystem for a project --
:mod:`femtoolkit.application.project`'s dataclasses stay pure data, and
:mod:`femtoolkit.gui` never calls ``json``/``open`` directly.
"""

from __future__ import annotations

import json
from pathlib import Path

from femtoolkit.application.project import Project
from femtoolkit.exceptions import ValidationError


class ProjectService:
    """Creates, serializes, and restores
    :class:`~femtoolkit.application.project.Project` instances.
    """

    def create_project(self, name: str, analysis_type: str = "linear_static") -> Project:
        """Create a new project with default configuration.

        Args:
            name: The project's display name.
            analysis_type: The initial analysis type key.

        Returns:
            A fresh :class:`~femtoolkit.application.project.Project`.

        Raises:
            ValidationError: If ``name`` is blank.
        """
        if not name or not name.strip():
            raise ValidationError("Project name must not be blank.")
        return Project(name=name.strip(), analysis_type=analysis_type)

    def reset(self) -> Project:
        """Return a brand-new default project, discarding any prior configuration."""
        return Project()

    def to_json(self, project: Project) -> str:
        """Serialize a project to a JSON string."""
        return json.dumps(project.to_dict(), indent=2, sort_keys=True)

    def from_json(self, text: str) -> Project:
        """Reconstruct a project from a JSON string.

        Raises:
            ValidationError: If ``text`` is not valid JSON.
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Project file is not valid JSON: {exc}") from exc
        return Project.from_dict(data)

    def save(self, project: Project, path: str | Path) -> Path:
        """Save a project to a JSON file.

        Args:
            project: The project to save.
            path: Destination file path.

        Returns:
            The resolved path written to.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(project), encoding="utf-8")
        return destination

    def load(self, path: str | Path) -> Project:
        """Load a project from a JSON file.

        Args:
            path: The project file to read.

        Returns:
            The loaded :class:`~femtoolkit.application.project.Project`.

        Raises:
            ValidationError: If the file does not exist or is not valid
                project JSON.
        """
        source = Path(path)
        if not source.exists():
            raise ValidationError(f"Project file not found: {source}")
        return self.from_json(source.read_text(encoding="utf-8"))
