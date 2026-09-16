"""Application state management (Version 24, spec section 16).

Streamlit reruns the whole script on every interaction, so any state
that must survive a rerun (the current project, simulation status,
results, visualization settings) has to live in ``st.session_state``
rather than a plain module-level variable. :class:`AppState` wraps that
mapping behind a small typed interface -- ``gui/pages`` never touches
``st.session_state`` directly, and this class never imports
:mod:`streamlit` itself, so it is fully testable with a plain ``dict``
standing in for the session store (see ``tests/test_gui_state.py``).
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from femtoolkit.application.project import Project
from femtoolkit.application.simulation_service import SimulationRunResult

_PROJECT_KEY = "femtoolkit_project"
_LAST_RUN_KEY = "femtoolkit_last_run"
_VIZ_SETTINGS_KEY = "femtoolkit_viz_settings"


@dataclass
class VisualizationSettings:
    """User-selected visualization controls (spec section 14), independent of any backend.

    Attributes:
        scalar_field: The selected scalar field name, or ``None``.
        vector_field: The selected vector field name, or ``None``.
        deformed: Whether to show the deformed geometry.
        deformation_scale: The cosmetic deformation scale factor.
        show_edges: Whether mesh edges are drawn.
    """

    scalar_field: str | None = None
    vector_field: str | None = None
    deformed: bool = False
    deformation_scale: float = 1.0
    show_edges: bool = True


class AppState:
    """A typed view over the Streamlit session store.

    Args:
        store: The underlying mutable mapping (``st.session_state`` in
            the running app; a plain ``dict`` in tests).
    """

    def __init__(self, store: MutableMapping[str, Any]) -> None:
        self._store = store

    @property
    def project(self) -> Project | None:
        """The currently active project, or ``None`` before one is created."""
        return self._store.get(_PROJECT_KEY)

    @project.setter
    def project(self, project: Project | None) -> None:
        self._store[_PROJECT_KEY] = project
        # A new/changed project invalidates any previously computed results.
        self._store.pop(_LAST_RUN_KEY, None)

    @property
    def last_run(self) -> SimulationRunResult | None:
        """The most recent
        :class:`~femtoolkit.application.simulation_service.SimulationRunResult`.
        """
        return self._store.get(_LAST_RUN_KEY)

    @last_run.setter
    def last_run(self, run_result: SimulationRunResult | None) -> None:
        self._store[_LAST_RUN_KEY] = run_result

    @property
    def visualization_settings(self) -> VisualizationSettings:
        """The current visualization control settings (created on first access)."""
        if _VIZ_SETTINGS_KEY not in self._store:
            self._store[_VIZ_SETTINGS_KEY] = VisualizationSettings()
        return self._store[_VIZ_SETTINGS_KEY]

    @visualization_settings.setter
    def visualization_settings(self, settings: VisualizationSettings) -> None:
        self._store[_VIZ_SETTINGS_KEY] = settings

    def reset(self) -> None:
        """Clear the project, results, and visualization settings."""
        self._store.pop(_PROJECT_KEY, None)
        self._store.pop(_LAST_RUN_KEY, None)
        self._store.pop(_VIZ_SETTINGS_KEY, None)

    def has_project(self) -> bool:
        """Whether a project currently exists."""
        return self.project is not None

    def has_results(self) -> bool:
        """Whether the last run completed with a usable simulation result."""
        run_result = self.last_run
        return run_result is not None and run_result.succeeded
