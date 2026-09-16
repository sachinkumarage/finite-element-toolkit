"""Tests for femtoolkit.gui.state.AppState (a plain dict stands in for st.session_state)."""

from femtoolkit.application.project import Project
from femtoolkit.application.simulation_service import SimulationRunResult
from femtoolkit.gui.state import AppState, VisualizationSettings


def test_no_project_by_default() -> None:
    state = AppState({})
    assert state.project is None
    assert not state.has_project()


def test_setting_project_updates_store() -> None:
    store = {}
    state = AppState(store)
    project = Project(name="Test")
    state.project = project

    assert state.project is project
    assert state.has_project()


def test_setting_project_clears_previous_run() -> None:
    store = {}
    state = AppState(store)
    state.project = Project(name="First")
    state.last_run = SimulationRunResult(status="completed")

    assert state.last_run is not None

    state.project = Project(name="Second")
    assert state.last_run is None


def test_has_results_false_without_a_run() -> None:
    state = AppState({})
    assert not state.has_results()


def test_has_results_false_for_failed_run() -> None:
    state = AppState({})
    state.last_run = SimulationRunResult(status="failed", errors=["boom"])
    assert not state.has_results()


def test_has_results_true_for_completed_run() -> None:
    state = AppState({})
    state.last_run = SimulationRunResult(status="completed")
    assert state.has_results()


def test_visualization_settings_created_lazily_with_defaults() -> None:
    state = AppState({})
    settings = state.visualization_settings

    assert isinstance(settings, VisualizationSettings)
    assert settings.scalar_field is None
    assert settings.deformed is False
    assert settings.deformation_scale == 1.0
    assert settings.show_edges is True


def test_visualization_settings_persist_across_access() -> None:
    state = AppState({})
    settings = state.visualization_settings
    settings.scalar_field = "von_mises_stress"

    assert state.visualization_settings.scalar_field == "von_mises_stress"


def test_reset_clears_project_run_and_visualization_settings() -> None:
    state = AppState({})
    state.project = Project(name="To Be Reset")
    state.last_run = SimulationRunResult(status="completed")
    state.visualization_settings.scalar_field = "temperature"

    state.reset()

    assert state.project is None
    assert state.last_run is None
    assert state.visualization_settings.scalar_field is None
