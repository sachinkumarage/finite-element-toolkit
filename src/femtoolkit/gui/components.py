"""Small reusable Streamlit rendering helpers shared across pages (Version 24).

Kept to a handful of plain functions -- not a widget class hierarchy --
since the GUI's actual complexity lives in the application/service
layer, not in its display code (spec section 23: "do not create
unnecessary abstractions simply to increase the number of classes").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from femtoolkit.application.simulation_service import SimulationRunResult

if TYPE_CHECKING:
    from femtoolkit.gui.state import AppState


def error_banner(title: str, messages: list[str]) -> None:
    """Render a list of problems as a single Streamlit error banner.

    Args:
        title: A one-line summary of the problem category.
        messages: Individual problem descriptions.
    """
    if not messages:
        return
    body = "\n".join(f"- {message}" for message in messages)
    st.error(f"**{title}**\n\n{body}")


def run_status_banner(run_result: SimulationRunResult | None) -> None:
    """Render the current simulation run status.

    Args:
        run_result: The last simulation run, or ``None`` if nothing has
            been run yet.
    """
    if run_result is None:
        st.info("No simulation has been run yet.")
        return
    if run_result.status == "completed":
        st.success("Simulation completed.")
    elif run_result.status == "invalid":
        error_banner("Project configuration is invalid", run_result.errors)
    else:
        error_banner("Simulation failed", run_result.errors)


def metric_row(metrics: list[tuple[str, str]]) -> None:
    """Render a row of Streamlit metrics.

    Args:
        metrics: ``(label, formatted_value)`` pairs.
    """
    columns = st.columns(len(metrics)) if metrics else []
    for column, (label, value) in zip(columns, metrics, strict=True):
        with column:
            st.metric(label, value)


def require_project(state: AppState) -> bool:
    """Warn and return ``False`` if no project exists yet; otherwise return ``True``."""
    if not state.has_project():
        st.warning("Create or load a project first, on the Project page.")
        return False
    return True
