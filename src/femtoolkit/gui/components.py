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
from femtoolkit.orchestration.config import OrchestrationConfig

if TYPE_CHECKING:
    from femtoolkit.gui.state import AppState
    from femtoolkit.orchestration.summary import ExecutionSummary


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


def render_execution_mode_controls(key_prefix: str) -> OrchestrationConfig | None:
    """Render the shared Version 34 Execution Mode controls.

    Used identically by the Simulation Studies, Uncertainty Analysis,
    and Optimization pages so parallel execution is configured the same
    way everywhere in the GUI, rather than three near-duplicate widget
    sets.

    **A known limitation of this integration.** Streamlit reruns the
    whole page script synchronously on every interaction and has no
    built-in background-task model; a button's callback blocks the UI
    until it returns. This means a parallel run here still blocks the
    page like a serial one does -- there is no live progress bar or
    mid-run cancel button while the batch is executing, only the
    :class:`~femtoolkit.orchestration.summary.ExecutionSummary`
    displayed once it finishes (see :func:`render_execution_summary`).
    Genuine non-blocking progress/cancellation would need a background
    thread or process reporting back into Streamlit's session state
    across reruns, which this version does not add.

    Args:
        key_prefix: A unique prefix for this page's Streamlit widget
            keys, so the Simulation Studies, Uncertainty Analysis, and
            Optimization pages' controls never collide.

    Returns:
        ``None`` if "Serial" is selected (the default -- identical to
        every prior version's behavior); otherwise a validated
        :class:`~femtoolkit.orchestration.config.OrchestrationConfig`
        with ``execution_mode="parallel"``.
    """
    st.caption(
        "**Execution Mode** (Version 34): run independent simulations across local "
        "worker processes instead of one at a time. Parallel execution still blocks "
        "this page until the whole batch finishes -- see this control's tooltip."
    )
    mode = st.radio(
        "Execution Mode", options=["Serial", "Parallel"], horizontal=True,
        key=f"{key_prefix}_execution_mode",
        help=(
            "Serial (default): run every simulation one at a time, in this process -- "
            "identical to every prior version. Parallel: spread simulations across "
            "local worker processes. The page still waits for the whole batch to "
            "finish either way; Streamlit has no background-task model for a live "
            "progress bar or a mid-run cancel button in this version."
        ),
    )
    if mode == "Serial":
        return None

    columns = st.columns(3)
    with columns[0]:
        workers = st.number_input(
            "Workers", min_value=1, max_value=16, value=4, step=1,
            key=f"{key_prefix}_workers",
        )
    with columns[1]:
        fail_fast = st.checkbox(
            "Fail fast", value=False, key=f"{key_prefix}_fail_fast",
            help="Stop submitting new tasks after the first failure.",
        )
    with columns[2]:
        preserve_order = st.checkbox(
            "Preserve order", value=True, key=f"{key_prefix}_preserve_order",
            help="Keep results in submission order rather than completion order.",
        )
    return OrchestrationConfig(
        execution_mode="parallel", max_workers=int(workers), fail_fast=fail_fast,
        preserve_order=preserve_order,
    )


def render_execution_summary(summary: ExecutionSummary) -> None:
    """Render a Version 34 :class:`~femtoolkit.orchestration.summary.ExecutionSummary`.

    Args:
        summary: The completed batch's execution summary.
    """
    st.caption(f"Execution mode: **{summary.execution_mode}** ({summary.worker_count} worker(s))")
    metrics = [
        ("Total", str(summary.total_tasks)),
        ("Completed", str(summary.completed_tasks)),
        ("Failed", str(summary.failed_tasks)),
        ("Cancelled", str(summary.cancelled_tasks)),
        ("Elapsed (s)", f"{summary.total_elapsed_seconds:.2f}"),
    ]
    if summary.parallel_speedup is not None:
        metrics.append(("Speedup", f"{summary.parallel_speedup:.2f}x"))
    if summary.parallel_efficiency is not None:
        metrics.append(("Efficiency", f"{summary.parallel_efficiency:.0%}"))
    metric_row(metrics)
    for note in summary.notes:
        st.info(note)
