"""Simulation Studies page (Version 30 spec section 22).

Surfaces the Version 30 parameter-study framework
(:mod:`femtoolkit.studies`, :mod:`femtoolkit.runs`) inside the existing
GUI: building a parameter sweep over the current project, reviewing the
scenario count *before* executing (spec section 18 -- a study must never
launch a surprise number of solves), running the study sequentially
through the existing solver pipeline, viewing run history, comparing
results, checking sensitivity, and generating a study report -- with no
FEA algorithm, scenario-generation, comparison, or reporting logic
implemented in this module itself.
"""

from __future__ import annotations

import streamlit as st

from femtoolkit.application.exceptions_display import describe_error
from femtoolkit.exceptions import FiniteElementToolkitError
from femtoolkit.gui.components import require_project
from femtoolkit.gui.state import AppState
from femtoolkit.runs.history import RunHistory, record_from_run
from femtoolkit.studies.extractors import EXTRACTORS, get_extractor
from femtoolkit.studies.parameter_sweep import DEFAULT_MAX_SCENARIOS, ParameterDefinition
from femtoolkit.studies.report import build_study_report, render_study_report_markdown
from femtoolkit.studies.results import StudyResult
from femtoolkit.studies.runner import SimulationStudy, StudyRunner

_PARAMETERS_KEY = "femtoolkit_study_parameters"
_STUDY_RESULT_KEY = "femtoolkit_study_result"
_RUN_HISTORY_KEY = "femtoolkit_run_history"


def _parameters(state_store) -> list[ParameterDefinition]:
    return state_store.setdefault(_PARAMETERS_KEY, [])


def _run_history(state_store) -> RunHistory:
    if _RUN_HISTORY_KEY not in state_store:
        state_store[_RUN_HISTORY_KEY] = RunHistory()
    return state_store[_RUN_HISTORY_KEY]


def render(state: AppState) -> None:
    """Render the Simulation Studies page."""
    st.header("Simulation Studies")
    st.caption(
        "Sweep -> Generate Scenarios -> Review Count -> Execute -> Run History -> "
        "Compare -> Sensitivity -> Report"
    )

    if not require_project(state):
        return

    _render_parameter_sweep_builder(state)
    st.divider()
    _render_execution(state)
    st.divider()
    _render_run_history(state)
    st.divider()
    _render_comparison_and_sensitivity(state)
    st.divider()
    _render_report_generation(state)


def _render_parameter_sweep_builder(state: AppState) -> None:
    st.subheader("1. Parameter Sweep")
    st.caption(
        "Each parameter is a dotted override path on the current project (e.g. "
        "'loads.0.magnitude', 'material.youngs_modulus') and a comma-separated list "
        "of values to sweep it over. Several parameters combine as a Cartesian "
        "product -- the scenario count grows fast."
    )
    parameters = _parameters(st.session_state)

    with st.form("add_parameter_form", clear_on_submit=True):
        col_path, col_label, col_values = st.columns(3)
        path = col_path.text_input("Override path", placeholder="loads.0.magnitude")
        label = col_label.text_input("Label", placeholder="Tip load (N)")
        values_text = col_values.text_input(
            "Values (comma-separated)", placeholder="-1000, -2000, -3000"
        )
        submitted = st.form_submit_button("Add Parameter")

    if submitted:
        try:
            values = _parse_values(values_text)
            parameters.append(ParameterDefinition(path=path, label=label or path, values=values))
            st.success(f"Added parameter '{label or path}' with {len(values)} value(s).")
        except FiniteElementToolkitError as exc:
            st.error(describe_error(exc))

    if not parameters:
        st.info("No parameters added yet -- add at least one to build a sweep.")
        return

    for index, parameter in enumerate(parameters):
        col_info, col_remove = st.columns([5, 1])
        col_info.write(f"**{parameter.label}** (`{parameter.path}`): {parameter.values}")
        if col_remove.button("Remove", key=f"remove_parameter_{index}"):
            parameters.pop(index)
            st.rerun()

    total = 1
    for parameter in parameters:
        total *= len(parameter.values)
    st.metric("Scenarios this sweep would generate", total)
    if total > DEFAULT_MAX_SCENARIOS:
        st.warning(
            f"{total} scenarios exceeds the default limit of {DEFAULT_MAX_SCENARIOS}. "
            "Raise 'Max scenarios' below deliberately, or remove a parameter/value."
        )


def _parse_values(values_text: str) -> list[float]:
    raw_values = [item.strip() for item in values_text.split(",") if item.strip()]
    if not raw_values:
        raise FiniteElementToolkitError("Enter at least one value.")
    parsed: list[float] = []
    for raw_value in raw_values:
        try:
            parsed.append(float(raw_value))
        except ValueError as exc:
            raise FiniteElementToolkitError(f"'{raw_value}' is not a number.") from exc
    return parsed


def _render_execution(state: AppState) -> None:
    st.subheader("2. Execute Study")
    parameters = _parameters(st.session_state)
    max_scenarios = st.number_input(
        "Max scenarios", min_value=1, value=DEFAULT_MAX_SCENARIOS, step=1
    )
    study_name = st.text_input("Study name", value=f"{state.project.name} Study")

    if st.button("Run Study", type="primary", disabled=not parameters):
        study = SimulationStudy(
            study_id=study_name,
            name=study_name,
            base_project=state.project,
            parameters=list(parameters),
            max_scenarios=int(max_scenarios),
        )
        try:
            with st.spinner(f"Running {study_name}..."):
                result = StudyRunner().run(study)
        except FiniteElementToolkitError as exc:
            st.error(describe_error(exc))
            return

        st.session_state[_STUDY_RESULT_KEY] = result
        history = _run_history(st.session_state)
        for run in result.runs:
            history.add(record_from_run(run))

        st.success(
            f"{len(result.successful_runs)}/{len(result.runs)} runs succeeded."
        )
        if result.failed_runs:
            st.warning(f"{len(result.failed_runs)} run(s) failed -- see Run History below.")


def _render_run_history(state: AppState) -> None:
    st.subheader("3. Run History")
    history = _run_history(st.session_state)

    col_save, col_load = st.columns(2)
    with col_save:
        if len(history) > 0:
            st.download_button(
                "Download run history JSON",
                data=history.to_json(),
                file_name="run_history.json",
                mime="application/json",
            )
        else:
            st.caption("No runs recorded yet.")
    with col_load:
        uploaded = st.file_uploader("Load run history JSON", type="json")
        if uploaded is not None:
            try:
                text = uploaded.getvalue().decode("utf-8")
                st.session_state[_RUN_HISTORY_KEY] = RunHistory.from_json(text)
                st.success(f"Loaded {len(st.session_state[_RUN_HISTORY_KEY])} run record(s).")
            except FiniteElementToolkitError as exc:
                st.error(describe_error(exc))

    history = _run_history(st.session_state)
    if len(history) == 0:
        return
    st.dataframe(
        [
            {
                "Run ID": record.run_id[:8],
                "Scenario": record.scenario_id or "-",
                "Status": record.status,
                "Verification": record.verification_status,
                "Time (s)": record.execution_time_seconds,
                **record.key_results,
            }
            for record in history.all()
        ],
        width="stretch",
    )


def _current_study_result() -> StudyResult | None:
    return st.session_state.get(_STUDY_RESULT_KEY)


def _render_comparison_and_sensitivity(state: AppState) -> None:
    st.subheader("4. Result Comparison & Sensitivity")
    result = _current_study_result()
    if result is None:
        st.info("Run a study above to compare and check sensitivity of its results.")
        return
    if len(result.successful_runs) < 2:
        st.info("At least two successful runs are needed to compare results.")
        return

    quantity_name = st.selectbox(
        "Quantity", options=sorted(EXTRACTORS), key="study_quantity_select"
    )
    extractor = get_extractor(quantity_name)

    try:
        comparison = result.compare(extractor, quantity_name)
    except FiniteElementToolkitError as exc:
        st.error(describe_error(exc))
        return

    st.write(f"Baseline run: `{comparison.baseline_run_id[:8]}`")
    st.dataframe(
        [
            {
                "Run 2": entry.run_id_2[:8],
                "Value 1": entry.value1,
                "Value 2": entry.value2,
                "Abs. diff": entry.absolute_difference,
                "% change": entry.percentage_change,
            }
            for entry in comparison.entries
        ],
        width="stretch",
    )

    if result.parameters:
        parameter_labels = {parameter.label: parameter for parameter in result.parameters}
        parameter_label = st.selectbox(
            "Sensitivity parameter", options=list(parameter_labels), key="study_sensitivity_select"
        )
        try:
            sensitivities = result.sensitivity(
                parameter_labels[parameter_label], extractor, quantity_name
            )
        except FiniteElementToolkitError as exc:
            st.error(describe_error(exc))
            return
        st.dataframe(
            [
                {
                    "p1": entry.p1,
                    "p2": entry.p2,
                    "y1": entry.y1,
                    "y2": entry.y2,
                    "S": entry.sensitivity,
                }
                for entry in sensitivities
            ],
            width="stretch",
        )


def _render_report_generation(state: AppState) -> None:
    st.subheader("5. Study Report")
    result = _current_study_result()
    if result is None:
        st.info("Run a study above to generate a report for it.")
        return

    summary = st.text_area(
        "Study summary", value=f"Parameter study of project '{state.project.name}'."
    )
    conclusions = st.text_area("Conclusions (optional, free text -- never generated automatically)")

    report = build_study_report(
        title=f"{result.study_name} Report",
        study_summary=summary,
        base_model_description=f"Project '{state.project.name}', {state.project.analysis_type}.",
        result=result,
        conclusions=conclusions,
    )
    markdown_text = render_study_report_markdown(report)

    st.download_button(
        "Download Markdown Report",
        data=markdown_text,
        file_name="study_report.md",
        mime="text/markdown",
    )
    st.text_area("Preview (Markdown)", value=markdown_text, height=300)
