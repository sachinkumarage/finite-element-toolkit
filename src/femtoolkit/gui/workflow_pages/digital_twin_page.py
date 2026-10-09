"""Digital Twin & Model Updating page (Version 38).

Surfaces the Version 38 digital twin framework (:mod:`femtoolkit.digital_twin`)
inside the existing GUI, against the current project: defining calibratable
model parameters, entering or generating measured reference data, running the
baseline simulation, calibrating parameters against the measurements (reusing
the unmodified Version 32/33 optimization machinery), and comparing
measured-vs-baseline-vs-updated results -- with no measurement-handling,
calibration-search, or accuracy-metric logic implemented in this module
itself.

**Scope note.** This page operates on whatever project is currently loaded
(the same convention the Adaptive Optimization and Multi-Fidelity pages use)
rather than a separate hardcoded "example" selector. A "Generate Synthetic
Measurement" action is provided purely for demonstration -- every value it
produces is labeled **SYNTHETIC**, never presented as real experimental data.
"""

from __future__ import annotations

import streamlit as st

from femtoolkit.application.exceptions_display import describe_error
from femtoolkit.digital_twin.measurements import MeasurementData, MeasurementPoint
from femtoolkit.digital_twin.parameters import ModelParameter
from femtoolkit.digital_twin.problem import ModelUpdateProblem
from femtoolkit.digital_twin.updating import ModelUpdateConfig, run_model_update
from femtoolkit.digital_twin.validation import compare_model_accuracy
from femtoolkit.exceptions import FiniteElementToolkitError
from femtoolkit.gui.components import render_execution_mode_controls, require_project
from femtoolkit.gui.state import AppState
from femtoolkit.optimization import SUPPORTED_ALGORITHMS
from femtoolkit.orchestration.simulation import SimulationTask, evaluate_simulation_batch
from femtoolkit.runs.models import RunStatus
from femtoolkit.studies.extractors import EXTRACTORS, get_extractor
from femtoolkit.studies.scenarios import Scenario, apply_scenario

_PARAMETERS_KEY = "femtoolkit_dt_parameters"
_MEASUREMENT_KEY = "femtoolkit_dt_measurement"
_BASELINE_KEY = "femtoolkit_dt_baseline_predictions"
_RESULT_KEY = "femtoolkit_dt_result"


def _parameters(state_store) -> list[ModelParameter]:
    return state_store.setdefault(_PARAMETERS_KEY, [])


def _measurement_data(state_store, quantity_name: str) -> MeasurementData:
    data = state_store.get(_MEASUREMENT_KEY)
    if data is None or data.quantity_name != quantity_name:
        data = MeasurementData(quantity_name=quantity_name)
        state_store[_MEASUREMENT_KEY] = data
    return data


def _run_single_point(project, overrides: dict[str, float], quantity_name: str) -> float | None:
    """Run one simulation with ``overrides`` applied and extract ``quantity_name``.

    Reuses the unmodified Version 30 ``Scenario``/``apply_scenario`` override
    mechanism and Version 34 ``evaluate_simulation_batch`` -- no new
    simulation-execution logic.
    """
    task = SimulationTask(
        task_id="digital-twin-single-point",
        project=apply_scenario(
            project,
            Scenario(
                scenario_id="digital-twin-single-point", name="Digital twin single point",
                parameter_overrides=overrides,
            ),
        ),
        scenario_id="digital-twin-single-point",
    )
    runs, _summary = evaluate_simulation_batch([task])
    run = runs[0]
    if run.status != RunStatus.COMPLETED:
        return None
    return get_extractor(quantity_name)(run)


def render(state: AppState) -> None:
    """Render the Digital Twin & Model Updating page."""
    st.header("Digital Twin & Model Updating")
    st.caption(
        "Measured Data -> Baseline Simulation -> Compare -> Update Parameters -> "
        "Re-run Simulation -> Validate Improved Agreement. Operates on the current "
        "project, which stands in for the 'engineering example' being twinned."
    )

    if not require_project(state):
        return

    _render_parameters(state)
    st.divider()
    _render_measurement_data(state)
    st.divider()
    _render_baseline(state)
    st.divider()
    _render_model_updating(state)
    st.divider()
    _render_compare_results()


def _render_parameters(state: AppState) -> None:
    st.subheader("1. Baseline Model Parameters")
    st.caption(
        "Each parameter is a dotted override path on the current project (e.g. "
        "'material.youngs_modulus'), its baseline value, and the bounds model "
        "updating may search within. Uncheck 'Updatable' for a parameter that "
        "should be reported but never calibrated."
    )
    parameters = _parameters(st.session_state)

    with st.form("add_dt_parameter_form", clear_on_submit=True):
        col_name, col_path, col_units = st.columns(3)
        name = col_name.text_input("Name", placeholder="youngs_modulus")
        path = col_path.text_input("Override path", placeholder="material.youngs_modulus")
        units = col_units.text_input("Units", placeholder="Pa")

        col_current, col_low, col_high, col_updatable = st.columns(4)
        current_value = col_current.number_input(
            "Current (baseline) value", value=1.0, format="%.6g"
        )
        lower_bound = col_low.number_input("Lower bound", value=0.0, format="%.6g")
        upper_bound = col_high.number_input("Upper bound", value=2.0, format="%.6g")
        updatable = col_updatable.checkbox("Updatable", value=True)
        submitted = st.form_submit_button("Add Parameter")

    if submitted:
        try:
            if not name:
                raise FiniteElementToolkitError("A parameter name is required.")
            parameters.append(
                ModelParameter(
                    name=name, current_value=float(current_value), lower_bound=float(lower_bound),
                    upper_bound=float(upper_bound), units=units, updatable=updatable,
                    path=path or name,
                )
            )
        except FiniteElementToolkitError as exc:
            st.error(describe_error(exc))

    if not parameters:
        st.info("Add at least one model parameter before continuing.")
        return

    for index, parameter in enumerate(parameters):
        col_info, col_remove = st.columns([5, 1])
        bounds = f"[{parameter.lower_bound}, {parameter.upper_bound}]"
        role = "updatable" if parameter.updatable else "fixed"
        col_info.write(
            f"**{parameter.name}** (`{parameter.path}`): {parameter.current_value} "
            f"{parameter.units}, bounds={bounds}, {role}"
        )
        if col_remove.button("Remove", key=f"remove_dt_parameter_{index}"):
            parameters.pop(index)
            st.rerun()


def _render_measurement_data(state: AppState) -> None:
    st.subheader("2. Measurement Data")
    parameters = _parameters(st.session_state)
    if not parameters:
        st.info("Add at least one model parameter above first.")
        return

    quantity_name = st.selectbox("Measured quantity", options=list(EXTRACTORS))
    measurement_data = _measurement_data(st.session_state, quantity_name)

    col_manual, col_synthetic = st.columns(2)

    with col_manual:
        st.markdown("**Enter a measurement manually**")
        with st.form("add_dt_measurement_form", clear_on_submit=True):
            measurement_id = st.text_input("Measurement ID", placeholder="m1")
            measured_value = st.number_input("Measured value", value=0.0, format="%.6g")
            uncertainty_text = st.text_input("Uncertainty (optional)", placeholder="")
            submitted_manual = st.form_submit_button("Add Measurement")
        if submitted_manual:
            try:
                if not measurement_id:
                    raise FiniteElementToolkitError("A measurement ID is required.")
                uncertainty = float(uncertainty_text) if uncertainty_text else None
                measurement_data.add_point(
                    MeasurementPoint(
                        measurement_id=measurement_id, measured_value=float(measured_value),
                        uncertainty=uncertainty,
                    )
                )
            except FiniteElementToolkitError as exc:
                st.error(describe_error(exc))
            except ValueError as exc:
                st.error(str(exc))

    with col_synthetic:
        st.markdown("**Or generate a SYNTHETIC measurement**")
        st.caption(
            "Runs the current project once at the 'true' parameter values below, for "
            "demonstration only -- clearly labeled synthetic, never real experimental data."
        )
        true_values: dict[str, float] = {}
        for parameter in parameters:
            if parameter.updatable:
                true_values[parameter.name] = st.number_input(
                    f"True value for '{parameter.name}'", value=parameter.current_value,
                    format="%.6g", key=f"dt_true_value_{parameter.name}",
                )
        if st.button("Generate Synthetic Measurement"):
            try:
                overrides = {
                    parameter.path: true_values[parameter.name]
                    for parameter in parameters
                    if parameter.updatable
                }
                with st.spinner("Running synthetic 'true' simulation..."):
                    value = _run_single_point(state.project, overrides, quantity_name)
                if value is None:
                    st.error("The synthetic simulation did not complete; no measurement added.")
                else:
                    index = measurement_data.n_points
                    measurement_data.add_point(
                        MeasurementPoint(
                            measurement_id=f"synthetic-{index}", measured_value=value,
                            metadata={"source": "synthetic", "true_values": true_values},
                        )
                    )
                    st.success(f"SYNTHETIC measurement added: {value:.6e}")
            except FiniteElementToolkitError as exc:
                st.error(describe_error(exc))

    if measurement_data.n_points == 0:
        st.info("Add at least one measurement point (manual or synthetic) to continue.")
        return

    st.caption(f"{measurement_data.n_points} measurement point(s) for '{quantity_name}':")
    for index, point in enumerate(measurement_data.points):
        col_info, col_remove = st.columns([5, 1])
        label = " (SYNTHETIC)" if point.metadata.get("source") == "synthetic" else ""
        col_info.write(f"**{point.measurement_id}**{label}: {point.measured_value:.6e}")
        if col_remove.button("Remove", key=f"remove_dt_measurement_{index}"):
            measurement_data.points.pop(index)
            st.rerun()


def _build_problem(state: AppState) -> ModelUpdateProblem | None:
    parameters = _parameters(st.session_state)
    measurement_data = st.session_state.get(_MEASUREMENT_KEY)
    if not parameters or measurement_data is None or measurement_data.n_points == 0:
        return None
    try:
        return ModelUpdateProblem(
            base_project=state.project, parameters=list(parameters),
            measurement_data=measurement_data,
        )
    except FiniteElementToolkitError as exc:
        st.error(describe_error(exc))
        return None


def _render_baseline(state: AppState) -> None:
    st.subheader("3. Baseline Simulation")
    problem = _build_problem(state)
    if problem is None:
        st.info("Define model parameters and measurement data above first.")
        return

    if st.button("Run Baseline Simulation", type="primary"):
        with st.spinner("Running the baseline simulation at every measurement point..."):
            predictions = problem.predict(problem.current_parameter_values())
        st.session_state[_BASELINE_KEY] = predictions

    predictions = st.session_state.get(_BASELINE_KEY)
    if predictions is None:
        st.info("Run the baseline simulation to see its prediction.")
        return

    st.markdown("**BASELINE PREDICTION vs. measured**")
    measured = problem.measurement_data.measured_values()
    st.table(
        [
            {
                "measurement_id": point.measurement_id, "baseline_prediction": predictions[i],
                "measured_value": measured[i], "residual": predictions[i] - measured[i],
            }
            for i, point in enumerate(problem.measurement_data.points)
        ]
    )


def _render_model_updating(state: AppState) -> None:
    st.subheader("4. Model Updating")
    problem = _build_problem(state)
    if problem is None:
        st.info("Define model parameters and measurement data above first.")
        return
    if not problem.updatable_parameters:
        st.warning("No updatable parameter is defined; model updating has nothing to calibrate.")
        return

    col_algorithm, col_max_eval, col_seed = st.columns(3)
    algorithm = col_algorithm.selectbox(
        "Algorithm", options=SUPPORTED_ALGORITHMS, key="dt_algorithm"
    )
    max_evaluations = col_max_eval.number_input(
        "Max evaluations", min_value=1, value=30, step=1, key="dt_max_evaluations"
    )
    seed = col_seed.number_input("Random seed", min_value=0, value=0, step=1, key="dt_seed")

    orchestration_config = render_execution_mode_controls(key_prefix="digital_twin")

    if st.button("Run Model Updating", type="primary"):
        config = ModelUpdateConfig(
            algorithm=algorithm, max_evaluations=int(max_evaluations), seed=int(seed)
        )
        with st.spinner("Calibrating parameters against measured data..."):
            result = run_model_update(problem, config, orchestration_config=orchestration_config)
        st.session_state[_RESULT_KEY] = result

    result = st.session_state.get(_RESULT_KEY)
    if result is None:
        st.info("Run model updating to see calibrated parameters.")
        return

    st.write(f"**Status:** {result.status.value} ({result.stopping_reason})")
    st.write(f"**Evaluations used:** {result.n_evaluations}")

    st.markdown("**UPDATED PARAMETERS** (the updated simulation is already included above)")
    st.table(
        [
            {
                "parameter": name, "initial_value": result.initial_parameters[name],
                "updated_value": result.updated_parameters[name],
            }
            for name in result.initial_parameters
        ]
    )


def _render_compare_results() -> None:
    st.subheader("5. Compare Measured vs. Baseline vs. Updated")
    result = st.session_state.get(_RESULT_KEY)
    if result is None or result.initial_objective is None:
        st.info("Run model updating above to compare results.")
        return

    comparison = compare_model_accuracy(result)
    col_before, col_after = st.columns(2)
    col_before.metric("RMSE before update", f"{comparison.before.rmse:.4e}")
    col_after.metric("RMSE after update", f"{comparison.after.rmse:.4e}")
    st.write(f"**Model updating improved agreement with measurements:** {comparison.improved}")

    st.caption("Measured vs. Simulated")
    st.scatter_chart(
        {
            "Baseline Prediction": list(result.initial_predictions),
            "Updated Prediction": list(result.updated_predictions),
            "Measured Value": list(result.measured_values),
        }
    )

    st.caption("Residual Before vs. After")
    st.bar_chart(
        {
            "Residual Before": list(result.residuals_before),
            "Residual After": list(result.residuals_after),
        }
    )

    st.caption("Parameter Before vs. After")
    st.bar_chart(
        {
            "Before": list(result.initial_parameters.values()),
            "After": list(result.updated_parameters.values()),
        }
    )

    st.caption("Calibration Objective J(theta): Before vs. After")
    st.bar_chart({"Objective": [result.initial_objective, result.final_objective]})

    if result.convergence_history:
        st.caption("Model-Update Convergence (best objective value found so far)")
        st.line_chart([value for value in result.convergence_history if value is not None])


__all__ = ["render"]
