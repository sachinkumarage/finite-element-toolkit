"""Surrogate / ROM Workspace page (Version 35).

Surfaces the Version 35 reduced-order-modeling and surrogate-analysis
framework (:mod:`femtoolkit.surrogate`) inside the existing GUI:
building a training dataset from a parameter sweep (reusing the
Version 30 sweep machinery and, optionally, the Version 34 parallel
execution layer), training a transparent surrogate model, reviewing its
held-out validation metrics and engineering-tolerance checks, fitting a
POD reduced-order model, checking a query point's applicability-domain
status, verifying a candidate against fresh high-fidelity FEA, and
generating a downloadable report -- with no surrogate, ROM, scaling, or
validation logic implemented in this module itself.
"""

from __future__ import annotations

import streamlit as st

from femtoolkit.application.exceptions_display import describe_error
from femtoolkit.exceptions import FiniteElementToolkitError
from femtoolkit.gui.components import render_execution_mode_controls, require_project
from femtoolkit.gui.state import AppState
from femtoolkit.studies.extractors import EXTRACTORS, get_extractor
from femtoolkit.studies.parameter_sweep import ParameterDefinition
from femtoolkit.surrogate.domain import DomainStatus
from femtoolkit.surrogate.models import SURROGATE_MODEL_TYPES
from femtoolkit.surrogate.report import SurrogateReport, render_surrogate_report_markdown
from femtoolkit.surrogate.rom.pod import PODModel
from femtoolkit.surrogate.scaling import SCALER_TYPES
from femtoolkit.surrogate.workflows.training import (
    TrainingConfig,
    generate_training_dataset,
    train_surrogate,
)
from femtoolkit.surrogate.workflows.verification import verify_against_high_fidelity

_PARAMETERS_KEY = "femtoolkit_surrogate_parameters"
_DATASET_KEY = "femtoolkit_surrogate_dataset"
_MODEL_KEY = "femtoolkit_surrogate_model"
_VALIDATION_KEY = "femtoolkit_surrogate_validation"
_ROM_KEY = "femtoolkit_surrogate_rom"
_VERIFICATION_KEY = "femtoolkit_surrogate_verification"


def _parameters(state_store) -> list[ParameterDefinition]:
    return state_store.setdefault(_PARAMETERS_KEY, [])


def render(state: AppState) -> None:
    """Render the Surrogate / ROM Workspace page."""
    st.header("Surrogate / ROM Workspace")
    st.caption(
        "Parameter Space -> Training Dataset -> Train Surrogate -> Validate -> "
        "Applicability Domain -> High-Fidelity Verification -> Report"
    )

    if not require_project(state):
        return

    _render_dataset_builder(state)
    st.divider()
    _render_surrogate_training(state)
    st.divider()
    _render_rom(state)
    st.divider()
    _render_validation()
    st.divider()
    _render_domain_and_prediction()
    st.divider()
    _render_verification(state)
    st.divider()
    _render_report()


def _render_dataset_builder(state: AppState) -> None:
    st.subheader("1. Dataset")
    st.caption(
        "Each parameter is a dotted override path (e.g. 'mesh.thickness') and a "
        "comma-separated list of values. The Cartesian product becomes the training "
        "design points, evaluated by the existing high-fidelity simulation pipeline."
    )
    parameters = _parameters(st.session_state)

    with st.form("add_surrogate_parameter_form", clear_on_submit=True):
        col_path, col_label, col_values = st.columns(3)
        path = col_path.text_input("Override path", placeholder="mesh.thickness")
        label = col_label.text_input("Label", placeholder="Thickness (m)")
        values_text = col_values.text_input(
            "Values (comma-separated)", placeholder="0.006, 0.008, 0.010, 0.012"
        )
        submitted = st.form_submit_button("Add Parameter")

    if submitted:
        try:
            values = [float(v.strip()) for v in values_text.split(",") if v.strip()]
            if not values:
                raise FiniteElementToolkitError("Enter at least one value.")
            parameters.append(ParameterDefinition(path=path, label=label or path, values=values))
        except (ValueError, FiniteElementToolkitError) as exc:
            if isinstance(exc, FiniteElementToolkitError):
                st.error(describe_error(exc))
            else:
                st.error(str(exc))

    for index, parameter in enumerate(parameters):
        col_info, col_remove = st.columns([5, 1])
        col_info.write(f"**{parameter.label}** (`{parameter.path}`): {parameter.values}")
        if col_remove.button("Remove", key=f"remove_surrogate_parameter_{index}"):
            parameters.pop(index)
            st.rerun()

    response_names = st.multiselect(
        "Response quantities", options=list(EXTRACTORS), default=["maximum_displacement"]
    )
    orchestration_config = render_execution_mode_controls(key_prefix="surrogate")

    can_generate = bool(parameters) and bool(response_names)
    if st.button("Generate Training Dataset", type="primary", disabled=not can_generate):
        try:
            with st.spinner("Running high-fidelity simulations..."):
                dataset = generate_training_dataset(
                    state.project,
                    list(parameters),
                    {name: get_extractor(name) for name in response_names},
                    orchestration_config=orchestration_config,
                )
            st.session_state[_DATASET_KEY] = dataset
            st.success(f"Collected {dataset.n_samples} snapshot(s).")
        except FiniteElementToolkitError as exc:
            st.error(describe_error(exc))

    dataset = st.session_state.get(_DATASET_KEY)
    if dataset is not None:
        st.metric("Snapshots", dataset.n_samples)
        st.write(f"**Dataset ID:** `{dataset.dataset_id}` (version {dataset.dataset_version})")
        st.write(f"**Input variables:** {', '.join(dataset.feature_names)}")
        st.write(f"**Response variables:** {', '.join(dataset.response_names)}")


def _render_surrogate_training(state: AppState) -> None:
    st.subheader("2. Surrogate")
    dataset = st.session_state.get(_DATASET_KEY)
    if dataset is None:
        st.info("Generate a training dataset above first.")
        return

    col_model, col_scaler_x, col_scaler_y = st.columns(3)
    model_type = col_model.selectbox("Model type", options=list(SURROGATE_MODEL_TYPES))
    feature_scaler = col_scaler_x.selectbox("Feature scaler", options=list(SCALER_TYPES))
    response_scaler = col_scaler_y.selectbox("Response scaler", options=list(SCALER_TYPES))

    model_kwargs: dict[str, object] = {}
    if model_type == "polynomial":
        model_kwargs["degree"] = st.selectbox("Polynomial degree", options=[1, 2])
    elif model_type == "rbf":
        model_kwargs["kernel"] = st.selectbox("RBF kernel", options=["gaussian", "multiquadric"])

    col_train, col_val, col_seed = st.columns(3)
    train_fraction = col_train.slider("Train fraction", 0.4, 0.9, 0.7, 0.05)
    validation_fraction = col_val.slider("Validation fraction", 0.0, 0.4, 0.15, 0.05)
    split_seed = col_seed.number_input("Split seed", min_value=0, value=0, step=1)
    cross_validate = st.checkbox("Also run k-fold cross-validation", value=False)

    if st.button("Train Surrogate", type="primary"):
        try:
            config = TrainingConfig(
                model_type=model_type,
                model_kwargs=model_kwargs,
                feature_scaler_type=feature_scaler,
                response_scaler_type=response_scaler,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
                split_seed=int(split_seed),
                cross_validate=cross_validate,
            )
            model, report = train_surrogate(dataset, config)
            st.session_state[_MODEL_KEY] = model
            st.session_state[_VALIDATION_KEY] = report
            st.success(f"Trained a '{model.name}' surrogate on {dataset.n_samples} snapshot(s).")
        except FiniteElementToolkitError as exc:
            st.error(describe_error(exc))


def _render_rom(_state: AppState) -> None:
    st.subheader("3. Reduced-Order Model (POD)")
    dataset = st.session_state.get(_DATASET_KEY)
    if dataset is None or dataset.n_responses < 2:
        st.info(
            "POD needs a dataset with at least two response quantities, treated here as a "
            "small demonstration field (one column per snapshot) -- see "
            "'examples/surrogate/pod_cantilever_displacement_field.py' for POD applied to a "
            "real full nodal-displacement field."
        )
        return

    col_mode, col_value = st.columns(2)
    selection_mode = col_mode.radio("Basis selection", options=["Fixed rank", "Energy threshold"])
    if selection_mode == "Fixed rank":
        rank = col_value.number_input(
            "Rank", min_value=1, max_value=dataset.n_responses, value=1, step=1
        )
        energy_threshold = None
    else:
        energy_threshold = col_value.slider("Energy threshold", 0.5, 1.0, 0.999, 0.001)
        rank = None

    if st.button("Fit POD Basis"):
        try:
            _, y = dataset.to_arrays()
            snapshot_matrix = y.T
            selected_rank = int(rank) if rank is not None else None
            pod = PODModel().fit(
                snapshot_matrix, rank=selected_rank, energy_threshold=energy_threshold
            )
            st.session_state[_ROM_KEY] = pod
        except FiniteElementToolkitError as exc:
            st.error(describe_error(exc))

    pod = st.session_state.get(_ROM_KEY)
    if pod is not None:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total modes", pod.total_modes)
        col_b.metric("Selected modes", pod.selected_modes)
        col_c.metric("Captured energy", f"{pod.captured_energy:.4%}")
        st.line_chart(pod.energy_spectrum())


def _render_validation() -> None:
    st.subheader("4. Validation")
    report = st.session_state.get(_VALIDATION_KEY)
    if report is None:
        st.info("Train a surrogate above first.")
        return

    rows = []
    for name, metrics in report.test_metrics.items():
        rows.append(
            {
                "Response": name,
                "N": metrics.n_samples,
                "MAE": metrics.mae,
                "RMSE": metrics.rmse,
                "R^2": metrics.r2,
                "Mean Rel. Error": metrics.mean_relative_error,
                "Max Rel. Error": metrics.max_relative_error,
            }
        )
    st.table(rows)

    for warning in report.warnings:
        st.warning(warning)

    st.caption("Optional engineering tolerance check (held-out test split):")
    col_response, col_tol = st.columns(2)
    response_name = col_response.selectbox(
        "Response", options=list(report.test_metrics), key="surrogate_tolerance_response"
    )
    max_relative = col_tol.number_input(
        "Max relative error", min_value=0.0, value=0.1, step=0.01, key="surrogate_tolerance_value"
    )
    if response_name and response_name in report.test_metrics:
        observed = report.test_metrics[response_name].max_relative_error
        passed = observed <= max_relative
        st.metric(
            f"Tolerance check: {response_name}",
            "PASS" if passed else "FAIL",
            delta=f"observed max relative error {observed:.2%}",
        )


def _render_domain_and_prediction() -> None:
    st.subheader("5. Applicability Domain")
    model = st.session_state.get(_MODEL_KEY)
    if model is None:
        st.info("Train a surrogate above first.")
        return

    metadata = model.training_metadata
    point: dict[str, float] = {}
    columns = st.columns(len(metadata.feature_names))
    for column, name in zip(columns, metadata.feature_names, strict=True):
        point[name] = column.number_input(name, value=0.0, key=f"surrogate_domain_{name}")

    if st.button("Predict"):
        prediction = model.predict_point(point)
        status_label = {
            DomainStatus.WITHIN_TRAINING_DOMAIN: "WITHIN_TRAINING_DOMAIN",
            DomainStatus.BOUNDARY: "BOUNDARY",
            DomainStatus.OUTSIDE_TRAINING_DOMAIN: "OUTSIDE_TRAINING_DOMAIN",
            DomainStatus.INVALID: "INVALID",
        }[prediction.domain_status]
        st.write(f"**Domain status:** {status_label}")
        st.write("**Surrogate prediction** (not a direct FEA result):")
        st.json(prediction.values)
        for warning in prediction.warnings:
            st.warning(warning)


def _render_verification(state: AppState) -> None:
    st.subheader("6. High-Fidelity Verification")
    model = st.session_state.get(_MODEL_KEY)
    if model is None:
        st.info("Train a surrogate above first.")
        return

    metadata = model.training_metadata
    point: dict[str, float] = {}
    columns = st.columns(len(metadata.feature_names))
    for column, name in zip(columns, metadata.feature_names, strict=True):
        point[name] = column.number_input(name, value=0.0, key=f"surrogate_verify_{name}")

    if st.button("Verify Against High-Fidelity FEA", type="primary"):
        try:
            with st.spinner("Running high-fidelity FEA for comparison..."):
                records = verify_against_high_fidelity(
                    model,
                    state.project,
                    [point],
                    {name: get_extractor(name) for name in metadata.response_names},
                )
            st.session_state[_VERIFICATION_KEY] = records
        except FiniteElementToolkitError as exc:
            st.error(describe_error(exc))

    records = st.session_state.get(_VERIFICATION_KEY)
    if records:
        for record in records:
            st.write(f"**Design point:** {record.design_point}")
            col_pred, col_actual, col_status = st.columns(3)
            col_pred.json({"surrogate prediction": record.predicted})
            col_actual.json({"high-fidelity verification": record.actual})
            col_status.write(f"**Acceptance:** {record.acceptance.value}")


def _render_report() -> None:
    st.subheader("7. Report")
    model = st.session_state.get(_MODEL_KEY)
    dataset = st.session_state.get(_DATASET_KEY)
    validation = st.session_state.get(_VALIDATION_KEY)
    rom = st.session_state.get(_ROM_KEY)
    verification_records = st.session_state.get(_VERIFICATION_KEY) or []

    if model is None and dataset is None:
        st.info("Nothing to report yet.")
        return

    report = SurrogateReport(
        title="Surrogate / ROM Report",
        summary="Generated from the current Surrogate / ROM Workspace session.",
        dataset=dataset,
        model=model,
        validation=validation,
        rom=rom,
        verification_records=verification_records,
    )
    markdown_text = render_surrogate_report_markdown(report)
    st.download_button(
        "Download Report (Markdown)",
        data=markdown_text,
        file_name="surrogate_report.md",
        mime="text/markdown",
    )
    with st.expander("Preview report"):
        st.markdown(markdown_text)


__all__ = ["render"]
