"""Multi-Fidelity page: Version 37 low-fidelity + high-fidelity + discrepancy fusion.

Reuses :mod:`femtoolkit.multifidelity` (Version 37) directly -- which itself
reuses :mod:`femtoolkit.surrogate` (Version 35) and the existing FEA pipeline. No
fidelity-model, discrepancy, or fusion logic is implemented in this module.

.. code-block:: text

    Design Point -> Low-Fidelity Model -> Low-Fidelity Result ->
    Discrepancy Model -> Fused Prediction -> Optional High-Fidelity FEA -> Validation

**The fused prediction is never shown as if it were an actual FEA result** --
every section below labels its numbers LOW-FIDELITY RESULT, MULTI-FIDELITY
PREDICTION, HIGH-FIDELITY FEA, or VERIFICATION RESULT.
"""

from __future__ import annotations

import streamlit as st

from app.streamlit.plotting import fidelity_error_comparison_frame, predicted_vs_actual_frame
from femtoolkit.application.exceptions_display import describe_error
from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.exceptions import FiniteElementToolkitError
from femtoolkit.multifidelity.dataset import MultiFidelityDataset, MultiFidelitySample
from femtoolkit.multifidelity.discrepancy import train_discrepancy_surrogate
from femtoolkit.multifidelity.fidelity import (
    HIGH_FIDELITY,
    LOW_FIDELITY,
    AnalyticalFidelityModel,
    SimulationFidelityModel,
    summarize_costs,
)
from femtoolkit.multifidelity.model import MultiFidelityModel, verify_fused_prediction
from femtoolkit.multifidelity.validation import compare_fidelity_accuracy
from femtoolkit.studies.extractors import get_extractor
from femtoolkit.surrogate.models import SURROGATE_MODEL_TYPES
from femtoolkit.surrogate.workflows.training import TrainingConfig

_DATASET_KEY = "quickstart_mf_dataset"
_MODEL_KEY = "quickstart_mf_model"
_REPORT_KEY = "quickstart_mf_report"
_VERIFICATION_KEY = "quickstart_mf_verification"

_RESPONSE_NAME = "maximum_displacement"
_BEAM_LENGTH = 2.0
_BEAM_HEIGHT = 0.4


def _base_project(tip_load: float) -> Project:
    project = Project(name="Multi-Fidelity Cantilever", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.width = _BEAM_LENGTH
    project.mesh.height = _BEAM_HEIGHT
    project.mesh.nx = 10
    project.mesh.ny = 3
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=tip_load)]
    return project


def _analytical_tip_deflection(tip_load: float):
    def evaluate(point: dict[str, float]) -> dict[str, float]:
        thickness = point["mesh.thickness"]
        moment_of_inertia = thickness * _BEAM_HEIGHT**3 / 12.0
        deflection = abs(tip_load) * _BEAM_LENGTH**3 / (3 * 200e9 * moment_of_inertia)
        return {_RESPONSE_NAME: deflection}

    return evaluate


def render() -> None:
    """Render the Multi-Fidelity page."""
    st.header("Multi-Fidelity")
    st.caption(
        "Low-Fidelity Model (Euler-Bernoulli beam, ignores shear deformation) + "
        "High-Fidelity Model (continuum FEA) + Discrepancy Surrogate -> Fused Prediction."
    )

    st.subheader("1/2. Low- and High-Fidelity Models")
    col_low, col_high = st.columns(2)
    col_low.metric("Low Fidelity", "Euler-Bernoulli beam")
    col_low.caption(f"Estimated cost: {LOW_FIDELITY.estimated_cost}")
    col_high.metric("High Fidelity", "Continuum FEA")
    col_high.caption(f"Estimated cost: {HIGH_FIDELITY.estimated_cost}")

    st.subheader("3. Design Parameters")
    col_t_low, col_t_high, col_load, col_n = st.columns(4)
    thickness_low = col_t_low.number_input("Thickness min (m)", value=0.006, format="%.4f")
    thickness_high = col_t_high.number_input("Thickness max (m)", value=0.020, format="%.4f")
    tip_load = col_load.number_input("Tip load (N)", value=-4000.0, step=500.0)
    n_samples = col_n.slider("Paired samples", 6, 30, 12)

    low_model = AnalyticalFidelityModel(
        name="Euler-Bernoulli beam", evaluate_fn=_analytical_tip_deflection(tip_load),
        level=LOW_FIDELITY,
    )
    high_model = SimulationFidelityModel(
        name="FEA cantilever", base_project=_base_project(tip_load),
        response_extractors={_RESPONSE_NAME: get_extractor(_RESPONSE_NAME)},
        level=HIGH_FIDELITY,
    )

    st.caption("Cost comparison:")
    st.table(summarize_costs([low_model, high_model]))

    if st.button("Generate Samples", type="primary"):
        try:
            dataset = MultiFidelityDataset(
                feature_names=["mesh.thickness"], response_names=[_RESPONSE_NAME]
            )
            step = (thickness_high - thickness_low) / max(n_samples - 1, 1)
            with st.spinner("Running low- and high-fidelity models..."):
                for index in range(n_samples):
                    point = {"mesh.thickness": thickness_low + index * step}
                    dataset.add_sample(
                        MultiFidelitySample(
                            sample_id=str(index), inputs=point,
                            low_result=low_model.evaluate(point),
                            high_result=high_model.evaluate(point),
                        )
                    )
            st.session_state[_DATASET_KEY] = dataset
            st.session_state[_MODEL_KEY] = None
            st.session_state[_REPORT_KEY] = None
            st.success(f"Generated {dataset.n_samples} paired samples.")
        except FiniteElementToolkitError as exc:
            st.error(describe_error(exc))

    dataset = st.session_state.get(_DATASET_KEY)
    if dataset is None:
        st.info("Set parameters above and click 'Generate Samples'.")
        return

    st.subheader("4/5. Paired Samples and Discrepancy")
    discrepancies = dataset.discrepancies()[_RESPONSE_NAME]
    st.line_chart({"Discrepancy (m)": list(discrepancies)})

    st.subheader("6. Discrepancy Surrogate")
    model_type = st.selectbox("Surrogate method", options=list(SURROGATE_MODEL_TYPES))
    if st.button("Train Discrepancy Model"):
        try:
            discrepancy_model, report = train_discrepancy_surrogate(
                dataset, TrainingConfig(model_type=model_type, split_seed=0)
            )
            st.session_state[_MODEL_KEY] = discrepancy_model
            st.session_state[_REPORT_KEY] = report
        except FiniteElementToolkitError as exc:
            st.error(describe_error(exc))

    discrepancy_model = st.session_state.get(_MODEL_KEY)
    report = st.session_state.get(_REPORT_KEY)
    if report is not None:
        metrics = report.test_metrics[_RESPONSE_NAME]
        row = {
            "Response": _RESPONSE_NAME, "MAE": metrics.mae, "RMSE": metrics.rmse, "R^2": metrics.r2,
        }
        st.table([row])

    if discrepancy_model is None:
        st.info("Train the discrepancy surrogate to continue.")
        return

    mf_model = MultiFidelityModel(low_fidelity_model=low_model, discrepancy_model=discrepancy_model)

    st.subheader("7/8. Fused Prediction and High-Fidelity Verification")
    query_thickness = st.number_input(
        "Query thickness (m)", value=float((thickness_low + thickness_high) / 2), format="%.4f"
    )
    col_predict, col_verify = st.columns(2)

    if col_predict.button("Predict"):
        prediction = mf_model.predict({"mesh.thickness": query_thickness})
        st.markdown("**LOW-FIDELITY RESULT**")
        st.json(prediction.low_fidelity_result)
        st.markdown("**MULTI-FIDELITY PREDICTION**")
        st.json(prediction.fused_prediction)

    if col_verify.button("Run High-Fidelity Verification"):
        try:
            with st.spinner("Running high-fidelity FEA for verification..."):
                records = verify_fused_prediction(
                    mf_model, high_model, [{"mesh.thickness": query_thickness}]
                )
            st.session_state[_VERIFICATION_KEY] = records[0]
        except FiniteElementToolkitError as exc:
            st.error(describe_error(exc))

    verification = st.session_state.get(_VERIFICATION_KEY)
    if verification is not None:
        st.markdown("**HIGH-FIDELITY FEA**")
        st.json(verification.high_fidelity_result)
        st.markdown("**VERIFICATION RESULT**")
        st.json(
            {
                "fused_absolute_error": verification.fused_absolute_error,
                "fused_relative_error": verification.fused_relative_error,
                "low_fidelity_relative_error": verification.low_fidelity_relative_error,
                "status": verification.status.value,
            }
        )

    st.subheader("9. Compare Results")
    comparison = compare_fidelity_accuracy(mf_model, dataset)
    low_rmse = comparison.low_fidelity_metrics[_RESPONSE_NAME].rmse
    fused_rmse = comparison.fused_metrics[_RESPONSE_NAME].rmse
    col_low_rmse, col_fused_rmse = st.columns(2)
    col_low_rmse.metric("Low-fidelity-only RMSE", f"{low_rmse:.4e}")
    col_fused_rmse.metric("Fused-model RMSE", f"{fused_rmse:.4e}")
    improves = comparison.improves_on_low_fidelity(_RESPONSE_NAME)
    st.write(f"**Fusion improves accuracy:** {improves}")

    paired = dataset.paired_samples()
    low_errors = [abs(s.high_result[_RESPONSE_NAME] - s.low_result[_RESPONSE_NAME]) for s in paired]
    fused_values = [mf_model.predict(s.inputs).fused_prediction[_RESPONSE_NAME] for s in paired]
    high_values = [s.high_result[_RESPONSE_NAME] for s in paired]
    fused_errors = [abs(h - f) for h, f in zip(high_values, fused_values, strict=True)]

    st.caption("Fused Prediction vs. High-Fidelity Result")
    st.scatter_chart(predicted_vs_actual_frame(fused_values, high_values))
    st.caption("Low-Fidelity Error vs. Fused-Model Error")
    st.line_chart(fidelity_error_comparison_frame(low_errors, fused_errors))


__all__ = ["render"]
