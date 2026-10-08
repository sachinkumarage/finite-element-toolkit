"""Results page: the consolidated engineering summary of the last adaptive study."""

from __future__ import annotations

import streamlit as st

_RESULT_KEY = "quickstart_adaptive_result"
_STUDY_KEY = "quickstart_adaptive_study_config"


def render() -> None:
    """Render the Results page."""
    st.header("Results")

    result = st.session_state.get(_RESULT_KEY)
    if result is None:
        st.info("Run the Surrogate-Assisted Optimization page first.")
        return

    st.subheader("Best Verified Design")
    if result.best_verified_design is None:
        st.warning("No high-fidelity verification completed successfully.")
    else:
        st.json(result.best_verified_design)

    col_objective, col_status = st.columns(2)
    col_objective.metric(
        "Objective value (mass, kg)",
        "-" if result.best_verified_objective is None else f"{result.best_verified_objective:.4f}",
    )
    col_status.metric("Engineering status", result.status.value.upper())

    st.subheader("Surrogate Prediction vs. High-Fidelity Result")
    if result.best_surrogate_predicted_design is not None:
        col_pred, col_actual = st.columns(2)
        with col_pred:
            st.markdown("**Surrogate Prediction**")
            st.json(
                {
                    "design": result.best_surrogate_predicted_design,
                    "predicted_objective": result.best_surrogate_predicted_objective,
                }
            )
        with col_actual:
            st.markdown("**High-Fidelity FEA**")
            st.json(
                {
                    "design": result.best_verified_design,
                    "actual_objective": result.best_verified_objective,
                }
            )
    else:
        st.info("No surrogate-predicted candidate is available.")

    st.subheader("Relative Error")
    if result.prediction_error:
        st.json(result.prediction_error)
    else:
        st.write("Not available.")

    st.subheader("Evaluation Budget")
    col_initial, col_total, col_iter = st.columns(3)
    col_initial.metric("Initial samples", result.initial_sample_count)
    col_total.metric("Total FEA evaluations", result.total_high_fidelity_evaluations)
    col_iter.metric("Refinement iterations", result.iteration_count)

    st.caption(f"Stopping reason: {result.stopping_reason}")


__all__ = ["render"]
