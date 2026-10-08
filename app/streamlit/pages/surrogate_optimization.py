"""Surrogate-Assisted Optimization page: the core Version 36 workflow.

Reuses :mod:`femtoolkit.adaptive` (Version 36) end to end, which itself reuses
:mod:`femtoolkit.surrogate` (Version 35), :mod:`femtoolkit.optimization`
(Version 33), and :mod:`femtoolkit.orchestration` (Version 34) -- no FEA,
surrogate, or adaptive-sampling logic is implemented in this module.

.. code-block:: text

    Initial FEA Samples -> Train Surrogate -> Candidate Search ->
    High-Fidelity Verification -> Add Sample -> Retrain -> Repeat

Every result is explicitly labeled **Surrogate Prediction** or **High-Fidelity
FEA** -- the two are never shown as if they were the same kind of result.
"""

from __future__ import annotations

import streamlit as st

from app.streamlit.plotting import convergence_frame, predicted_vs_actual_frame
from femtoolkit.adaptive.refinement import RefinementConfig, SurrogateAcceptanceState
from femtoolkit.adaptive.sampling import SamplingStrategy
from femtoolkit.adaptive.study import AdaptiveStudy
from femtoolkit.application.exceptions_display import describe_error
from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.exceptions import FiniteElementToolkitError
from femtoolkit.optimization.constraints import Constraint, ConstraintRelation
from femtoolkit.optimization.objectives import Objective, ObjectiveDirection, from_result_extractor
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType
from femtoolkit.studies.extractors import get_extractor
from femtoolkit.surrogate.models import SURROGATE_MODEL_TYPES

_RESULT_KEY = "quickstart_adaptive_result"
_STUDY_KEY = "quickstart_adaptive_study_config"


def _mass_extractor(run) -> float | None:
    """Mass of the simulated rectangular domain: ``width * height * thickness * density``.

    Reads the run's frozen configuration directly (no FEA solve is needed for mass),
    mirroring :func:`femtoolkit.optimization.objectives.rectangular_mass`'s formula
    for the Version 35/36 "response from a completed SimulationRun" convention.
    """
    snapshot = run.configuration_snapshot
    if snapshot is None or snapshot.material.density is None:
        return None
    mesh = snapshot.mesh
    return mesh.width * mesh.height * mesh.thickness * snapshot.material.density


def _base_project() -> Project:
    project = Project(name="Quickstart Adaptive Cantilever", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.height = 0.4
    project.mesh.nx = 10
    project.mesh.ny = 3
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-4000.0)]
    return project


def render() -> None:
    """Render the Surrogate-Assisted Optimization page."""
    st.header("Surrogate-Assisted Optimization")
    st.caption(
        "Minimize beam mass subject to a maximum displacement limit. The surrogate "
        "accelerates candidate search; every reported result is clearly labeled "
        "either a surrogate prediction or a real high-fidelity FEA verification."
    )

    st.subheader("1. Design Variables")
    col_t_low, col_t_high, col_w_low, col_w_high = st.columns(4)
    thickness_low = col_t_low.number_input("Thickness min (m)", value=0.006, format="%.4f")
    thickness_high = col_t_high.number_input("Thickness max (m)", value=0.020, format="%.4f")
    width_low = col_w_low.number_input("Width min (m)", value=1.0, format="%.3f")
    width_high = col_w_high.number_input("Width max (m)", value=3.0, format="%.3f")

    allowable_displacement = st.number_input(
        "Allowable maximum displacement (m)", value=0.003, step=0.0005, format="%.4f"
    )

    st.subheader("2/3. Surrogate Model")
    col_model, col_samples = st.columns(2)
    model_type = col_model.selectbox("Surrogate method", options=list(SURROGATE_MODEL_TYPES))
    n_initial_samples = col_samples.slider("Initial high-fidelity samples", 6, 30, 10)

    st.subheader("Adaptive Refinement Settings")
    col_iter, col_cand, col_strategy = st.columns(3)
    max_iterations = col_iter.slider("Refinement iterations", 1, 10, 3)
    n_candidates = col_cand.slider("Candidates scored per iteration", 10, 100, 30)
    strategy = col_strategy.selectbox(
        "Sampling strategy", options=[s.value for s in SamplingStrategy]
    )
    col_explore, col_exploit, col_tol = st.columns(3)
    exploration_weight = col_explore.slider("Exploration weight", 0.0, 1.0, 0.5)
    exploitation_weight = col_exploit.slider("Exploitation weight", 0.0, 1.0, 0.5)
    error_tolerance = col_tol.slider("Error tolerance (relative)", 0.01, 0.5, 0.1)
    seed = st.number_input("Random seed", min_value=0, value=0, step=1)

    if st.button("Run Surrogate-Assisted Optimization", type="primary"):
        design_variables = [
            DesignVariable(
                name="thickness", path="mesh.thickness",
                variable_type=DesignVariableType.CONTINUOUS,
                lower_bound=thickness_low, upper_bound=thickness_high,
            ),
            DesignVariable(
                name="width", path="mesh.width", variable_type=DesignVariableType.CONTINUOUS,
                lower_bound=width_low, upper_bound=width_high,
            ),
        ]
        response_extractors = {
            "mass": _mass_extractor,
            "maximum_displacement": get_extractor("maximum_displacement"),
        }
        objective = Objective(
            name="mass", direction=ObjectiveDirection.MINIMIZE,
            evaluate=from_result_extractor(_mass_extractor),
        )
        constraint = Constraint(
            name="maximum_displacement",
            evaluate=from_result_extractor(get_extractor("maximum_displacement")),
            relation=ConstraintRelation.LESS_EQUAL, limit=allowable_displacement,
        )
        study = AdaptiveStudy(
            base_project=_base_project(),
            design_variables=design_variables,
            objective=objective,
            constraints=[constraint],
            response_extractors=response_extractors,
            model_type=model_type,
            refinement_config=RefinementConfig(
                max_iterations=max_iterations,
                n_candidates=n_candidates,
                sampling_strategy=SamplingStrategy(strategy),
                exploration_weight=exploration_weight,
                exploitation_weight=exploitation_weight,
                error_tolerance=error_tolerance,
                seed=int(seed),
            ),
            random_seed=int(seed),
        )
        try:
            with st.spinner("Generating initial samples, training, and refining..."):
                result = study.run(n_initial_samples=n_initial_samples)
            st.session_state[_RESULT_KEY] = result
            st.session_state[_STUDY_KEY] = {
                "design_variables": design_variables, "objective": objective,
                "constraint": constraint, "allowable_displacement": allowable_displacement,
            }
        except FiniteElementToolkitError as exc:
            st.error(describe_error(exc))

    result = st.session_state.get(_RESULT_KEY)
    if result is None:
        st.info("Configure the study above and click 'Run Surrogate-Assisted Optimization'.")
        return

    st.subheader("4. Validation Metrics")
    if result.final_surrogate_metrics is not None:
        rows = []
        for name, metrics in result.final_surrogate_metrics.test_metrics.items():
            rows.append(
                {"Response": name, "MAE": metrics.mae, "RMSE": metrics.rmse, "R^2": metrics.r2}
            )
        st.table(rows)

    st.subheader("5-8. Candidate Search & High-Fidelity Verification")
    st.write(f"Ran {result.iteration_count} refinement iteration(s).")
    predicted_mass, actual_mass = [], []
    for step in result.iteration_history:
        predicted = step.candidate.evaluation.objective_values.get("mass")
        actual = step.verification.actual.get("mass")
        if predicted is not None and actual is not None:
            predicted_mass.append(predicted)
            actual_mass.append(actual)
        with st.expander(f"Iteration {step.iteration}: {step.status.value}"):
            col_pred, col_actual = st.columns(2)
            col_pred.markdown("**Surrogate Prediction**")
            col_pred.json(step.candidate.evaluation.objective_values)
            col_actual.markdown("**High-Fidelity FEA**")
            col_actual.json(step.verification.actual)
            if step.verification.relative_error:
                st.write(f"Relative error: {step.verification.relative_error}")

    if predicted_mass:
        st.subheader("Surrogate Prediction vs. High-Fidelity FEA")
        st.scatter_chart(predicted_vs_actual_frame(predicted_mass, actual_mass))

    if result.convergence_history:
        st.subheader("Convergence")
        st.line_chart(convergence_frame(result.convergence_history, "Best verified mass (kg)"))

    st.subheader("9. Study Summary")
    status_label = {
        SurrogateAcceptanceState.VERIFIED: "VERIFIED",
        SurrogateAcceptanceState.REQUIRES_REFINEMENT: "REQUIRES_REFINEMENT",
        SurrogateAcceptanceState.VERIFICATION_FAILED: "VERIFICATION_FAILED",
        SurrogateAcceptanceState.SURROGATE_ONLY: "SURROGATE_ONLY",
        SurrogateAcceptanceState.PENDING_VERIFICATION: "PENDING_VERIFICATION",
    }[result.status]
    st.write(f"**Engineering status:** {status_label}")
    st.write(f"**Stopping reason:** {result.stopping_reason}")
    st.caption(
        "Go to the Results page for the consolidated engineering summary of the best "
        "verified design found."
    )


__all__ = ["render"]
