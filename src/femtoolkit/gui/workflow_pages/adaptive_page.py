"""Surrogate-Assisted Adaptive Optimization page (Version 36).

Surfaces the Version 36 adaptive-sampling and refinement framework
(:mod:`femtoolkit.adaptive`) inside the existing GUI, against the current
project: defining design variables and an objective/constraint, running the
initial-sample-generation + surrogate-training + candidate-search +
high-fidelity-verification + retrain loop (:class:`~femtoolkit.adaptive.study.AdaptiveStudy`),
and reviewing the iteration history, convergence, and best verified design --
with no sampling, surrogate, or optimization logic implemented in this module
itself.
"""

from __future__ import annotations

import streamlit as st

from femtoolkit.adaptive.refinement import RefinementConfig, SurrogateAcceptanceState
from femtoolkit.adaptive.sampling import SamplingStrategy
from femtoolkit.adaptive.study import AdaptiveStudy
from femtoolkit.application.exceptions_display import describe_error
from femtoolkit.exceptions import FiniteElementToolkitError
from femtoolkit.gui.components import render_execution_mode_controls, require_project
from femtoolkit.gui.state import AppState
from femtoolkit.optimization.constraints import Constraint, ConstraintRelation
from femtoolkit.optimization.objectives import Objective, ObjectiveDirection, from_result_extractor
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType
from femtoolkit.studies.extractors import EXTRACTORS, get_extractor

_VARIABLES_KEY = "femtoolkit_adaptive_variables"
_RESULT_KEY = "femtoolkit_adaptive_result"

_NO_CONSTRAINT = "(none)"


def _variables(state_store) -> list[DesignVariable]:
    return state_store.setdefault(_VARIABLES_KEY, [])


def render(state: AppState) -> None:
    """Render the Surrogate-Assisted Adaptive Optimization page."""
    st.header("Surrogate-Assisted Adaptive Optimization")
    st.caption(
        "Initial Samples -> Surrogate -> Candidate Search -> High-Fidelity Verification -> "
        "Add Sample -> Retrain -> Repeat. The surrogate accelerates the search; the "
        "reported best design is always the best VERIFIED one, never just predicted."
    )

    if not require_project(state):
        return

    _render_design_variables(state)
    st.divider()
    _render_objective_and_run(state)
    st.divider()
    _render_results()


def _render_design_variables(state: AppState) -> None:
    st.subheader("1. Design Variables")
    st.caption(
        "Each variable is a dotted override path (e.g. 'mesh.thickness') on the current "
        "project, with a name used to refer to it, and a continuous lower/upper bound."
    )
    variables = _variables(st.session_state)

    with st.form("add_adaptive_variable_form", clear_on_submit=True):
        col_path, col_name, col_low, col_high = st.columns(4)
        path = col_path.text_input("Override path", placeholder="mesh.thickness")
        name = col_name.text_input("Variable name", placeholder="thickness")
        lower_bound = col_low.number_input("Lower bound", value=0.0, format="%.5f")
        upper_bound = col_high.number_input("Upper bound", value=1.0, format="%.5f")
        submitted = st.form_submit_button("Add Variable")

    if submitted:
        try:
            if not path or not name:
                raise FiniteElementToolkitError("Both an override path and a name are required.")
            variables.append(
                DesignVariable(
                    name=name, path=path, variable_type=DesignVariableType.CONTINUOUS,
                    lower_bound=lower_bound, upper_bound=upper_bound,
                )
            )
        except FiniteElementToolkitError as exc:
            st.error(describe_error(exc))

    for index, variable in enumerate(variables):
        col_info, col_remove = st.columns([5, 1])
        bounds = f"[{variable.lower_bound}, {variable.upper_bound}]"
        col_info.write(f"**{variable.name}** (`{variable.path}`): {bounds}")
        if col_remove.button("Remove", key=f"remove_adaptive_variable_{index}"):
            variables.pop(index)
            st.rerun()


def _render_objective_and_run(state: AppState) -> None:
    st.subheader("2. Objective, Constraint, and Refinement Settings")
    variables = _variables(st.session_state)
    if not variables:
        st.info("Add at least one design variable above first.")
        return

    col_objective, col_direction = st.columns(2)
    objective_name = col_objective.selectbox("Objective response", options=list(EXTRACTORS))
    direction = col_direction.selectbox("Direction", options=["Minimize", "Maximize"])

    col_constraint, col_relation, col_limit = st.columns(3)
    constraint_name = col_constraint.selectbox(
        "Constraint response", options=[_NO_CONSTRAINT, *EXTRACTORS]
    )
    relation = col_relation.selectbox("Relation", options=["<=", ">=", "=="])
    limit = col_limit.number_input("Limit", value=0.0, format="%.5f")

    col_init, col_iter, col_cand = st.columns(3)
    n_initial_samples = col_init.slider("Initial high-fidelity samples", 6, 40, 10)
    max_iterations = col_iter.slider("Refinement iterations", 1, 10, 3)
    n_candidates = col_cand.slider("Candidates scored per iteration", 10, 100, 30)

    col_strategy, col_explore, col_exploit = st.columns(3)
    strategy = col_strategy.selectbox(
        "Sampling strategy", options=[s.value for s in SamplingStrategy]
    )
    exploration_weight = col_explore.slider("Exploration weight", 0.0, 1.0, 0.5)
    exploitation_weight = col_exploit.slider("Exploitation weight", 0.0, 1.0, 0.5)

    col_tol, col_seed = st.columns(2)
    error_tolerance = col_tol.slider("Error tolerance (relative)", 0.01, 0.5, 0.1)
    seed = col_seed.number_input("Random seed", min_value=0, value=0, step=1)

    orchestration_config = render_execution_mode_controls(key_prefix="adaptive")

    if st.button("Run Adaptive Study", type="primary"):
        try:
            objective = Objective(
                name=objective_name,
                direction=ObjectiveDirection.MINIMIZE if direction == "Minimize"
                else ObjectiveDirection.MAXIMIZE,
                evaluate=from_result_extractor(get_extractor(objective_name)),
            )
            constraints = []
            response_extractors = {objective_name: get_extractor(objective_name)}
            if constraint_name != _NO_CONSTRAINT:
                constraints.append(
                    Constraint(
                        name=constraint_name,
                        evaluate=from_result_extractor(get_extractor(constraint_name)),
                        relation=ConstraintRelation(relation),
                        limit=limit,
                    )
                )
                response_extractors[constraint_name] = get_extractor(constraint_name)

            study = AdaptiveStudy(
                base_project=state.project,
                design_variables=list(variables),
                objective=objective,
                constraints=constraints,
                response_extractors=response_extractors,
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
            with st.spinner("Generating initial samples, training, and refining..."):
                result = study.run(
                    n_initial_samples=n_initial_samples, orchestration_config=orchestration_config
                )
            st.session_state[_RESULT_KEY] = result
        except FiniteElementToolkitError as exc:
            st.error(describe_error(exc))


def _render_results() -> None:
    st.subheader("3. Results")
    result = st.session_state.get(_RESULT_KEY)
    if result is None:
        st.info("Run the adaptive study above first.")
        return

    col_initial, col_total, col_iter, col_status = st.columns(4)
    col_initial.metric("Initial samples", result.initial_sample_count)
    col_total.metric("Total FEA evaluations", result.total_high_fidelity_evaluations)
    col_iter.metric("Iterations run", result.iteration_count)
    status_label = {
        SurrogateAcceptanceState.SURROGATE_ONLY: "SURROGATE_ONLY",
        SurrogateAcceptanceState.PENDING_VERIFICATION: "PENDING_VERIFICATION",
        SurrogateAcceptanceState.VERIFIED: "VERIFIED",
        SurrogateAcceptanceState.VERIFICATION_FAILED: "VERIFICATION_FAILED",
        SurrogateAcceptanceState.REQUIRES_REFINEMENT: "REQUIRES_REFINEMENT",
    }[result.status]
    col_status.metric("Status", status_label)

    st.write(f"**Stopping reason:** {result.stopping_reason}")
    st.write("**Best verified design** (from real FEA, never just predicted):")
    st.json(
        {
            "design": result.best_verified_design,
            "objective_value": result.best_verified_objective,
        }
    )

    if result.final_surrogate_metrics is not None:
        st.caption("Final surrogate validation metrics:")
        rows = [
            {"Response": name, "MAE": m.mae, "RMSE": m.rmse, "R^2": m.r2}
            for name, m in result.final_surrogate_metrics.test_metrics.items()
        ]
        st.table(rows)

    if result.convergence_history:
        st.caption("Best verified objective value per iteration:")
        st.line_chart([value for value in result.convergence_history if value is not None])

    for step in result.iteration_history:
        with st.expander(f"Iteration {step.iteration}: {step.status.value}"):
            col_predicted, col_actual = st.columns(2)
            col_predicted.markdown("**Surrogate Prediction**")
            col_predicted.json(step.candidate.evaluation.objective_values)
            col_actual.markdown("**High-Fidelity FEA**")
            col_actual.json(step.verification.actual)
            if step.verification.relative_error:
                st.write(f"Relative error: {step.verification.relative_error}")


__all__ = ["render"]
