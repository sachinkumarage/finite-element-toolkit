"""Optimization & Design Exploration page (Version 32).

Surfaces the Version 32 optimization framework
(:mod:`femtoolkit.optimization`) inside the existing GUI: defining
design variables, objectives, and constraints; choosing and configuring
a search algorithm; evaluating the baseline design; running the search;
and reviewing results -- with no design-variable, objective,
constraint, or search logic implemented in this module itself.

**Best feasible, never "best overall."** For a single-objective
problem, this page reports the best *feasible* candidate found relative
to the defined objective and constraints -- never as an unqualified
"best" or "optimal" design. For a multi-objective problem, it reports
the full non-dominated (Pareto) set with no design singled out as a
winner; choosing among trade-offs is left to the engineer.

**On run cancellation.** As with the Version 31 Uncertainty Analysis
page, this GUI executes each page synchronously within one Streamlit
script run -- there is no in-process mechanism here to safely interrupt
a search partway through and preserve its partial history. A modest
default evaluation budget and the configurable `evaluation_limit` are
this page's safeguard against an unexpectedly long-running search
instead of a genuine cancel button.
"""

from __future__ import annotations

import streamlit as st

from femtoolkit.application.exceptions_display import describe_error
from femtoolkit.exceptions import FiniteElementToolkitError
from femtoolkit.gui.components import require_project
from femtoolkit.gui.state import AppState
from femtoolkit.optimization import (
    SUPPORTED_ALGORITHMS,
    Constraint,
    ConstraintRelation,
    DesignVariable,
    DesignVariableType,
    Objective,
    ObjectiveDirection,
    OptimizationConfig,
    OptimizationProblem,
    OptimizationRunner,
    build_optimization_report,
    from_result_extractor,
    plot_pareto_front,
    rectangular_mass,
    render_optimization_report_markdown,
)
from femtoolkit.studies.extractors import EXTRACTORS, get_extractor

_VARIABLES_KEY = "femtoolkit_optimization_variables"
_OBJECTIVES_KEY = "femtoolkit_optimization_objectives"
_CONSTRAINTS_KEY = "femtoolkit_optimization_constraints"
_RESULT_KEY = "femtoolkit_optimization_result"

_MASS_QUANTITY = "mass (rectangular, from project geometry)"
_QUANTITY_OPTIONS = [_MASS_QUANTITY, *sorted(EXTRACTORS)]


def _variables(state_store) -> list[DesignVariable]:
    return state_store.setdefault(_VARIABLES_KEY, [])


def _objectives(state_store) -> list[Objective]:
    return state_store.setdefault(_OBJECTIVES_KEY, [])


def _constraints(state_store) -> list[Constraint]:
    return state_store.setdefault(_CONSTRAINTS_KEY, [])


def _evaluate_fn(quantity_name: str):
    if quantity_name == _MASS_QUANTITY:
        return rectangular_mass
    return from_result_extractor(get_extractor(quantity_name))


def render(state: AppState) -> None:
    """Render the Optimization & Design Exploration page."""
    st.header("Optimization & Design Exploration")
    st.caption(
        "Design Variables -> Build Scenario -> Run FEA -> Extract Results -> "
        "Calculate Objectives -> Evaluate Constraints -> Design Evaluation"
    )

    if not require_project(state):
        return

    _render_variable_builder(state)
    st.divider()
    _render_objective_builder(state)
    st.divider()
    _render_constraint_builder(state)
    st.divider()
    _render_execution(state)
    st.divider()
    _render_results(state)
    st.divider()
    _render_report_generation(state)


def _render_variable_builder(state: AppState) -> None:
    st.subheader("1. Design Variables")
    st.caption(
        "Each variable is a dotted override path on the current project (e.g. "
        "'mesh.thickness') paired with a domain the search is allowed to explore."
    )
    variables = _variables(st.session_state)

    col_name, col_path, col_units = st.columns(3)
    name = col_name.text_input("Name", key="opt_var_name", placeholder="thickness")
    path = col_path.text_input("Override path", key="opt_var_path", placeholder="mesh.thickness")
    units = col_units.text_input("Units", key="opt_var_units", placeholder="m")

    variable_type = st.selectbox(
        "Type", options=[t.value for t in DesignVariableType], key="opt_var_type"
    )

    if variable_type == "categorical":
        categories_text = st.text_input(
            "Categories (comma-separated)", key="opt_var_categories", placeholder="quad, cst"
        )
        lower_bound, upper_bound = None, None
    else:
        col_lower, col_upper = st.columns(2)
        lower_bound = col_lower.number_input("Lower bound", value=0.0, format="%.6g")
        upper_bound = col_upper.number_input("Upper bound", value=1.0, format="%.6g")
        categories_text = ""

    if st.button("Add Design Variable"):
        try:
            categories = (
                [c.strip() for c in categories_text.split(",") if c.strip()]
                if variable_type == "categorical"
                else None
            )
            variables.append(
                DesignVariable(
                    name=name,
                    path=path,
                    variable_type=DesignVariableType(variable_type),
                    lower_bound=(
                        float(lower_bound) if variable_type == "continuous" else lower_bound
                    ),
                    upper_bound=(
                        float(upper_bound) if variable_type == "continuous" else upper_bound
                    ),
                    categories=categories,
                    units=units,
                )
            )
            st.success(f"Added design variable '{name}'.")
        except FiniteElementToolkitError as exc:
            st.error(describe_error(exc))
        except ValueError as exc:
            st.error(str(exc))

    if not variables:
        st.info("No design variables added yet -- add at least one before running a search.")
        return

    for index, variable in enumerate(variables):
        col_info, col_remove = st.columns([5, 1])
        domain = (
            f"[{variable.lower_bound}, {variable.upper_bound}]"
            if variable.variable_type is not DesignVariableType.CATEGORICAL
            else str(variable.categories)
        )
        col_info.write(
            f"**{variable.name}** (`{variable.path}`) -- {variable.variable_type.value}, "
            f"domain={domain}, units={variable.units or '-'}"
        )
        if col_remove.button("Remove", key=f"remove_variable_{index}"):
            variables.pop(index)
            st.rerun()


def _render_objective_builder(state: AppState) -> None:
    st.subheader("2. Objectives")
    objectives = _objectives(st.session_state)

    col_name, col_quantity, col_direction = st.columns(3)
    name = col_name.text_input("Name", key="opt_obj_name", placeholder="maximum_displacement")
    quantity = col_quantity.selectbox("Quantity", options=_QUANTITY_OPTIONS, key="opt_obj_quantity")
    direction = col_direction.selectbox(
        "Direction", options=[d.value for d in ObjectiveDirection], key="opt_obj_direction"
    )
    units = st.text_input("Units", key="opt_obj_units", placeholder="m")

    if st.button("Add Objective"):
        try:
            objectives.append(
                Objective(
                    name=name or quantity,
                    direction=ObjectiveDirection(direction),
                    evaluate=_evaluate_fn(quantity),
                    units=units,
                )
            )
            st.success(f"Added objective '{name or quantity}'.")
        except FiniteElementToolkitError as exc:
            st.error(describe_error(exc))

    if not objectives:
        st.info("No objectives added yet -- add at least one before running a search.")
        return

    for index, objective in enumerate(objectives):
        col_info, col_remove = st.columns([5, 1])
        col_info.write(
            f"**{objective.name}** -- {objective.direction.value}, "
            f"units={objective.units or '-'}"
        )
        if col_remove.button("Remove", key=f"remove_objective_{index}"):
            objectives.pop(index)
            st.rerun()


def _render_constraint_builder(state: AppState) -> None:
    st.subheader("3. Constraints (optional)")
    constraints = _constraints(st.session_state)

    col_name, col_quantity, col_relation, col_limit = st.columns(4)
    name = col_name.text_input("Name", key="opt_con_name", placeholder="stress_limit")
    quantity = col_quantity.selectbox("Quantity", options=_QUANTITY_OPTIONS, key="opt_con_quantity")
    relation = col_relation.selectbox(
        "Relation",
        options=[r.value for r in ConstraintRelation],
        key="opt_con_relation",
    )
    limit = col_limit.number_input("Limit", value=0.0, format="%.6g", key="opt_con_limit")
    units = st.text_input("Units", key="opt_con_units", placeholder="Pa")

    if st.button("Add Constraint"):
        try:
            constraints.append(
                Constraint(
                    name=name or quantity,
                    evaluate=_evaluate_fn(quantity),
                    relation=ConstraintRelation(relation),
                    limit=float(limit),
                    units=units,
                )
            )
            st.success(f"Added constraint '{name or quantity}'.")
        except FiniteElementToolkitError as exc:
            st.error(describe_error(exc))

    if not constraints:
        st.info("No constraints added -- the search will explore the design space unconstrained.")
        return

    for index, constraint in enumerate(constraints):
        col_info, col_remove = st.columns([5, 1])
        col_info.write(
            f"**{constraint.name}** -- {constraint.relation.value} {constraint.limit:g} "
            f"{constraint.units or ''}"
        )
        if col_remove.button("Remove", key=f"remove_constraint_{index}"):
            constraints.pop(index)
            st.rerun()


def _render_execution(state: AppState) -> None:
    st.subheader("4. Algorithm & Execution")
    variables = _variables(st.session_state)
    objectives = _objectives(st.session_state)
    constraints = _constraints(st.session_state)

    col_algorithm, col_max_eval, col_seed = st.columns(3)
    algorithm = col_algorithm.selectbox("Algorithm", options=SUPPORTED_ALGORITHMS)
    max_evaluations = col_max_eval.number_input("Max evaluations", min_value=1, value=40, step=1)
    seed = col_seed.number_input("Random seed", min_value=0, value=42, step=1)

    col_tolerance, col_step = st.columns(2)
    tolerance = col_tolerance.number_input(
        "Convergence tolerance", min_value=0.0, value=1e-6, format="%.2e"
    )
    step_size = col_step.slider(
        "Step size (coordinate search only)", min_value=0.01, max_value=1.0, value=0.1
    )

    problem_name = st.text_input("Problem name", value=f"{state.project.name} Optimization")
    ready = bool(variables and objectives)

    if st.button("Run Optimization", type="primary", disabled=not ready):
        try:
            problem = OptimizationProblem(
                name=problem_name,
                base_project=state.project,
                design_variables=list(variables),
                objectives=list(objectives),
                constraints=list(constraints),
            )
            config = OptimizationConfig(
                algorithm=algorithm,
                max_evaluations=int(max_evaluations),
                tolerance=float(tolerance) if tolerance > 0 else 1e-6,
                seed=int(seed),
                step_size=float(step_size),
            )
            with st.spinner(f"Running up to {config.max_evaluations} evaluations..."):
                result = OptimizationRunner().run(problem, config)
        except FiniteElementToolkitError as exc:
            st.error(describe_error(exc))
            return

        st.session_state[_RESULT_KEY] = result
        st.success(
            f"Stopped: {result.stop_reason.value}. {result.history.n_evaluations} evaluations "
            f"recorded, {len(result.history.feasible_evaluations())} feasible."
        )

    if not ready:
        st.info("Add at least one design variable and one objective to run a search.")


def _current_result():
    return st.session_state.get(_RESULT_KEY)


def _render_results(state: AppState) -> None:
    st.subheader("5. Results")
    result = _current_result()
    if result is None:
        st.info("Run an optimization above to see its results.")
        return

    st.write(
        f"**Baseline:** {result.baseline.design_variables}, "
        f"status={result.baseline.status.value}, "
        f"objectives={result.baseline.objective_values}"
    )

    if result.is_multi_objective:
        front = result.pareto_front()
        st.write(
            f"**Non-dominated (Pareto) set:** {len(front)} design(s). "
            "No single design is labeled 'best overall' -- each trades objectives "
            "against each other; selecting one is an engineering judgment."
        )
        st.dataframe(
            [
                {"design_id": e.design_id, **e.design_variables, **e.objective_values}
                for e in front
            ],
            width="stretch",
        )
        if len(result.objectives) == 2:
            st.pyplot(
                plot_pareto_front(
                    result.history.evaluations, result.objectives, front, baseline=result.baseline
                )
            )
    else:
        best = result.best_feasible()
        if best is None:
            st.warning("No feasible design was found -- every evaluation was infeasible or failed.")
            return
        st.write(
            f"**Best feasible candidate** (per the defined objective and constraints, "
            f"never labeled 'best overall'): {best.design_id}"
        )
        st.write(f"Design variables: {best.design_variables}")
        st.write(f"Objective values: {best.objective_values}")

        improvement = result.improvement_over_baseline()
        if improvement is not None:
            st.write(
                f"Change vs. baseline: {improvement['absolute_difference']:.6g} "
                f"({improvement['percentage_change']:+.2f}%) -- a quantitative "
                "comparison only, not a claim of a global or guaranteed optimum."
            )

    with st.expander("Full evaluation history"):
        st.dataframe(
            [
                {
                    "design_id": e.design_id,
                    "status": e.status.value,
                    **e.design_variables,
                    **e.objective_values,
                    "total_violation": e.total_violation,
                }
                for e in result.history.evaluations
            ],
            width="stretch",
        )


def _render_report_generation(state: AppState) -> None:
    st.subheader("6. Optimization Report")
    result = _current_result()
    if result is None:
        st.info("Run an optimization above to generate a report.")
        return

    summary = st.text_area(
        "Study summary", value=f"Optimization study of project '{state.project.name}'."
    )
    conclusions = st.text_area("Conclusions (optional, free text -- never generated automatically)")

    report = build_optimization_report(
        title=f"{result.problem_name} Report",
        summary=summary,
        base_model_description=f"Project '{state.project.name}', {state.project.analysis_type}.",
        result=result,
        conclusions=conclusions,
    )
    markdown_text = render_optimization_report_markdown(report)

    st.download_button(
        "Download Markdown Report",
        data=markdown_text,
        file_name="optimization_report.md",
        mime="text/markdown",
    )
    st.text_area("Preview (Markdown)", value=markdown_text, height=300)
