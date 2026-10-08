"""Optimization page: a plain (non-surrogate) Version 33 optimization run.

Reuses :mod:`femtoolkit.optimization` directly -- every candidate here is
evaluated with real, high-fidelity FEA. See the Surrogate-Assisted Optimization
page for the surrogate-accelerated workflow.
"""

from __future__ import annotations

import streamlit as st

from app.streamlit.plotting import objective_history_frame
from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.optimization.algorithms import OptimizationConfig
from femtoolkit.optimization.constraints import Constraint, ConstraintRelation
from femtoolkit.optimization.evaluation import DesignStatus
from femtoolkit.optimization.history import compute_convergence
from femtoolkit.optimization.objectives import Objective, ObjectiveDirection, from_result_extractor
from femtoolkit.optimization.problems import OptimizationProblem
from femtoolkit.optimization.runner import OptimizationRunner
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType
from femtoolkit.studies.extractors import get_extractor

_RESULT_KEY = "quickstart_optimization_result"


def _base_project() -> Project:
    project = Project(name="Quickstart Optimization Cantilever", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.width = 2.0
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
    """Render the Optimization page."""
    st.header("Optimization")
    st.caption(
        "Minimize cantilever beam thickness-driven displacement, subject to a stress "
        "limit -- every candidate is evaluated with real, high-fidelity FEA."
    )

    col_low, col_high = st.columns(2)
    lower_bound = col_low.number_input(
        "Thickness lower bound (m)", value=0.006, step=0.001, format="%.4f"
    )
    upper_bound = col_high.number_input(
        "Thickness upper bound (m)", value=0.020, step=0.001, format="%.4f"
    )

    stress_limit = st.number_input(
        "Maximum allowable stress (Pa)", value=1.5e8, step=1.0e7, format="%.2e"
    )
    algorithm = st.selectbox("Algorithm", options=["random_search", "coordinate_search"])
    max_evaluations = st.slider("Max evaluations", min_value=5, max_value=60, value=20)
    seed = st.number_input("Random seed", min_value=0, value=0, step=1)

    if st.button("Run Optimization", type="primary"):
        design_variable = DesignVariable(
            name="thickness", path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
            lower_bound=lower_bound, upper_bound=upper_bound,
        )
        objective = Objective(
            name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
            evaluate=from_result_extractor(get_extractor("maximum_displacement")),
        )
        constraint = Constraint(
            name="maximum_von_mises_stress",
            evaluate=from_result_extractor(get_extractor("maximum_von_mises_stress")),
            relation=ConstraintRelation.LESS_EQUAL, limit=stress_limit,
        )
        problem = OptimizationProblem(
            name="Quickstart Beam Optimization", base_project=_base_project(),
            design_variables=[design_variable], objectives=[objective], constraints=[constraint],
        )
        config = OptimizationConfig(
            algorithm=algorithm, max_evaluations=int(max_evaluations), seed=int(seed)
        )
        with st.spinner("Running optimization..."):
            result = OptimizationRunner().run(problem, config)
        st.session_state[_RESULT_KEY] = result

    result = st.session_state.get(_RESULT_KEY)
    if result is None:
        st.info("Configure the problem above and click 'Run Optimization'.")
        return

    objective = result.objectives[0]
    best = result.history.best_feasible(objective)

    st.subheader("Outcome")
    col_status, col_stop = st.columns(2)
    col_status.metric("Best design status", "FEASIBLE" if best is not None else "NONE FOUND")
    col_stop.metric("Stop reason", result.stop_reason.value)

    if best is not None:
        col_thickness, col_obj = st.columns(2)
        col_thickness.metric("Best thickness (m)", f"{best.design_variables['thickness']:.5f}")
        col_obj.metric(
            f"Best {objective.name}", f"{best.objective_values[objective.name]:.4e}"
        )

    feasible_count = sum(1 for e in result.history.evaluations if e.status is DesignStatus.FEASIBLE)
    st.write(f"Evaluations: {result.history.n_evaluations} total, {feasible_count} feasible.")

    steps = compute_convergence(result.history, objective)
    best_values = [step.best_value for step in steps if step.best_value is not None]
    if best_values:
        st.subheader("Objective History")
        st.line_chart(objective_history_frame(best_values, f"Best {objective.name}"))


__all__ = ["render"]
