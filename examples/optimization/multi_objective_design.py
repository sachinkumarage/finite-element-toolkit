"""Example: multi-objective cantilever beam design (Version 32).

**Procedure.** Minimizes mass AND maximum tip displacement
simultaneously -- two objectives that trade off against each other
(a thinner beam is lighter but deflects more). Generates candidate
designs, computes the non-dominated (Pareto) set, and visualizes the
Pareto front. No single "best overall" design is selected: choosing
between mass and stiffness is an engineering judgment call outside
the scope of this framework.
"""

from __future__ import annotations

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.optimization import (
    Objective,
    ObjectiveDirection,
    OptimizationConfig,
    OptimizationProblem,
    OptimizationRunner,
    build_optimization_report,
    from_result_extractor,
    plot_pareto_front,
    rectangular_mass,
    save_optimization_report,
)
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType
from femtoolkit.studies.extractors import get_extractor


def main() -> None:
    """Run the multi-objective optimization and print/save its results."""
    print("Finite Element Toolkit")
    print("Version 32 -- Multi-Objective Design (Mass vs. Displacement)")
    print("=" * 60)

    base_project = Project(name="Multi-Objective Design", analysis_type="linear_static")
    base_project.material.youngs_modulus = 200e9
    base_project.material.poisson_ratio = 0.3
    base_project.material.density = 7850.0
    base_project.mesh.width = 2.0
    base_project.mesh.height = 0.4
    base_project.mesh.nx = 16
    base_project.mesh.ny = 4
    base_project.mesh.thickness = 0.010
    base_project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    base_project.loads = [LoadConfig(region="right", dof="Y", magnitude=-2500.0)]

    thickness = DesignVariable(
        name="thickness", path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=0.003, upper_bound=0.020, default_value=0.010, units="m",
    )

    mass_objective = Objective(
        name="mass", direction=ObjectiveDirection.MINIMIZE, evaluate=rectangular_mass, units="kg"
    )
    displacement_objective = Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")), units="m",
    )

    problem = OptimizationProblem(
        name="multi-objective-design", base_project=base_project, design_variables=[thickness],
        objectives=[mass_objective, displacement_objective],
        description="Minimize mass and maximum tip displacement simultaneously.",
    )

    config = OptimizationConfig(algorithm="random_search", max_evaluations=40, seed=11)
    result = OptimizationRunner().run(problem, config)

    print(f"\nBaseline: thickness={result.baseline.design_variables['thickness']:.4f} m")
    print(f"  mass={result.baseline.objective_values['mass']:.3f} kg")
    print(
        f"  displacement={result.baseline.objective_values['maximum_displacement']:.6e} m"
    )

    print(
        f"\n{result.history.n_evaluations} evaluations recorded. "
        f"Stop reason: {result.stop_reason.value}"
    )
    print(f"Feasible: {len(result.history.feasible_evaluations())}")

    front = result.pareto_front()
    print(f"\nNon-dominated (Pareto) set: {len(front)} design(s)")
    print("No single design is labeled 'best' -- each trades mass against stiffness.")
    for evaluation in sorted(front, key=lambda e: e.objective_values["mass"]):
        print(
            f"  {evaluation.design_id}: thickness="
            f"{evaluation.design_variables['thickness']:.4f} m, "
            f"mass={evaluation.objective_values['mass']:.3f} kg, "
            f"displacement={evaluation.objective_values['maximum_displacement']:.6e} m"
        )

    figure = plot_pareto_front(
        result.history.evaluations, [mass_objective, displacement_objective], front,
        baseline=result.baseline,
    )
    plot_path = "examples/optimization/multi_objective_pareto_front.png"
    figure.savefig(plot_path)
    print(f"\nSaved Pareto front plot: {plot_path}")

    report = build_optimization_report(
        title="Multi-Objective Design Report",
        summary="Minimizes mass and maximum tip displacement simultaneously.",
        base_model_description=(
            "2.0 m x 0.4 m structural steel cantilever beam, fixed (X, Y) at the left "
            "edge, a 2500 N tip load on the right edge."
        ),
        result=result,
        plot_paths=[plot_path],
        conclusions=(
            f"The search produced {len(front)} non-dominated design(s) trading mass "
            "against tip displacement. Selecting a single preferred design from this "
            "set requires an engineering judgment about the relative importance of "
            "mass and stiffness that this framework does not make automatically."
        ),
    )
    save_optimization_report(report, "examples/optimization/multi_objective_report.md", "markdown")
    print("Saved report: examples/optimization/multi_objective_report.md")


if __name__ == "__main__":
    main()
