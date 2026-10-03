"""Example: multi-objective cantilever beam design with NSGA-II (Version 33).

**Procedure.** The same mass-vs-displacement trade-off as Version 32's
``multi_objective_design.py``, but searched with **NSGA-II**
(non-dominated sorting + crowding distance) instead of plain random
sampling -- a population specifically evolved toward the trade-off
curve, rather than hoped to land on it by chance. As with every
multi-objective example in this toolkit, no single design from the
resulting Pareto front is selected as "best"; the front is reported in
full so an engineer can weigh mass against stiffness against real
requirements this toolkit has no way to know.
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
    """Run the NSGA-II multi-objective optimization and print/save its results."""
    print("Finite Element Toolkit")
    print("Version 33 -- Multi-Objective Design (NSGA-II)")
    print("=" * 50)

    base_project = Project(name="NSGA-II Multi-Objective Design", analysis_type="linear_static")
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
        name="nsga2-multi-objective-design", base_project=base_project,
        design_variables=[thickness], objectives=[mass_objective, displacement_objective],
        description="Minimize mass and maximum tip displacement simultaneously (NSGA-II).",
    )

    config = OptimizationConfig(
        algorithm="nsga2", population_size=16, max_evaluations=160, max_generations=10,
        crossover_probability=0.9, mutation_probability=0.1, tournament_size=3, seed=11,
        tolerance=1e-9, patience=150,
    )
    result = OptimizationRunner().run(problem, config)

    print(f"\nBaseline: thickness={result.baseline.design_variables['thickness']:.4f} m")
    print(f"  mass={result.baseline.objective_values['mass']:.3f} kg")
    print(
        f"  displacement={result.baseline.objective_values['maximum_displacement']:.6e} m"
    )

    print(
        f"\n{result.history.n_evaluations} evaluations recorded across "
        f"{len({e.generation for e in result.history.evaluations})} generations. "
        f"Stop reason: {result.stop_reason.value}"
    )
    print(f"Feasible: {len(result.history.feasible_evaluations())}")

    front = result.pareto_front()
    print(f"\nNon-dominated (Pareto) set: {len(front)} design(s)")
    print("No single design is labeled 'best' -- each trades mass against stiffness.")
    for evaluation in sorted(front, key=lambda e: e.objective_values["mass"])[:10]:
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
    plot_path = "examples/optimization/multi_objective_nsga2_pareto_front.png"
    figure.savefig(plot_path)
    print(f"\nSaved Pareto front plot: {plot_path}")

    report = build_optimization_report(
        title="Multi-Objective Design Report (NSGA-II)",
        summary="Minimizes mass and maximum tip displacement simultaneously, using NSGA-II.",
        base_model_description=(
            "2.0 m x 0.4 m structural steel cantilever beam, fixed (X, Y) at the left "
            "edge, a 2500 N tip load on the right edge."
        ),
        result=result,
        plot_paths=[plot_path],
        conclusions=(
            f"NSGA-II produced {len(front)} non-dominated design(s) tracing the mass-"
            "displacement trade-off curve. Selecting a single preferred design from this "
            "set requires an engineering judgment about the relative importance of mass "
            "and stiffness that this framework does not make automatically."
        ),
    )
    save_optimization_report(
        report, "examples/optimization/multi_objective_nsga2_report.md", "markdown"
    )
    print("Saved report: examples/optimization/multi_objective_nsga2_report.md")


if __name__ == "__main__":
    main()
