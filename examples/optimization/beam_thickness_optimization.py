"""Example: cantilever beam thickness optimization (Version 32).

**Procedure.** Defines the baseline design, one continuous design
variable (plate thickness, 5-20 mm), one objective (minimize tip
displacement), and one constraint (maximum von Mises stress must not
exceed an allowable value). Evaluates the baseline, runs coordinate
search, records every evaluation, compares the best feasible candidate
against the baseline, and generates a twenty-section optimization
report. No result is hard-coded -- everything below is computed from
the search itself.
"""

from __future__ import annotations

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.optimization import (
    Constraint,
    ConstraintRelation,
    Objective,
    ObjectiveDirection,
    OptimizationConfig,
    OptimizationProblem,
    OptimizationRunner,
    build_optimization_report,
    from_result_extractor,
    plot_constraint_violation_history,
    plot_design_variable_history,
    plot_objective_history,
    save_optimization_report,
)
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType
from femtoolkit.studies.extractors import get_extractor


def main() -> None:
    """Run the beam thickness optimization and print/save its results."""
    print("Finite Element Toolkit")
    print("Version 32 -- Beam Thickness Optimization")
    print("=" * 45)

    # 1. Baseline definition: a thin, under-designed cantilever beam.
    base_project = Project(name="Beam Thickness Optimization", analysis_type="linear_static")
    base_project.material.youngs_modulus = 200e9
    base_project.material.poisson_ratio = 0.3
    base_project.material.density = 7850.0
    base_project.mesh.width = 2.0
    base_project.mesh.height = 0.4
    base_project.mesh.nx = 16
    base_project.mesh.ny = 4
    base_project.mesh.thickness = 0.005
    base_project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    base_project.loads = [LoadConfig(region="right", dof="Y", magnitude=-3000.0)]

    # 2. Design variable: plate thickness.
    thickness = DesignVariable(
        name="thickness", path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=0.005, upper_bound=0.020, default_value=0.005, units="m",
        description="Plate thickness.",
    )

    # 3. Objective: minimize tip displacement.
    displacement_objective = Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")), units="m",
    )

    # 4. Constraint: stay within the material's allowable stress.
    allowable_stress = 250e6
    stress_constraint = Constraint(
        name="stress_limit",
        evaluate=from_result_extractor(get_extractor("maximum_von_mises_stress")),
        relation=ConstraintRelation.LESS_EQUAL, limit=allowable_stress, units="Pa",
        description="Maximum von Mises stress must not exceed the allowable stress.",
    )

    problem = OptimizationProblem(
        name="beam-thickness-optimization", base_project=base_project,
        design_variables=[thickness], objectives=[displacement_objective],
        constraints=[stress_constraint],
        description="Minimize tip displacement of a cantilever beam by varying thickness.",
    )

    # 5. Run optimization.
    config = OptimizationConfig(
        algorithm="coordinate_search", max_evaluations=60, step_size=0.15, seed=42
    )
    result = OptimizationRunner().run(problem, config)

    # 6. Evaluation history is already recorded in result.history.
    print(f"\nBaseline: thickness={result.baseline.design_variables['thickness']:.4f} m")
    print(f"  status={result.baseline.status.value}")
    print(f"  displacement={result.baseline.objective_values['maximum_displacement']:.6e} m")
    for ce in result.baseline.constraint_evaluations:
        print(f"  {ce.name}: {ce.value:.4e} Pa (limit {ce.limit:.4e} Pa, satisfied={ce.satisfied})")

    print(
        f"\n{result.history.n_evaluations} evaluations recorded. "
        f"Stop reason: {result.stop_reason.value}"
    )
    print(f"Feasible: {len(result.history.feasible_evaluations())}, "
          f"Infeasible: {len(result.history.infeasible_evaluations())}, "
          f"Failed: {len(result.history.failed_evaluations())}")

    # 7. Compare candidate designs.
    best = result.best_feasible()
    print(f"\nBest feasible candidate: {best.design_id}")
    print(f"  thickness={best.design_variables['thickness']:.4f} m")
    print(f"  displacement={best.objective_values['maximum_displacement']:.6e} m")

    improvement = result.improvement_over_baseline()
    print(
        f"\nChange vs. baseline: {improvement['absolute_difference']:.6e} m "
        f"({improvement['percentage_change']:+.2f}%)"
    )
    print(
        "This is a quantitative comparison only -- no claim of a global or guaranteed "
        "optimum is made."
    )

    # 8. Generate an optimization report.
    plot_paths = []
    for name, figure in (
        ("objective", plot_objective_history(result.history, displacement_objective)),
        ("constraint", plot_constraint_violation_history(result.history)),
        ("thickness", plot_design_variable_history(result.history, thickness)),
    ):
        path = f"examples/optimization/beam_thickness_{name}_history.png"
        figure.savefig(path)
        plot_paths.append(path)

    report = build_optimization_report(
        title="Beam Thickness Optimization Report",
        summary=(
            "Minimizes cantilever beam tip displacement by varying plate thickness, "
            "subject to a maximum von Mises stress constraint."
        ),
        base_model_description=(
            "2.0 m x 0.4 m structural steel cantilever beam, fixed (X, Y) at the left "
            "edge, a 3000 N tip load on the right edge."
        ),
        result=result,
        plot_paths=plot_paths,
        conclusions=(
            f"The best feasible candidate reduces maximum displacement by "
            f"{abs(improvement['percentage_change']):.1f}% relative to the baseline, "
            f"at a thickness of {best.design_variables['thickness']:.4f} m -- the search "
            "moved toward the thickest allowed design, as expected when minimizing "
            "displacement with no mass penalty. See the multi-objective example for the "
            "displacement-vs-mass trade-off this optimization ignores."
        ),
    )
    save_optimization_report(report, "examples/optimization/beam_thickness_report.md", "markdown")
    print("\nSaved report: examples/optimization/beam_thickness_report.md")


if __name__ == "__main__":
    main()
