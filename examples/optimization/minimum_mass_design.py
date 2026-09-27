"""Example: minimum-mass cantilever beam design (Version 32).

**Procedure.** The reverse of the beam thickness optimization example:
here the objective is to **minimize mass** while satisfying both a
displacement limit and a stress limit. This demonstrates that the same
design-variable/objective/constraint framework applies regardless of
which physical quantity is being minimized -- the optimization engine
never hard-codes what "the objective" means.
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
    rectangular_mass,
    save_optimization_report,
)
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType
from femtoolkit.studies.extractors import get_extractor


def main() -> None:
    """Run the minimum-mass design optimization and print/save its results."""
    print("Finite Element Toolkit")
    print("Version 32 -- Minimum-Mass Design")
    print("=" * 45)

    base_project = Project(name="Minimum-Mass Design", analysis_type="linear_static")
    base_project.material.youngs_modulus = 200e9
    base_project.material.poisson_ratio = 0.3
    base_project.material.density = 7850.0
    base_project.mesh.width = 2.0
    base_project.mesh.height = 0.4
    base_project.mesh.nx = 16
    base_project.mesh.ny = 4
    base_project.mesh.thickness = 0.020  # start heavy, over-designed
    base_project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    base_project.loads = [LoadConfig(region="right", dof="Y", magnitude=-2000.0)]

    thickness = DesignVariable(
        name="thickness", path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=0.003, upper_bound=0.020, default_value=0.020, units="m",
    )

    mass_objective = Objective(
        name="mass", direction=ObjectiveDirection.MINIMIZE, evaluate=rectangular_mass, units="kg"
    )

    displacement_limit = 0.005  # 5 mm
    displacement_constraint = Constraint(
        name="displacement_limit",
        evaluate=from_result_extractor(get_extractor("maximum_displacement")),
        relation=ConstraintRelation.LESS_EQUAL, limit=displacement_limit, units="m",
    )
    stress_limit = 250e6
    stress_constraint = Constraint(
        name="stress_limit",
        evaluate=from_result_extractor(get_extractor("maximum_von_mises_stress")),
        relation=ConstraintRelation.LESS_EQUAL, limit=stress_limit, units="Pa",
    )

    problem = OptimizationProblem(
        name="minimum-mass-design", base_project=base_project, design_variables=[thickness],
        objectives=[mass_objective], constraints=[displacement_constraint, stress_constraint],
        description="Minimize mass subject to displacement and stress limits.",
    )

    config = OptimizationConfig(
        algorithm="coordinate_search", max_evaluations=60, step_size=0.1, seed=7
    )
    result = OptimizationRunner().run(problem, config)

    print(f"\nBaseline: thickness={result.baseline.design_variables['thickness']:.4f} m, "
          f"mass={result.baseline.objective_values['mass']:.3f} kg")
    for ce in result.baseline.constraint_evaluations:
        print(f"  {ce.name}: {ce.value:.4e} (limit {ce.limit:.4e}, satisfied={ce.satisfied})")

    best = result.best_feasible()
    print(f"\nBest feasible candidate: {best.design_id}")
    print(f"  thickness={best.design_variables['thickness']:.4f} m")
    print(f"  mass={best.objective_values['mass']:.3f} kg")
    for ce in best.constraint_evaluations:
        print(f"  {ce.name}: {ce.value:.4e} (limit {ce.limit:.4e}, satisfied={ce.satisfied})")

    improvement = result.improvement_over_baseline()
    print(
        f"\nMass change vs. baseline: {improvement['absolute_difference']:.3f} kg "
        f"({improvement['percentage_change']:+.2f}%)"
    )
    print(f"Stop reason: {result.stop_reason.value}, {result.history.n_evaluations} evaluations.")

    report = build_optimization_report(
        title="Minimum-Mass Design Report",
        summary="Minimizes cantilever beam mass subject to displacement and stress limits.",
        base_model_description=(
            "2.0 m x 0.4 m structural steel cantilever beam, fixed (X, Y) at the left "
            "edge, a 2000 N tip load on the right edge."
        ),
        result=result,
        conclusions=(
            f"The best feasible candidate reduces mass by "
            f"{abs(improvement['percentage_change']):.1f}% relative to the over-designed "
            "baseline while still satisfying both the displacement and stress limits."
        ),
    )
    save_optimization_report(report, "examples/optimization/minimum_mass_report.md", "markdown")
    print("\nSaved report: examples/optimization/minimum_mass_report.md")


if __name__ == "__main__":
    main()
