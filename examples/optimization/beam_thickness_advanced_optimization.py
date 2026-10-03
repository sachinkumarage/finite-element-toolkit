"""Example: beam thickness optimization with differential evolution (Version 33).

**Procedure.** The same engineering problem as Version 32's
``beam_thickness_optimization.py`` -- minimize mass subject to a
displacement limit and a stress limit -- but searched with
**differential evolution**, a population-based algorithm, instead of
single-point coordinate search. Demonstrates that a Version 33
algorithm is a drop-in replacement for a Version 32 one: the same
`OptimizationProblem`, the same `OptimizationRunner`, only the
`OptimizationConfig.algorithm` and its population-specific settings
change.
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
    """Run the differential-evolution beam thickness optimization and print/save results."""
    print("Finite Element Toolkit")
    print("Version 33 -- Beam Thickness Optimization (Differential Evolution)")
    print("=" * 68)

    base_project = Project(
        name="Beam Thickness Optimization (DE)", analysis_type="linear_static"
    )
    base_project.material.youngs_modulus = 200e9
    base_project.material.poisson_ratio = 0.3
    base_project.material.density = 7850.0
    base_project.mesh.width = 2.0
    base_project.mesh.height = 0.4
    base_project.mesh.nx = 16
    base_project.mesh.ny = 4
    base_project.mesh.thickness = 0.015
    base_project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    base_project.loads = [LoadConfig(region="right", dof="Y", magnitude=-3000.0)]

    thickness = DesignVariable(
        name="thickness", path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=0.005, upper_bound=0.020, default_value=0.015, units="m",
    )

    mass_objective = Objective(
        name="mass", direction=ObjectiveDirection.MINIMIZE, evaluate=rectangular_mass, units="kg"
    )
    displacement_constraint = Constraint(
        name="displacement_limit",
        evaluate=from_result_extractor(get_extractor("maximum_displacement")),
        relation=ConstraintRelation.LESS_EQUAL, limit=0.005, units="m",
    )
    stress_constraint = Constraint(
        name="stress_limit",
        evaluate=from_result_extractor(get_extractor("maximum_von_mises_stress")),
        relation=ConstraintRelation.LESS_EQUAL, limit=250e6, units="Pa",
    )

    problem = OptimizationProblem(
        name="beam-thickness-de", base_project=base_project, design_variables=[thickness],
        objectives=[mass_objective], constraints=[displacement_constraint, stress_constraint],
        description=(
            "Minimize mass subject to displacement and stress limits "
            "(differential evolution)."
        ),
    )

    config = OptimizationConfig(
        algorithm="differential_evolution", population_size=10, max_evaluations=100,
        max_generations=20, mutation_factor=0.8, crossover_probability=0.9, seed=42,
        tolerance=1e-9, patience=80,
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
    print(
        f"Stop reason: {result.stop_reason.value}, {result.history.n_evaluations} evaluations "
        f"across {len({e.generation for e in result.history.evaluations})} generations."
    )

    report = build_optimization_report(
        title="Beam Thickness Optimization Report (Differential Evolution)",
        summary=(
            "Minimizes cantilever beam mass subject to displacement and stress limits, "
            "searched with differential evolution instead of a single-point algorithm."
        ),
        base_model_description=(
            "2.0 m x 0.4 m structural steel cantilever beam, fixed (X, Y) at the left "
            "edge, a 3000 N tip load on the right edge."
        ),
        result=result,
        conclusions=(
            f"The best feasible candidate found by differential evolution reduces mass by "
            f"{abs(improvement['percentage_change']):.1f}% relative to the baseline while "
            "satisfying both the displacement and stress limits. This is a quantitative "
            "comparison only -- derivative-free population search carries no guarantee of "
            "global optimality."
        ),
    )
    save_optimization_report(
        report, "examples/optimization/beam_thickness_advanced_report.md", "markdown"
    )
    print("\nSaved report: examples/optimization/beam_thickness_advanced_report.md")


if __name__ == "__main__":
    main()
