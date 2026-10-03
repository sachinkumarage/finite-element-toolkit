"""Example: robust (uncertainty-aware) beam thickness design (Version 33).

**Procedure.** Introduces uncertainty in material properties (Young's
modulus) and the applied load, then optimizes beam thickness to
control two uncertainty-aware quantities instead of their deterministic
equivalents:

- **Objective:** minimize the *mean* tip displacement across the
  uncertainty distribution, not a single deterministic value.
- **Constraint:** the *95th percentile* von Mises stress must stay
  within the allowable stress -- a high-percentile response, not the
  mean, is the appropriate quantity to control against an occasional
  worst-case combination of load and material properties.

Every "evaluation" here is actually a small Monte Carlo study (ten
samples) at that design point, so the nested-cost safeguard
(`RobustDesignConfig.maximum_total_evaluations`) is set explicitly and
the run's total estimated FEA count is printed before execution.
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
    RobustDesignConfig,
    build_optimization_report,
    estimate_total_fea_count,
    robust_constraint_statistic,
    robust_objective_statistic,
    save_optimization_report,
)
from femtoolkit.optimization.problems import OptimizationMode
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType
from femtoolkit.uncertainty.distributions import NormalDistribution, UniformDistribution
from femtoolkit.uncertainty.parameters import UncertainParameter, UncertaintyCategory


def _build_uncertain_parameters(context):
    """Material and load uncertainty sampled around the current design point."""
    return [
        UncertainParameter(
            path="material.youngs_modulus", label="Young's Modulus",
            distribution=NormalDistribution(mean_value=200e9, std_value=8e9),
            units="Pa", category=UncertaintyCategory.ALEATORY, physical_lower_bound=1.0,
        ),
        UncertainParameter(
            path="loads.0.magnitude", label="Tip Load",
            distribution=UniformDistribution(low=-3500.0, high=-2500.0),
            units="N", category=UncertaintyCategory.ALEATORY,
        ),
    ]


def main() -> None:
    """Run the robust beam thickness optimization and print/save its results."""
    print("Finite Element Toolkit")
    print("Version 33 -- Robust Beam Design")
    print("=" * 45)

    base_project = Project(name="Robust Beam Design", analysis_type="linear_static")
    base_project.material.youngs_modulus = 200e9
    base_project.material.poisson_ratio = 0.3
    base_project.material.density = 7850.0
    base_project.mesh.width = 2.0
    base_project.mesh.height = 0.4
    base_project.mesh.nx = 12
    base_project.mesh.ny = 3
    base_project.mesh.thickness = 0.012
    base_project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    base_project.loads = [LoadConfig(region="right", dof="Y", magnitude=-3000.0)]

    thickness = DesignVariable(
        name="thickness", path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=0.006, upper_bound=0.020, default_value=0.012, units="m",
    )

    robust_config = RobustDesignConfig(
        uncertainty_enabled=True,
        sampling_method="latin_hypercube",
        sample_count=10,
        random_seed=42,
        objective_statistic="mean",
        constraint_statistic="percentile",
        percentile=95.0,
        maximum_total_evaluations=2000,
    )

    mean_displacement_objective = Objective(
        name="mean_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=robust_objective_statistic(
            _build_uncertain_parameters, "maximum_displacement", robust_config
        ),
        units="m",
    )
    p95_stress_constraint = Constraint(
        name="p95_stress_limit",
        evaluate=robust_constraint_statistic(
            _build_uncertain_parameters, "maximum_von_mises_stress", robust_config
        ),
        relation=ConstraintRelation.LESS_EQUAL, limit=250e6, units="Pa",
    )

    problem = OptimizationProblem(
        name="robust-beam-design", base_project=base_project, design_variables=[thickness],
        objectives=[mean_displacement_objective], constraints=[p95_stress_constraint],
        mode=OptimizationMode.UNCERTAINTY_AWARE,
        description=(
            "Minimize mean tip displacement subject to a 95th-percentile stress limit, "
            "under Young's modulus and load uncertainty."
        ),
    )

    config = OptimizationConfig(
        algorithm="coordinate_search", max_evaluations=10, step_size=0.2, seed=1
    )
    print(
        f"\nEstimated total FEA evaluations: up to "
        f"{estimate_total_fea_count(config.max_evaluations, robust_config)} "
        f"({config.max_evaluations} optimization evaluations x "
        f"{robust_config.sample_count} uncertainty samples each)"
    )

    result = OptimizationRunner().run(problem, config, robust_config=robust_config)

    print(f"\nBaseline: thickness={result.baseline.design_variables['thickness']:.4f} m")
    print(f"  mean displacement={result.baseline.objective_values['mean_displacement']:.6e} m")
    for ce in result.baseline.constraint_evaluations:
        print(
            f"  {ce.name}: p95={ce.value:.4e} Pa (limit {ce.limit:.4e} Pa, "
            f"satisfied={ce.satisfied})"
        )

    best = result.best_feasible()
    print(f"\nBest feasible candidate: {best.design_id}")
    print(f"  thickness={best.design_variables['thickness']:.4f} m")
    print(f"  mean displacement={best.objective_values['mean_displacement']:.6e} m")
    for ce in best.constraint_evaluations:
        print(f"  {ce.name}: p95={ce.value:.4e} Pa (satisfied={ce.satisfied})")

    improvement = result.improvement_over_baseline()
    print(
        f"\nChange vs. baseline: {improvement['absolute_difference']:.6e} m "
        f"({improvement['percentage_change']:+.2f}%)"
    )
    print(
        "Every statistic above is an empirical estimate from a 10-sample Monte Carlo "
        "study at each design point -- not a rigorous reliability probability."
    )

    report = build_optimization_report(
        title="Robust Beam Design Report",
        summary=(
            "Minimizes mean tip displacement subject to a 95th-percentile stress "
            "constraint, under material and load uncertainty."
        ),
        base_model_description=(
            "2.0 m x 0.4 m structural steel cantilever beam, fixed (X, Y) at the left "
            "edge, a nominally 3000 N tip load (uniformly uncertain 2500-3500 N) on the "
            "right edge; Young's modulus normally distributed (mean 200 GPa, std 8 GPa)."
        ),
        result=result,
        conclusions=(
            "The best feasible candidate reduces mean displacement relative to the "
            "baseline while keeping the estimated 95th-percentile stress within the "
            "allowable limit, based on a 10-sample Monte Carlo study per design point."
        ),
    )
    save_optimization_report(report, "examples/optimization/robust_beam_report.md", "markdown")
    print("\nSaved report: examples/optimization/robust_beam_report.md")


if __name__ == "__main__":
    main()
