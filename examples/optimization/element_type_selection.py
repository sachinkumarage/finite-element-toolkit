"""Example: categorical design variable -- element formulation selection (Version 32).

**Why this example is not "steel vs. aluminum vs. titanium."** A
genuine multi-property material catalog choice (e.g. Steel/Aluminum/
Titanium) must change *several* project fields at once -- Young's
modulus, Poisson's ratio, and density -- atomically. The current
:class:`~femtoolkit.optimization.variables.DesignVariable` is
deliberately a single-path abstraction (one variable controls exactly
one dotted override path, the same convention
:class:`~femtoolkit.studies.scenarios.Scenario` uses), matching the
project's own guidance to avoid an unnecessarily complicated type
system. Extending it to atomically override a *group* of fields for one
categorical choice is real, useful future work, documented in
``docs/optimization.md`` -- it is not implemented here.

What this example demonstrates instead is a categorical design
variable that genuinely, honestly drives the real FEA solve through the
existing single-path override mechanism: ``mesh.element_type``, a
choice between the bilinear quadrilateral (``"quad"``) and the
constant-strain triangle (``"cst"``) element formulations. This is a
real discretization choice with a measurable effect on the computed
result, combined here with a continuous thickness variable to show a
mixed continuous/categorical design space.
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
    save_optimization_report,
)
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType
from femtoolkit.studies.extractors import get_extractor


def main() -> None:
    """Run the categorical element-type optimization and print/save its results."""
    print("Finite Element Toolkit")
    print("Version 32 -- Categorical Design Variable (Element Formulation)")
    print("=" * 65)

    base_project = Project(name="Element Type Selection", analysis_type="linear_static")
    base_project.material.youngs_modulus = 200e9
    base_project.material.poisson_ratio = 0.3
    base_project.material.density = 7850.0
    base_project.mesh.width = 2.0
    base_project.mesh.height = 0.4
    base_project.mesh.nx = 16
    base_project.mesh.ny = 4
    base_project.mesh.thickness = 0.008
    base_project.mesh.element_type = "quad"
    base_project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    base_project.loads = [LoadConfig(region="right", dof="Y", magnitude=-2500.0)]

    thickness = DesignVariable(
        name="thickness", path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=0.005, upper_bound=0.015, default_value=0.008, units="m",
    )
    element_type = DesignVariable(
        name="element_type", path="mesh.element_type",
        variable_type=DesignVariableType.CATEGORICAL, categories=["quad", "cst"],
        default_value="quad", description="Element formulation used to discretize the beam.",
    )

    displacement_objective = Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")), units="m",
    )
    stress_constraint = Constraint(
        name="stress_limit",
        evaluate=from_result_extractor(get_extractor("maximum_von_mises_stress")),
        relation=ConstraintRelation.LESS_EQUAL, limit=250e6, units="Pa",
    )

    problem = OptimizationProblem(
        name="element-type-selection", base_project=base_project,
        design_variables=[thickness, element_type], objectives=[displacement_objective],
        constraints=[stress_constraint],
        description="Minimize tip displacement over a mixed continuous/categorical design space.",
    )

    config = OptimizationConfig(
        algorithm="coordinate_search", max_evaluations=40, step_size=0.2, seed=3
    )
    result = OptimizationRunner().run(problem, config)

    print(f"\nBaseline: thickness={result.baseline.design_variables['thickness']:.4f} m, "
          f"element_type={result.baseline.design_variables['element_type']}")
    print(f"  displacement={result.baseline.objective_values['maximum_displacement']:.6e} m")

    best = result.best_feasible()
    print(f"\nBest feasible candidate: {best.design_id}")
    print(f"  thickness={best.design_variables['thickness']:.4f} m")
    print(f"  element_type={best.design_variables['element_type']}")
    print(f"  displacement={best.objective_values['maximum_displacement']:.6e} m")

    improvement = result.improvement_over_baseline()
    print(
        f"\nChange vs. baseline: {improvement['absolute_difference']:.6e} m "
        f"({improvement['percentage_change']:+.2f}%)"
    )
    print(f"Stop reason: {result.stop_reason.value}, {result.history.n_evaluations} evaluations.")

    report = build_optimization_report(
        title="Element Type Selection Report",
        summary=(
            "Minimizes tip displacement over a mixed continuous (thickness) and "
            "categorical (element formulation) design space."
        ),
        base_model_description=(
            "2.0 m x 0.4 m structural steel cantilever beam, fixed (X, Y) at the left "
            "edge, a 2500 N tip load on the right edge."
        ),
        result=result,
        conclusions=(
            "This example demonstrates a genuinely working categorical design variable. "
            "Full multi-property material catalog selection (e.g. Steel/Aluminum/"
            "Titanium, which must change several project fields atomically) is "
            "documented as future work rather than implemented here."
        ),
    )
    save_optimization_report(report, "examples/optimization/element_type_report.md", "markdown")
    print("\nSaved report: examples/optimization/element_type_report.md")


if __name__ == "__main__":
    main()
