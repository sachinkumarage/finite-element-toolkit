"""Example: surrogate-assisted adaptive optimization of a cantilever beam (Version 36).

**Procedure.** Minimizes beam mass (thickness and width are the design variables)
subject to a maximum-displacement constraint, using
:class:`~femtoolkit.adaptive.study.AdaptiveStudy`:

.. code-block:: text

    Initial FEA Samples -> Train Surrogate -> Candidate Search (hybrid
    exploration/exploitation) -> High-Fidelity Verification -> Add Sample ->
    Retrain -> Repeat

Every iteration's result is printed as a clearly labeled surrogate prediction and
a separately labeled high-fidelity FEA verification -- the surrogate accelerates
the search, but the reported best design is always the best *verified* one.
"""

from __future__ import annotations

from femtoolkit.adaptive.refinement import RefinementConfig
from femtoolkit.adaptive.sampling import SamplingStrategy
from femtoolkit.adaptive.study import AdaptiveStudy
from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.optimization.constraints import Constraint, ConstraintRelation
from femtoolkit.optimization.objectives import Objective, ObjectiveDirection, from_result_extractor
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType
from femtoolkit.studies.extractors import get_extractor


def _mass_extractor(run) -> float | None:
    """Mass of the simulated rectangular domain: ``width * height * thickness * density``."""
    snapshot = run.configuration_snapshot
    if snapshot is None or snapshot.material.density is None:
        return None
    mesh = snapshot.mesh
    return mesh.width * mesh.height * mesh.thickness * snapshot.material.density


def build_base_project() -> Project:
    project = Project(name="Adaptive Cantilever Beam", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.height = 0.4
    project.mesh.nx = 10
    project.mesh.ny = 3
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-4000.0)]
    return project


def main() -> None:
    base_project = build_base_project()
    design_variables = [
        DesignVariable(
            name="thickness", path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
            lower_bound=0.006, upper_bound=0.020,
        ),
        DesignVariable(
            name="width", path="mesh.width", variable_type=DesignVariableType.CONTINUOUS,
            lower_bound=1.0, upper_bound=3.0,
        ),
    ]
    response_extractors = {
        "mass": _mass_extractor,
        "maximum_displacement": get_extractor("maximum_displacement"),
    }
    objective = Objective(
        name="mass", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(_mass_extractor),
    )
    constraint = Constraint(
        name="maximum_displacement",
        evaluate=from_result_extractor(get_extractor("maximum_displacement")),
        relation=ConstraintRelation.LESS_EQUAL, limit=0.003,
    )

    study = AdaptiveStudy(
        base_project=base_project,
        design_variables=design_variables,
        objective=objective,
        constraints=[constraint],
        response_extractors=response_extractors,
        model_type="polynomial",
        refinement_config=RefinementConfig(
            max_iterations=4, n_candidates=40, sampling_strategy=SamplingStrategy.HYBRID,
            exploration_weight=0.4, exploitation_weight=0.6, seed=0,
        ),
        random_seed=0,
    )

    print("Generating initial high-fidelity samples and training the surrogate...")
    result = study.run(n_initial_samples=10)

    print(f"\nRan {result.iteration_count} adaptive refinement iteration(s).")
    for step in result.iteration_history:
        predicted = step.candidate.evaluation.objective_values.get("mass")
        actual = step.verification.actual.get("mass")
        print(
            f"  Iteration {step.iteration}: surrogate prediction mass={predicted!r}, "
            f"high-fidelity FEA mass={actual!r}, status={step.status.value}"
        )

    print(f"\nTotal high-fidelity evaluations: {result.total_high_fidelity_evaluations}")
    print(f"Best verified design: {result.best_verified_design}")
    print(f"Best verified mass (kg): {result.best_verified_objective}")
    print(f"Engineering status: {result.status.value}")
    print(f"Stopping reason: {result.stopping_reason}")


if __name__ == "__main__":
    main()
