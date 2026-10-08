"""Example: surrogate-assisted design search, verified with high-fidelity FEA (Version 35).

**Procedure.** Trains one polynomial surrogate that predicts both
maximum displacement and maximum stress from beam thickness, then uses
:class:`~femtoolkit.surrogate.workflows.evaluator.SurrogateEvaluator`
(the Version 35 evaluator foundation -- see that module's docstring for
why this is deliberately *not* wired into a Version 33 optimization
algorithm yet) to screen a candidate pool of thickness values entirely
from surrogate predictions, selecting the candidate with the lowest
predicted displacement that still satisfies a stress constraint. The
selected candidate is then, and only then, verified against real
high-fidelity FEA with
:func:`~femtoolkit.surrogate.workflows.verification.verify_against_high_fidelity`
-- every result printed below is explicitly labeled surrogate prediction
or high-fidelity verification, never blurred together (spec sections
29-31).
"""

from __future__ import annotations

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.optimization.constraints import Constraint, ConstraintRelation
from femtoolkit.optimization.objectives import Objective, ObjectiveDirection, from_result_extractor
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType
from femtoolkit.studies.extractors import get_extractor
from femtoolkit.studies.parameter_sweep import ParameterDefinition
from femtoolkit.surrogate.validation import EngineeringTolerance
from femtoolkit.surrogate.workflows.evaluator import SurrogateEvaluator
from femtoolkit.surrogate.workflows.training import (
    TrainingConfig,
    generate_training_dataset,
    train_surrogate,
)
from femtoolkit.surrogate.workflows.verification import verify_against_high_fidelity


def build_base_project() -> Project:
    project = Project(name="Surrogate-Assisted Beam Design", analysis_type="linear_static")
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


def main() -> None:
    base_project = build_base_project()
    stress_limit = 8.0e7  # Pa

    thickness = ParameterDefinition(
        path="mesh.thickness", label="Thickness", values=[0.006 + 0.001 * i for i in range(13)]
    )
    dataset = generate_training_dataset(
        base_project,
        [thickness],
        {
            "maximum_displacement": get_extractor("maximum_displacement"),
            "maximum_von_mises_stress": get_extractor("maximum_von_mises_stress"),
        },
    )
    model, _report = train_surrogate(
        dataset, TrainingConfig(model_type="polynomial", model_kwargs={"degree": 2}, split_seed=0)
    )

    design_variable = DesignVariable(
        name="thickness",
        path="mesh.thickness",
        variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=0.006,
        upper_bound=0.018,
    )
    objective = Objective(
        name="maximum_displacement",
        direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")),
    )
    constraint = Constraint(
        name="maximum_von_mises_stress",
        evaluate=from_result_extractor(get_extractor("maximum_von_mises_stress")),
        relation=ConstraintRelation.LESS_EQUAL,
        limit=stress_limit,
    )
    evaluator = SurrogateEvaluator(
        {"maximum_displacement": model, "maximum_von_mises_stress": model}
    )

    print(f"Screening candidates with the surrogate (backend={evaluator.backend.value})...")
    best = None
    for index, value in enumerate([0.0065 + 0.0005 * i for i in range(24)]):
        evaluation = evaluator.evaluate(
            f"candidate-{index}", {"thickness": value}, [design_variable], [objective], [constraint]
        )
        if evaluation.status.value != "feasible":
            continue
        predicted_displacement = evaluation.objective_values["maximum_displacement"]
        if best is None or predicted_displacement < best.objective_values["maximum_displacement"]:
            best = evaluation

    if best is None:
        print("No surrogate-feasible candidate found in the screened range.")
        return

    thickness_value = best.design_variables["thickness"]
    print(
        f"Surrogate-selected candidate: thickness={thickness_value:.5f} m, "
        f"predicted displacement={best.objective_values['maximum_displacement']:.4e} m "
        f"(surrogate prediction, metadata={best.metadata['surrogate_derived']})"
    )

    print("\nVerifying the selected candidate with high-fidelity FEA...")
    records = verify_against_high_fidelity(
        model,
        base_project,
        [{"mesh.thickness": thickness_value}],
        {
            "maximum_displacement": get_extractor("maximum_displacement"),
            "maximum_von_mises_stress": get_extractor("maximum_von_mises_stress"),
        },
        tolerances=[EngineeringTolerance("maximum_von_mises_stress", max_relative_error=0.10)],
    )
    record = records[0]
    print(
        f"High-fidelity verification: displacement={record.actual['maximum_displacement']:.4e} m, "
        f"stress={record.actual['maximum_von_mises_stress']:.4e} Pa, "
        f"acceptance={record.acceptance.value} "
        "(this is the real FEA result -- the final engineering acceptance decision)"
    )


if __name__ == "__main__":
    main()
