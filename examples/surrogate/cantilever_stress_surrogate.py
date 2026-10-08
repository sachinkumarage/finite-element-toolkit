"""Example: a surrogate for cantilever maximum stress, verified against FEA (Version 35).

**Procedure.** Trains a polynomial surrogate for maximum von Mises
stress from a thickness sweep, then compares its predictions against
freshly-run high-fidelity FEA on held-out thickness values the training
data never saw, using
:func:`~femtoolkit.surrogate.workflows.verification.verify_against_high_fidelity`
-- the core "surrogate vs. FEA" engineering check (spec section 28).
"""

from __future__ import annotations

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.studies.extractors import get_extractor
from femtoolkit.studies.parameter_sweep import ParameterDefinition
from femtoolkit.surrogate.validation import EngineeringTolerance
from femtoolkit.surrogate.workflows.training import (
    TrainingConfig,
    generate_training_dataset,
    train_surrogate,
)
from femtoolkit.surrogate.workflows.verification import verify_against_high_fidelity


def build_base_project() -> Project:
    project = Project(name="Cantilever Stress Surrogate", analysis_type="linear_static")
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
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-5000.0)]
    return project


def main() -> None:
    base_project = build_base_project()
    thickness = ParameterDefinition(
        path="mesh.thickness", label="Thickness", values=[0.005 + 0.001 * i for i in range(16)]
    )

    dataset = generate_training_dataset(
        base_project,
        [thickness],
        {"maximum_von_mises_stress": get_extractor("maximum_von_mises_stress")},
    )
    model, report = train_surrogate(
        dataset,
        TrainingConfig(
            model_type="polynomial",
            model_kwargs={"degree": 2},
            split_seed=0,
            tolerances=[EngineeringTolerance("maximum_von_mises_stress", max_relative_error=0.10)],
        ),
    )
    metrics = report.test_metrics["maximum_von_mises_stress"]
    print(f"Held-out test R^2 for maximum_von_mises_stress: {metrics.r2:.4f}")
    print(f"Meets configured engineering tolerance: {report.meets_engineering_tolerances}")

    held_out = [{"mesh.thickness": 0.0085}, {"mesh.thickness": 0.0135}, {"mesh.thickness": 0.019}]
    records = verify_against_high_fidelity(
        model,
        base_project,
        held_out,
        {"maximum_von_mises_stress": get_extractor("maximum_von_mises_stress")},
        tolerances=[EngineeringTolerance("maximum_von_mises_stress", max_relative_error=0.10)],
    )

    print("\nSurrogate prediction vs. high-fidelity FEA (held-out thicknesses):")
    for record in records:
        predicted = record.predicted.get("maximum_von_mises_stress")
        actual = record.actual.get("maximum_von_mises_stress")
        rel = record.relative_error.get("maximum_von_mises_stress")
        print(
            f"  thickness={record.design_point['mesh.thickness']:.4f} m: "
            f"surrogate prediction={predicted:.4e} Pa, high-fidelity verification={actual:.4e} Pa, "
            f"relative error={rel:.2%}, acceptance={record.acceptance.value}"
        )


if __name__ == "__main__":
    main()
