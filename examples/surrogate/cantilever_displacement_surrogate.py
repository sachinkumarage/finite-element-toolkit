"""Example: a surrogate for cantilever maximum displacement (Version 35).

**Procedure.** Sweeps beam thickness and Young's modulus over a small
design-of-experiments grid, generates the corresponding high-fidelity
FEA training data with
:func:`~femtoolkit.surrogate.workflows.training.generate_training_dataset`
(Version 30 parameter sweeps, reused unchanged), trains a polynomial
surrogate for maximum displacement with
:func:`~femtoolkit.surrogate.workflows.training.train_surrogate`, and
reports its held-out accuracy.
"""

from __future__ import annotations

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.studies.extractors import get_extractor
from femtoolkit.studies.parameter_sweep import ParameterDefinition
from femtoolkit.surrogate.workflows.training import (
    TrainingConfig,
    generate_training_dataset,
    train_surrogate,
)


def build_base_project() -> Project:
    project = Project(name="Cantilever Displacement Surrogate", analysis_type="linear_static")
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
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-3000.0)]
    return project


def main() -> None:
    base_project = build_base_project()
    thickness = ParameterDefinition(
        path="mesh.thickness", label="Thickness", values=[0.006 + 0.002 * i for i in range(8)]
    )
    modulus = ParameterDefinition(
        path="material.youngs_modulus",
        label="Young's modulus",
        values=[150e9, 200e9, 250e9],
    )

    dataset = generate_training_dataset(
        base_project,
        [thickness, modulus],
        {"maximum_displacement": get_extractor("maximum_displacement")},
    )
    print(f"Training dataset: {dataset.n_samples} snapshots, features={dataset.feature_names}")

    model, report = train_surrogate(
        dataset,
        TrainingConfig(model_type="polynomial", model_kwargs={"degree": 2}, split_seed=0),
    )
    metrics = report.test_metrics["maximum_displacement"]
    print(
        f"Held-out test metrics for maximum_displacement: "
        f"MAE={metrics.mae:.3e} m, RMSE={metrics.rmse:.3e} m, R^2={metrics.r2:.4f}"
    )

    prediction = model.predict_point({"mesh.thickness": 0.01, "material.youngs_modulus": 200e9})
    print(
        f"Surrogate prediction at thickness=0.01 m, E=200 GPa: "
        f"{prediction.values['maximum_displacement']:.6e} m "
        f"(domain status: {prediction.domain_status.value})"
    )


if __name__ == "__main__":
    main()
