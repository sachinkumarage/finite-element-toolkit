"""Example: multi-fidelity cantilever beam deflection (Version 37).

**Procedure.** Compares a cheap Euler-Bernoulli analytical beam formula (low
fidelity -- ignores shear deformation) against the existing continuum FEA model
(high fidelity), trains a discrepancy surrogate on
``delta(x) = y_H(x) - y_L(x)``, and reports whether the fused prediction
``y_hat_H(x) = y_L(x) + delta_hat(x)`` is actually more accurate than the raw
low-fidelity model on held-out thickness values.
"""

from __future__ import annotations

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.multifidelity.dataset import MultiFidelityDataset, MultiFidelitySample
from femtoolkit.multifidelity.discrepancy import train_discrepancy_surrogate
from femtoolkit.multifidelity.fidelity import (
    HIGH_FIDELITY,
    LOW_FIDELITY,
    AnalyticalFidelityModel,
    SimulationFidelityModel,
    summarize_costs,
)
from femtoolkit.multifidelity.model import MultiFidelityModel, verify_fused_prediction
from femtoolkit.multifidelity.validation import compare_fidelity_accuracy
from femtoolkit.studies.extractors import get_extractor
from femtoolkit.surrogate.workflows.training import TrainingConfig

_BEAM_LENGTH = 2.0  # mesh.width
_BEAM_HEIGHT = 0.4  # mesh.height (fixed; the bending-plane cross-section dimension)
_TIP_LOAD = -4000.0


def build_base_project() -> Project:
    project = Project(name="Multi-Fidelity Cantilever", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.width = _BEAM_LENGTH
    project.mesh.height = _BEAM_HEIGHT
    project.mesh.nx = 10
    project.mesh.ny = 3
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=_TIP_LOAD)]
    return project


def analytical_tip_deflection(point: dict[str, float]) -> dict[str, float]:
    """Euler-Bernoulli cantilever tip deflection: ``P * L^3 / (3 * E * I)``.

    ``I = thickness * height^3 / 12`` -- ``thickness`` (the out-of-plane depth)
    enters linearly, matching how the 2D plane-stress FEA model's stiffness scales
    with thickness; ``height`` (the in-plane bending dimension) is held fixed.
    Deliberately ignores shear deformation -- the gap between this and the real
    FEA result is exactly the discrepancy this example corrects for.
    """
    thickness = point["mesh.thickness"]
    youngs_modulus = 200e9
    moment_of_inertia = thickness * _BEAM_HEIGHT**3 / 12.0
    deflection = abs(_TIP_LOAD) * _BEAM_LENGTH**3 / (3 * youngs_modulus * moment_of_inertia)
    return {"maximum_displacement": deflection}


def main() -> None:
    base_project = build_base_project()
    low_model = AnalyticalFidelityModel(
        name="Euler-Bernoulli beam", evaluate_fn=analytical_tip_deflection, level=LOW_FIDELITY
    )
    high_model = SimulationFidelityModel(
        name="FEA cantilever", base_project=base_project,
        response_extractors={"maximum_displacement": get_extractor("maximum_displacement")},
        level=HIGH_FIDELITY,
    )

    print("Cost comparison:")
    for entry in summarize_costs([low_model, high_model]):
        print(f"  {entry['name']} ({entry['level']}): estimated_cost={entry['estimated_cost']}")

    dataset = MultiFidelityDataset(
        feature_names=["mesh.thickness"], response_names=["maximum_displacement"]
    )
    training_thicknesses = [0.006 + 0.001 * i for i in range(12)]
    for index, thickness in enumerate(training_thicknesses):
        point = {"mesh.thickness": thickness}
        dataset.add_sample(
            MultiFidelitySample(
                sample_id=str(index), inputs=point, low_result=low_model.evaluate(point),
                high_result=high_model.evaluate(point),
            )
        )
    print(f"\nCollected {len(dataset.paired_samples())} paired low-/high-fidelity samples.")

    discrepancy_model, report = train_discrepancy_surrogate(
        dataset, TrainingConfig(model_type="polynomial", model_kwargs={"degree": 2}, split_seed=0)
    )
    r_squared = report.test_metrics["maximum_displacement"].r2
    print(f"Discrepancy surrogate held-out R^2: {r_squared:.4f}")

    mf_model = MultiFidelityModel(low_fidelity_model=low_model, discrepancy_model=discrepancy_model)

    held_out = [{"mesh.thickness": 0.0085}, {"mesh.thickness": 0.013}, {"mesh.thickness": 0.019}]
    records = verify_fused_prediction(mf_model, high_model, held_out)
    print("\nFused prediction vs. real high-fidelity FEA (held-out thicknesses):")
    for record in records:
        thickness = record.design_point["mesh.thickness"]
        print(
            f"  thickness={thickness:.4f} m: LOW-FIDELITY RESULT={record.low_fidelity_result} "
            f"MULTI-FIDELITY PREDICTION={record.fused_prediction} "
            f"HIGH-FIDELITY FEA={record.high_fidelity_result} "
            f"status={record.status.value}"
        )

    comparison = compare_fidelity_accuracy(mf_model, dataset)
    low_rmse = comparison.low_fidelity_metrics["maximum_displacement"].rmse
    fused_rmse = comparison.fused_metrics["maximum_displacement"].rmse
    print(f"\nLow-fidelity-only RMSE: {low_rmse:.4e} m")
    print(f"Fused-model RMSE:       {fused_rmse:.4e} m")
    improves = comparison.improves_on_low_fidelity("maximum_displacement")
    print(f"Fusion improves accuracy: {improves}")


if __name__ == "__main__":
    main()
