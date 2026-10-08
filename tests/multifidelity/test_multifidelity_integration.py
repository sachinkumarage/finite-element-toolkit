"""Integration tests: femtoolkit.multifidelity <-> real (small) FEA simulations.

Mirrors the cantilever beam example (Euler-Bernoulli low fidelity, continuum FEA
high fidelity) end to end.
"""

from __future__ import annotations

import pytest

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.multifidelity.dataset import MultiFidelityDataset, MultiFidelitySample
from femtoolkit.multifidelity.discrepancy import train_discrepancy_surrogate
from femtoolkit.multifidelity.fidelity import (
    HIGH_FIDELITY,
    LOW_FIDELITY,
    AnalyticalFidelityModel,
    SimulationFidelityModel,
)
from femtoolkit.multifidelity.model import (
    FusionAcceptanceStatus,
    MultiFidelityModel,
    verify_fused_prediction,
)
from femtoolkit.multifidelity.validation import compare_fidelity_accuracy
from femtoolkit.studies.extractors import get_extractor
from femtoolkit.surrogate.workflows.training import TrainingConfig

_BEAM_LENGTH = 2.0
_BEAM_HEIGHT = 0.4
_TIP_LOAD = -4000.0


def _base_project() -> Project:
    project = Project(name="MF Integration Cantilever", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.width = _BEAM_LENGTH
    project.mesh.height = _BEAM_HEIGHT
    project.mesh.nx = 8
    project.mesh.ny = 2
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=_TIP_LOAD)]
    return project


def _analytical_tip_deflection(point: dict[str, float]) -> dict[str, float]:
    thickness = point["mesh.thickness"]
    moment_of_inertia = thickness * _BEAM_HEIGHT**3 / 12.0
    deflection = abs(_TIP_LOAD) * _BEAM_LENGTH**3 / (3 * 200e9 * moment_of_inertia)
    return {"maximum_displacement": deflection}


@pytest.mark.slow
def test_cantilever_multi_fidelity_fusion_improves_on_low_fidelity() -> None:
    base_project = _base_project()
    low_model = AnalyticalFidelityModel(
        name="Euler-Bernoulli beam", evaluate_fn=_analytical_tip_deflection, level=LOW_FIDELITY
    )
    high_model = SimulationFidelityModel(
        name="FEA cantilever", base_project=base_project,
        response_extractors={"maximum_displacement": get_extractor("maximum_displacement")},
        level=HIGH_FIDELITY,
    )

    dataset = MultiFidelityDataset(
        feature_names=["mesh.thickness"], response_names=["maximum_displacement"]
    )
    for i in range(10):
        thickness = 0.006 + 0.0015 * i
        point = {"mesh.thickness": thickness}
        dataset.add_sample(
            MultiFidelitySample(
                sample_id=str(i), inputs=point, low_result=low_model.evaluate(point),
                high_result=high_model.evaluate(point),
            )
        )

    discrepancy_model, report = train_discrepancy_surrogate(
        dataset, TrainingConfig(model_type="polynomial", model_kwargs={"degree": 2}, split_seed=0)
    )
    assert discrepancy_model.is_fitted
    assert report.test_metrics["maximum_displacement"].n_samples > 0

    mf_model = MultiFidelityModel(low_fidelity_model=low_model, discrepancy_model=discrepancy_model)
    held_out = [{"mesh.thickness": 0.011}]
    records = verify_fused_prediction(mf_model, high_model, held_out)
    assert records[0].status in (
        FusionAcceptanceStatus.IMPROVED,
        FusionAcceptanceStatus.NOT_IMPROVED,
    )
    assert records[0].high_fidelity_result

    comparison = compare_fidelity_accuracy(mf_model, dataset)
    assert comparison.n_samples == dataset.n_samples
    assert comparison.improves_on_low_fidelity("maximum_displacement") is True
