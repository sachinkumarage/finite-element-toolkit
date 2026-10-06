"""Tests for femtoolkit.optimization.objectives."""

import pytest

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.context import DesignContext
from femtoolkit.optimization.objectives import (
    Objective,
    ObjectiveDirection,
    from_result_extractor,
    rectangular_mass,
    robust_objective_mean,
    validate_unique_objective_names,
)
from femtoolkit.runs.manager import SimulationRunManager
from femtoolkit.studies.extractors import get_extractor
from femtoolkit.uncertainty.distributions import NormalDistribution
from femtoolkit.uncertainty.parameters import UncertainParameter


def _base_project() -> Project:
    project = Project(name="Objective Test", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.width, project.mesh.height, project.mesh.thickness = 2.0, 0.4, 0.02
    project.mesh.nx, project.mesh.ny = 8, 2
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-1000.0)]
    return project


def test_objective_requires_nonempty_name() -> None:
    with pytest.raises(ValidationError):
        Objective(name="", direction=ObjectiveDirection.MINIMIZE, evaluate=lambda ctx: 1.0)


def test_objective_direction_values() -> None:
    assert ObjectiveDirection.MINIMIZE.value == "minimize"
    assert ObjectiveDirection.MAXIMIZE.value == "maximize"


def test_validate_unique_objective_names_rejects_duplicates() -> None:
    a = Objective(name="a", direction=ObjectiveDirection.MINIMIZE, evaluate=lambda ctx: 1.0)
    dup = Objective(name="a", direction=ObjectiveDirection.MAXIMIZE, evaluate=lambda ctx: 1.0)
    with pytest.raises(ValidationError):
        validate_unique_objective_names([a, dup])


def test_from_result_extractor_reads_run_result() -> None:
    project = _base_project()
    run = SimulationRunManager().execute(project)
    context = DesignContext(design_variables={}, project=project, run=run)
    objective_fn = from_result_extractor(get_extractor("maximum_displacement"))
    assert objective_fn(context) == run.result.summary.maximum_displacement


def test_rectangular_mass_matches_hand_calculation() -> None:
    project = _base_project()
    run = SimulationRunManager().execute(project)
    context = DesignContext(design_variables={}, project=project, run=run)
    expected = 2.0 * 0.4 * 0.02 * 7850.0
    assert rectangular_mass(context) == pytest.approx(expected)


def test_rectangular_mass_returns_none_without_density() -> None:
    project = _base_project()
    project.material.density = None
    run = SimulationRunManager().execute(project)
    context = DesignContext(design_variables={}, project=project, run=run)
    assert rectangular_mass(context) is None


def test_robust_objective_mean_returns_positive_value() -> None:
    project = _base_project()
    run = SimulationRunManager().execute(project)
    context = DesignContext(design_variables={}, project=project, run=run)

    def build_parameters(ctx: DesignContext) -> list[UncertainParameter]:
        return [
            UncertainParameter(
                path="material.youngs_modulus",
                label="E",
                distribution=NormalDistribution(
                    ctx.project.material.youngs_modulus,
                    ctx.project.material.youngs_modulus * 0.02,
                ),
                physical_lower_bound=0.0,
            )
        ]

    objective_fn = robust_objective_mean(
        build_parameters, "maximum_displacement", n_samples=10, seed=1
    )
    value = objective_fn(context)
    assert value is not None
    assert value > 0.0


def test_robust_objective_mean_returns_none_for_empty_parameters() -> None:
    project = _base_project()
    run = SimulationRunManager().execute(project)
    context = DesignContext(design_variables={}, project=project, run=run)
    objective_fn = robust_objective_mean(lambda ctx: [], "maximum_displacement")
    assert objective_fn(context) is None


# --- Picklability (Version 34: required for parallel optimization evaluation) ---


def test_from_result_extractor_is_picklable() -> None:
    import pickle

    objective_fn = from_result_extractor(get_extractor("maximum_displacement"))
    pickled = pickle.dumps(objective_fn)
    restored = pickle.loads(pickled)

    project = _base_project()
    run = SimulationRunManager().execute(project)
    context = DesignContext(design_variables={}, project=project, run=run)
    assert restored(context) == objective_fn(context)


def test_robust_objective_mean_is_picklable_with_module_level_build_parameters() -> None:
    import pickle

    objective_fn = robust_objective_mean(
        _build_parameters_module_level, "maximum_displacement", n_samples=5, seed=1
    )
    pickle.dumps(objective_fn)


def test_rectangular_mass_is_already_picklable() -> None:
    import pickle

    pickle.dumps(rectangular_mass)


def _build_parameters_module_level(context: DesignContext) -> list[UncertainParameter]:
    return [
        UncertainParameter(
            path="material.youngs_modulus",
            label="E",
            distribution=NormalDistribution(
                context.project.material.youngs_modulus,
                context.project.material.youngs_modulus * 0.02,
            ),
            physical_lower_bound=0.0,
        )
    ]
