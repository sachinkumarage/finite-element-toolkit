"""The surrogate training workflow: dataset generation, fitting, and validation (Version 35).

.. code-block:: text

    Parameter Space
          |
          v
    Version 34 parallel FEA execution
          |
          v
    Snapshot Dataset
          |
          v
    Train Surrogate
          |
          v
    Validation
          |
          v
    Fast Prediction

The high-fidelity FEA model remains the source of truth throughout:
:func:`generate_training_dataset` only ever *collects* high-fidelity
results that were actually computed by the existing simulation
infrastructure (:class:`~femtoolkit.runs.manager.SimulationRunManager`,
optionally parallelized by
:func:`~femtoolkit.orchestration.simulation.evaluate_simulation_batch`)
-- training a surrogate never substitutes for running the FEA model
itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from femtoolkit.application.project import Project
from femtoolkit.orchestration.config import OrchestrationConfig
from femtoolkit.orchestration.simulation import SimulationTask, evaluate_simulation_batch
from femtoolkit.studies.extractors import Extractor
from femtoolkit.studies.parameter_sweep import (
    DEFAULT_MAX_SCENARIOS,
    ParameterDefinition,
    generate_scenarios,
)
from femtoolkit.studies.scenarios import apply_scenario
from femtoolkit.surrogate.datasets import SnapshotDataset, collect_snapshots_from_runs
from femtoolkit.surrogate.models import SurrogateModel, build_surrogate_model
from femtoolkit.surrogate.validation import (
    CrossValidationResult,
    EngineeringTolerance,
    SurrogateValidationReport,
    check_engineering_tolerances,
    k_fold_cross_validate,
)


def generate_training_dataset(
    base_project: Project,
    parameters: list[ParameterDefinition],
    response_quantities: dict[str, Extractor],
    orchestration_config: OrchestrationConfig | None = None,
    max_scenarios: int = DEFAULT_MAX_SCENARIOS,
) -> SnapshotDataset:
    """Generate a training snapshot dataset from a parameter sweep.

    Builds one scenario per combination of ``parameters`` (see
    :func:`~femtoolkit.studies.parameter_sweep.generate_scenarios`),
    executes every scenario through
    :func:`~femtoolkit.orchestration.simulation.evaluate_simulation_batch`
    (Version 34 -- serial by default, or parallel if
    ``orchestration_config`` requests it), and collects the results
    into a dataset.

    Args:
        base_project: The unmodified base project every scenario overrides.
        parameters: The parameters to sweep (the Cartesian product of
            their values becomes the training design points).
        response_quantities: The named result-quantity extractors to
            collect, keyed by response name.
        orchestration_config: An optional Version 34 orchestration
            configuration. ``None`` (the default) runs every scenario
            sequentially.
        max_scenarios: The ceiling on generated scenarios, checked
            before any FEA solve -- see
            :data:`~femtoolkit.studies.parameter_sweep.DEFAULT_MAX_SCENARIOS`.

    Returns:
        A :class:`~femtoolkit.surrogate.datasets.SnapshotDataset`.
    """
    scenarios = generate_scenarios("surrogate-training", parameters, max_scenarios=max_scenarios)
    tasks = [
        SimulationTask(
            task_id=scenario.scenario_id,
            project=apply_scenario(base_project, scenario),
            scenario_id=scenario.scenario_id,
        )
        for scenario in scenarios
    ]
    runs, _summary = evaluate_simulation_batch(tasks, config=orchestration_config)
    design_points = [dict(scenario.parameter_overrides) for scenario in scenarios]
    return collect_snapshots_from_runs(runs, design_points, response_quantities)


@dataclass
class TrainingConfig:
    """Configuration for :func:`train_surrogate`.

    Attributes:
        model_type: One of
            :data:`~femtoolkit.surrogate.models.SURROGATE_MODEL_TYPES`.
        model_kwargs: Extra keyword arguments forwarded to the model's constructor.
        feature_scaler_type: One of
            :data:`~femtoolkit.surrogate.scaling.SCALER_TYPES`.
        response_scaler_type: One of
            :data:`~femtoolkit.surrogate.scaling.SCALER_TYPES`.
        train_fraction: See :meth:`~femtoolkit.surrogate.datasets.SnapshotDataset.split`.
        validation_fraction: See :meth:`~femtoolkit.surrogate.datasets.SnapshotDataset.split`.
        split_seed: The random seed used for the train/validation/test split.
        cross_validate: Whether to also run k-fold cross-validation per response.
        cross_validation_folds: The number of folds, if ``cross_validate`` is ``True``.
        cross_validation_seed: The random seed used to assign cross-validation folds.
        tolerances: Engineering tolerances to check against the held-out test split.
    """

    model_type: str = "polynomial"
    model_kwargs: dict[str, object] = field(default_factory=dict)
    feature_scaler_type: str = "standard"
    response_scaler_type: str = "standard"
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    split_seed: int | None = None
    cross_validate: bool = False
    cross_validation_folds: int = 5
    cross_validation_seed: int | None = None
    tolerances: list[EngineeringTolerance] = field(default_factory=list)


def train_surrogate(
    dataset: SnapshotDataset, config: TrainingConfig | None = None
) -> tuple[SurrogateModel, SurrogateValidationReport]:
    """Split, fit, and validate a surrogate model on ``dataset``.

    Reproducibility: the train/validation/test split, the surrogate's
    own fitting (deterministic for both
    :class:`~femtoolkit.surrogate.models.polynomial.PolynomialRegressionSurrogate`
    and :class:`~femtoolkit.surrogate.models.rbf.RBFSurrogate` --
    neither uses randomness during fitting itself), and the optional
    cross-validation fold assignment are each controlled by an explicit
    seed recorded on the returned
    :class:`~femtoolkit.surrogate.validation.SurrogateValidationReport`.

    Args:
        dataset: The dataset to train on.
        config: The training configuration. ``None`` uses every
            :class:`TrainingConfig` default.

    Returns:
        A ``(model, report)`` pair: the fitted model (never evaluated
        only on its own training data -- see
        :attr:`~femtoolkit.surrogate.validation.SurrogateValidationReport`)
        and its validation report.
    """
    config = config or TrainingConfig()
    split = dataset.split(config.train_fraction, config.validation_fraction, seed=config.split_seed)

    model = build_surrogate_model(
        config.model_type,
        feature_scaler_type=config.feature_scaler_type,
        response_scaler_type=config.response_scaler_type,
        **config.model_kwargs,
    )
    x_train, y_train = split.train.to_arrays()
    model.fit(
        x_train,
        y_train,
        feature_names=dataset.feature_names,
        response_names=dataset.response_names,
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.dataset_version,
        random_seed=config.split_seed,
    )

    warnings: list[str] = []
    training_metrics = model.validate(x_train, y_train)

    validation_metrics: dict = {}
    if split.validation.n_samples > 0:
        x_val, y_val = split.validation.to_arrays()
        validation_metrics = model.validate(x_val, y_val)
    else:
        warnings.append(
            "Dataset too small for a separate validation split; only train/test metrics "
            "are available. A surrogate should still never be evaluated only on its "
            "training data -- see the test metrics below."
        )

    x_test, y_test = split.test.to_arrays()
    test_metrics = model.validate(x_test, y_test)
    test_predicted = model.predict(x_test)

    cross_validation: dict[str, CrossValidationResult] = {}
    if config.cross_validate:
        if dataset.n_samples < config.cross_validation_folds:
            warnings.append(
                f"Dataset has only {dataset.n_samples} snapshot(s), fewer than the requested "
                f"{config.cross_validation_folds} cross-validation folds; cross-validation "
                "was skipped."
            )
        else:
            for response_name in dataset.response_names:
                cross_validation[response_name] = k_fold_cross_validate(
                    model_factory=lambda: build_surrogate_model(
                        config.model_type,
                        feature_scaler_type=config.feature_scaler_type,
                        response_scaler_type=config.response_scaler_type,
                        **config.model_kwargs,
                    ),
                    dataset=dataset,
                    response_name=response_name,
                    k=config.cross_validation_folds,
                    seed=config.cross_validation_seed,
                )

    tolerance_checks = []
    if config.tolerances:
        response_names = dataset.response_names
        actual_by_response = {name: y_test[:, i] for i, name in enumerate(response_names)}
        predicted_by_response = {
            name: test_predicted[:, i] for i, name in enumerate(response_names)
        }
        tolerance_checks = check_engineering_tolerances(
            config.tolerances, actual_by_response, predicted_by_response
        )
        if any(not check.passed for check in tolerance_checks):
            warnings.append(
                "At least one engineering tolerance was not met on the held-out test split -- "
                "see tolerance_checks. A good statistical score does not by itself make this "
                "surrogate acceptable for the responses that failed."
            )

    report = SurrogateValidationReport(
        model_name=model.name,
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.dataset_version,
        n_samples=dataset.n_samples,
        n_features=dataset.n_features,
        n_responses=dataset.n_responses,
        training_metrics=training_metrics,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        cross_validation=cross_validation,
        tolerance_checks=tolerance_checks,
        warnings=warnings,
    )
    return model, report


__all__ = ["TrainingConfig", "generate_training_dataset", "train_surrogate"]
