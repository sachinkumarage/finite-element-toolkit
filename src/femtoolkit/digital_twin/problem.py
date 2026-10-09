"""`ModelUpdateProblem`: ties a project, its calibratable parameters, and
measured data together (Version 38).

No FEA solver logic lives here -- :meth:`ModelUpdateProblem.predict` reuses the
unmodified Version 30 :class:`~femtoolkit.studies.scenarios.Scenario`/
:func:`~femtoolkit.studies.scenarios.apply_scenario` override mechanism and
Version 34 :func:`~femtoolkit.orchestration.simulation.evaluate_simulation_batch`
to run one simulation per measurement point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from femtoolkit.digital_twin.measurements import MeasurementData
from femtoolkit.digital_twin.parameters import ModelParameter, validate_unique_parameter_names
from femtoolkit.exceptions import ValidationError
from femtoolkit.orchestration.simulation import SimulationTask, evaluate_simulation_batch
from femtoolkit.runs.models import RunStatus
from femtoolkit.studies.extractors import get_extractor
from femtoolkit.studies.scenarios import Scenario, apply_scenario

if TYPE_CHECKING:
    from femtoolkit.application.project import Project
    from femtoolkit.orchestration.config import OrchestrationConfig


@dataclass
class ModelUpdateProblem:
    """The complete definition of one model-updating (calibration) problem.

    Attributes:
        base_project: The unmodified base project -- every non-updatable
            parameter's value is whatever this project's own configuration
            already holds; :attr:`parameters` describes it for display, but
            only *updatable* parameters are ever overridden.
        parameters: Every model parameter under consideration (updatable and
            fixed). Names must be unique.
        measurement_data: The measured reference data this problem calibrates
            against. :attr:`~femtoolkit.digital_twin.measurements.MeasurementData.quantity_name`
            must be one of :data:`~femtoolkit.studies.extractors.EXTRACTORS`.
        tolerance: An engineering-acceptable relative error used by
            downstream validation (not by the simulation itself).
    """

    base_project: Project
    parameters: list[ModelParameter]
    measurement_data: MeasurementData
    tolerance: float = 0.05

    def __post_init__(self) -> None:
        if not self.parameters:
            raise ValidationError("ModelUpdateProblem requires at least one model parameter.")
        validate_unique_parameter_names(self.parameters)
        if self.measurement_data.n_points == 0:
            raise ValidationError(
                "ModelUpdateProblem requires at least one measurement point."
            )
        if self.tolerance <= 0.0:
            raise ValidationError(f"tolerance must be positive, got {self.tolerance!r}.")
        # Fails fast (ValidationError, via get_extractor) if quantity_name is unknown.
        get_extractor(self.measurement_data.quantity_name)

    @property
    def updatable_parameters(self) -> list[ModelParameter]:
        """Every parameter calibration is allowed to adjust."""
        return [parameter for parameter in self.parameters if parameter.updatable]

    def current_parameter_values(self) -> dict[str, float]:
        """Every parameter's current value, keyed by name."""
        return {parameter.name: parameter.current_value for parameter in self.parameters}

    def predict(
        self,
        parameter_values: dict[str, float],
        orchestration_config: OrchestrationConfig | None = None,
    ) -> np.ndarray:
        """Run the simulation at every measurement point with the given parameter values.

        Args:
            parameter_values: Updatable-parameter values to apply, keyed by
                :attr:`~femtoolkit.digital_twin.parameters.ModelParameter.name`.
                A name not present in this dict keeps its current value.
            orchestration_config: An optional Version 34 orchestration
                configuration for running the batch in parallel. ``None``
                runs every point sequentially.

        Returns:
            One prediction per measurement point, in :attr:`measurement_data`
            order; a point whose simulation did not complete or whose
            response could not be extracted is ``NaN``.
        """
        overrides = {
            parameter.path: parameter_values.get(parameter.name, parameter.current_value)
            for parameter in self.parameters
            if parameter.updatable
        }
        points = self.measurement_data.points
        tasks = [
            SimulationTask(
                task_id=f"digital-twin-{index}",
                project=apply_scenario(
                    self.base_project,
                    Scenario(
                        scenario_id=f"digital-twin-{index}",
                        name=f"Digital twin point {index}",
                        parameter_overrides={**overrides, **point.input_conditions},
                    ),
                ),
                scenario_id=f"digital-twin-{index}",
            )
            for index, point in enumerate(points)
        ]
        runs, _summary = evaluate_simulation_batch(tasks, config=orchestration_config)
        extractor = get_extractor(self.measurement_data.quantity_name)

        predictions = np.full(len(points), np.nan)
        for index, run in enumerate(runs):
            if run.status != RunStatus.COMPLETED:
                continue
            value = extractor(run)
            if value is not None:
                predictions[index] = value
        return predictions


__all__ = ["ModelUpdateProblem"]
