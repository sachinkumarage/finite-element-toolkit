"""Engineering digital twin foundation: model updating against measured data (Version 38).

**Engineering concept.** A digital twin is a computational representation of
a physical engineering system that can be compared with measured behavior:

.. code-block:: text

    Physical/Measured Data -> Simulation Model -> Parameter Updating -> Model Validation

- **Physical system** -- the actual engineering component or structure.
- **Measurement data** -- observed quantities (displacement, temperature,
  strain, natural frequency, stress, load response), stored here as
  :class:`~femtoolkit.digital_twin.measurements.MeasurementData`.
- **Simulation model** -- the existing FEA/mathematical representation of the
  physical system (an unmodified :class:`~femtoolkit.application.project.Project`
  -- this package never duplicates the solver).
- **Model parameters** -- quantities that may require calibration (Young's
  modulus, density, thickness, cross-sectional area, thermal conductivity,
  damping coefficient, ...), stored as :class:`~femtoolkit.digital_twin.parameters.ModelParameter`.
- **Model updating** -- adjusting selected parameters so simulation
  predictions better agree with measured data
  (:func:`~femtoolkit.digital_twin.updating.run_model_update`).

**Mathematical foundation.** For input/loading conditions :math:`x` and model
parameters :math:`\\theta`, the simulation prediction is

.. math::

    y_{sim} = f(x, \\theta)

compared against measured data :math:`y_{meas}` via the residual
:math:`r = y_{sim} - y_{meas}`, absolute error :math:`e_{abs} = |r|`, and
relative error :math:`e_{rel} = |r| / |y_{meas}|`. Model updating minimizes
the calibration objective

.. math::

    J(\\theta) = \\frac{1}{N} \\sum_{i=1}^{N} \\left(y_{sim,i}(\\theta) - y_{meas,i}\\right)^2

-- **model updating is a parameter-calibration problem**, solved here by
reusing the unmodified Version 32/33 optimization machinery
(:mod:`femtoolkit.optimization`), never a second optimizer, Bayesian
calibration, or machine-learning parameter identification.

.. code-block:: text

    Physical Reference Data -> MeasurementData -> Baseline Simulation Model ->
    Baseline Prediction -> Error Evaluation -> Parameter Updating ->
    Updated Simulation Model -> Updated Prediction -> Validation -> Digital Twin Result

Built entirely on top of the existing Version 30 simulation workflows
(:mod:`femtoolkit.studies`, :mod:`femtoolkit.runs`), Version 32/33 optimization
(:mod:`femtoolkit.optimization`), Version 34 parallel execution
(:mod:`femtoolkit.orchestration`), and Version 35 accuracy metrics
(:mod:`femtoolkit.surrogate.metrics`) -- no dataset, optimization, or metric
logic is duplicated here.

**This is a foundational engineering digital twin, not an industrial
platform.** It is explicitly *not* a real-time industrial digital twin, an
IoT platform, a sensor-streaming system, a cloud digital-twin platform, or an
autonomous control system -- see ``docs/releases/v38.0.0.md`` for the full
scope boundary and the Version 39 preview.
"""

from __future__ import annotations

from femtoolkit.digital_twin.measurements import MeasurementData, MeasurementPoint
from femtoolkit.digital_twin.parameters import ModelParameter, validate_unique_parameter_names
from femtoolkit.digital_twin.problem import ModelUpdateProblem
from femtoolkit.digital_twin.results import DigitalTwinStatus, ModelUpdateResult, ModelUpdateStatus
from femtoolkit.digital_twin.updating import ModelUpdateConfig, run_model_update
from femtoolkit.digital_twin.validation import ModelAccuracyComparison, compare_model_accuracy

__all__ = [
    "DigitalTwinStatus",
    "MeasurementData",
    "MeasurementPoint",
    "ModelAccuracyComparison",
    "ModelParameter",
    "ModelUpdateConfig",
    "ModelUpdateProblem",
    "ModelUpdateResult",
    "ModelUpdateStatus",
    "compare_model_accuracy",
    "run_model_update",
    "validate_unique_parameter_names",
]
