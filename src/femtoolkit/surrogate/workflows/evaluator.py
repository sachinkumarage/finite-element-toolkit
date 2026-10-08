"""Evaluator abstraction: high-fidelity vs. surrogate evaluation (Version 35, foundation only).

**Optimization (spec sections 29/30/31).**

.. code-block:: text

    OptimizationProblem
           |
           v
    Evaluator
           |-- HighFidelityEvaluator  (wraps femtoolkit.optimization.evaluation.evaluate_design)
           `-- SurrogateEvaluator     (predicts objectives/constraints from a trained surrogate)

This is only the *architecture* a future version needs to run a
surrogate-assisted optimization algorithm -- this version does not
implement such an algorithm. Every
:class:`~femtoolkit.optimization.evaluation.DesignEvaluation` produced
by :class:`SurrogateEvaluator` is explicitly marked surrogate-derived
(``metadata["surrogate_derived"] = True``, plus the model/dataset
identity and applicability-domain status) so a caller can never mistake
it for a verified high-fidelity result; see
:mod:`femtoolkit.surrogate.workflows.verification` for how a
surrogate-derived candidate is checked against real FEA before any
final engineering acceptance.

**Uncertainty quantification (spec section 27).**
:class:`EvaluationBackend` is reused for a Monte Carlo study's own
configuration: :func:`describe_uncertainty_backend` makes explicit,
wherever a report is built, whether a study's repeated evaluations came
from ``HIGH_FIDELITY`` FEA or a ``SURROGATE`` -- a Version 31
:class:`~femtoolkit.uncertainty.monte_carlo.MonteCarloRunner` is never
silently pointed at a surrogate; using one requires the caller to
explicitly acknowledge it through this same enum.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

from femtoolkit.optimization.constraints import Constraint
from femtoolkit.optimization.evaluation import ConstraintEvaluation, DesignEvaluation, DesignStatus
from femtoolkit.optimization.objectives import Objective
from femtoolkit.optimization.variables import DesignVariable
from femtoolkit.runs.manager import SimulationRunManager
from femtoolkit.surrogate.models.base import PredictionStatus, SurrogateModel

if TYPE_CHECKING:
    from femtoolkit.application.project import Project


class EvaluationBackend(Enum):
    """Which engine actually produced a design evaluation or a repeated UQ sample.

    Attributes:
        HIGH_FIDELITY: A real FEA simulation was run.
        SURROGATE: A trained surrogate model predicted the result --
            never silently substituted for high-fidelity FEA; see the
            module docstring.
    """

    HIGH_FIDELITY = "high_fidelity"
    SURROGATE = "surrogate"


def describe_uncertainty_backend(backend: EvaluationBackend) -> str:
    """A one-line, report-ready description of which backend a UQ study used.

    Args:
        backend: The backend actually used.

    Returns:
        A human-readable description, explicit about the trade-off
        being made.
    """
    if backend is EvaluationBackend.HIGH_FIDELITY:
        return "Every sample was evaluated with the high-fidelity FEA model."
    return (
        "Every sample was evaluated with a surrogate model prediction, not the "
        "high-fidelity FEA model -- faster, but only as trustworthy as the surrogate's "
        "own validation report."
    )


class Evaluator(ABC):
    """The shared interface an optimization algorithm evaluates one design through.

    Attributes:
        backend: Which :class:`EvaluationBackend` this evaluator uses.
    """

    backend: EvaluationBackend

    @abstractmethod
    def evaluate(
        self,
        design_id: str,
        values: dict[str, Any],
        design_variables: list[DesignVariable],
        objectives: list[Objective],
        constraints: list[Constraint],
        generation: int | None = None,
    ) -> DesignEvaluation:
        """Evaluate one design and return a fully-populated :class:`DesignEvaluation`."""
        raise NotImplementedError


class HighFidelityEvaluator(Evaluator):
    """Evaluates a design with the real FEA model (wraps the existing Version 32 pipeline)."""

    backend = EvaluationBackend.HIGH_FIDELITY

    def __init__(
        self, base_project: Project, run_manager: SimulationRunManager | None = None
    ) -> None:
        """Create a high-fidelity evaluator.

        Args:
            base_project: The unmodified base project every design overrides.
            run_manager: The run manager to execute each design with.
                ``None`` constructs a fresh
                :class:`~femtoolkit.runs.manager.SimulationRunManager`.
        """
        self._base_project = base_project
        self._run_manager = run_manager or SimulationRunManager()

    def evaluate(
        self,
        design_id: str,
        values: dict[str, Any],
        design_variables: list[DesignVariable],
        objectives: list[Objective],
        constraints: list[Constraint],
        generation: int | None = None,
    ) -> DesignEvaluation:
        """Delegate directly to :func:`femtoolkit.optimization.evaluation.evaluate_design`."""
        from femtoolkit.optimization.evaluation import evaluate_design

        return evaluate_design(
            design_id, values, design_variables, self._base_project, objectives, constraints,
            self._run_manager, generation=generation,
        )


class SurrogateEvaluator(Evaluator):
    """Evaluates a design by predicting its objectives/constraints from trained surrogates.

    **Never a verified result on its own.** Every
    :class:`~femtoolkit.optimization.evaluation.DesignEvaluation` this
    evaluator returns carries ``metadata["surrogate_derived"] = True``
    plus the originating model/dataset identity and applicability-
    domain status -- see
    :mod:`femtoolkit.surrogate.workflows.verification` for checking a
    surrogate-selected candidate against real FEA before final
    engineering acceptance (spec sections 30/31).
    """

    backend = EvaluationBackend.SURROGATE

    def __init__(self, surrogates: dict[str, SurrogateModel]) -> None:
        """Create a surrogate evaluator.

        Args:
            surrogates: A fitted surrogate model per quantity name,
                keyed by the quantity it predicts. An objective or
                constraint is evaluated by looking up a surrogate under
                its own ``name`` and reading the matching response from
                that surrogate's prediction -- every
                :class:`~femtoolkit.optimization.objectives.Objective`/
                :class:`~femtoolkit.optimization.constraints.Constraint`
                name used with this evaluator must therefore also be
                one of the registered surrogate's response names.
        """
        self._surrogates = surrogates

    def _predict(self, name: str, values: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
        model = self._surrogates.get(name)
        if model is None:
            return None, {"error": f"No surrogate registered for quantity {name!r}."}
        prediction = model.predict_point(values)
        if prediction.status is not PredictionStatus.OK:
            return None, {"error": f"Surrogate could not predict {name!r}: {prediction.warnings}."}
        if name not in prediction.values:
            return None, {"error": f"Surrogate for {name!r} does not report that response."}
        info = {
            "model_name": prediction.model_name,
            "model_version": prediction.model_version,
            "dataset_id": prediction.dataset_id,
            "dataset_version": prediction.dataset_version,
            "domain_status": prediction.domain_status.value,
            "warnings": list(prediction.warnings),
        }
        return prediction.values[name], info

    def evaluate(
        self,
        design_id: str,
        values: dict[str, Any],
        design_variables: list[DesignVariable],
        objectives: list[Objective],
        constraints: list[Constraint],
        generation: int | None = None,
    ) -> DesignEvaluation:
        """Predict every objective/constraint for this design from the registered surrogates."""
        for variable in design_variables:
            if variable.name not in values or not variable.is_valid_value(values[variable.name]):
                return DesignEvaluation(
                    design_id=design_id,
                    design_variables=dict(values),
                    run_id=None,
                    status=DesignStatus.INVALID,
                    error_message=f"Invalid value for design variable {variable.name!r}.",
                    generation=generation,
                    metadata={"surrogate_derived": True},
                )

        feature_point = {variable.path: values[variable.name] for variable in design_variables}

        metadata: dict[str, Any] = {"surrogate_derived": True, "surrogate_info": {}}
        objective_values: dict[str, float] = {}
        for objective in objectives:
            value, info = self._predict(objective.name, feature_point)
            metadata["surrogate_info"][objective.name] = info
            if value is None:
                return DesignEvaluation(
                    design_id=design_id, design_variables=dict(values), run_id=None,
                    status=DesignStatus.FAILED, error_message=info.get("error"),
                    generation=generation, metadata=metadata,
                )
            objective_values[objective.name] = value

        constraint_evaluations: list[ConstraintEvaluation] = []
        total_violation = 0.0
        for constraint in constraints:
            value, info = self._predict(constraint.name, feature_point)
            metadata["surrogate_info"][constraint.name] = info
            if value is None:
                return DesignEvaluation(
                    design_id=design_id, design_variables=dict(values), run_id=None,
                    status=DesignStatus.FAILED, error_message=info.get("error"),
                    generation=generation, metadata=metadata,
                )
            violation = constraint.violation(value)
            total_violation += violation
            constraint_evaluations.append(
                ConstraintEvaluation(
                    name=constraint.name, value=value, limit=constraint.limit,
                    relation=constraint.relation, violation=violation, satisfied=violation <= 0.0,
                )
            )

        status = DesignStatus.FEASIBLE if total_violation <= 0.0 else DesignStatus.INFEASIBLE
        return DesignEvaluation(
            design_id=design_id, design_variables=dict(values), run_id=None, status=status,
            objective_values=objective_values, constraint_evaluations=constraint_evaluations,
            total_violation=total_violation, generation=generation, metadata=metadata,
        )


__all__ = [
    "EvaluationBackend",
    "Evaluator",
    "HighFidelityEvaluator",
    "SurrogateEvaluator",
    "describe_uncertainty_backend",
]
