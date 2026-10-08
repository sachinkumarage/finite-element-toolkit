"""Fidelity levels and fidelity models (Version 37).

**Engineering concept.** The same physical quantity can be estimated by models of
very different accuracy, physical detail, and computational cost:

.. code-block:: text

    Low Fidelity   -> fast but less accurate
    High Fidelity  -> slower but more accurate

A :class:`FidelityLevel` is just a label with an estimated relative cost; a
:class:`FidelityModel` is the thing that actually produces a result at that level.
This module deliberately keeps the hierarchy flat -- :data:`LOW_FIDELITY` and
:data:`HIGH_FIDELITY` today, with room for a caller to define another
:class:`FidelityLevel` later without this toolkit needing a complex fidelity
hierarchy to support it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from femtoolkit.exceptions import ValidationError

if TYPE_CHECKING:
    from femtoolkit.application.project import Project
    from femtoolkit.runs.manager import SimulationRunManager
    from femtoolkit.studies.extractors import Extractor


@dataclass(frozen=True)
class FidelityLevel:
    """A named level of modeling fidelity.

    Attributes:
        name: A short, unique identifier (e.g. ``"LOW"``, ``"HIGH"``).
        description: A human-readable description of what this level represents.
        estimated_cost: A relative computational-cost figure (arbitrary units --
            only meaningful compared against another :class:`FidelityLevel`'s
            value, never as an absolute runtime guarantee).
    """

    name: str
    description: str
    estimated_cost: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("FidelityLevel requires a non-empty name.")
        if self.estimated_cost < 0.0:
            raise ValidationError(
                f"FidelityLevel {self.name!r} estimated_cost must be non-negative, "
                f"got {self.estimated_cost!r}."
            )


LOW_FIDELITY = FidelityLevel(
    name="LOW",
    description="A fast, approximate model (e.g. an analytical formula) used for screening.",
    estimated_cost=1.0,
)
"""The standard low-fidelity level: cheap, approximate, used for screening."""

HIGH_FIDELITY = FidelityLevel(
    name="HIGH",
    description="The expensive, high-accuracy FEA model used as the engineering reference.",
    estimated_cost=100.0,
)
"""The standard high-fidelity level: expensive, accurate, the engineering reference."""


class FidelityModel(ABC):
    """An engineering model that evaluates a design point at one fidelity level.

    Attributes:
        level: This model's :class:`FidelityLevel`.
        name: A short, human-readable model name.
    """

    def __init__(
        self, level: FidelityLevel, name: str, estimated_cost: float | None = None
    ) -> None:
        """Create a fidelity model.

        Args:
            level: This model's fidelity level.
            name: A short, human-readable model name.
            estimated_cost: This specific model instance's own estimated cost, if it
                differs from ``level.estimated_cost`` (e.g. a particularly coarse or
                fine mesh at the same nominal fidelity level). ``None`` uses
                ``level.estimated_cost``.
        """
        self.level = level
        self.name = name
        self._estimated_cost = level.estimated_cost if estimated_cost is None else estimated_cost

    @property
    def estimated_cost(self) -> float:
        """This model's estimated relative computational cost."""
        return self._estimated_cost

    @abstractmethod
    def evaluate(self, point: dict[str, float]) -> dict[str, float]:
        """Evaluate this model at one design point.

        Args:
            point: The design point's input values, keyed by name.

        Returns:
            The model's response value(s), keyed by response name. A response this
            model could not compute is simply omitted (never a fabricated value).
        """
        raise NotImplementedError


class AnalyticalFidelityModel(FidelityModel):
    """A fidelity model backed by a plain, cheap callable (typically low fidelity).

    No FEA solve is performed -- this is intended for closed-form or simplified
    engineering formulas (e.g. Euler-Bernoulli beam deflection), not a replacement
    for the real solver.
    """

    def __init__(
        self,
        name: str,
        evaluate_fn: Callable[[dict[str, float]], dict[str, float]],
        level: FidelityLevel = LOW_FIDELITY,
        estimated_cost: float | None = None,
    ) -> None:
        """Create an analytical fidelity model.

        Args:
            name: A short, human-readable model name.
            evaluate_fn: A callable computing every response from a design point.
            level: This model's fidelity level (defaults to :data:`LOW_FIDELITY`).
            estimated_cost: An override for this instance's estimated cost.
        """
        super().__init__(level, name, estimated_cost)
        self._evaluate_fn = evaluate_fn

    def evaluate(self, point: dict[str, float]) -> dict[str, float]:
        """Evaluate the wrapped callable at ``point``."""
        return self._evaluate_fn(point)


class SimulationFidelityModel(FidelityModel):
    """A fidelity model backed by the existing high-fidelity FEA simulation pipeline.

    Reuses :class:`~femtoolkit.studies.scenarios.Scenario`/
    :func:`~femtoolkit.studies.scenarios.apply_scenario` (Version 30) and
    :class:`~femtoolkit.runs.manager.SimulationRunManager` (also Version 30)
    unchanged -- no FEA solver logic is duplicated here.
    """

    def __init__(
        self,
        name: str,
        base_project: Project,
        response_extractors: dict[str, Extractor],
        run_manager: SimulationRunManager | None = None,
        level: FidelityLevel = HIGH_FIDELITY,
        estimated_cost: float | None = None,
    ) -> None:
        """Create a simulation-backed fidelity model.

        Args:
            name: A short, human-readable model name.
            base_project: The unmodified base project every design point overrides.
            response_extractors: Named result-quantity extractors, keyed by response
                name.
            run_manager: The run manager to execute each design point with. ``None``
                constructs a fresh
                :class:`~femtoolkit.runs.manager.SimulationRunManager`.
            level: This model's fidelity level (defaults to :data:`HIGH_FIDELITY`).
            estimated_cost: An override for this instance's estimated cost.
        """
        super().__init__(level, name, estimated_cost)
        self._base_project = base_project
        self._response_extractors = response_extractors
        self._run_manager = run_manager

    def evaluate(self, point: dict[str, float]) -> dict[str, float]:
        """Run the real FEA simulation at ``point`` and extract every configured response."""
        from femtoolkit.runs.manager import SimulationRunManager
        from femtoolkit.studies.scenarios import Scenario, apply_scenario

        manager = self._run_manager or SimulationRunManager()
        scenario = Scenario(
            scenario_id=f"multifidelity-{id(point)}",
            name=f"{self.name} evaluation",
            parameter_overrides=dict(point),
        )
        project = apply_scenario(self._base_project, scenario)
        run = manager.execute(project, scenario_id=scenario.scenario_id)

        values: dict[str, float] = {}
        for response_name, extractor in self._response_extractors.items():
            value = extractor(run)
            if value is not None:
                values[response_name] = value
        return values


def summarize_costs(models: list[FidelityModel]) -> list[dict[str, object]]:
    """Return a simple, plain-dict cost/accuracy summary for a list of fidelity models.

    Args:
        models: The models to summarize.

    Returns:
        One dict per model, in order, with ``name``, ``level``, ``description``, and
        ``estimated_cost`` -- intended for a quick textual or tabular comparison
        (spec section 13), not a cost-optimization algorithm.
    """
    return [
        {
            "name": model.name,
            "level": model.level.name,
            "description": model.level.description,
            "estimated_cost": model.estimated_cost,
        }
        for model in models
    ]


__all__ = [
    "HIGH_FIDELITY",
    "LOW_FIDELITY",
    "AnalyticalFidelityModel",
    "FidelityLevel",
    "FidelityModel",
    "SimulationFidelityModel",
    "summarize_costs",
]
