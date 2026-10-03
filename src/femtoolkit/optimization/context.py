"""`DesignContext`: everything an objective/constraint evaluation function may need (Version 32).

Kept in its own module (rather than in :mod:`femtoolkit.optimization.objectives`
or :mod:`.constraints`) since both of those modules need it, and neither
should depend on the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from femtoolkit.application.project import Project
    from femtoolkit.runs.models import SimulationRun


@dataclass(frozen=True)
class DesignContext:
    """The full context one evaluated design provides to an objective or constraint.

    Attributes:
        design_variables: This design's variable values, keyed by
            :attr:`~femtoolkit.optimization.variables.DesignVariable.name`.
        project: The resolved :class:`~femtoolkit.application.project.Project`
            -- the base project with this design's variable values
            applied (via the same dotted-path override mechanism
            :class:`~femtoolkit.studies.scenarios.Scenario` uses) --
            available for any quantity derived directly from the
            configuration itself (e.g. mass from geometry and density),
            not only from the solved FEA result.
        run: The executed :class:`~femtoolkit.runs.models.SimulationRun`.
            Always present, but only carries a usable ``result`` when
            ``run.status == RunStatus.COMPLETED`` -- an objective or
            constraint function must handle a non-completed run itself
            (typically by returning ``None``, which
            :mod:`femtoolkit.optimization.evaluation` treats as an
            evaluation failure).
        metadata: A mutable scratch dictionary an objective or
            constraint function may write diagnostic information into
            (Version 33) -- for example, a robust/uncertainty-aware
            objective (:mod:`femtoolkit.optimization.robust`) records
            the statistic used and the underlying Monte Carlo sample
            count here. ``evaluate_design`` copies this dictionary into
            the resulting
            :class:`~femtoolkit.optimization.evaluation.DesignEvaluation`'s
            own ``metadata`` field. The dataclass itself is frozen (its
            fields cannot be reassigned), but this dictionary's
            *contents* remain mutable by design, exactly like a shared
            scratch pad passed to every objective/constraint call for
            one design.
    """

    design_variables: dict[str, Any]
    project: Project
    run: SimulationRun
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["DesignContext"]
