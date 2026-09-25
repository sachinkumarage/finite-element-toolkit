"""`UncertainParameter`: one uncertain input to a simulation (Version 31).

An uncertain parameter pairs a :class:`~femtoolkit.uncertainty.distributions.Distribution`
with the same dotted override path
:class:`~femtoolkit.studies.scenarios.Scenario`/
:class:`~femtoolkit.studies.parameter_sweep.ParameterDefinition` already
use to identify a project field (e.g. ``"material.youngs_modulus"``) --
so an uncertain parameter is a drop-in replacement for a deterministic
one, and a Monte Carlo sample becomes a :class:`~femtoolkit.studies.scenarios.Scenario`
the exact same way a parameter-sweep value does. See
:mod:`femtoolkit.uncertainty.monte_carlo`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from femtoolkit.exceptions import ValidationError
from femtoolkit.uncertainty.distributions import (
    Distribution,
    distribution_from_dict,
    distribution_to_dict,
)


class UncertaintyCategory(Enum):
    """Whether an uncertain parameter's variability is aleatory or epistemic.

    Attributes:
        ALEATORY: Natural, irreducible variability -- e.g. sample-to-
            sample variation in a manufactured material's strength.
        EPISTEMIC: Uncertainty from a lack of knowledge -- e.g. an
            incompletely characterized boundary-condition stiffness.
            More data could, in principle, reduce it; aleatory
            variability cannot be reduced the same way.
    """

    ALEATORY = "aleatory"
    EPISTEMIC = "epistemic"


@dataclass
class UncertainParameter:
    """One uncertain input parameter to a simulation.

    Attributes:
        path: The dotted override path this parameter controls on a
            :class:`~femtoolkit.application.project.Project` (the same
            convention :class:`~femtoolkit.studies.scenarios.Scenario`
            uses), e.g. ``"material.youngs_modulus"``.
        label: A short, human-readable name (e.g. ``"Young's Modulus"``).
        distribution: The parameter's probability distribution.
        units: A units string for display (e.g. ``"Pa"``), purely
            descriptive.
        description: A longer, free-text description.
        category: Whether this parameter's uncertainty is aleatory or
            epistemic.
        physical_lower_bound: A hard physical lower bound (e.g. ``0.0``
            for a Young's modulus, since ``E > 0``), or ``None`` for no
            additional bound beyond the distribution's own. A sampled
            value at or below this bound is physically invalid.
        physical_upper_bound: A hard physical upper bound, or ``None``.
            A sampled value at or above this bound is physically
            invalid.
        reference_value: The deterministic/nominal value this parameter
            takes in a non-uncertain analysis, if relevant (e.g. for
            comparison against the uncertainty study's mean output).
    """

    path: str
    label: str
    distribution: Distribution
    units: str = ""
    description: str = ""
    category: UncertaintyCategory = UncertaintyCategory.ALEATORY
    physical_lower_bound: float | None = None
    physical_upper_bound: float | None = None
    reference_value: float | None = None

    def __post_init__(self) -> None:
        if self.physical_lower_bound is not None and not math.isfinite(self.physical_lower_bound):
            raise ValidationError(f"Physical lower bound for {self.label!r} must be finite.")
        if self.physical_upper_bound is not None and not math.isfinite(self.physical_upper_bound):
            raise ValidationError(f"Physical upper bound for {self.label!r} must be finite.")
        if (
            self.physical_lower_bound is not None
            and self.physical_upper_bound is not None
            and not self.physical_lower_bound < self.physical_upper_bound
        ):
            raise ValidationError(
                f"Physical lower bound must be less than the upper bound for {self.label!r} "
                f"(got lower={self.physical_lower_bound}, upper={self.physical_upper_bound})."
            )
        if self.reference_value is not None and not self.is_physically_valid(self.reference_value):
            raise ValidationError(
                f"Reference value {self.reference_value!r} for {self.label!r} violates its "
                f"physical bounds {self.effective_bounds()!r}."
            )

    def effective_bounds(self) -> tuple[float | None, float | None]:
        """The combined bounds from this parameter's distribution and its own physical bounds.

        Returns:
            ``(lower, upper)``, taking the tighter of the distribution's
            own :meth:`~femtoolkit.uncertainty.distributions.Distribution.bounds`
            and :attr:`physical_lower_bound`/:attr:`physical_upper_bound`
            on each side. ``None`` on a side means no bound at all.
        """
        distribution_low, distribution_high = self.distribution.bounds()
        lows = [b for b in (distribution_low, self.physical_lower_bound) if b is not None]
        highs = [b for b in (distribution_high, self.physical_upper_bound) if b is not None]
        return (max(lows) if lows else None, min(highs) if highs else None)

    def is_physically_valid(self, value: float) -> bool:
        """Whether ``value`` satisfies this parameter's effective bounds.

        Args:
            value: A candidate sampled value.

        Returns:
            ``False`` if ``value`` is non-finite or lies at/beyond
            either effective bound; ``True`` otherwise.
        """
        if not math.isfinite(value):
            return False
        lower, upper = self.effective_bounds()
        below_lower = lower is not None and value <= lower
        above_upper = upper is not None and value >= upper
        return not (below_lower or above_upper)

    def to_dict(self) -> dict:
        """Return a plain, JSON-serializable representation of this parameter."""
        return {
            "path": self.path,
            "label": self.label,
            "distribution": distribution_to_dict(self.distribution),
            "units": self.units,
            "description": self.description,
            "category": self.category.value,
            "physical_lower_bound": self.physical_lower_bound,
            "physical_upper_bound": self.physical_upper_bound,
            "reference_value": self.reference_value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> UncertainParameter:
        """Reconstruct an :class:`UncertainParameter` from :meth:`to_dict`'s output."""
        return cls(
            path=data["path"],
            label=data["label"],
            distribution=distribution_from_dict(data["distribution"]),
            units=data.get("units", ""),
            description=data.get("description", ""),
            category=UncertaintyCategory(data.get("category", UncertaintyCategory.ALEATORY.value)),
            physical_lower_bound=data.get("physical_lower_bound"),
            physical_upper_bound=data.get("physical_upper_bound"),
            reference_value=data.get("reference_value"),
        )


__all__ = ["UncertainParameter", "UncertaintyCategory"]
