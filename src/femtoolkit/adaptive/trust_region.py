"""A simple trust-region foundation for adaptive refinement (Version 36).

**Engineering concept.** A trust region is the ball

.. math::

    \\|x - x_c\\| \\leq \\Delta

around a current design center :math:`x_c`, with radius :math:`\\Delta`, inside which
the surrogate is currently trusted to guide the search. Distance is measured with
:func:`~femtoolkit.adaptive.sampling.normalized_distance` so the radius is one single,
unit-independent number regardless of how many physically different design variables
are involved.

This is deliberately a simple foundation, not a full trust-region optimization
algorithm: callers decide when to recenter and when to call :meth:`TrustRegion.expand`/
:meth:`TrustRegion.contract`, typically from the agreement ratio computed by
:func:`~femtoolkit.adaptive.refinement.prediction_agreement`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from femtoolkit.adaptive.sampling import normalized_distance
from femtoolkit.exceptions import InvalidTrustRegionConfigurationError


@dataclass(frozen=True)
class TrustRegion:
    """A simple trust region: a center point and a normalized-distance radius.

    Attributes:
        center: The current design center, keyed by design-variable path.
        radius: The trust region's radius, in normalized design-space distance.
        min_radius: The smallest radius :meth:`contract` may shrink to.
        max_radius: The largest radius :meth:`expand` may grow to.
        expansion_factor: The multiplier :meth:`expand` applies (``> 1.0``).
        contraction_factor: The multiplier :meth:`contract` applies (``< 1.0``).
    """

    center: dict[str, float]
    radius: float
    min_radius: float = 0.01
    max_radius: float = 2.0
    expansion_factor: float = 2.0
    contraction_factor: float = 0.5

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise InvalidTrustRegionConfigurationError(
                f"TrustRegion radius must be positive, got {self.radius!r}."
            )
        if self.min_radius <= 0.0 or self.min_radius > self.max_radius:
            raise InvalidTrustRegionConfigurationError(
                f"TrustRegion requires 0 < min_radius <= max_radius; got "
                f"min_radius={self.min_radius!r}, max_radius={self.max_radius!r}."
            )
        if not (0.0 < self.contraction_factor < 1.0 < self.expansion_factor):
            raise InvalidTrustRegionConfigurationError(
                "TrustRegion requires 0 < contraction_factor < 1 < expansion_factor; got "
                f"contraction_factor={self.contraction_factor!r}, "
                f"expansion_factor={self.expansion_factor!r}."
            )

    def contains(self, point: dict[str, float], bounds: dict[str, tuple[float, float]]) -> bool:
        """Whether ``point`` lies within this trust region's radius of its center.

        Args:
            point: The point to check, keyed by design-variable path.
            bounds: Each dimension's ``(lower, upper)`` bound (see
                :func:`~femtoolkit.adaptive.sampling.design_variable_bounds`).

        Returns:
            ``True`` if the normalized distance from ``point`` to :attr:`center` is
            at most :attr:`radius`.
        """
        return normalized_distance(point, self.center, bounds) <= self.radius

    def expand(self) -> TrustRegion:
        """Return a new trust region with the radius grown by :attr:`expansion_factor`.

        Used when a prediction agreed well with the high-fidelity result -- the
        surrogate is reliable here, so a larger neighborhood can be trusted next time.
        """
        return replace(self, radius=min(self.radius * self.expansion_factor, self.max_radius))

    def contract(self) -> TrustRegion:
        """Return a new trust region with the radius shrunk by :attr:`contraction_factor`.

        Used when a prediction disagreed with the high-fidelity result -- the
        surrogate is unreliable here, so the search should stay closer to the center
        until it is retrained.
        """
        return replace(self, radius=max(self.radius * self.contraction_factor, self.min_radius))

    def recenter(self, new_center: dict[str, float]) -> TrustRegion:
        """Return a new trust region centered on ``new_center``, radius unchanged."""
        return replace(self, center=dict(new_center))


__all__ = ["TrustRegion"]
