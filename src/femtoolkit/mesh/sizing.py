"""Mesh sizing controls (Version 25, spec section 10).

A minimal, engineering-oriented abstraction for expressing "how fine
should the mesh be" in physical units (meters, matching every other
length in this toolkit -- see the module docstring's note on units)
rather than the raw subdivision counts
:func:`~femtoolkit.mesh.generator.create_quad_mesh`/
:func:`~femtoolkit.mesh.generator.create_triangular_mesh` take directly.
This is a clean foundation for a future, more capable mesh generator
(see the Version 26 preview) -- not a sophisticated commercial meshing
algorithm: for this version's structured rectangular generators, a
target size simply becomes ``round(extent / target_size)`` subdivisions
along each axis.

**Regional sizing.** ``regional_sizes`` lets a caller name a finer or
coarser target size for a named boundary region (e.g. ``{"left":
0.05}``), matching the region-naming convention already used throughout
the toolkit (:mod:`femtoolkit.geometry.boundary`). This version's
structured generators do not yet vary element size within a mesh (they
produce a single uniform grid), so ``regional_sizes`` is accepted and
validated here but not yet consumed by
:mod:`femtoolkit.mesh.generation` -- an explicit, honest interface for a
future unstructured or graded generator to use, not a fabricated
capability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from femtoolkit.exceptions import ValidationError


@dataclass(frozen=True)
class MeshSizingParameters:
    """Target element sizing, in meters (this toolkit's base unit -- see below).

    **Units.** Every length in this toolkit -- node coordinates, domain
    width/height, element thickness -- is a plain ``float`` interpreted
    as **meters** (SI base unit); nothing in the codebase attaches a
    unit tag to a number. ``target_size``/``minimum_size``/
    ``maximum_size`` follow the same convention: a caller working in
    millimeters or inches must convert to meters before constructing
    this object (:mod:`femtoolkit.units` provides the toolkit's existing
    SI unit constants for this -- there is no second unit system here).

    Attributes:
        target_size: The desired characteristic element edge length, in
            meters. Must be positive.
        minimum_size: A lower bound on element size, in meters, for a
            future adaptive/graded generator to respect. Optional.
        maximum_size: An upper bound on element size, in meters. Optional.
        regional_sizes: Maps a named boundary region (e.g. ``"left"``)
            to a target size, in meters, for that region. See the
            module docstring -- not yet consumed by this version's
            structured generators.

    Raises:
        ValidationError: If ``target_size`` is not positive and finite,
            if ``minimum_size``/``maximum_size`` is given and not
            positive and finite, or if
            ``minimum_size > target_size > maximum_size`` is violated.

    Example:
        >>> sizing = MeshSizingParameters(target_size=0.1, minimum_size=0.02, maximum_size=0.5)
        >>> sizing.subdivisions(width=2.0, height=0.4)
        (20, 4)
    """

    target_size: float
    minimum_size: float | None = None
    maximum_size: float | None = None
    regional_sizes: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate every sizing bound immediately after construction.

        Raises:
            ValidationError: See the class docstring.
        """
        if not math.isfinite(self.target_size) or self.target_size <= 0:
            raise ValidationError(f"target_size must be positive, got {self.target_size}.")
        if self.minimum_size is not None:
            if not math.isfinite(self.minimum_size) or self.minimum_size <= 0:
                raise ValidationError(f"minimum_size must be positive, got {self.minimum_size}.")
            if self.minimum_size > self.target_size:
                raise ValidationError(
                    f"minimum_size ({self.minimum_size}) must not exceed "
                    f"target_size ({self.target_size})."
                )
        if self.maximum_size is not None:
            if not math.isfinite(self.maximum_size) or self.maximum_size <= 0:
                raise ValidationError(f"maximum_size must be positive, got {self.maximum_size}.")
            if self.maximum_size < self.target_size:
                raise ValidationError(
                    f"maximum_size ({self.maximum_size}) must not be less than "
                    f"target_size ({self.target_size})."
                )
        for region, size in self.regional_sizes.items():
            if not math.isfinite(size) or size <= 0:
                raise ValidationError(
                    f"regional_sizes[{region!r}] must be positive, got {size}."
                )

    def subdivisions(self, width: float, height: float) -> tuple[int, int]:
        """Convert this sizing to ``(nx, ny)`` structured-grid subdivisions.

        Args:
            width: Domain width (X extent), in meters. Must be positive.
            height: Domain height (Y extent), in meters. Must be positive.

        Returns:
            ``(nx, ny)``, each at least 1.

        Raises:
            ValidationError: If ``width`` or ``height`` is not positive and finite.
        """
        if not math.isfinite(width) or width <= 0:
            raise ValidationError(f"width must be positive, got {width}.")
        if not math.isfinite(height) or height <= 0:
            raise ValidationError(f"height must be positive, got {height}.")
        nx = max(1, round(width / self.target_size))
        ny = max(1, round(height / self.target_size))
        return nx, ny


__all__ = ["MeshSizingParameters"]
