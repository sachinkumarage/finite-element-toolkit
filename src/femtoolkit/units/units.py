"""Engineering unit constants for the Finite Element Toolkit.

The toolkit assumes that all numerical quantities passed to its domain
classes (:class:`~femtoolkit.materials.Material`, :class:`~femtoolkit.mesh.Node`,
etc.) are expressed in SI base and derived units. This module defines named
constants for those units so that calling code can write self-documenting
values instead of bare numbers.

Version 1 does not provide unit conversion. Every constant below is defined
as ``1.0`` because it represents "one unit of itself" in the assumed SI
system. A future version may introduce a full unit-conversion engine built
on top of this foundation.

Example:
    >>> from femtoolkit.units import PASCAL
    >>> youngs_modulus = 200e9 * PASCAL  # 200 GPa, expressed in pascals
"""

from __future__ import annotations

METER: float = 1.0
"""SI base unit of length."""

KILOGRAM: float = 1.0
"""SI base unit of mass."""

SECOND: float = 1.0
"""SI base unit of time."""

NEWTON: float = 1.0
"""SI derived unit of force (kg*m/s^2)."""

PASCAL: float = 1.0
"""SI derived unit of pressure and stress (N/m^2)."""
