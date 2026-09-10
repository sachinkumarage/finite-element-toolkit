"""Thermal boundary conditions (Version 20).

Two kinds of boundary condition close the heat-conduction problem,
directly mirroring the two kinds every mechanical analysis in this
toolkit already uses (a prescribed *value* vs. a prescribed *flow*):

* **Prescribed temperature** (``T = T_bar``, a Dirichlet condition) --
  the thermal analogue of
  :class:`~femtoolkit.analysis.boundary_conditions.BoundaryCondition`
  (prescribed displacement). Eliminated from the linear system exactly
  the same way, via
  :func:`~femtoolkit.analysis.system.solve`'s free/constrained DOF
  partition.
* **Prescribed heat flux** (``q . n = q_bar``, a Neumann condition) --
  the thermal analogue of an applied nodal force. Represented here, like
  :class:`~femtoolkit.thermal.thermal_loads.ThermalLoad`, as a direct
  nodal heat-flow value (watts) added to the load vector -- the standard
  finite-element Neumann-condition representation once a boundary
  traction/flux has been discretized down to nodal degrees of freedom
  (see :mod:`femtoolkit.thermal.thermal_loads`'s module docstring for
  the same representation used for heat generation).

**Designed for future extension to convection.** A convective boundary
(``q . n = h * (T - T_infinity)``, Newton's law of cooling) is neither
of these: it is *temperature-dependent* (the flux itself depends on the
unknown surface temperature), making the problem's left-hand side pick
up an extra term (a convection "stiffness"). This version deliberately
keeps :class:`PrescribedHeatFlux` a simple, temperature-*independent*
value, so that a future convection boundary condition can be added as a
genuinely new type (contributing to both ``K_T`` and ``F_T``) without
needing to change this one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from femtoolkit.exceptions import ValidationError


@dataclass(frozen=True)
class PrescribedTemperature:
    """A prescribed (Dirichlet) temperature boundary condition, ``T = T_bar``.

    Attributes:
        node_id: The node this condition applies to.
        value: The prescribed temperature, in kelvin.

    Raises:
        ValidationError: If ``value`` is not finite.

    Example:
        >>> fixed = PrescribedTemperature(node_id=1, value=373.15)
    """

    node_id: int
    value: float

    def __post_init__(self) -> None:
        """Validate the prescribed temperature immediately after construction.

        Raises:
            ValidationError: If ``value`` is not finite.
        """
        if not math.isfinite(self.value):
            raise ValidationError(f"PrescribedTemperature value must be finite, got {self.value}.")


@dataclass(frozen=True)
class PrescribedHeatFlux:
    """A prescribed (Neumann) heat-flux boundary condition, ``q . n = q_bar``.

    Represented as a direct nodal heat-flow value (see the module
    docstring for why): a caller wanting a uniform surface flux over a
    specific boundary applies this at each node on that boundary, with
    a value equal to that node's own equivalent share of the total flux
    (e.g. computed the same way
    :mod:`femtoolkit.thermal.thermal_loads` converts a volumetric source
    to nodal shares).

    Attributes:
        node_id: The node this condition applies to.
        value: The prescribed heat flow, in watts. Positive means heat
            flowing *into* the body at this node.

    Raises:
        ValidationError: If ``value`` is not finite.

    Example:
        >>> heated_edge = PrescribedHeatFlux(node_id=3, value=250.0)
    """

    node_id: int
    value: float

    def __post_init__(self) -> None:
        """Validate the prescribed heat flux immediately after construction.

        Raises:
            ValidationError: If ``value`` is not finite.
        """
        if not math.isfinite(self.value):
            raise ValidationError(f"PrescribedHeatFlux value must be finite, got {self.value}.")
