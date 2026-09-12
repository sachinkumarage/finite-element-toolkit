"""Thermal boundary conditions (Version 20-21).

Four kinds of boundary condition can close the heat-conduction problem,
forming a small conceptual taxonomy:

.. code-block:: text

    ThermalBoundaryCondition
    +-- PrescribedTemperature   (Dirichlet: T = T_bar)
    +-- PrescribedHeatFlux      (Neumann: a fixed nodal heat flow)
    +-- ConvectionBoundaryCondition   (Robin/Newton cooling: q = h(T - T_inf))
    +-- RadiationBoundaryCondition    (nonlinear Robin: q = eps*sigma*(T^4 - Tsur^4))

There is no single shared base *class* for these four -- deliberately.
:class:`PrescribedTemperature`/:class:`PrescribedHeatFlux` apply to a
*single node*, exactly mirroring the mechanical
:class:`~femtoolkit.analysis.boundary_conditions.BoundaryCondition`/
:class:`~femtoolkit.analysis.loads.NodalLoad` pair; a convective or
radiative boundary instead applies over a *surface* (several nodes at
once, integrated via shape functions -- see
:mod:`femtoolkit.thermal.thermal_surfaces`), so unifying all four under
one polymorphic interface would force either the node-based pair to
carry unused surface machinery, or the surface-based pair to fake a
single-node interface -- exactly the kind of forced unification this
project avoids elsewhere (see, e.g., Version 19's decision not to thread
temperature through every material's ``trial_state``). The tree above is
this module's *documented* taxonomy of what a "thermal boundary
condition" is conceptually, without an artificial common class.

**The two kinds already built (Version 20), kept unchanged:**

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
  nodal heat-flow value (watts) added to the load vector.

**Convection (Version 21).** Newton's law of cooling relates the heat
flux leaving a surface to the temperature difference between that
surface and the surrounding fluid:

.. code-block:: text

    q = h * (T - T_infinity)

``q`` is the heat flux (W/m^2) leaving the surface; ``h`` is the
convection coefficient (W/(m^2*K)), a lumped measure of how effectively
the surrounding fluid carries heat away (larger for a fast-moving fluid
like forced air or water, smaller for still air); ``T`` is the surface's
own temperature; ``T_infinity`` is the ambient (far-field) fluid
temperature, unaffected by the small amount of heat this surface adds to
it. Multiplying by area and integrating over the finite element mesh's
shape functions gives:

.. code-block:: text

    K_conv = integral_Gamma( h * N^T @ N ) dGamma
    F_conv = integral_Gamma( h * T_infinity * N^T ) dGamma

``K_conv`` is a *conductance* matrix: exactly like ``K_T`` resists a
temperature *gradient* inside the body, ``K_conv`` resists the surface
temperature itself rising above ambient, adding directly to the global
conductivity matrix (``K_T = K_conduction + K_convection``, as this
version's brief specifies). ``F_conv`` is a *forcing* term -- the
"pull" toward the ambient temperature -- adding directly to the global
thermal load vector. When ``h`` is a plain constant, both are exactly
linear in the unknown temperature ``T``, so a convective boundary with
constant ``h`` costs nothing extra: it is folded directly into
``K_T``/``F_T`` and solved with the same one-shot linear solve Version
20 already uses (see :mod:`femtoolkit.thermal.thermal_analysis`).

**Why convection can also be temperature-dependent.** Real convection
coefficients are not always constant -- natural convection, boiling, and
several other regimes have ``h`` that itself depends on the surface
temperature, ``h = h(T)``. :class:`ConvectionBoundaryCondition` reuses
Version 19/20's :data:`~femtoolkit.materials.thermal_properties.ThermalProperty`
machinery directly for this (a constant ``float``, or any
``Callable[[float], float]``, including a tabulated
:class:`~femtoolkit.materials.thermal_properties.TemperatureDependentProperty`)
-- when ``h`` is temperature-dependent, the convective term becomes
genuinely nonlinear in ``T``, and :mod:`femtoolkit.thermal.thermal_analysis`
switches from a direct linear solve to Newton-Raphson iteration to handle it.

**Time-dependent ambient temperature.** The ambient/fluid temperature
can also vary with time (a furnace heating up, an ambient cycle) --
represented by reusing
:class:`~femtoolkit.analysis.dynamic_loads.TimeDependentLoad` directly
(the same ``ConstantLoad``/``StepLoad``/``SinusoidalLoad`` shapes already
used for time-varying mechanical/thermal loads), since a time history has
no notion of "watts" vs. "kelvin" baked into its shape.

**Radiation (Version 21).** The Stefan-Boltzmann law gives the net
radiative heat flux leaving a surface exchanging heat with surroundings
at a different temperature:

.. code-block:: text

    q = epsilon * sigma * (T^4 - T_surroundings^4)

``epsilon`` (dimensionless, ``0 < epsilon <= 1``) is the surface's
emissivity -- how effectively it radiates compared to an ideal ("black
body") radiator; ``sigma`` is the Stefan-Boltzmann constant
(:data:`STEFAN_BOLTZMANN_CONSTANT`); ``T`` and ``T_surroundings`` **must**
be absolute (kelvin) temperatures, since the ``T^4`` dependence makes
radiation acutely sensitive to the temperature scale used (a Celsius
value raised to the fourth power is physically meaningless here).

**Why radiation makes the thermal problem nonlinear.** Every other term
in the heat equation (conduction, capacity, generation, prescribed flux,
constant-``h`` convection) is *linear* in the unknown temperature -- the
system is exactly ``K_T @ T = F_T``, solved once. Radiation's ``T^4``
term is not linear in ``T`` at all: doubling the surface temperature
scales the radiative loss by roughly sixteen times, not two. This means
``K_T @ T = F_T`` no longer holds as a fixed linear system -- the
"resistance" radiation offers depends on the very temperature being
solved for. :mod:`femtoolkit.thermal.thermal_analysis` handles this the
same way :class:`~femtoolkit.analysis.nonlinear_analysis.NonlinearAnalysis`
already handles material nonlinearity: Newton-Raphson iteration,
linearizing the ``T^4`` term about the current trial temperature at each
step (``d(T^4)/dT = 4*T^3``, a simple closed-form derivative) until the
residual vanishes.

**Designed for future extension.** :class:`PrescribedHeatFlux` stays a
simple, temperature-*independent* value, and
:class:`ConvectionBoundaryCondition` deliberately has no notion of
radiation or vice versa -- each new boundary-condition type has been
added here without modifying any of the ones that came before it,
exactly the extensibility this taxonomy was designed to preserve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from femtoolkit.analysis.dynamic_loads import TimeDependentLoad
from femtoolkit.exceptions import ValidationError
from femtoolkit.materials.thermal_properties import ThermalProperty, evaluate_thermal_property
from femtoolkit.thermal.thermal_surfaces import ThermalSurface

STEFAN_BOLTZMANN_CONSTANT: float = 5.670374419e-8
"""The Stefan-Boltzmann constant, sigma, in W/(m^2*K^4)."""


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


def convective_heat_flux(
    coefficient: float, surface_temperature: float, ambient_temperature: float
) -> float:
    """Return the convective heat flux, ``q = h * (T - T_infinity)`` (Newton's law of cooling).

    Distinct from the *conductive* flux,
    :func:`~femtoolkit.thermal.thermal_elements.element_heat_flux`
    (``q = -k * grad(T)``): conductive flux describes heat moving
    *through* a solid, while this describes heat leaving a *surface*
    into a surrounding fluid.

    Args:
        coefficient: The convection coefficient ``h``, in W/(m^2*K).
        surface_temperature: The surface temperature ``T``, in kelvin.
        ambient_temperature: The ambient fluid temperature
            ``T_infinity``, in kelvin.

    Returns:
        The heat flux, in W/m^2. Positive means heat leaving the surface
        (``T > T_infinity``); negative means the surface is being heated
        by the surrounding fluid (``T < T_infinity``).
    """
    return coefficient * (surface_temperature - ambient_temperature)


def radiative_heat_flux(
    emissivity: float, surface_temperature: float, surrounding_temperature: float
) -> float:
    """Return the net radiative heat flux, ``q = eps*sigma*(T^4 - T_surroundings^4)``.

    Args:
        emissivity: The surface's emissivity, ``epsilon`` (dimensionless,
            ``0 < epsilon <= 1``).
        surface_temperature: The surface's absolute temperature ``T``,
            in **kelvin**.
        surrounding_temperature: The surroundings' absolute temperature,
            in **kelvin**.

    Returns:
        The heat flux, in W/m^2. Positive means heat leaving the surface
        (the surface is hotter than its surroundings).

    Raises:
        ValidationError: If either temperature is negative (an absolute
            temperature cannot be negative on the kelvin scale).
    """
    if surface_temperature < 0 or surrounding_temperature < 0:
        raise ValidationError(
            "radiative_heat_flux requires absolute (kelvin) temperatures, got "
            f"surface_temperature={surface_temperature}, "
            f"surrounding_temperature={surrounding_temperature}."
        )
    return emissivity * STEFAN_BOLTZMANN_CONSTANT * (
        surface_temperature**4 - surrounding_temperature**4
    )


@dataclass(frozen=True)
class ConvectionBoundaryCondition:
    """A convective (Robin) boundary condition, ``q = h * (T - T_infinity)``.

    Attributes:
        surface: The mesh surface (edge or face) exposed to convection.
        convection_coefficient: The convection coefficient ``h``, in
            W/(m^2*K), or a temperature-dependent property (see the
            module docstring). Must be positive at every temperature it
            is evaluated at.
        ambient_temperature: The ambient fluid temperature, in kelvin,
            or a :class:`~femtoolkit.analysis.dynamic_loads.TimeDependentLoad`
            for a time-varying ambient condition (e.g. a furnace cycle).

    Raises:
        ValidationError: A constant-valued ``convection_coefficient`` is
            validated immediately; a temperature-dependent one is
            validated lazily, when evaluated (see
            :meth:`convection_coefficient_at`). A constant-valued
            ``ambient_temperature`` must be finite.

    Example:
        >>> air = ConvectionBoundaryCondition(
        ...     surface=ThermalSurface(element_id=1, local_face_index=0),
        ...     convection_coefficient=25.0,
        ...     ambient_temperature=293.15,
        ... )
    """

    surface: ThermalSurface
    convection_coefficient: ThermalProperty
    ambient_temperature: float | TimeDependentLoad

    def __post_init__(self) -> None:
        """Validate constant-valued fields immediately after construction.

        Raises:
            ValidationError: If a constant-valued
                ``convection_coefficient`` is not positive and finite,
                or a constant-valued ``ambient_temperature`` is not finite.
        """
        if not callable(self.convection_coefficient):
            self._validate_coefficient(self.convection_coefficient)
        if not isinstance(self.ambient_temperature, TimeDependentLoad) and not math.isfinite(
            self.ambient_temperature
        ):
            raise ValidationError(
                "ConvectionBoundaryCondition ambient_temperature must be finite, got "
                f"{self.ambient_temperature}."
            )

    @staticmethod
    def _validate_coefficient(value: float) -> None:
        if not math.isfinite(value) or value <= 0:
            raise ValidationError(
                f"ConvectionBoundaryCondition convection_coefficient must be positive, got {value}."
            )

    @property
    def is_temperature_dependent(self) -> bool:
        """Whether ``convection_coefficient`` varies with temperature (``h = h(T)``)."""
        return callable(self.convection_coefficient)

    def convection_coefficient_at(self, temperature: float) -> float:
        """Return the convection coefficient at ``temperature``.

        Raises:
            ValidationError: If the resulting value is not positive and finite.
        """
        value = evaluate_thermal_property(self.convection_coefficient, temperature)
        self._validate_coefficient(value)
        return value

    def ambient_temperature_at(self, time: float) -> float:
        """Return the ambient temperature at time ``time`` (seconds).

        For a constant ``ambient_temperature``, ``time`` is ignored.
        """
        if isinstance(self.ambient_temperature, TimeDependentLoad):
            return self.ambient_temperature.value_at(time)
        return self.ambient_temperature


@dataclass(frozen=True)
class RadiationBoundaryCondition:
    """A radiative boundary condition, ``q = epsilon*sigma*(T^4 - T_surroundings^4)``.

    Attributes:
        surface: The mesh surface (edge or face) exchanging radiative heat.
        emissivity: The surface's emissivity, ``epsilon`` (dimensionless).
            Must satisfy ``0 < epsilon <= 1``.
        surrounding_temperature: The surroundings' absolute temperature,
            in **kelvin**. Must be non-negative and finite.

    Raises:
        ValidationError: If ``emissivity`` is outside ``(0, 1]``, or
            ``surrounding_temperature`` is negative or not finite.

    Example:
        >>> furnace_wall = RadiationBoundaryCondition(
        ...     surface=ThermalSurface(element_id=1, local_face_index=0),
        ...     emissivity=0.85,
        ...     surrounding_temperature=293.15,
        ... )
    """

    surface: ThermalSurface
    emissivity: float
    surrounding_temperature: float

    def __post_init__(self) -> None:
        """Validate the radiation boundary condition immediately after construction.

        Raises:
            ValidationError: If ``emissivity`` is outside ``(0, 1]``, or
                ``surrounding_temperature`` is negative or not finite.
        """
        if not math.isfinite(self.emissivity) or not (0.0 < self.emissivity <= 1.0):
            raise ValidationError(
                f"RadiationBoundaryCondition emissivity must be within (0, 1], got "
                f"{self.emissivity}."
            )
        if not math.isfinite(self.surrounding_temperature) or self.surrounding_temperature < 0:
            raise ValidationError(
                "RadiationBoundaryCondition surrounding_temperature must be a non-negative, "
                f"finite kelvin value, got {self.surrounding_temperature}."
            )
