"""Thermal material properties for heat conduction (Version 20).

A heat-conduction analysis needs exactly three material properties,
independent of the mechanical properties (Young's modulus, Poisson's
ratio) used elsewhere in this toolkit:

* **Thermal conductivity** ``k`` (W/(m*K)) -- how readily the material
  conducts heat, appearing in Fourier's law and the conductivity matrix
  ``K_T``.
* **Density** ``rho`` (kg/m^3) -- reused, conceptually, from
  :class:`~femtoolkit.materials.linear_elastic_3d.LinearElastic3D` and
  friends, but kept as an independent field here (this class has no
  mechanical properties at all -- see the module docstring for why
  thermal and mechanical materials are deliberately kept separate).
* **Specific heat capacity** ``c`` (J/(kg*K)) -- how much energy a unit
  mass must absorb to raise its temperature by one kelvin; ``rho * c``
  is the volumetric heat capacity appearing in the capacity matrix
  ``C_T``.

**Why a separate class, not an extension of `ThermoelasticMaterial3D`.**
Section 14's brief is explicit: keep thermal properties separate from
mechanical internal variables, and do not mix thermal state with
plasticity state unnecessarily. Physically, thermal conduction and
mechanical (thermoelastic) response are two different physical
phenomena that happen to share a temperature field as a *coupling*
variable -- not properties of "one enlarged material." A single body
might reasonably be modeled with a :class:`ThermalMaterial` (for a heat-
conduction analysis) *and*, separately,
:class:`~femtoolkit.materials.thermoelastic.ThermoelasticMaterial3D`
(for the resulting thermal-stress analysis) -- the workflow this version
is building toward (see :mod:`femtoolkit.thermal.thermal_analysis`'s
module docstring for the sequential thermal -> mechanical connection).

**Isotropic conductivity.** ``k`` is a single scalar here -- the same
value in every direction (Fourier's law, ``q = -k * grad(T)``, uses it
as a scalar multiplying the gradient vector directly). A future version
could generalize ``k`` to a full 3x3 conductivity *tensor* for
anisotropic materials (e.g. composites, wood); this class's
``conductivity_at(T) -> float`` return type would become
``conductivity_at(T) -> np.ndarray`` at that point, but every *element*
formula in :mod:`femtoolkit.thermal.thermal_elements` already isolates
``k`` into a single ``k * identity`` factor, so that generalization
would only touch this one class, not the element formulas.

**Temperature-dependent conductivity.** Reuses Version 19's
:data:`~femtoolkit.materials.thermal_properties.ThermalProperty`
machinery directly rather than reinventing it: ``conductivity`` (and
``specific_heat``) may be a constant ``float`` or any
``Callable[[float], float]`` (including a
:class:`~femtoolkit.materials.thermal_properties.TemperatureDependentProperty`
lookup table).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials.thermal_properties import ThermalProperty, evaluate_thermal_property


@dataclass(frozen=True)
class ThermalMaterial:
    """An isotropic material's thermal conduction properties.

    Attributes:
        thermal_conductivity: Thermal conductivity ``k``, in W/(m*K), or
            a temperature-dependent property. Must be positive at every
            temperature it is evaluated at.
        density: Mass density ``rho``, in kg/m^3. Must be positive.
        specific_heat: Specific heat capacity ``c``, in J/(kg*K), or a
            temperature-dependent property. Must be positive at every
            temperature it is evaluated at.

    Raises:
        ValidationError: If ``density`` is not positive and finite.
            Constant-valued conductivity/specific heat are validated
            immediately; temperature-dependent ones are validated
            lazily, when evaluated at a specific temperature (see
            :meth:`conductivity_at`/:meth:`specific_heat_at`).

    Example:
        >>> steel = ThermalMaterial(thermal_conductivity=50.0, density=7850.0, specific_heat=460.0)
        >>> steel.conductivity_at(293.15)
        50.0
    """

    thermal_conductivity: ThermalProperty
    density: float
    specific_heat: ThermalProperty

    def __post_init__(self) -> None:
        """Validate ``density`` and any constant-valued property immediately.

        Raises:
            ValidationError: If ``density`` is not positive and finite,
                or a constant-valued ``thermal_conductivity``/``specific_heat``
                is not positive and finite.
        """
        if not math.isfinite(self.density) or self.density <= 0:
            raise ValidationError(f"ThermalMaterial density must be positive, got {self.density}.")
        if not callable(self.thermal_conductivity):
            self._validate_conductivity(self.thermal_conductivity)
        if not callable(self.specific_heat):
            self._validate_specific_heat(self.specific_heat)

    @staticmethod
    def _validate_conductivity(value: float) -> None:
        if not math.isfinite(value) or value <= 0:
            raise ValidationError(
                f"ThermalMaterial thermal_conductivity must be positive, got {value}."
            )

    @staticmethod
    def _validate_specific_heat(value: float) -> None:
        if not math.isfinite(value) or value <= 0:
            raise ValidationError(f"ThermalMaterial specific_heat must be positive, got {value}.")

    def conductivity_at(self, temperature: float) -> float:
        """Return the thermal conductivity ``k`` at ``temperature``.

        Raises:
            ValidationError: If the resulting value is not positive and
                finite (e.g. a tabulated property outside its range).
        """
        value = evaluate_thermal_property(self.thermal_conductivity, temperature)
        self._validate_conductivity(value)
        return value

    def specific_heat_at(self, temperature: float) -> float:
        """Return the specific heat capacity ``c`` at ``temperature``.

        Raises:
            ValidationError: If the resulting value is not positive and
                finite.
        """
        value = evaluate_thermal_property(self.specific_heat, temperature)
        self._validate_specific_heat(value)
        return value

    def volumetric_heat_capacity_at(self, temperature: float) -> float:
        """Return ``rho * c`` at ``temperature`` -- the capacity matrix's material factor."""
        return self.density * self.specific_heat_at(temperature)
