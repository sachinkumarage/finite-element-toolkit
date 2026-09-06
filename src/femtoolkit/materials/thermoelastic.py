"""Isotropic linear thermoelasticity for 3D solids (Version 19).

**Temperature vs. displacement fields.** Every material through Version
18 evaluates stress purely as a function of the *displacement* field
(through strain) -- temperature has never entered the picture. Physically,
a body's temperature ``T(X)`` is a second, independent scalar field over
the reference configuration, exactly as legitimate an input to a
material's response as the displacement field ``u(X)`` is -- but distinct
in kind: ``u(X)`` is *solved for* (the analysis's whole purpose), while
``T(X)`` in this version is *prescribed* (a known, given input, like a
load) -- reflecting real thermal-stress analysis practice: apply a known
temperature distribution, solve for the resulting displacement, strain,
and stress. Solving for the temperature field *itself* (heat conduction)
is a separate, future capability (see the Version 20 preview).

**Thermal strain.** Heating or cooling a *free* (unrestrained) chunk of
isotropic material by ``dT = T - T_ref`` makes it expand or contract
uniformly in every direction, with no accompanying shear -- a pure
volume change, no shape change:

.. code-block:: text

    epsilon_thermal = alpha * dT * I

``alpha`` is the material's coefficient of thermal expansion (1/K);
``T_ref`` is the temperature at which the material is in its natural,
stress-free state (``dT = 0 => epsilon_thermal = 0``). This is an
**eigenstrain**: a strain that exists with *no* associated stress,
exactly like the plastic strain in
:mod:`femtoolkit.materials.j2_plasticity` is a strain with no immediate
associated stress -- both are strains the material "wants" to have
regardless of loading, only becoming a source of *stress* when something
(a support, a neighboring, differently-heated region) prevents the body
from actually reaching them.

**Strain decomposition and thermoelastic stress.** For small-strain
thermoelasticity, the total strain (from displacements, exactly as
before) splits additively:

.. code-block:: text

    epsilon = epsilon_mechanical + epsilon_thermal
    epsilon_mechanical = epsilon - epsilon_thermal

    sigma = C : epsilon_mechanical = C : (epsilon - epsilon_thermal)

Critically, stress depends on the **mechanical** strain, never the total
strain directly -- a free thermal expansion (``epsilon = epsilon_thermal``
exactly, nothing resisting it) gives ``epsilon_mechanical = 0`` and
therefore **zero stress**, even though the total strain and displacement
are both very much nonzero. A **fully restrained** body, by contrast, has
``epsilon = 0`` everywhere (no displacement is possible at all), so
``epsilon_mechanical = -epsilon_thermal`` -- the entire thermal eigenstrain
becomes mechanical strain, producing genuine **thermal stress**. Both
cases are mandatory validation tests in this version (see
``tests/validation/test_thermoelastic_free_expansion.py`` and
``tests/validation/test_thermoelastic_constrained.py``).

**Why temperature is a construction-time parameter, not a `trial_state`
argument.** :class:`~femtoolkit.materials.nonlinear.NonlinearMaterial`'s
interface is ``trial_state(strain, committed_state) -> MaterialState``,
with no way to pass a third, independently-varying quantity -- and
retrofitting one in would force *every* existing material (Versions
13-18) to accept and ignore a parameter that means nothing to them,
exactly the "forcing temperature into unrelated APIs" this version's
brief warns against. Since temperature in this version is a *prescribed*
field, not a solved unknown, it does not need to vary during a Newton-
Raphson iteration at all -- so it can simply be a **fixed parameter, set
once, at construction time**, exactly like Young's modulus or Poisson's
ratio already are. :meth:`ThermoelasticMaterial3D.at_temperature` builds
exactly this: a :class:`~femtoolkit.materials.nonlinear.NonlinearMaterial`
with a specific temperature baked in, usable anywhere the existing TET4/HEX8
dispatch (:mod:`femtoolkit.analysis.nonlinear_elements`) already expects
one, with **zero changes** to that dispatch or to
:class:`~femtoolkit.analysis.nonlinear_analysis.NonlinearAnalysis` --
since both already accept a *different* material per element,
spatially-varying temperature is simply a different ``at_temperature(T)``
instance per element (see
:mod:`femtoolkit.analysis.temperature_field` for the mesh-wide
convenience that builds exactly this mapping). This mirrors the existing
precedent of :class:`~femtoolkit.materials.nonlinear.ElasticMaterialAdapter`,
which likewise exists purely to expose an otherwise-plain elastic
material (:class:`~femtoolkit.materials.linear_elastic_3d.LinearElastic3D`)
through the :class:`~femtoolkit.materials.nonlinear.NonlinearMaterial`
interface.

**Temperature-dependent properties.** ``youngs_modulus``, ``poisson_ratio``,
and ``thermal_expansion_coefficient`` are each a
:data:`~femtoolkit.materials.thermal_properties.ThermalProperty`: a
constant ``float``, or any ``Callable[[float], float]`` (e.g. a
:class:`~femtoolkit.materials.thermal_properties.TemperatureDependentProperty`
lookup table) evaluated at the material's current (fixed) temperature.
This is intentionally *not* a temperature-dependent tangent in the
plasticity sense -- the material remains linear-elastic at any single,
fixed temperature; only *which* linear-elastic constants apply changes
with temperature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from femtoolkit.continuum.constitutive import isotropic_3d_matrix
from femtoolkit.exceptions import ValidationError
from femtoolkit.materials.nonlinear import MaterialState, NonlinearMaterial
from femtoolkit.materials.thermal_properties import ThermalProperty, evaluate_thermal_property

_MIN_POISSONS_RATIO = -1.0
_MAX_POISSONS_RATIO = 0.5


@dataclass(frozen=True)
class ThermoelasticMaterial3D:
    """An isotropic linear thermoelastic material for 3D solid (TET4/HEX8) analysis.

    Not itself a :class:`~femtoolkit.materials.nonlinear.NonlinearMaterial`
    (there is no single, fixed temperature to evaluate a response at
    until one is chosen -- see :meth:`at_temperature`), exactly like
    :class:`~femtoolkit.materials.linear_elastic_3d.LinearElastic3D` is
    not one either without
    :meth:`~femtoolkit.materials.nonlinear.ElasticMaterialAdapter.from_linear_elastic_3d`.

    Attributes:
        youngs_modulus: Young's modulus ``E``, in pascals, or a
            temperature-dependent property. Must be positive at every
            temperature it is evaluated at.
        poisson_ratio: Poisson's ratio ``v`` (dimensionless), or a
            temperature-dependent property. Must lie within
            ``(-1.0, 0.5)`` at every temperature it is evaluated at.
        thermal_expansion_coefficient: Coefficient of linear thermal
            expansion ``alpha``, in 1/K, or a temperature-dependent
            property. Must be finite at every temperature it is
            evaluated at (negative values -- materials that contract on
            heating -- are physically valid and permitted).
        reference_temperature: The temperature ``T_ref``, in kelvin, at
            which the material is in its natural, stress-free state
            (``dT = 0``). Must be finite.
        density: Mass density, in kg/m^3. Must be positive.

    Raises:
        ValidationError: If ``reference_temperature`` is not finite, or
            ``density`` is not positive and finite. Constant-valued
            elastic/thermal properties are validated immediately;
            temperature-dependent ones are validated lazily, when
            actually evaluated at a specific temperature (see
            :meth:`youngs_modulus_at` etc.).

    Example:
        >>> steel = ThermoelasticMaterial3D(
        ...     youngs_modulus=200e9,
        ...     poisson_ratio=0.3,
        ...     thermal_expansion_coefficient=12e-6,
        ...     reference_temperature=293.15,
        ...     density=7850.0,
        ... )
        >>> heated_steel = steel.at_temperature(393.15)
        >>> heated_steel.temperature
        393.15
    """

    youngs_modulus: ThermalProperty
    poisson_ratio: ThermalProperty
    thermal_expansion_coefficient: ThermalProperty
    reference_temperature: float
    density: float

    def __post_init__(self) -> None:
        """Validate constant-valued properties and required scalars immediately.

        Raises:
            ValidationError: If ``reference_temperature`` is not finite,
                ``density`` is not positive and finite, or any
                *constant-valued* (non-callable) elastic/thermal
                property is not physically valid.
        """
        if not math.isfinite(self.reference_temperature):
            raise ValidationError(
                "ThermoelasticMaterial3D reference_temperature must be finite, got "
                f"{self.reference_temperature}."
            )
        if not math.isfinite(self.density) or self.density <= 0:
            raise ValidationError(
                f"ThermoelasticMaterial3D density must be positive, got {self.density}."
            )
        # Constant-valued properties can (and should) be validated now; a
        # temperature-dependent one is validated lazily, at each temperature
        # it is actually evaluated at (see youngs_modulus_at/poisson_ratio_at/
        # thermal_expansion_coefficient_at) -- there is no single "the" value
        # to check yet.
        if not callable(self.youngs_modulus):
            self._validate_youngs_modulus(self.youngs_modulus)
        if not callable(self.poisson_ratio):
            self._validate_poisson_ratio(self.poisson_ratio)
        if not callable(self.thermal_expansion_coefficient):
            self._validate_thermal_expansion_coefficient(self.thermal_expansion_coefficient)

    @staticmethod
    def _validate_youngs_modulus(value: float) -> None:
        if not math.isfinite(value) or value <= 0:
            raise ValidationError(
                f"ThermoelasticMaterial3D youngs_modulus must be positive, got {value}."
            )

    @staticmethod
    def _validate_poisson_ratio(value: float) -> None:
        if not math.isfinite(value) or not (_MIN_POISSONS_RATIO < value < _MAX_POISSONS_RATIO):
            raise ValidationError(
                "ThermoelasticMaterial3D poisson_ratio must be within "
                f"({_MIN_POISSONS_RATIO}, {_MAX_POISSONS_RATIO}), got {value}."
            )

    @staticmethod
    def _validate_thermal_expansion_coefficient(value: float) -> None:
        if not math.isfinite(value):
            raise ValidationError(
                "ThermoelasticMaterial3D thermal_expansion_coefficient must be finite, got "
                f"{value}."
            )

    def youngs_modulus_at(self, temperature: float) -> float:
        """Return Young's modulus at ``temperature``.

        Raises:
            ValidationError: If the resulting value is not positive and
                finite (e.g. a tabulated property outside its range).
        """
        value = evaluate_thermal_property(self.youngs_modulus, temperature)
        self._validate_youngs_modulus(value)
        return value

    def poisson_ratio_at(self, temperature: float) -> float:
        """Return Poisson's ratio at ``temperature``.

        Raises:
            ValidationError: If the resulting value is outside the
                physically valid range.
        """
        value = evaluate_thermal_property(self.poisson_ratio, temperature)
        self._validate_poisson_ratio(value)
        return value

    def thermal_expansion_coefficient_at(self, temperature: float) -> float:
        """Return the thermal expansion coefficient at ``temperature``.

        Raises:
            ValidationError: If the resulting value is not finite.
        """
        value = evaluate_thermal_property(self.thermal_expansion_coefficient, temperature)
        self._validate_thermal_expansion_coefficient(value)
        return value

    def constitutive_matrix_at(self, temperature: float) -> np.ndarray:
        """Return the 6x6 isotropic elastic constitutive matrix ``C`` at ``temperature``."""
        return isotropic_3d_matrix(
            self.youngs_modulus_at(temperature), self.poisson_ratio_at(temperature)
        )

    def thermal_strain_voigt(self, temperature: float) -> np.ndarray:
        """Return the thermal (eigen)strain, Voigt ``[a*dT, a*dT, a*dT, 0, 0, 0]``.

        No thermal *shear* strain: isotropic thermal expansion is a pure
        volume change (see the module docstring).
        """
        alpha = self.thermal_expansion_coefficient_at(temperature)
        delta_temperature = temperature - self.reference_temperature
        normal_strain = alpha * delta_temperature
        return np.array([normal_strain, normal_strain, normal_strain, 0.0, 0.0, 0.0])

    def mechanical_strain_voigt(
        self, total_strain_voigt: np.ndarray, temperature: float
    ) -> np.ndarray:
        """Return ``epsilon_mechanical = epsilon_total - epsilon_thermal``, at ``temperature``."""
        return np.asarray(total_strain_voigt, dtype=float) - self.thermal_strain_voigt(temperature)

    def stress_at(self, total_strain_voigt: np.ndarray, temperature: float) -> np.ndarray:
        """Return ``sigma = C : (epsilon_total - epsilon_thermal)``, at ``temperature``.

        The core thermoelastic constitutive law -- a pure function of
        total strain and temperature, needing no material *state* at
        all (linear thermoelasticity is exactly as path-independent as
        plain linear elasticity).
        """
        mechanical_strain = self.mechanical_strain_voigt(total_strain_voigt, temperature)
        return self.constitutive_matrix_at(temperature) @ mechanical_strain

    def at_temperature(self, temperature: float) -> ThermoelasticMaterialAtTemperature:
        """Bind this material to a fixed ``temperature``, as a usable :class:`NonlinearMaterial`.

        Args:
            temperature: The (fixed) temperature to evaluate this
                material's response at, in kelvin.

        Returns:
            A :class:`ThermoelasticMaterialAtTemperature` wrapping this
            material and ``temperature``.
        """
        return ThermoelasticMaterialAtTemperature(base_material=self, temperature=temperature)


@dataclass(frozen=True)
class ThermoelasticMaterialAtTemperature(NonlinearMaterial):
    """A :class:`ThermoelasticMaterial3D` evaluated at one fixed temperature.

    The actual :class:`~femtoolkit.materials.nonlinear.NonlinearMaterial`
    supplied to :class:`~femtoolkit.analysis.nonlinear_analysis.NonlinearAnalysis`
    for a thermoelastic TET4/HEX8 analysis -- see the module docstring
    for why temperature is fixed here rather than threaded through
    ``trial_state``. Path-independent (like any linear elastic material):
    ``trial_state`` ignores ``committed_state`` entirely.

    Attributes:
        base_material: The underlying temperature-dependent material.
        temperature: The fixed temperature this instance evaluates
            ``base_material`` at, in kelvin.
    """

    base_material: ThermoelasticMaterial3D
    temperature: float

    def initial_state(self) -> MaterialState:
        """Return the zero-strain, zero-(mechanical)-stress initial state at :attr:`temperature`."""
        return MaterialState(
            strain=np.zeros(6),
            stress=np.zeros(6),
            plastic_strain=np.zeros(6),
            yielded=False,
            temperature=self.temperature,
        )

    def trial_state(self, strain: np.ndarray, committed_state: MaterialState) -> MaterialState:
        """Compute ``sigma = C : (epsilon - epsilon_thermal)`` at the fixed :attr:`temperature`.

        Args:
            strain: The proposed total strain, Voigt
                ``[epsilon_xx, epsilon_yy, epsilon_zz, gamma_xy, gamma_yz, gamma_xz]``.
            committed_state: Unused (path-independent; see the class docstring).

        Returns:
            A new :class:`~femtoolkit.materials.nonlinear.MaterialState` at ``strain``.
        """
        del committed_state
        strain_array = np.asarray(strain, dtype=float)
        stress = self.base_material.stress_at(strain_array, self.temperature)
        return MaterialState(
            strain=strain_array,
            stress=stress,
            plastic_strain=np.zeros(6),
            yielded=False,
            temperature=self.temperature,
        )

    def tangent_modulus(self, state: MaterialState) -> np.ndarray:
        """Return the (constant, at this fixed temperature) elastic constitutive matrix."""
        del state
        return self.base_material.constitutive_matrix_at(self.temperature)
