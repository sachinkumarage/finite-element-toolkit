"""Example: temperature-dependent material properties, Version 19.

Demonstrates femtoolkit.materials.thermal_properties.TemperatureDependentProperty:
Young's modulus and the thermal expansion coefficient both given as
tabulated, temperature-interpolated values rather than single constants
-- a more realistic model of how real materials actually behave as they
heat up (typically softening: lower stiffness, higher expansion
coefficient, at higher temperature).
"""

import numpy as np

from femtoolkit.materials.thermal_properties import TemperatureDependentProperty
from femtoolkit.materials.thermoelastic import ThermoelasticMaterial3D

REFERENCE_TEMPERATURE = 293.15  # K (20 C)

# A simple, illustrative steel-like softening/expansion table.
YOUNGS_MODULUS_TABLE = TemperatureDependentProperty(
    temperatures=(293.15, 473.15, 673.15, 873.15),
    values=(200e9, 190e9, 170e9, 140e9),
)
THERMAL_EXPANSION_TABLE = TemperatureDependentProperty(
    temperatures=(293.15, 473.15, 673.15, 873.15),
    values=(11e-6, 12.5e-6, 14e-6, 15.5e-6),
)

TEMPERATURE_SWEEP = [293.15, 373.15, 473.15, 573.15, 673.15, 773.15, 873.15]


def main() -> None:
    """Sweep temperature and report how the material's evaluated properties change."""
    material = ThermoelasticMaterial3D(
        youngs_modulus=YOUNGS_MODULUS_TABLE,
        poisson_ratio=0.3,
        thermal_expansion_coefficient=THERMAL_EXPANSION_TABLE,
        reference_temperature=REFERENCE_TEMPERATURE,
        density=7850.0,
    )

    print("Finite Element Toolkit")
    print("Version 19 -- Temperature-Dependent Material Properties")
    print("=" * 60)
    print(f"\n{'T (K)':>8}  {'E(T) (Pa)':>14}  {'alpha(T) (1/K)':>16}  {'thermal strain':>16}")
    for temperature in TEMPERATURE_SWEEP:
        youngs_modulus = material.youngs_modulus_at(temperature)
        alpha = material.thermal_expansion_coefficient_at(temperature)
        thermal_strain = material.thermal_strain_voigt(temperature)[0]
        print(
            f"{temperature:>8.2f}  {youngs_modulus:>14.4e}  {alpha:>16.4e}  {thermal_strain:>16.4e}"
        )

    print(
        "\nBoth properties are linearly interpolated between the tabulated points -- "
        "note E(T) decreasing (softening) and alpha(T) increasing (more expansive) "
        "with temperature, a common trend for structural metals."
    )

    print("\nRequesting a temperature outside the tabulated range raises an error:")
    try:
        material.youngs_modulus_at(1000.0)
    except Exception as error:  # noqa: BLE001 -- intentionally broad, for the demonstration
        print(f"    {type(error).__name__}: {error}")

    print("\nA constant (non-tabulated) property is also supported directly:")
    simple_material = ThermoelasticMaterial3D(
        youngs_modulus=200e9,
        poisson_ratio=0.3,
        thermal_expansion_coefficient=12e-6,
        reference_temperature=REFERENCE_TEMPERATURE,
        density=7850.0,
    )
    print(
        f"    E at 293.15 K: {simple_material.youngs_modulus_at(293.15):.4e} Pa, "
        f"E at 873.15 K: {simple_material.youngs_modulus_at(873.15):.4e} Pa (unchanged)"
    )

    print("\nA user-defined function is equally valid (no lookup table required):")

    def linear_softening(temperature: float) -> float:
        return 200e9 * (1.0 - 5e-4 * (temperature - REFERENCE_TEMPERATURE))

    functional_material = ThermoelasticMaterial3D(
        youngs_modulus=linear_softening,
        poisson_ratio=0.3,
        thermal_expansion_coefficient=12e-6,
        reference_temperature=REFERENCE_TEMPERATURE,
        density=7850.0,
    )
    for temperature in (293.15, 573.15, 873.15):
        youngs_modulus = functional_material.youngs_modulus_at(temperature)
        print(f"    E at {temperature:.2f} K: {youngs_modulus:.4e} Pa")

    assert np.isclose(functional_material.youngs_modulus_at(REFERENCE_TEMPERATURE), 200e9)


if __name__ == "__main__":
    main()
