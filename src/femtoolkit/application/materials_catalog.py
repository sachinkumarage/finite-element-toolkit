"""A small catalog of preset engineering materials (Version 24).

Section 7 of this version's brief asks the GUI to let a user "select a
material" rather than only type in raw numbers. This module provides a
handful of common structural/thermal materials with standard textbook
property values (room-temperature, isotropic, approximate -- the same
level of precision every prior version's examples already use), so a
new project starts from a sensible default instead of an empty form.
``"Custom"`` is always available for a user-entered material.

These are plain :class:`~femtoolkit.application.project.MaterialConfig`
values -- data, not solver objects -- consistent with the rest of the
project/application layer never holding a live material instance.
"""

from __future__ import annotations

from femtoolkit.application.project import MaterialConfig

CUSTOM_MATERIAL_KEY = "custom"
"""The catalog key reserved for a user-entered material."""

MATERIAL_CATALOG: dict[str, MaterialConfig] = {
    "structural_steel": MaterialConfig(
        name="Structural Steel",
        youngs_modulus=200e9,
        poisson_ratio=0.30,
        density=7850.0,
        thermal_conductivity=45.0,
        specific_heat=460.0,
    ),
    "aluminum_6061": MaterialConfig(
        name="Aluminum 6061-T6",
        youngs_modulus=69e9,
        poisson_ratio=0.33,
        density=2700.0,
        thermal_conductivity=167.0,
        specific_heat=896.0,
    ),
    "copper": MaterialConfig(
        name="Copper (C11000)",
        youngs_modulus=117e9,
        poisson_ratio=0.34,
        density=8960.0,
        thermal_conductivity=390.0,
        specific_heat=385.0,
    ),
    "titanium_ti6al4v": MaterialConfig(
        name="Titanium Ti-6Al-4V",
        youngs_modulus=114e9,
        poisson_ratio=0.34,
        density=4430.0,
        thermal_conductivity=6.7,
        specific_heat=526.0,
    ),
    CUSTOM_MATERIAL_KEY: MaterialConfig(name="Custom Material"),
}


def material_options() -> list[str]:
    """Return every catalog key, in a stable display order."""
    return list(MATERIAL_CATALOG)


def get_material_preset(key: str) -> MaterialConfig:
    """Look up one preset material by its catalog key.

    Args:
        key: A key from :func:`material_options`.

    Returns:
        A fresh copy of the matching :class:`~femtoolkit.application.project.MaterialConfig`
        (a copy, not the shared catalog instance, so a caller can freely
        edit it without mutating the catalog).

    Raises:
        KeyError: If ``key`` is not a known catalog entry.
    """
    if key not in MATERIAL_CATALOG:
        raise KeyError(f"Unknown material preset {key!r}; known presets: {material_options()}.")
    preset = MATERIAL_CATALOG[key]
    return MaterialConfig(
        name=preset.name,
        youngs_modulus=preset.youngs_modulus,
        poisson_ratio=preset.poisson_ratio,
        density=preset.density,
        thermal_conductivity=preset.thermal_conductivity,
        specific_heat=preset.specific_heat,
    )
