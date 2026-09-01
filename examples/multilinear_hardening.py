"""Example: multilinear isotropic hardening curve evaluation, Version 14.

Demonstrates :class:`~femtoolkit.materials.hardening.MultilinearIsotropicHardeningMaterial1D`,
which represents a full stress-strain curve directly (elastic segment
included) rather than a single hardening slope -- useful when a
material's real hardening response has more than one distinct region
(e.g. rapid initial hardening followed by a flatter region, a common
shape for structural steels beyond their first yield plateau).
"""

import numpy as np

from femtoolkit.materials import MultilinearIsotropicHardeningMaterial1D

STRAIN_POINTS = (0.0, 0.001, 0.005, 0.02)
STRESS_POINTS = (0.0, 200e6, 250e6, 300e6)


def main() -> None:
    """Evaluate the multilinear curve across its elastic segment and every hardening segment."""
    material = MultilinearIsotropicHardeningMaterial1D(
        strain_points=STRAIN_POINTS, stress_points=STRESS_POINTS
    )

    print("Finite Element Toolkit")
    print("Version 14 -- Multilinear Isotropic Hardening")
    print("=" * 40)
    curve_points = list(zip(STRAIN_POINTS, STRESS_POINTS, strict=True))
    print(f"\nCurve points (strain, stress): {curve_points}")
    print(f"Derived elastic modulus (segment 1): {material.youngs_modulus / 1e9:.1f} GPa")

    print(f"\n{'strain':>14} {'stress (Pa)':>16} {'tangent (Pa)':>16} {'yielded':>8}")

    state = material.initial_state()
    strains = np.concatenate(
        [
            np.linspace(0.0, STRAIN_POINTS[1], 3),
            np.linspace(STRAIN_POINTS[1], STRAIN_POINTS[2], 3)[1:],
            np.linspace(STRAIN_POINTS[2], STRAIN_POINTS[3], 3)[1:],
            [STRAIN_POINTS[3] * 1.25],  # extrapolation beyond the last tabulated point
        ]
    )
    for strain in strains:
        state = material.trial_state(float(strain), state)
        tangent = material.tangent_modulus(state)
        print(f"{strain:14.6e} {state.stress:16.6e} {tangent:16.6e} {state.yielded!s:>8}")

    print(f"\nFinal plastic strain (elastic-unload approximation): {state.plastic_strain:.6e}")


if __name__ == "__main__":
    main()
