"""Example: bilinear isotropic hardening material, Version 14.

Demonstrates :class:`~femtoolkit.materials.hardening.BilinearIsotropicHardeningMaterial1D`
in isolation -- no mesh, no solver, just the constitutive law -- sweeping
a strain history through its two regimes:

.. code-block:: text

    Elastic:  sigma = E * epsilon               while |sigma| <= sigma_y0 + H*alpha
    Plastic:  sigma = sigma_trial - E*d_gamma*sign(sigma_trial)   (return-mapped)

Unlike Version 13's perfectly-plastic material (where stress plateaus
forever once yielded), the yield surface here **expands** with
accumulated plastic strain ``alpha``, so stress keeps growing -- just
more slowly (at the reduced tangent ``E*H/(E+H)``) than in the elastic
region.
"""

import numpy as np

from femtoolkit.materials import BilinearIsotropicHardeningMaterial1D

YOUNGS_MODULUS = 200e9  # Pa
YIELD_STRESS = 250e6  # Pa
HARDENING_MODULUS = 20e9  # Pa


def main() -> None:
    """Sweep a monotonically increasing strain history through elastic and hardening regimes."""
    material = BilinearIsotropicHardeningMaterial1D(
        youngs_modulus=YOUNGS_MODULUS,
        yield_stress=YIELD_STRESS,
        hardening_modulus=HARDENING_MODULUS,
    )
    yield_strain = YIELD_STRESS / YOUNGS_MODULUS

    print("Finite Element Toolkit")
    print("Version 14 -- Bilinear Isotropic Hardening Material")
    print("=" * 40)
    print(f"\nE = {YOUNGS_MODULUS / 1e9:.0f} GPa, sigma_y0 = {YIELD_STRESS / 1e6:.0f} MPa, "
          f"H = {HARDENING_MODULUS / 1e9:.0f} GPa")
    print(f"Yield strain: {yield_strain:.6e}")

    print(f"\n{'strain':>14} {'stress (Pa)':>16} {'tangent (Pa)':>16} {'alpha':>12} {'yielded':>8}")

    state = material.initial_state()
    strains = np.concatenate(
        [
            np.linspace(0.0, yield_strain, 4),
            np.linspace(yield_strain, 6.0 * yield_strain, 6)[1:],
        ]
    )
    for strain in strains:
        state = material.trial_state(float(strain), state)
        tangent = material.tangent_modulus(state)
        print(
            f"{strain:14.6e} {state.stress:16.6e} {tangent:16.6e} "
            f"{state.hardening_variable:12.6e} {state.yielded!s:>8}"
        )

    print(f"\nFinal plastic strain: {state.plastic_strain:.6e}")
    print(f"Final stress ({state.stress / 1e6:.1f} MPa) exceeds sigma_y0 -- this is hardening,")
    print("not a plateau: perfectly-plastic behavior would have stayed at exactly 250 MPa.")


if __name__ == "__main__":
    main()
