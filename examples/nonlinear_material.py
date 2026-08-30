"""Example: the 1D elastic-perfectly-plastic material model, Version 13.

Demonstrates :class:`~femtoolkit.materials.nonlinear.ElasticPerfectlyPlasticMaterial1D`
in isolation -- no mesh, no solver, just the constitutive law itself --
sweeping a strain history through its three regimes:

.. code-block:: text

    Elastic:   sigma = E * epsilon                       while |sigma| <= sigma_y
    Yielded:   sigma = sign(epsilon_elastic) * sigma_y    once |sigma| would exceed sigma_y

At each strain, the material also reports its tangent modulus
``D_t = d(sigma)/d(epsilon)`` (``E`` while elastic, a small regularized
value once yielded -- see the material module's docstring for why an
exact zero is avoided) and whether the point has yielded, following the
trial/committed state pattern every Version 13 nonlinear material and
solver relies on (see :mod:`femtoolkit.materials.nonlinear`).
"""

import numpy as np

from femtoolkit.materials import ElasticPerfectlyPlasticMaterial1D

YOUNGS_MODULUS = 200e9  # Pa
YIELD_STRESS = 250e6  # Pa


def main() -> None:
    """Sweep a strain history from zero, through yield, and back (unload/reload)."""
    material = ElasticPerfectlyPlasticMaterial1D(
        youngs_modulus=YOUNGS_MODULUS, yield_stress=YIELD_STRESS
    )
    yield_strain = YIELD_STRESS / YOUNGS_MODULUS

    print("Finite Element Toolkit")
    print("Version 13 -- Elastic-Perfectly-Plastic 1D Material")
    print("=" * 40)
    print(f"\nE = {YOUNGS_MODULUS / 1e9:.0f} GPa, sigma_y = {YIELD_STRESS / 1e6:.0f} MPa")
    print(f"Yield strain: {yield_strain:.6e}")

    # --- Loading: elastic region, then well past yield ---
    print("\nLoading (strain increasing from 0 to 4x yield strain):")
    print(f"{'strain':>14} {'stress (Pa)':>16} {'tangent (Pa)':>16} {'yielded':>8}")

    state = material.initial_state()
    loading_strains = np.concatenate(
        [
            np.linspace(0.0, yield_strain, 5),
            np.linspace(yield_strain, 4.0 * yield_strain, 5)[1:],
        ]
    )
    for strain in loading_strains:
        state = material.trial_state(float(strain), state)
        tangent = material.tangent_modulus(state)
        print(f"{strain:14.6e} {state.stress:16.6e} {tangent:16.6e} {state.yielded!s:>8}")

    peak_state = state
    print(f"\nAfter loading:\n    plastic strain = {peak_state.plastic_strain:.6e}")

    # --- Unloading: elastic slope back down from the plastic point ---
    print("\nUnloading (strain decreasing back to the plastic strain):")
    print(f"{'strain':>14} {'stress (Pa)':>16} {'tangent (Pa)':>16} {'yielded':>8}")

    unloading_strains = np.linspace(peak_state.strain, peak_state.plastic_strain, 5)
    for strain in unloading_strains:
        state = material.trial_state(float(strain), peak_state)
        tangent = material.tangent_modulus(state)
        print(f"{strain:14.6e} {state.stress:16.6e} {tangent:16.6e} {state.yielded!s:>8}")

    print(f"\nStress at plastic strain (should be ~0): {state.stress:.6e} Pa")
    print("\nModel limitation: no hardening -- stress never exceeds +/- sigma_y.")


if __name__ == "__main__":
    main()
