"""Example: bilinear kinematic hardening and the Bauschinger effect, Version 14.

Demonstrates :class:`~femtoolkit.materials.hardening.BilinearKinematicHardeningMaterial1D`
through a full cyclic loading path:

.. code-block:: text

    0 -> tension (past yield) -> unload -> compression (past reverse yield)
        -> unload -> tension again

Unlike isotropic hardening (which grows the yield surface symmetrically
about zero), kinematic hardening *translates* a fixed-size yield surface
by a back stress ``X``. The practical consequence -- the **Bauschinger
effect** -- is the point of this example: after yielding in tension, the
material re-yields in *compression* at a smaller stress magnitude than
its virgin compressive yield stress, because the yield surface has
shifted with it.
"""

from femtoolkit.materials import BilinearKinematicHardeningMaterial1D

YOUNGS_MODULUS = 200e9  # Pa
YIELD_STRESS = 250e6  # Pa
HARDENING_MODULUS = 20e9  # Pa


def main() -> None:
    """Run a tension -> unload -> compression -> unload -> tension cycle."""
    material = BilinearKinematicHardeningMaterial1D(
        youngs_modulus=YOUNGS_MODULUS,
        yield_stress=YIELD_STRESS,
        hardening_modulus=HARDENING_MODULUS,
    )
    yield_strain = YIELD_STRESS / YOUNGS_MODULUS

    print("Finite Element Toolkit")
    print("Version 14 -- Bilinear Kinematic Hardening (Bauschinger Effect)")
    print("=" * 40)
    print(f"\nE = {YOUNGS_MODULUS / 1e9:.0f} GPa, sigma_y0 = {YIELD_STRESS / 1e6:.0f} MPa, "
          f"H = {HARDENING_MODULUS / 1e9:.0f} GPa")

    state = material.initial_state()
    print(f"\n{'stage':>22} {'strain':>14} {'stress (Pa)':>14} {'back stress':>14} {'yielded':>8}")

    def step(stage: str, strain: float) -> None:
        nonlocal state
        state = material.trial_state(strain, state)
        print(
            f"{stage:>22} {strain:14.6e} {state.stress:14.4e} "
            f"{state.back_stress:14.4e} {state.yielded!s:>8}"
        )

    step("tension load", 3.0 * yield_strain)
    tension_back_stress = state.back_stress

    step("unload to zero stress", state.plastic_strain)

    reverse_yield_stress = tension_back_stress - YIELD_STRESS
    reverse_yield_strain = state.plastic_strain + reverse_yield_stress / YOUNGS_MODULUS
    step("reverse yield onset", reverse_yield_strain)

    step("compression load", reverse_yield_strain - 2.0 * yield_strain)

    step("unload to zero stress", state.plastic_strain)
    step("reload in tension", state.plastic_strain + 3.0 * yield_strain)

    print("\nBauschinger effect:")
    print(f"    Virgin compressive yield stress:  {-YIELD_STRESS / 1e6:8.2f} MPa")
    print(f"    Actual reverse yield stress:       {reverse_yield_stress / 1e6:8.2f} MPa")
    reduced = abs(reverse_yield_stress) < YIELD_STRESS
    print(f"    Magnitude reduced by kinematic hardening: {reduced}")


if __name__ == "__main__":
    main()
