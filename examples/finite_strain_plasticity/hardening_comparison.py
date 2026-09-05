"""Example: finite-strain vs. small-strain J2 plasticity, Version 18.

Directly compares femtoolkit.materials.finite_strain_plasticity.J2FiniteStrainPlasticity3D
against femtoolkit.materials.j2_plasticity.J2Plasticity3D (Version 15,
small-strain) under the same uniaxial strain history, with identical
elastic and hardening parameters -- demonstrating the connection this
version's README documents: the two models agree closely at small
strain (where the finite-strain formulation's logarithmic strain measure
and the small-strain formulation's linear strain measure coincide), then
diverge once strain grows large enough that the distinction between
"engineering" and "true" (logarithmic) strain becomes significant.
"""

import numpy as np

from femtoolkit.materials.finite_strain_plasticity import J2FiniteStrainPlasticity3D
from femtoolkit.materials.j2_plasticity import J2Plasticity3D

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
YIELD_STRESS = 250e6  # Pa
HARDENING_MODULUS = 10e9  # Pa

STRAIN_SWEEP = [1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]


def main() -> None:
    """Sweep uniaxial strain through both material models and compare hardening evolution."""
    finite_strain_material = J2FiniteStrainPlasticity3D(
        youngs_modulus=YOUNGS_MODULUS,
        poisson_ratio=POISSON_RATIO,
        yield_stress=YIELD_STRESS,
        hardening_modulus=HARDENING_MODULUS,
    )
    small_strain_material = J2Plasticity3D(
        youngs_modulus=YOUNGS_MODULUS,
        poisson_ratio=POISSON_RATIO,
        yield_stress=YIELD_STRESS,
        hardening_modulus=HARDENING_MODULUS,
    )

    print("Finite Element Toolkit")
    print("Version 18 -- Finite-Strain vs. Small-Strain J2 Plasticity")
    print("=" * 65)
    print(f"\nE = {YOUNGS_MODULUS:.3e} Pa, v = {POISSON_RATIO}")
    print(f"sigma_y0 = {YIELD_STRESS:.3e} Pa, H = {HARDENING_MODULUS:.3e} Pa")

    print(
        f"\n{'E_xx':>9}  {'S_xx (finite-strain)':>21}  {'sigma_xx (small-strain)':>24}  "
        f"{'rel. diff':>10}"
    )
    fs_state = finite_strain_material.initial_state()
    ss_state = small_strain_material.initial_state()
    for axial_strain in STRAIN_SWEEP:
        strain = np.array([axial_strain, -0.3 * axial_strain, -0.3 * axial_strain, 0, 0, 0])
        fs_state = finite_strain_material.trial_state(strain, fs_state)
        ss_state = small_strain_material.trial_state(strain, ss_state)

        relative_difference = abs(fs_state.stress[0] - ss_state.stress[0]) / abs(ss_state.stress[0])
        print(
            f"{axial_strain:>9.5f}  {fs_state.stress[0]:>21.6e}  {ss_state.stress[0]:>24.6e}  "
            f"{relative_difference:>10.4%}"
        )

    print(
        "\nAt small strain, the relative difference is a small fraction of a percent -- "
        "the finite-strain model's logarithmic strain measure reduces to the small-"
        "strain model's linear one, as proven in this project's own test suite "
        "(tests/validation/test_finite_strain_plasticity_small_strain_limit.py). At "
        "larger strain, the two models diverge: only the finite-strain model remains "
        "valid once rotation and large stretch genuinely matter."
    )


if __name__ == "__main__":
    main()
