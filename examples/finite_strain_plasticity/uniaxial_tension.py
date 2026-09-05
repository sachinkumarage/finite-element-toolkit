"""Example: finite-strain J2 plasticity under uniaxial tension, Version 18.

Sweeps a large uniaxial stretch (well beyond the small-strain regime)
through femtoolkit.materials.finite_strain_plasticity.J2FiniteStrainPlasticity3D,
demonstrating elastic loading, yield onset, and isotropic hardening --
mirroring examples/finite_strain_elastic.py's (Version 16) style, but for
a genuinely dissipative (plastic) finite-strain material rather than an
elastic one.
"""

import numpy as np

from femtoolkit.materials.finite_strain_plasticity import J2FiniteStrainPlasticity3D

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
YIELD_STRESS = 250e6  # Pa
HARDENING_MODULUS = 10e9  # Pa

# Uniaxial stretch sweep -- lambda=1 is the reference (undeformed) state.
STRETCH_SWEEP = [1.0005, 1.001, 1.002, 1.005, 1.01, 1.02, 1.05, 1.1]


def _uniaxial_deformation_gradient(stretch: float, lateral_stretch: float) -> np.ndarray:
    """A pure-stretch (no rotation) uniaxial deformation gradient."""
    return np.diag([stretch, lateral_stretch, lateral_stretch])


def main() -> None:
    """Sweep uniaxial stretch and report the elastic-to-plastic stress response."""
    steel = J2FiniteStrainPlasticity3D(
        youngs_modulus=YOUNGS_MODULUS,
        poisson_ratio=POISSON_RATIO,
        yield_stress=YIELD_STRESS,
        hardening_modulus=HARDENING_MODULUS,
    )

    print("Finite Element Toolkit")
    print("Version 18 -- Finite-Strain J2 Plasticity: Uniaxial Tension")
    print("=" * 65)
    print(f"\nE = {YOUNGS_MODULUS:.3e} Pa, v = {POISSON_RATIO}")
    print(f"sigma_y0 = {YIELD_STRESS:.3e} Pa, H = {HARDENING_MODULUS:.3e} Pa")

    print(f"\n{'stretch':>9}  {'sigma_xx (Cauchy)':>18}  {'yielded':>8}  {'alpha':>12}")
    committed = steel.initial_state()
    for stretch in STRETCH_SWEEP:
        # An isochoric-like lateral contraction, a reasonable approximation of a
        # free-lateral-surface uniaxial tension test (not enforced exactly, since
        # this is a single-material-point sweep, not a boundary-value solve --
        # see examples/finite_strain_plasticity/plastic_block_hex8.py for that).
        lateral_stretch = stretch ** (-0.3)
        deformation_gradient = _uniaxial_deformation_gradient(stretch, lateral_stretch)
        green_lagrange = 0.5 * (deformation_gradient.T @ deformation_gradient - np.eye(3))
        strain_voigt = np.array(
            [green_lagrange[0, 0], green_lagrange[1, 1], green_lagrange[2, 2], 0.0, 0.0, 0.0]
        )

        state = steel.trial_state(strain_voigt, committed)
        cauchy = steel.cauchy_stress(state, deformation_gradient)
        print(
            f"{stretch:>9.4f}  {cauchy[0, 0]:>18.6e}  {str(state.yielded):>8}  "
            f"{state.hardening_variable:>12.6e}"
        )
        committed = state

    print(
        "\nOnce yielded, the Cauchy stress continues to rise -- isotropic hardening --"
        " but far more slowly than the initial elastic slope, and the equivalent"
        " plastic strain (alpha) accumulates monotonically: permanent, unrecoverable"
        " deformation, unlike the purely elastic Version 17 hyperelastic materials."
    )


if __name__ == "__main__":
    main()
