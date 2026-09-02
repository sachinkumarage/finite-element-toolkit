"""Example: J2 (von Mises) material behavior in isolation, Version 15.

Demonstrates the J2Plasticity3D material model directly -- no elements or
mesh involved -- sweeping a uniaxial strain history through the elastic
region, first yield, and progressive hardening, and reporting the von
Mises equivalent stress, yield status, and accumulated equivalent plastic
strain at each step. See femtoolkit.materials.j2_plasticity for the full
radial-return derivation this class implements.
"""

from femtoolkit.continuum.stress import von_mises_3d
from femtoolkit.materials import J2Plasticity3D

YOUNGS_MODULUS = 200e9  # Pa (steel)
POISSON_RATIO = 0.3
YIELD_STRESS = 250e6  # Pa
HARDENING_MODULUS = 10e9  # Pa

# A monotonically increasing uniaxial strain sweep.
STRAIN_SWEEP = [0.0005, 0.001, 0.0015, 0.002, 0.004, 0.006, 0.01, 0.015, 0.02]


def main() -> None:
    """Sweep a uniaxial strain history through the J2 material and report each step."""
    steel = J2Plasticity3D(
        youngs_modulus=YOUNGS_MODULUS,
        poisson_ratio=POISSON_RATIO,
        yield_stress=YIELD_STRESS,
        hardening_modulus=HARDENING_MODULUS,
    )

    print("Finite Element Toolkit")
    print("Version 15 -- J2 (von Mises) Material Uniaxial Strain Sweep")
    print("=" * 60)
    print(f"\nE = {YOUNGS_MODULUS:.3e} Pa, v = {POISSON_RATIO}, "
          f"sigma_y0 = {YIELD_STRESS:.3e} Pa, H = {HARDENING_MODULUS:.3e} Pa")

    state = steel.initial_state()
    print(f"\n{'strain':>10}  {'sigma_xx':>14}  {'von Mises':>14}  {'yielded':>8}  {'alpha':>12}")
    for axial_strain in STRAIN_SWEEP:
        # Free lateral contraction (uniaxial *stress*-like boundary condition,
        # approximated here by imposing the elastic Poisson ratio laterally --
        # see examples/tet4_j2_plasticity.py for the true FE uniaxial-stress test).
        lateral_strain = -steel.poisson_ratio * axial_strain
        strain = [axial_strain, lateral_strain, lateral_strain, 0.0, 0.0, 0.0]

        state = steel.trial_state(strain, state)
        von_mises = von_mises_3d(*state.stress)
        print(
            f"{axial_strain:>10.5f}  {state.stress[0]:>14.6e}  {von_mises:>14.6e}  "
            f"{str(state.yielded):>8}  {state.hardening_variable:>12.6e}"
        )

    print("\nFinal tangent modulus (6x6):")
    tangent = steel.tangent_modulus(state)
    for row in tangent:
        print("    " + "  ".join(f"{value:>12.4e}" for value in row))


if __name__ == "__main__":
    main()
