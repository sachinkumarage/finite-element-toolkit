"""Example: St. Venant-Kirchhoff finite-strain elastic response, Version 16.

Demonstrates femtoolkit.materials.finite_strain.SaintVenantKirchhoff3D in
isolation: sweeps a uniaxial Green-Lagrange strain history and reports the
resulting second Piola-Kirchhoff stress, comparing against the constant
constitutive matrix (S = C : E is exactly linear -- St. Venant-Kirchhoff's
defining, simplifying property) and against the small-strain limit where
Green-Lagrange strain and engineering strain coincide.
"""

import numpy as np

from femtoolkit.materials.finite_strain import SaintVenantKirchhoff3D

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3

STRAIN_SWEEP = [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2]


def main() -> None:
    """Sweep a uniaxial Green-Lagrange strain history and report the SVK stress response."""
    material = SaintVenantKirchhoff3D(youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO)

    print("Finite Element Toolkit")
    print("Version 16 -- St. Venant-Kirchhoff Finite-Strain Elastic Material")
    print("=" * 60)
    print(f"\nE = {YOUNGS_MODULUS:.3e} Pa, v = {POISSON_RATIO}")

    print(f"\n{'E_xx':>10}  {'S_xx':>14}  {'S_yy':>14}  {'S_zz':>14}")
    for axial_strain in STRAIN_SWEEP:
        # Uniaxial Green-Lagrange strain (no lateral strain prescribed here --
        # this is a pure material-response sweep, not a boundary-value problem;
        # see examples/nonlinear_tet4.py for a genuine uniaxial-stress solve).
        strain = np.array([axial_strain, 0.0, 0.0, 0.0, 0.0, 0.0])
        state = material.trial_state(strain, material.initial_state())
        print(
            f"{axial_strain:>10.5f}  {state.stress[0]:>14.6e}  {state.stress[1]:>14.6e}  "
            f"{state.stress[2]:>14.6e}"
        )

    print("\nTangent modulus (constant, since S is linear in E):")
    tangent = material.tangent_modulus(material.initial_state())
    for row in tangent:
        print("    " + "  ".join(f"{value:>12.4e}" for value in row))

    print(
        "\nAt small strain, S_xx/E_xx approaches the constitutive matrix's "
        f"[0,0] entry ({tangent[0, 0]:.4e} Pa) -- the St. Venant-Kirchhoff model "
        "recovers ordinary linear elasticity in that limit."
    )


if __name__ == "__main__":
    main()
