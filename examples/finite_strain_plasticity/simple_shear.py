"""Example: finite-strain J2 plasticity under simple shear, Version 18.

Simple shear is a classic large-deformation test case: unlike uniaxial
tension, it involves genuine *rotation* of material fibers even though
the deformation gradient itself is applied "statically" (no rigid-body
motion is superposed) -- a good demonstration that this material's
return-mapping algorithm, built entirely on the (rotation-invariant)
right Cauchy-Green tensor, handles it correctly without any special
casing.
"""

import numpy as np

from femtoolkit.materials.finite_strain_plasticity import J2FiniteStrainPlasticity3D

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
YIELD_STRESS = 250e6  # Pa
HARDENING_MODULUS = 10e9  # Pa

SHEAR_SWEEP = [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05]


def _simple_shear_deformation_gradient(gamma: float) -> np.ndarray:
    """F = I + gamma * e_x (x) e_y (the classic simple-shear deformation gradient)."""
    f = np.eye(3)
    f[0, 1] = gamma
    return f


def main() -> None:
    """Sweep an engineering shear strain and report the elastic-to-plastic shear response."""
    steel = J2FiniteStrainPlasticity3D(
        youngs_modulus=YOUNGS_MODULUS,
        poisson_ratio=POISSON_RATIO,
        yield_stress=YIELD_STRESS,
        hardening_modulus=HARDENING_MODULUS,
    )

    print("Finite Element Toolkit")
    print("Version 18 -- Finite-Strain J2 Plasticity: Simple Shear")
    print("=" * 60)
    print(f"\nmu (shear modulus) = {steel.shear_modulus:.4e} Pa")
    print(f"sigma_y0 = {YIELD_STRESS:.3e} Pa, H = {HARDENING_MODULUS:.3e} Pa")

    print(f"\n{'gamma':>8}  {'sigma_xy (Cauchy)':>18}  {'yielded':>8}  {'alpha':>12}")
    committed = steel.initial_state()
    for gamma in SHEAR_SWEEP:
        deformation_gradient = _simple_shear_deformation_gradient(gamma)
        green_lagrange = 0.5 * (deformation_gradient.T @ deformation_gradient - np.eye(3))
        strain_voigt = np.array(
            [
                green_lagrange[0, 0], green_lagrange[1, 1], green_lagrange[2, 2],
                2.0 * green_lagrange[0, 1], 0.0, 0.0,
            ]
        )

        state = steel.trial_state(strain_voigt, committed)
        cauchy = steel.cauchy_stress(state, deformation_gradient)
        print(
            f"{gamma:>8.4f}  {cauchy[0, 1]:>18.6e}  {str(state.yielded):>8}  "
            f"{state.hardening_variable:>12.6e}"
        )
        committed = state

    print(
        "\nElastic shear stress grows linearly with gamma (sigma_xy ~= mu*gamma) until "
        "yield, then grows far more slowly under hardening -- the same qualitative "
        "elastic-to-plastic transition seen in uniaxial_tension.py, here driven by a "
        "deformation mode that genuinely rotates material fibers."
    )


if __name__ == "__main__":
    main()
