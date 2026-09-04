"""Example: St. Venant-Kirchhoff vs. Neo-Hookean vs. Mooney-Rivlin, Version 17.

Sweeps a uniaxial stretch (with a physically appropriate lateral Poisson-
like contraction, not a boundary-value solve -- a pure material-response
comparison, matching examples/finite_strain_elastic.py's style) through
all three finite-strain material models implemented so far
(SaintVenantKirchhoff3D from Version 16, NeoHookean3D and MooneyRivlin3D
from Version 17), all parameterized to match at small strain (same
initial shear modulus), and reports how far they diverge as the stretch
grows -- demonstrating that all three finite-strain models agree closely
in the small-strain limit, but diverge sharply at genuinely large strain,
where St. Venant-Kirchhoff's simple linear ``S = C:E`` law (extrapolated
far past the small-strain regime it was derived for) makes stress grow
far faster than Neo-Hookean or Mooney-Rivlin's genuinely nonlinear,
rubber-appropriate response.
"""

import numpy as np

from femtoolkit.continuum.stress import cauchy_stress_from_second_piola_kirchhoff
from femtoolkit.continuum.tensor import voigt_stress_to_tensor
from femtoolkit.materials import MooneyRivlin3D, NeoHookean3D, SaintVenantKirchhoff3D

YOUNGS_MODULUS = 5.0e6  # Pa
POISSON_RATIO = 0.45

STRETCH_SWEEP = [1.01, 1.1, 1.3, 1.6, 2.0, 2.5, 3.0]


def main() -> None:
    """Sweep uniaxial stretch through all three finite-strain materials and compare stress."""
    svk = SaintVenantKirchhoff3D(youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO)
    neo_hookean = NeoHookean3D(youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO)

    # Match Mooney-Rivlin's small-strain shear modulus to Neo-Hookean's mu, with a
    # modest C01 contribution (see the module docstring for mooney_rivlin.py), and
    # a bulk modulus derived from the same Young's modulus/Poisson ratio pair.
    shear_modulus = neo_hookean.shear_modulus
    bulk_modulus = YOUNGS_MODULUS / (3.0 * (1.0 - 2.0 * POISSON_RATIO))
    c01 = 0.2 * shear_modulus / 2.0
    c10 = shear_modulus / 2.0 - c01
    mooney_rivlin = MooneyRivlin3D(c10=c10, c01=c01, bulk_modulus=bulk_modulus)

    print("Finite Element Toolkit")
    print("Version 17 -- Comparing Finite-Strain Material Models")
    print("=" * 70)
    print(f"\nE = {YOUNGS_MODULUS:.3e} Pa, v = {POISSON_RATIO}")
    print(f"Neo-Hookean:     mu={neo_hookean.shear_modulus:.4e} Pa")
    print(f"Mooney-Rivlin:   C10={mooney_rivlin.c10:.4e} Pa, C01={mooney_rivlin.c01:.4e} Pa")
    print(f"                 (initial shear modulus {mooney_rivlin.initial_shear_modulus:.4e} Pa)")

    print(f"\n{'stretch':>8}  {'SVK sigma_11':>14}  {'NeoHookean sigma_11':>20}  "
          f"{'MooneyRivlin sigma_11':>22}")
    for stretch in STRETCH_SWEEP:
        lateral = stretch ** (-0.5)  # isochoric-like lateral contraction
        f = np.diag([stretch, lateral, lateral])

        axial_strain = 0.5 * (stretch**2 - 1.0)
        svk_strain = np.array([axial_strain, 0.5 * (lateral**2 - 1.0), 0.5 * (lateral**2 - 1.0),
                                0.0, 0.0, 0.0])
        svk_state = svk.trial_state(svk_strain, svk.initial_state())
        # SVK's Cauchy stress via the standard S -> sigma conversion at this F.
        svk_sigma = cauchy_stress_from_second_piola_kirchhoff(
            f, voigt_stress_to_tensor(svk_state.stress)
        )[0, 0]

        neo_hookean_sigma = neo_hookean.cauchy_stress(f)[0, 0]
        mooney_rivlin_sigma = mooney_rivlin.cauchy_stress(f)[0, 0]

        print(
            f"{stretch:>8.2f}  {svk_sigma:>14.4e}  {neo_hookean_sigma:>20.4e}  "
            f"{mooney_rivlin_sigma:>22.4e}"
        )

    print(
        "\nAt small stretch, all three models agree closely (matched small-strain "
        "shear modulus). At large stretch they diverge sharply: St. Venant-"
        "Kirchhoff's stress grows unboundedly fast (its linear S=C:E law, "
        "extrapolated far past the small-strain regime it was derived for, makes "
        "stress grow roughly with the *square* of Green-Lagrange strain), while "
        "Neo-Hookean and Mooney-Rivlin grow much more modestly, as a real rubber "
        "does -- the defining reason genuinely nonlinear hyperelastic models exist."
    )


if __name__ == "__main__":
    main()
