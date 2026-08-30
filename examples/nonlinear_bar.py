"""Example: incremental Newton-Raphson solution of a nonlinear bar system, Version 13.

Demonstrates incremental loading, the Newton-Raphson iteration itself,
and the resulting nonlinear displacement response for a 1D
elastic-perfectly-plastic system, following the workflow:

.. code-block:: text

    Load Increment -> Predict Displacement -> Calculate Strain ->
        Calculate Stress -> Calculate Internal Force ->
        Calculate Tangent Stiffness -> Calculate Residual ->
        Check Convergence -> Repeat if necessary -> Next Load Increment

:class:`~femtoolkit.analysis.nonlinear_analysis.NonlinearAnalysis` only
drives CST/Q4 continuum elements (see :mod:`femtoolkit.analysis.nonlinear_elements`
for why -- both require a 2D, 3-component material). A genuinely 1D bar
is instead solved directly here with
:class:`~femtoolkit.materials.nonlinear.ElasticPerfectlyPlasticMaterial1D`
and the same public convergence primitives
(:mod:`femtoolkit.analysis.convergence`) the orchestrator itself uses --
the "equivalent reduced system" the Version 13 spec explicitly permits.

Model: two parallel bars, both length ``L``, sharing a single
displacement DOF, with staggered yield capacities (bar 1 yields first).
A single perfectly-plastic bar loaded by pure force *past* its own yield
capacity has no equilibrium solution (its resisting force cannot exceed
``A * sigma_y`` no matter how far it stretches) -- two bars with
different yield stresses stay well-posed past the first bar's yield
point, because the second, still-elastic bar supplies residual
stiffness. This lets the example show a genuine elastic region,
yielding, a reduced tangent stiffness, and converged Newton-Raphson
iterations, then compare the final state against the closed-form
bilinear solution.
"""

import numpy as np

from femtoolkit.analysis.convergence import residual_norm_ratio
from femtoolkit.materials import ElasticPerfectlyPlasticMaterial1D

YOUNGS_MODULUS = 200e9  # Pa
LENGTH = 1.0  # m
AREA_1 = 0.001  # m^2
AREA_2 = 0.001  # m^2
YIELD_STRESS_1 = 150e6  # Pa, yields first
YIELD_STRESS_2 = 400e6  # Pa, stays elastic longer
LOAD_STEPS = 10
MAX_ITERATIONS = 30
TOLERANCE = 1e-10


def main() -> None:
    """Push a two-bar reduced system past bar 1's yield point and report convergence."""
    bar1 = ElasticPerfectlyPlasticMaterial1D(
        youngs_modulus=YOUNGS_MODULUS, yield_stress=YIELD_STRESS_1
    )
    bar2 = ElasticPerfectlyPlasticMaterial1D(
        youngs_modulus=YOUNGS_MODULUS, yield_stress=YIELD_STRESS_2
    )

    f_yield_1 = AREA_1 * YIELD_STRESS_1
    f_target = f_yield_1 + AREA_2 * 200e6  # comfortably exceeds bar 1's capacity

    print("Finite Element Toolkit")
    print("Version 13 -- Nonlinear Bar (Newton-Raphson, Two-Bar Reduced System)")
    print("=" * 40)
    print(f"\nBar 1: A = {AREA_1} m^2, sigma_y = {YIELD_STRESS_1 / 1e6:.0f} MPa")
    print(f"Bar 2: A = {AREA_2} m^2, sigma_y = {YIELD_STRESS_2 / 1e6:.0f} MPa")
    print(f"Target force: {f_target:.1f} N over {LOAD_STEPS} load steps")

    u_committed = 0.0
    state1 = bar1.initial_state()
    state2 = bar2.initial_state()

    print(f"\n{'step':>4} {'F (N)':>12} {'u (m)':>14} {'sigma1':>12} {'sigma2':>12} {'iters':>6}")
    for step in range(1, LOAD_STEPS + 1):
        load_factor = step / LOAD_STEPS
        f_ext = load_factor * f_target
        u_trial = u_committed
        converged = False
        iterations_used = 0

        for iteration in range(1, MAX_ITERATIONS + 1):
            iterations_used = iteration
            strain = u_trial / LENGTH
            trial1 = bar1.trial_state(strain, state1)
            trial2 = bar2.trial_state(strain, state2)
            f_int = AREA_1 * trial1.stress + AREA_2 * trial2.stress
            residual = f_ext - f_int

            ratio = residual_norm_ratio(
                np.array([residual]), np.array([f_ext if f_ext != 0.0 else 1.0])
            )
            if ratio < TOLERANCE:
                converged = True
                break

            tangent_1 = AREA_1 * bar1.tangent_modulus(trial1)
            tangent_2 = AREA_2 * bar2.tangent_modulus(trial2)
            u_trial += residual / ((tangent_1 + tangent_2) / LENGTH)

        assert converged, f"load step {step} failed to converge"
        u_committed = u_trial
        state1, state2 = trial1, trial2

        print(
            f"{step:4d} {f_ext:12.1f} {u_committed:14.6e} "
            f"{state1.stress:12.4e} {state2.stress:12.4e} {iterations_used:6d}"
        )

    print(f"\nBar 1 yielded: {state1.yielded}, Bar 2 yielded: {state2.yielded}")

    # Analytical cross-check: once bar 1 yields, F = A1*sigma_y1 + A2*E*(u/L).
    analytical_u = (f_target - AREA_1 * YIELD_STRESS_1) * LENGTH / (AREA_2 * YOUNGS_MODULUS)
    print(f"\nFinal displacement:      {u_committed:.6e} m")
    print(f"Analytical displacement: {analytical_u:.6e} m")
    print(f"Match: {abs(u_committed - analytical_u) / analytical_u < 1e-6}")


if __name__ == "__main__":
    main()
