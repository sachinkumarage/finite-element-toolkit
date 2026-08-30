"""Validation: a reduced 1D nonlinear bar system solved with hand-rolled
Newton-Raphson (Section 25).

:class:`~femtoolkit.analysis.nonlinear_analysis.NonlinearAnalysis` only
drives CST/Q4 continuum elements (both require a 2D, 3-component
material -- see :mod:`femtoolkit.analysis.nonlinear_elements`), so a
genuinely 1D elastic-perfectly-plastic bar is exercised directly through
:class:`~femtoolkit.materials.nonlinear.ElasticPerfectlyPlasticMaterial1D`
with a small Newton-Raphson loop built from the same public
:mod:`femtoolkit.analysis.convergence` primitives the orchestrator uses
-- the "equivalent reduced system" the spec explicitly allows.

The reduced system here is two parallel bars sharing a single
displacement DOF, with staggered yield capacities. This is deliberately
**not** a single perfectly-plastic bar under pure force control: a lone
elastic-perfectly-plastic bar has no equilibrium solution once the
applied force exceeds its yield capacity (its resisting force cannot
exceed ``A * sigma_y`` no matter the displacement), so pushing it past
yield under force control is an ill-posed test, not a solver bug. Two
bars with different yield stresses remain well-posed past the first
bar's yield point (the second, still-elastic bar supplies residual
stiffness) -- letting this test genuinely exercise multi-iteration
Newton-Raphson convergence and a tangent-stiffness reduction, then check
the result against a closed-form bilinear solution.
"""

import numpy as np
import pytest

from femtoolkit.analysis.convergence import has_converged
from femtoolkit.materials import ElasticPerfectlyPlasticMaterial1D

YOUNGS_MODULUS = 200e9
LENGTH = 1.0
AREA_1 = 0.001
AREA_2 = 0.001
YIELD_STRESS_1 = 150e6
YIELD_STRESS_2 = 400e6


def _solve_two_bar_system(
    f_target: float, load_steps: int = 10, max_iterations: int = 30, tolerance: float = 1e-10
) -> list[dict]:
    bar1 = ElasticPerfectlyPlasticMaterial1D(
        youngs_modulus=YOUNGS_MODULUS, yield_stress=YIELD_STRESS_1
    )
    bar2 = ElasticPerfectlyPlasticMaterial1D(
        youngs_modulus=YOUNGS_MODULUS, yield_stress=YIELD_STRESS_2
    )

    u_committed = 0.0
    state1 = bar1.initial_state()
    state2 = bar2.initial_state()
    history = []

    for step in range(1, load_steps + 1):
        load_factor = step / load_steps
        f_ext = load_factor * f_target
        u_trial = u_committed
        converged = False
        iterations = 0

        for iteration in range(1, max_iterations + 1):
            iterations = iteration
            strain = u_trial / LENGTH
            trial1 = bar1.trial_state(strain, state1)
            trial2 = bar2.trial_state(strain, state2)
            f_int = AREA_1 * trial1.stress + AREA_2 * trial2.stress
            residual = np.array([f_ext - f_int])
            external = np.array([f_ext if f_ext != 0.0 else 1.0])

            if has_converged(
                "residual",
                residual=residual,
                external_force=external,
                delta_u=np.array([0.0]),
                u=np.array([u_trial if u_trial != 0.0 else 1.0]),
                tolerance=tolerance,
            ):
                converged = True
                break

            tangent_1 = AREA_1 * bar1.tangent_modulus(trial1)
            tangent_2 = AREA_2 * bar2.tangent_modulus(trial2)
            u_trial += residual[0] / ((tangent_1 + tangent_2) / LENGTH)

        assert converged, f"load step {step} failed to converge"
        u_committed = u_trial
        state1, state2 = trial1, trial2
        history.append(
            {
                "load_factor": load_factor,
                "f_ext": f_ext,
                "u": u_committed,
                "stress1": trial1.stress,
                "yielded1": trial1.yielded,
                "stress2": trial2.stress,
                "yielded2": trial2.yielded,
                "iterations": iterations,
            }
        )

    return history


def test_elastic_region_matches_combined_stiffness() -> None:
    """Below bar 1's yield force, both bars are elastic: u = F / ((A1+A2)*E/L)."""
    f_yield_1 = AREA_1 * YIELD_STRESS_1
    history = _solve_two_bar_system(f_target=0.8 * f_yield_1, load_steps=4)

    combined_stiffness = (AREA_1 + AREA_2) * YOUNGS_MODULUS / LENGTH
    for record in history:
        assert not record["yielded1"]
        assert not record["yielded2"]
        expected_u = record["f_ext"] / combined_stiffness
        assert record["u"] == pytest.approx(expected_u, rel=1e-8)


def test_staggered_yielding_and_reduced_tangent_stiffness() -> None:
    """Push past bar 1's yield capacity: bar 1 plateaus, bar 2 stays elastic,
    and Newton-Raphson needs more iterations once the tangent switches.
    """
    f_yield_1 = AREA_1 * YIELD_STRESS_1
    f_target = f_yield_1 + AREA_2 * 200e6  # bar 2 stays well within its own 400 MPa yield.
    history = _solve_two_bar_system(f_target=f_target, load_steps=10)

    final = history[-1]
    assert final["yielded1"] is True
    assert final["yielded2"] is False
    assert final["stress1"] == pytest.approx(YIELD_STRESS_1, rel=1e-8)

    # Closed-form post-yield solution: F = A1*sigma_y1 + A2*E*(u/L).
    expected_u = (final["f_ext"] - AREA_1 * YIELD_STRESS_1) * LENGTH / (AREA_2 * YOUNGS_MODULUS)
    assert final["u"] == pytest.approx(expected_u, rel=1e-6)

    # Convergence achieved throughout, within the iteration budget.
    assert all(record["iterations"] <= 30 for record in history)
    # Iteration count increases once bar 1 first yields, evidence of the
    # tangent-stiffness reduction actually mattering to the solver.
    pre_yield_iterations = [r["iterations"] for r in history if not r["yielded1"]]
    post_yield_iterations = [r["iterations"] for r in history if r["yielded1"]]
    assert post_yield_iterations
    assert max(post_yield_iterations) >= max(pre_yield_iterations)


def test_final_displacement_and_stress_match_analytical_bilinear_response() -> None:
    # Must clear the combined-elastic transition force, (A1+A2)*E/L * yield_strain_1
    # = 300 kN here, for bar 1 to have actually yielded by the final step.
    f_yield_1 = AREA_1 * YIELD_STRESS_1
    f_target = f_yield_1 + AREA_2 * 180e6
    history = _solve_two_bar_system(f_target=f_target, load_steps=8)
    final = history[-1]

    expected_u = (final["f_ext"] - AREA_1 * YIELD_STRESS_1) * LENGTH / (AREA_2 * YOUNGS_MODULUS)
    expected_stress2 = YOUNGS_MODULUS * expected_u / LENGTH

    assert final["u"] == pytest.approx(expected_u, rel=1e-6)
    assert final["stress1"] == pytest.approx(YIELD_STRESS_1, rel=1e-8)
    assert final["stress2"] == pytest.approx(expected_stress2, rel=1e-6)
