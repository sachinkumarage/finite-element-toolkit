"""Validation: a 1D nonlinear bar with bilinear isotropic hardening (Section 15).

Unlike Version 13's perfectly-plastic bar (which needed a two-bar
reduced system to stay well-posed under force control past yield, since
a perfectly-plastic bar's resisting force cannot exceed ``A*sigma_y``),
a *hardening* bar's resisting force keeps increasing with displacement
indefinitely -- so a single bar under simple force control is already
well-posed here. This uses the same hand-rolled Newton-Raphson pattern
as ``examples/nonlinear_bar.py`` (:class:`~femtoolkit.materials.nonlinear.NonlinearMaterial`
and :mod:`femtoolkit.analysis.convergence` primitives directly, since
:class:`~femtoolkit.analysis.nonlinear_analysis.NonlinearAnalysis` only
drives CST/Q4 continuum elements).
"""

import numpy as np
import pytest

from femtoolkit.analysis.convergence import residual_norm_ratio
from femtoolkit.materials import BilinearIsotropicHardeningMaterial1D

YOUNGS_MODULUS = 200e9
YIELD_STRESS = 250e6
HARDENING_MODULUS = 20e9
AREA = 0.001
LENGTH = 1.0
MAX_ITERATIONS = 30
TOLERANCE = 1e-10


def _solve_bar(f_target: float, load_steps: int) -> list[dict]:
    material = BilinearIsotropicHardeningMaterial1D(
        youngs_modulus=YOUNGS_MODULUS,
        yield_stress=YIELD_STRESS,
        hardening_modulus=HARDENING_MODULUS,
    )
    u_committed = 0.0
    state = material.initial_state()
    history = []

    for step in range(1, load_steps + 1):
        load_factor = step / load_steps
        f_ext = load_factor * f_target
        u_trial = u_committed
        converged = False
        iterations = 0

        for iteration in range(1, MAX_ITERATIONS + 1):
            iterations = iteration
            strain = u_trial / LENGTH
            trial = material.trial_state(strain, state)
            f_int = AREA * trial.stress
            residual = f_ext - f_int
            external = f_ext if f_ext != 0.0 else 1.0
            ratio = residual_norm_ratio(np.array([residual]), np.array([external]))
            if ratio < TOLERANCE:
                converged = True
                break
            tangent = AREA * material.tangent_modulus(trial) / LENGTH
            u_trial += residual / tangent

        assert converged, f"load step {step} failed to converge"
        u_committed = u_trial
        state = trial
        history.append(
            {
                "load_factor": load_factor,
                "f_ext": f_ext,
                "u": u_committed,
                "stress": trial.stress,
                "plastic_strain": trial.plastic_strain,
                "yielded": trial.yielded,
                "iterations": iterations,
            }
        )

    return history


def test_elastic_displacement_matches_hookes_law() -> None:
    f_yield = AREA * YIELD_STRESS
    history = _solve_bar(f_target=0.8 * f_yield, load_steps=4)

    for record in history:
        assert not record["yielded"]
        expected_u = record["f_ext"] * LENGTH / (AREA * YOUNGS_MODULUS)
        assert record["u"] == pytest.approx(expected_u, rel=1e-8)


def test_yielding_and_post_yield_stiffness_and_plastic_strain() -> None:
    f_yield = AREA * YIELD_STRESS
    history = _solve_bar(f_target=2.0 * f_yield, load_steps=10)

    yielded_records = [record for record in history if record["yielded"]]
    elastic_records = [record for record in history if not record["yielded"]]
    assert yielded_records
    assert elastic_records
    for record in yielded_records:
        assert record["plastic_strain"] > 0.0

    # Closed-form single-bar monotonic post-yield solution:
    # strain = [(F/A)*(1 + H/E) - sigma_y0] / H
    for record in yielded_records:
        analytical_strain = (
            (record["f_ext"] / AREA) * (1.0 + HARDENING_MODULUS / YOUNGS_MODULUS) - YIELD_STRESS
        ) / HARDENING_MODULUS
        assert record["u"] / LENGTH == pytest.approx(analytical_strain, rel=1e-8)


def test_final_stress_and_convergence() -> None:
    f_yield = AREA * YIELD_STRESS
    history = _solve_bar(f_target=2.0 * f_yield, load_steps=10)
    final = history[-1]

    assert final["yielded"]
    assert final["stress"] > YIELD_STRESS  # hardening: stress exceeds the initial yield stress
    assert all(record["iterations"] <= MAX_ITERATIONS for record in history)
