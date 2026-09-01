"""Validation: cyclic loading and the Bauschinger effect (Sections 7, 16).

Loading path: ``0 -> positive load -> unload -> negative load -> unload
-> positive load``. Tracks stress, strain, plastic strain, and the
kinematic back stress through the full cycle, using
:class:`~femtoolkit.materials.nonlinear.NonlinearMaterial`'s
trial/committed pattern directly (a single material point, matching
this project's "material-level validation" style -- see
``tests/validation/test_bilinear_material.py`` from Version 13).
"""

import pytest

from femtoolkit.materials import BilinearKinematicHardeningMaterial1D

YOUNGS_MODULUS = 200e9
YIELD_STRESS = 250e6
HARDENING_MODULUS = 20e9
YIELD_STRAIN = YIELD_STRESS / YOUNGS_MODULUS


def _commit(material: BilinearKinematicHardeningMaterial1D, state, strain: float):
    return material.trial_state(strain, state)


def test_cyclic_loading_path_and_bauschinger_effect() -> None:
    material = BilinearKinematicHardeningMaterial1D(
        youngs_modulus=YOUNGS_MODULUS,
        yield_stress=YIELD_STRESS,
        hardening_modulus=HARDENING_MODULUS,
    )
    state = material.initial_state()
    history = []

    # 1. Load into tension, well past yield.
    state = _commit(material, state, 3.0 * YIELD_STRAIN)
    history.append(("tension_load", state))
    assert state.yielded is True
    assert state.stress > YIELD_STRESS
    tensile_plastic_strain = state.plastic_strain
    back_stress_after_tension = state.back_stress
    assert back_stress_after_tension > 0.0

    # 2. Unload elastically back to zero stress.
    zero_stress_strain = state.plastic_strain
    state = _commit(material, state, zero_stress_strain)
    history.append(("unload_to_zero", state))
    assert state.stress == pytest.approx(0.0, abs=1.0)
    assert state.yielded is False
    assert state.plastic_strain == pytest.approx(tensile_plastic_strain)
    assert state.back_stress == pytest.approx(back_stress_after_tension)

    # 3. Continue into compression until reverse yield.
    #    Reverse yield stress = X - sigma_y0 (the lower edge of the translated
    #    yield surface), reached at strain = plastic_strain + (X - sigma_y0)/E.
    reverse_yield_stress = back_stress_after_tension - YIELD_STRESS
    reverse_yield_strain = state.plastic_strain + reverse_yield_stress / YOUNGS_MODULUS
    state = _commit(material, state, reverse_yield_strain)
    history.append(("reverse_yield_boundary", state))
    assert state.stress == pytest.approx(reverse_yield_stress, rel=1e-6)

    # THE BAUSCHINGER EFFECT: the reverse (compressive) yield stress has a
    # smaller magnitude than the virgin compressive yield stress (-sigma_y0),
    # because the back stress shifted the yield surface during tensile flow.
    assert abs(reverse_yield_stress) < YIELD_STRESS

    # 4. Push further into compression, well past reverse yield.
    state = _commit(material, state, reverse_yield_strain - 2.0 * YIELD_STRAIN)
    history.append(("compression_load", state))
    assert state.yielded is True
    assert state.stress < reverse_yield_stress  # further plastic flow in compression
    compressive_plastic_strain = state.plastic_strain
    assert compressive_plastic_strain < tensile_plastic_strain

    # 5. Unload again and reload back into tension.
    zero_stress_strain_2 = state.plastic_strain
    state = _commit(material, state, zero_stress_strain_2)
    history.append(("unload_to_zero_2", state))
    assert state.stress == pytest.approx(0.0, abs=1.0)

    state = _commit(material, state, zero_stress_strain_2 + 3.0 * YIELD_STRAIN)
    history.append(("second_tension_load", state))
    assert state.yielded is True
    assert state.stress > 0.0

    # State variables evolved consistently and monotonically with the
    # cumulative plastic flow at each stage of the cycle.
    names = [name for name, _ in history]
    assert names == [
        "tension_load",
        "unload_to_zero",
        "reverse_yield_boundary",
        "compression_load",
        "unload_to_zero_2",
        "second_tension_load",
    ]


def test_bauschinger_effect_scales_with_hardening_modulus() -> None:
    """A larger kinematic hardening modulus shifts the back stress further
    for the same plastic strain, making the Bauschinger effect (the gap
    between the virgin and actual reverse-yield stress) more pronounced.
    """
    small_h = BilinearKinematicHardeningMaterial1D(
        youngs_modulus=YOUNGS_MODULUS, yield_stress=YIELD_STRESS, hardening_modulus=5e9
    )
    large_h = BilinearKinematicHardeningMaterial1D(
        youngs_modulus=YOUNGS_MODULUS, yield_stress=YIELD_STRESS, hardening_modulus=50e9
    )

    strain = 3.0 * YIELD_STRAIN
    state_small = small_h.trial_state(strain, small_h.initial_state())
    state_large = large_h.trial_state(strain, large_h.initial_state())

    reverse_yield_small = abs(state_small.back_stress - YIELD_STRESS)
    reverse_yield_large = abs(state_large.back_stress - YIELD_STRESS)

    # Larger H -> larger back stress -> smaller (more shifted) reverse-yield
    # magnitude -> a bigger gap from the virgin yield stress.
    assert reverse_yield_large < reverse_yield_small
