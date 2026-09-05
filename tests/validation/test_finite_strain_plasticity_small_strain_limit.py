"""Validation: finite-strain J2 plasticity reduces to small-strain J2Plasticity3D (spec section 12).

The module docstring for :mod:`femtoolkit.materials.finite_strain_plasticity`
claims this is not a coincidence: at small strain, ``ln(1+x) ~= x``, so the
principal-log-strain elastic law and radial return reduce exactly to
:class:`~femtoolkit.materials.j2_plasticity.J2Plasticity3D`'s closed-form
formulas. This is tested quantitatively here -- not just qualitatively --
by confirming the relative error between the two models shrinks linearly
with strain magnitude (the expected leading-order discrepancy between a
logarithmic and a linear strain measure), and that yield onset and
hardening evolution track closely at small strain.
"""

import numpy as np
import pytest

from femtoolkit.materials import J2FiniteStrainPlasticity3D, J2Plasticity3D

_MATERIAL_PARAMS = dict(
    youngs_modulus=200e9, poisson_ratio=0.3, yield_stress=250e6, hardening_modulus=10e9
)


@pytest.fixture
def finite_strain_material() -> J2FiniteStrainPlasticity3D:
    return J2FiniteStrainPlasticity3D(**_MATERIAL_PARAMS)


@pytest.fixture
def small_strain_material() -> J2Plasticity3D:
    return J2Plasticity3D(**_MATERIAL_PARAMS)


@pytest.mark.parametrize("axial_strain", [1e-6, 1e-5, 1e-4, 1e-3])
def test_elastic_regime_relative_error_shrinks_linearly_with_strain(
    finite_strain_material: J2FiniteStrainPlasticity3D,
    small_strain_material: J2Plasticity3D,
    axial_strain: float,
) -> None:
    strain = np.array([axial_strain, -0.3 * axial_strain, -0.3 * axial_strain, 0, 0, 0])
    fs_state = finite_strain_material.trial_state(strain, finite_strain_material.initial_state())
    ss_state = small_strain_material.trial_state(strain, small_strain_material.initial_state())

    assert not fs_state.yielded
    assert not ss_state.yielded

    relative_error = np.abs(fs_state.stress - ss_state.stress).max() / np.abs(ss_state.stress).max()
    # The leading-order log-strain-vs-linear-strain discrepancy is O(axial_strain);
    # a generous factor-of-10 margin above the empirically observed ~3.45x slope.
    assert relative_error < 40.0 * axial_strain


def test_yield_onset_strain_matches_small_strain_model_closely(
    finite_strain_material: J2FiniteStrainPlasticity3D, small_strain_material: J2Plasticity3D
) -> None:
    """Both models should first report yielding at essentially the same applied strain."""

    def _first_yield_strain(material) -> float:
        state = material.initial_state()
        for axial_strain in np.linspace(1e-4, 5e-3, 200):
            strain = np.array([axial_strain, -0.3 * axial_strain, -0.3 * axial_strain, 0, 0, 0])
            state = material.trial_state(strain, material.initial_state())
            if state.yielded:
                return float(axial_strain)
        raise AssertionError("Material never yielded in the swept range.")

    fs_yield_strain = _first_yield_strain(finite_strain_material)
    ss_yield_strain = _first_yield_strain(small_strain_material)
    assert fs_yield_strain == pytest.approx(ss_yield_strain, rel=0.02)


def test_hardening_evolution_matches_closely_at_small_strain(
    finite_strain_material: J2FiniteStrainPlasticity3D, small_strain_material: J2Plasticity3D
) -> None:
    fs_state = finite_strain_material.initial_state()
    ss_state = small_strain_material.initial_state()
    for axial_strain in (0.0005, 0.001, 0.0015, 0.002, 0.003):
        strain = np.array([axial_strain, -0.3 * axial_strain, -0.3 * axial_strain, 0, 0, 0])
        fs_state = finite_strain_material.trial_state(strain, fs_state)
        ss_state = small_strain_material.trial_state(strain, ss_state)

        assert fs_state.hardening_variable == pytest.approx(ss_state.hardening_variable, rel=0.05)
