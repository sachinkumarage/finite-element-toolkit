"""Tests for time-dependent load abstractions."""

import math

import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.dof import TranslationDOF
from femtoolkit.analysis.dynamic_loads import (
    ConstantLoad,
    SinusoidalLoad,
    StepLoad,
    TimeDependentNodalLoad,
)
from femtoolkit.exceptions import ValidationError

# --- ConstantLoad ---


def test_constant_load_value_independent_of_time() -> None:
    load = ConstantLoad(magnitude=1000.0)
    assert load.value_at(0.0) == 1000.0
    assert load.value_at(1e6) == 1000.0


def test_constant_load_rejects_non_finite() -> None:
    with pytest.raises(ValidationError):
        ConstantLoad(magnitude=float("inf"))


# --- StepLoad ---


def test_step_load_switches_on_at_step_time() -> None:
    load = StepLoad(magnitude=500.0, step_time=0.1)
    assert load.value_at(0.05) == 0.0
    assert load.value_at(0.1) == 500.0
    assert load.value_at(0.2) == 500.0


def test_step_load_default_step_time_is_zero() -> None:
    load = StepLoad(magnitude=200.0)
    assert load.value_at(0.0) == 200.0
    assert load.value_at(-0.001) == 0.0


def test_step_load_rejects_non_finite_magnitude() -> None:
    with pytest.raises(ValidationError):
        StepLoad(magnitude=float("nan"))


# --- SinusoidalLoad ---


def test_sinusoidal_load_formula() -> None:
    load = SinusoidalLoad(amplitude=100.0, angular_frequency=10.0, phase=0.0)
    t = 0.05
    assert_allclose(load.value_at(t), 100.0 * math.sin(10.0 * t))


def test_sinusoidal_load_zero_at_t_zero_with_no_phase() -> None:
    load = SinusoidalLoad(amplitude=50.0, angular_frequency=20.0)
    assert_allclose(load.value_at(0.0), 0.0, atol=1e-12)


def test_sinusoidal_load_peaks_at_quarter_period() -> None:
    omega = 10.0
    load = SinusoidalLoad(amplitude=100.0, angular_frequency=omega)
    quarter_period = (math.pi / 2.0) / omega
    assert_allclose(load.value_at(quarter_period), 100.0, atol=1e-9)


def test_sinusoidal_load_rejects_negative_angular_frequency() -> None:
    with pytest.raises(ValidationError):
        SinusoidalLoad(amplitude=1.0, angular_frequency=-1.0)


def test_sinusoidal_load_accepts_zero_angular_frequency() -> None:
    load = SinusoidalLoad(amplitude=1.0, angular_frequency=0.0)
    assert load.value_at(5.0) == 0.0


# --- TimeDependentNodalLoad ---


def test_time_dependent_nodal_load_evaluates_at_time() -> None:
    tdl = TimeDependentNodalLoad(
        node_id=3, dof=TranslationDOF.Y, load=SinusoidalLoad(1000.0, 50.0)
    )
    nodal_load = tdl.nodal_load_at(0.01)

    assert nodal_load.node_id == 3
    assert nodal_load.dof == TranslationDOF.Y
    assert_allclose(nodal_load.value, 1000.0 * math.sin(50.0 * 0.01))


def test_time_dependent_nodal_load_with_constant() -> None:
    tdl = TimeDependentNodalLoad(node_id=1, dof=TranslationDOF.X, load=ConstantLoad(500.0))
    assert tdl.nodal_load_at(0.0).value == 500.0
    assert tdl.nodal_load_at(10.0).value == 500.0


def test_time_dependent_nodal_load_rejects_invalid_node_id() -> None:
    with pytest.raises(ValidationError):
        TimeDependentNodalLoad(node_id=0, dof=TranslationDOF.X, load=ConstantLoad(1.0))


def test_time_dependent_nodal_load_rejects_invalid_dof() -> None:
    with pytest.raises(ValidationError):
        TimeDependentNodalLoad(node_id=1, dof=5, load=ConstantLoad(1.0))
