"""Tests for femtoolkit.uncertainty.reliability."""

import numpy as np
import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.uncertainty.reliability import exceedance_probability


def test_exceedance_probability_known_dataset() -> None:
    values = np.concatenate([np.full(977, 3.0), np.full(23, 6.0)])
    result = exceedance_probability(values, threshold=5.0, quantity_label="Displacement")
    assert result.n_samples == 1000
    assert result.n_exceeding == 23
    assert result.exceedance_frequency == pytest.approx(0.023)


def test_exceedance_probability_below_direction() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = exceedance_probability(values, threshold=3.0, direction="below")
    assert result.n_exceeding == 2
    assert result.exceedance_frequency == pytest.approx(0.4)


def test_exceedance_probability_none_exceeding() -> None:
    values = np.array([1.0, 2.0, 3.0])
    result = exceedance_probability(values, threshold=100.0)
    assert result.n_exceeding == 0
    assert result.exceedance_frequency == 0.0


def test_exceedance_probability_all_exceeding() -> None:
    values = np.array([10.0, 20.0, 30.0])
    result = exceedance_probability(values, threshold=5.0)
    assert result.n_exceeding == 3
    assert result.exceedance_frequency == 1.0


def test_exceedance_probability_rejects_empty_values() -> None:
    with pytest.raises(ValidationError):
        exceedance_probability(np.array([]), threshold=5.0)


def test_exceedance_probability_rejects_invalid_direction() -> None:
    with pytest.raises(ValidationError):
        exceedance_probability(np.array([1.0, 2.0]), threshold=1.0, direction="sideways")
