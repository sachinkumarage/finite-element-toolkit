"""The Rosenbrock function: a curved, narrow valley (Version 33).

.. math::

    f(x) = \\sum_{i=1}^{n-1} \\left[100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2\\right]

Unlike :mod:`.sphere`, the minimum ``f(1, ..., 1) = 0`` sits inside a
long, curved, narrow valley -- easy to find the valley itself, much
harder to converge precisely along it, which makes this a much more
discriminating test of an algorithm's exploitation behavior than the
sphere function.

This module has no dependency on finite element analysis or this
toolkit's optimization framework -- ``rosenbrock`` is a plain NumPy
function, directly unit-testable on its own.
"""

from __future__ import annotations

import numpy as np

from femtoolkit.exceptions import ValidationError

ROSENBROCK_OPTIMUM = 0.0
"""The known global minimum value, ``f(1, ..., 1) = 0``."""


def rosenbrock(x: np.ndarray) -> float:
    """Evaluate the Rosenbrock function at ``x``.

    Args:
        x: A 1D array of at least length 2.

    Returns:
        ``sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2)``.

    Raises:
        ValidationError: If ``x`` has fewer than two elements.
    """
    vector = np.asarray(x, dtype=float)
    if vector.size < 2:
        raise ValidationError("rosenbrock requires at least two dimensions.")
    return float(np.sum(100.0 * (vector[1:] - vector[:-1] ** 2) ** 2 + (1.0 - vector[:-1]) ** 2))


__all__ = ["ROSENBROCK_OPTIMUM", "rosenbrock"]
