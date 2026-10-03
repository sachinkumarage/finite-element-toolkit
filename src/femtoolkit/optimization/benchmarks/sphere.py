"""The sphere function: the simplest convex optimization benchmark (Version 33).

.. math::

    f(x) = \\sum_i x_i^2

A smooth, convex, separable bowl with a single global minimum
``f(0, ..., 0) = 0``. Every reasonable derivative-free algorithm should
find this minimum easily; it exists here as the simplest possible sanity
check that an algorithm's mechanics (bounds, crossover, mutation,
selection) are not broken, not as a meaningful difficulty benchmark.

This module has no dependency on finite element analysis or this
toolkit's optimization framework -- ``sphere`` is a plain NumPy
function, directly unit-testable on its own.
"""

from __future__ import annotations

import numpy as np

SPHERE_OPTIMUM = 0.0
"""The known global minimum value, ``f(0, ..., 0) = 0``."""


def sphere(x: np.ndarray) -> float:
    """Evaluate the sphere function at ``x``.

    Args:
        x: A 1D array of any length.

    Returns:
        ``sum(x_i ** 2)``.
    """
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


__all__ = ["SPHERE_OPTIMUM", "sphere"]
