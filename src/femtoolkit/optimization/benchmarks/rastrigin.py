"""The Rastrigin function: a bowl covered in many local minima (Version 33).

.. math::

    f(x) = 10n + \\sum_i \\left[x_i^2 - 10 \\cos(2 \\pi x_i)\\right]

A highly multimodal benchmark: the cosine term adds a regular lattice
of local minima on top of the same underlying bowl shape as
:mod:`.sphere`, with the single global minimum still at
``f(0, ..., 0) = 0``. A purely local search (e.g. coordinate search)
can easily get trapped in one of the surrounding local minima; this
benchmark exists specifically to demonstrate that difference against a
population-based algorithm's broader exploration.

This module has no dependency on finite element analysis or this
toolkit's optimization framework -- ``rastrigin`` is a plain NumPy
function, directly unit-testable on its own.
"""

from __future__ import annotations

import numpy as np

RASTRIGIN_OPTIMUM = 0.0
"""The known global minimum value, ``f(0, ..., 0) = 0``."""


def rastrigin(x: np.ndarray) -> float:
    """Evaluate the Rastrigin function at ``x``.

    Args:
        x: A 1D array of any length.

    Returns:
        ``10 * n + sum(x_i**2 - 10 * cos(2 * pi * x_i))``.
    """
    vector = np.asarray(x, dtype=float)
    n = vector.size
    return float(10.0 * n + np.sum(vector**2 - 10.0 * np.cos(2.0 * np.pi * vector)))


__all__ = ["RASTRIGIN_OPTIMUM", "rastrigin"]
