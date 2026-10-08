"""Candidate generation for adaptive sampling and surrogate-assisted search (Version 36).

A candidate pool is a set of points in the design space, each evaluated by the
surrogate first (never by the expensive high-fidelity model -- see
:mod:`femtoolkit.adaptive.sampling`). This module only *generates* that pool; scoring
and ranking it is :mod:`femtoolkit.adaptive.sampling`'s job, and deciding which
candidate actually gets a high-fidelity evaluation is
:mod:`femtoolkit.adaptive.refinement`'s.
"""

from __future__ import annotations

import numpy as np

from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.variables import DesignVariable


def generate_candidate_pool(
    design_variables: list[DesignVariable], n_candidates: int, seed: int | None = None
) -> list[dict[str, float]]:
    """Draw a pool of random candidate points from the design space, within bounds.

    Reuses :meth:`~femtoolkit.optimization.variables.DesignVariable.sample` directly,
    so every candidate is drawn exactly the way a Version 32
    :class:`~femtoolkit.optimization.algorithms.random_search.BoundedRandomSearch`
    candidate already is -- a continuous variable uniformly within
    ``[lower_bound, upper_bound]``, an integer variable uniformly over its integer
    range, a categorical variable uniformly over its categories.

    Args:
        design_variables: The design variables to sample. Every name must be unique
            (see :func:`~femtoolkit.optimization.variables.validate_unique_variable_names`).
        n_candidates: How many candidate points to generate.
        seed: A random seed; the same seed reproduces the same candidate pool.

    Returns:
        ``n_candidates`` points, each a mapping from design-variable *name* to a
        valid sampled value.

    Raises:
        ValidationError: If ``design_variables`` is empty or ``n_candidates`` is not
            positive.
    """
    if not design_variables:
        raise ValidationError("generate_candidate_pool() requires at least one design variable.")
    if n_candidates <= 0:
        raise ValidationError(f"n_candidates must be positive, got {n_candidates}.")

    rng = np.random.default_rng(seed)
    return [
        {variable.name: variable.sample(rng) for variable in design_variables}
        for _ in range(n_candidates)
    ]


def clip_to_bounds(
    point: dict[str, float], design_variables: list[DesignVariable]
) -> dict[str, float]:
    """Clamp every value in ``point`` into its design variable's valid domain.

    Args:
        point: A candidate point, keyed by design-variable name.
        design_variables: The design variables ``point`` was drawn from.

    Returns:
        A new dict with every value clamped via
        :meth:`~femtoolkit.optimization.variables.DesignVariable.clip`. A name in
        ``point`` with no matching design variable is passed through unchanged.
    """
    variables_by_name = {variable.name: variable for variable in design_variables}
    clipped: dict[str, float] = {}
    for name, value in point.items():
        variable = variables_by_name.get(name)
        clipped[name] = variable.clip(value) if variable is not None else value
    return clipped


__all__ = ["clip_to_bounds", "generate_candidate_pool"]
