"""Element-level mass matrix dispatch: consistent or lumped, CST or Q4.

Mirrors the isinstance-dispatch pattern established by
:mod:`femtoolkit.analysis.body_load` and
:mod:`femtoolkit.analysis.thermal_load`: this module reads an element's
public attributes (``material``, ``area``, ``thickness``, node
coordinates) from the outside and delegates the actual integration to
the pure functions in :mod:`femtoolkit.continuum.mass`, without adding
any new methods to :class:`~femtoolkit.mesh.cst_element.CSTElement2D` or
:class:`~femtoolkit.mesh.quad_element.QuadElement2D` themselves.

:data:`MassMatrixType` selects between the two element mass
formulations documented in :mod:`femtoolkit.continuum.mass`:

* ``"consistent"`` (default) -- the physically accurate, fully coupled
  mass matrix built from the same shape functions used for displacement.
* ``"lumped"`` -- a diagonal approximation, via row-sum lumping.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from femtoolkit.continuum.mass import (
    lumped_mass_matrix,
    quad_consistent_mass_matrix,
    triangle_consistent_mass_matrix,
)
from femtoolkit.exceptions import ValidationError
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.quad_element import QuadElement2D

MassMatrixType = Literal["consistent", "lumped"]

_MASS_CAPABLE_ELEMENT_TYPES = (CSTElement2D, QuadElement2D)


def _element_density(element: CSTElement2D | QuadElement2D) -> float:
    density = element.material.density
    if density is None:
        raise ValidationError(
            f"Element {element.id} ({type(element).__name__})'s material has no density "
            "set; density is required for mass matrix computation."
        )
    return density


def _consistent_mass_matrix(element: CSTElement2D | QuadElement2D) -> np.ndarray:
    density = _element_density(element)
    if isinstance(element, CSTElement2D):
        return triangle_consistent_mass_matrix(
            density=density, area=element.area, thickness=element.thickness
        )
    x_coords = tuple(node.x for node in element.nodes)
    y_coords = tuple(node.y for node in element.nodes)
    return quad_consistent_mass_matrix(
        x_coords, y_coords, density=density, thickness=element.thickness
    )


def element_mass_matrix(
    element: CSTElement2D | QuadElement2D, mass_matrix_type: MassMatrixType = "consistent"
) -> np.ndarray:
    """Compute an element's mass matrix, consistent or lumped.

    Args:
        element: The continuum element to compute a mass matrix for
            (:class:`~femtoolkit.mesh.cst_element.CSTElement2D` or
            :class:`~femtoolkit.mesh.quad_element.QuadElement2D`).
        mass_matrix_type: ``"consistent"`` (default, see
            :func:`~femtoolkit.continuum.mass.triangle_consistent_mass_matrix`/
            :func:`~femtoolkit.continuum.mass.quad_consistent_mass_matrix`)
            or ``"lumped"`` (see
            :func:`~femtoolkit.continuum.mass.lumped_mass_matrix`).

    Returns:
        A square NumPy array (6x6 for CST, 8x8 for Q4), ordered to match
        ``element.dof_keys()``.

    Raises:
        ValidationError: If ``element``'s material has no density set,
            or ``mass_matrix_type`` is not ``"consistent"`` or ``"lumped"``.
    """
    consistent = _consistent_mass_matrix(element)
    if mass_matrix_type == "consistent":
        return consistent
    if mass_matrix_type == "lumped":
        return lumped_mass_matrix(consistent)
    raise ValidationError(
        f'mass_matrix_type must be "consistent" or "lumped", got {mass_matrix_type!r}.'
    )


def element_total_mass(element: CSTElement2D | QuadElement2D) -> float:
    """Return an element's total physical mass, ``rho * area * thickness``.

    Independent of ``mass_matrix_type`` -- both the consistent and
    lumped mass matrices conserve this same total (see
    :func:`~femtoolkit.continuum.mass.lumped_mass_matrix`), so this is
    the reference value engineering validation compares either against.

    Args:
        element: The continuum element to query.

    Returns:
        Total mass, in kilograms.

    Raises:
        ValidationError: If ``element``'s material has no density set.
    """
    return _element_density(element) * element.area * element.thickness
