"""2D continuum mechanics building blocks.

This package holds the reusable mathematics behind the toolkit's
continuum elements, kept independently testable from the element classes
that coordinate them
(:class:`~femtoolkit.mesh.cst_element.CSTElement2D`,
:class:`~femtoolkit.mesh.quad_element.QuadElement2D`):

.. code-block:: text

    Geometry / natural coordinates (geometry.py, gauss.py)
        -> Shape functions (shape_functions.py)
        -> Isoparametric mapping / Jacobian (jacobian.py, Q4 only)
        -> Strain-displacement matrix (strain.py)
        -> Constitutive matrix (constitutive.py)
        -> Stress recovery (stress.py)

Every quantity here uses **engineering shear strain**
(``gamma_xy = du/dy + dv/dx``), not tensorial shear strain -- see
:mod:`femtoolkit.continuum.strain` for why this matters.
"""

from femtoolkit.continuum.constitutive import plane_strain_matrix, plane_stress_matrix
from femtoolkit.continuum.gauss import GAUSS_2X2_POINTS, GaussPoint
from femtoolkit.continuum.geometry import MIN_TRIANGLE_AREA, triangle_signed_area
from femtoolkit.continuum.jacobian import (
    MIN_JACOBIAN_DETERMINANT,
    inverse_jacobian,
    jacobian_determinant,
    jacobian_matrix,
    physical_shape_function_derivatives,
)
from femtoolkit.continuum.shape_functions import (
    quad_shape_function_derivatives,
    quad_shape_functions,
    triangle_shape_functions,
)
from femtoolkit.continuum.strain import (
    quad_strain_displacement_matrix,
    strain_from_displacements,
    triangle_strain_displacement_matrix,
)
from femtoolkit.continuum.stress import (
    principal_stresses_2d,
    stress_from_strain,
    von_mises_3d,
    von_mises_plane_strain,
    von_mises_plane_stress,
)

__all__ = [
    "GAUSS_2X2_POINTS",
    "MIN_JACOBIAN_DETERMINANT",
    "MIN_TRIANGLE_AREA",
    "GaussPoint",
    "inverse_jacobian",
    "jacobian_determinant",
    "jacobian_matrix",
    "physical_shape_function_derivatives",
    "plane_strain_matrix",
    "plane_stress_matrix",
    "principal_stresses_2d",
    "quad_shape_function_derivatives",
    "quad_shape_functions",
    "quad_strain_displacement_matrix",
    "strain_from_displacements",
    "stress_from_strain",
    "triangle_shape_functions",
    "triangle_signed_area",
    "triangle_strain_displacement_matrix",
    "von_mises_3d",
    "von_mises_plane_strain",
    "von_mises_plane_stress",
]
