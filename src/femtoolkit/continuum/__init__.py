"""2D continuum mechanics building blocks.

This package holds the reusable mathematics behind the constant strain
triangle (CST) element, kept independently testable from the element
class that coordinates them (:class:`~femtoolkit.mesh.cst_element.CSTElement2D`):

.. code-block:: text

    Geometry (geometry.py)
        -> Shape functions (shape_functions.py)
        -> Strain-displacement matrix (strain.py)
        -> Constitutive matrix (constitutive.py)
        -> Stress recovery (stress.py)

Every quantity here uses **engineering shear strain**
(``gamma_xy = du/dy + dv/dx``), not tensorial shear strain -- see
:mod:`femtoolkit.continuum.strain` for why this matters.
"""

from femtoolkit.continuum.constitutive import plane_strain_matrix, plane_stress_matrix
from femtoolkit.continuum.geometry import MIN_TRIANGLE_AREA, triangle_signed_area
from femtoolkit.continuum.shape_functions import triangle_shape_functions
from femtoolkit.continuum.strain import (
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
    "MIN_TRIANGLE_AREA",
    "plane_strain_matrix",
    "plane_stress_matrix",
    "principal_stresses_2d",
    "strain_from_displacements",
    "stress_from_strain",
    "triangle_shape_functions",
    "triangle_signed_area",
    "triangle_strain_displacement_matrix",
    "von_mises_3d",
    "von_mises_plane_strain",
    "von_mises_plane_stress",
]
