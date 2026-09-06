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

from femtoolkit.continuum.constitutive import (
    isotropic_3d_matrix,
    plane_strain_matrix,
    plane_stress_matrix,
)
from femtoolkit.continuum.deformation import (
    MIN_DEFORMATION_GRADIENT_DETERMINANT,
    deformation_gradient,
    displacement_gradient,
    green_lagrange_strain_tensor,
    green_lagrange_strain_voigt,
    isochoric_deformation_gradient,
    right_cauchy_green,
    validate_deformation_gradient,
)
from femtoolkit.continuum.edge import (
    GAUSS_1D_2POINT,
    edge_equivalent_nodal_force,
    edge_shape_functions,
)
from femtoolkit.continuum.gauss import (
    GAUSS_2X2_POINTS,
    GAUSS_2X2X2_POINTS,
    GaussPoint,
    GaussPoint3D,
)
from femtoolkit.continuum.geometry import MIN_TRIANGLE_AREA, triangle_signed_area
from femtoolkit.continuum.invariants import (
    MIN_RIGHT_CAUCHY_GREEN_DETERMINANT,
    first_invariant,
    isochoric_first_invariant,
    isochoric_second_invariant,
    jacobian_from_right_cauchy_green,
    second_invariant,
    third_invariant,
    validate_right_cauchy_green,
)
from femtoolkit.continuum.jacobian import (
    MIN_JACOBIAN_DETERMINANT,
    MIN_JACOBIAN_DETERMINANT_3D,
    inverse_jacobian,
    inverse_jacobian_3d,
    jacobian_determinant,
    jacobian_determinant_3d,
    jacobian_matrix,
    jacobian_matrix_3d,
    physical_shape_function_derivatives,
    physical_shape_function_derivatives_3d,
)
from femtoolkit.continuum.mass import (
    hex8_consistent_mass_matrix,
    hex8_shape_function_matrix,
    lumped_mass_matrix,
    quad_consistent_mass_matrix,
    quad_shape_function_matrix,
    tetrahedron_consistent_mass_matrix,
    triangle_consistent_mass_matrix,
)
from femtoolkit.continuum.shape_functions import (
    hex8_shape_function_derivatives,
    hex8_shape_functions,
    quad_shape_function_derivatives,
    quad_shape_functions,
    tet4_shape_function_derivatives,
    tet4_shape_functions,
    triangle_shape_functions,
)
from femtoolkit.continuum.strain import (
    hex8_strain_displacement_matrix,
    quad_strain_displacement_matrix,
    strain_from_displacements,
    tet4_strain_displacement_matrix,
    triangle_strain_displacement_matrix,
)
from femtoolkit.continuum.stress import (
    cauchy_stress_from_second_piola_kirchhoff,
    first_piola_kirchhoff_from_second,
    principal_stresses_2d,
    stress_from_strain,
    von_mises_3d,
    von_mises_plane_strain,
    von_mises_plane_stress,
)
from femtoolkit.continuum.tensor import (
    deviatoric_stress,
    hydrostatic_stress,
    j2_invariant,
    mean_stress,
    principal_stresses_3d,
    tensor_to_voigt_strain,
    tensor_to_voigt_stress,
    trace,
    voigt_strain_to_tensor,
    voigt_stress_to_tensor,
    von_mises_stress_from_tensor,
)
from femtoolkit.continuum.thermal import (
    MIN_THERMAL_STRETCH,
    elastic_deformation_gradient_from_thermal_split,
    thermal_deformation_gradient,
)

__all__ = [
    "GAUSS_1D_2POINT",
    "GAUSS_2X2_POINTS",
    "GAUSS_2X2X2_POINTS",
    "MIN_DEFORMATION_GRADIENT_DETERMINANT",
    "MIN_JACOBIAN_DETERMINANT",
    "MIN_JACOBIAN_DETERMINANT_3D",
    "MIN_RIGHT_CAUCHY_GREEN_DETERMINANT",
    "MIN_THERMAL_STRETCH",
    "MIN_TRIANGLE_AREA",
    "GaussPoint",
    "GaussPoint3D",
    "cauchy_stress_from_second_piola_kirchhoff",
    "deformation_gradient",
    "deviatoric_stress",
    "displacement_gradient",
    "edge_equivalent_nodal_force",
    "edge_shape_functions",
    "elastic_deformation_gradient_from_thermal_split",
    "first_invariant",
    "first_piola_kirchhoff_from_second",
    "green_lagrange_strain_tensor",
    "green_lagrange_strain_voigt",
    "hex8_consistent_mass_matrix",
    "hex8_shape_function_derivatives",
    "hex8_shape_function_matrix",
    "hex8_shape_functions",
    "hex8_strain_displacement_matrix",
    "hydrostatic_stress",
    "inverse_jacobian",
    "inverse_jacobian_3d",
    "isochoric_deformation_gradient",
    "isochoric_first_invariant",
    "isochoric_second_invariant",
    "isotropic_3d_matrix",
    "j2_invariant",
    "jacobian_determinant",
    "jacobian_determinant_3d",
    "jacobian_from_right_cauchy_green",
    "jacobian_matrix",
    "jacobian_matrix_3d",
    "lumped_mass_matrix",
    "mean_stress",
    "physical_shape_function_derivatives",
    "physical_shape_function_derivatives_3d",
    "plane_strain_matrix",
    "plane_stress_matrix",
    "principal_stresses_2d",
    "principal_stresses_3d",
    "quad_consistent_mass_matrix",
    "quad_shape_function_derivatives",
    "quad_shape_function_matrix",
    "quad_shape_functions",
    "quad_strain_displacement_matrix",
    "right_cauchy_green",
    "second_invariant",
    "strain_from_displacements",
    "stress_from_strain",
    "tensor_to_voigt_strain",
    "tensor_to_voigt_stress",
    "tet4_shape_function_derivatives",
    "tet4_shape_functions",
    "tet4_strain_displacement_matrix",
    "tetrahedron_consistent_mass_matrix",
    "thermal_deformation_gradient",
    "third_invariant",
    "trace",
    "triangle_consistent_mass_matrix",
    "triangle_shape_functions",
    "triangle_signed_area",
    "triangle_strain_displacement_matrix",
    "validate_deformation_gradient",
    "validate_right_cauchy_green",
    "voigt_strain_to_tensor",
    "voigt_stress_to_tensor",
    "von_mises_3d",
    "von_mises_plane_strain",
    "von_mises_plane_stress",
    "von_mises_stress_from_tensor",
]
