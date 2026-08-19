"""Uniform temperature-change loading, converted into equivalent nodal loads.

A uniform temperature change ``delta_temperature`` produces a free
(stress-free) thermal strain in an isotropic material:

.. code-block:: text

    epsilon_thermal = [ alpha * dT, alpha * dT, 0 ]

(no thermal shear strain for an isotropic material under a uniform
temperature change -- see :mod:`femtoolkit.continuum.strain` for the
``[epsilon_x, epsilon_y, gamma_xy]`` convention). Because a *restrained*
structure resists this free strain, it is converted to an
**initial-strain equivalent nodal force**, the standard finite-element
technique for injecting a stress-free eigenstrain into a linear elastic
model:

.. code-block:: text

    fe = integral( B^T * D * epsilon_thermal ) * thickness dA

This follows directly from the same virtual-work derivation as the
element stiffness matrix: with ``sigma = D * (epsilon - epsilon_thermal)``,
the internal virtual work ``integral(delta_epsilon^T sigma) dV`` splits
into a stiffness term ``Ke * d`` and this initial-strain force term,
which is added to the external load vector like any other applied force.

For a **CST element** (constant ``B``), this collapses to a single
multiplication by the element's physical area, exactly like its
stiffness matrix. For a **Q4 element** (``B`` varies by point), the same
2x2 Gauss quadrature used for its stiffness matrix is reused here.

Only a spatially uniform temperature change is supported --
temperature *gradients* within an element are out of scope for this
version.

**Stress recovery under a thermal load.**
:meth:`~femtoolkit.results.analysis_result.AnalysisResult.element_stress`
is unchanged since Version 6: it reports ``sigma = D @ epsilon_total``
from the *total* mechanical strain implied by nodal displacements, with
no notion of "thermal" loading. Under a thermal load, ``epsilon_total``
already includes the free thermal strain, so ``element_stress`` reports
what the material's stress *would be if none of that strain were
thermal* -- correct for a fully restrained element (where the free
thermal strain cannot develop and the total strain genuinely is
mechanical), but not the physically meaningful "thermal stress" for a
partially or fully unrestrained element, where some or all of the total
strain is stress-free. :func:`thermal_corrected_stress` (and
:func:`thermal_corrected_strain`) isolate the mechanical strain/stress by
explicitly subtracting the thermal eigenstrain, ``sigma = D @ (epsilon_total
- epsilon_thermal)`` -- the physically correct stress recovery formula
whenever thermal loading is present, without modifying the shared
:class:`~femtoolkit.analysis.element.ContinuumElement` protocol or
:class:`~femtoolkit.results.analysis_result.AnalysisResult` used by every
other version.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from femtoolkit.analysis.dof import TranslationDOF
from femtoolkit.analysis.loads import NodalLoad
from femtoolkit.exceptions import ValidationError
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.mesh.quad_element import QuadElement2D

if TYPE_CHECKING:
    from femtoolkit.results.analysis_result import AnalysisResult


@dataclass
class TemperatureLoad:
    """A uniform temperature change applied to every continuum element in a mesh.

    Attributes:
        delta_temperature: Uniform temperature change, ``dT``, in kelvin
            (equivalently degrees Celsius, since only the *difference*
            matters). Positive means heating.

    Raises:
        ValidationError: If ``delta_temperature`` is not finite.

    Example:
        >>> thermal_load = TemperatureLoad(delta_temperature=100.0)
    """

    delta_temperature: float

    def __post_init__(self) -> None:
        """Validate the thermal load immediately after construction.

        Raises:
            ValidationError: If ``delta_temperature`` is not finite.
        """
        if not math.isfinite(self.delta_temperature):
            raise ValidationError(
                "TemperatureLoad delta_temperature must be finite, got "
                f"{self.delta_temperature}."
            )


def _element_expansion_coefficient(element: CSTElement2D | QuadElement2D) -> float:
    alpha = element.material.thermal_expansion_coefficient
    if alpha is None:
        raise ValidationError(
            f"Element {element.id} ({type(element).__name__})'s material has no "
            "thermal_expansion_coefficient set; it is required for thermal loading."
        )
    return alpha


def _thermal_strain_vector(alpha: float, delta_temperature: float) -> np.ndarray:
    thermal_strain = alpha * delta_temperature
    return np.array([thermal_strain, thermal_strain, 0.0])


def _nodal_loads_from_force_vector(
    element: CSTElement2D | QuadElement2D, force: np.ndarray
) -> list[NodalLoad]:
    loads: list[NodalLoad] = []
    for i, node in enumerate(element.nodes):
        loads.append(NodalLoad(node.id, TranslationDOF.X, float(force[2 * i])))
        loads.append(NodalLoad(node.id, TranslationDOF.Y, float(force[2 * i + 1])))
    return loads


def _cst_thermal_nodal_loads(element: CSTElement2D, delta_temperature: float) -> list[NodalLoad]:
    """Closed-form ``t * A * B^T * D * epsilon0`` (B and D are constant over a CST)."""
    alpha = _element_expansion_coefficient(element)
    epsilon0 = _thermal_strain_vector(alpha, delta_temperature)
    d_matrix = element.material.constitutive_matrix
    force = element.thickness * element.area * (element.b_matrix.T @ d_matrix @ epsilon0)
    return _nodal_loads_from_force_vector(element, force)


def _quad_thermal_nodal_loads(element: QuadElement2D, delta_temperature: float) -> list[NodalLoad]:
    """2x2 Gauss quadrature of ``t * B^T * D * epsilon0`` (B varies over a Q4)."""
    from femtoolkit.continuum.gauss import GAUSS_2X2_POINTS
    from femtoolkit.continuum.jacobian import physical_shape_function_derivatives
    from femtoolkit.continuum.shape_functions import quad_shape_function_derivatives
    from femtoolkit.continuum.strain import quad_strain_displacement_matrix

    alpha = _element_expansion_coefficient(element)
    epsilon0 = _thermal_strain_vector(alpha, delta_temperature)
    d_matrix = element.material.constitutive_matrix
    x_coords = tuple(node.x for node in element.nodes)
    y_coords = tuple(node.y for node in element.nodes)

    force = np.zeros(8)
    for point in GAUSS_2X2_POINTS:
        dn_dxi, dn_deta = quad_shape_function_derivatives(point.xi, point.eta)
        dn_dx, dn_dy, det_j = physical_shape_function_derivatives(
            dn_dxi, dn_deta, x_coords, y_coords
        )
        b_matrix = quad_strain_displacement_matrix(dn_dx, dn_dy)
        force += point.weight * det_j * element.thickness * (b_matrix.T @ d_matrix @ epsilon0)

    return _nodal_loads_from_force_vector(element, force)


def thermal_load_to_nodal_loads(mesh: Mesh, thermal_load: TemperatureLoad) -> list[NodalLoad]:
    """Convert a uniform temperature-change load into equivalent nodal loads.

    Every continuum element (CST or Q4) in ``mesh`` contributes its own
    thermal-strain equivalent nodal force; elements sharing a node each
    add their own contribution, which
    :func:`~femtoolkit.analysis.system.build_force_vector` sums.

    Args:
        mesh: The mesh to compute thermal loads for. Only CST and Q4
            elements contribute; other element types are skipped.
        thermal_load: The uniform temperature change to convert.

    Returns:
        One :class:`~femtoolkit.analysis.loads.NodalLoad` per DOF
        contribution (2 per node per contributing element).

    Raises:
        ValidationError: If any contributing element's material has no
            thermal expansion coefficient set.
    """
    nodal_loads: list[NodalLoad] = []
    for element in mesh.elements:
        if isinstance(element, CSTElement2D):
            nodal_loads.extend(_cst_thermal_nodal_loads(element, thermal_load.delta_temperature))
        elif isinstance(element, QuadElement2D):
            nodal_loads.extend(_quad_thermal_nodal_loads(element, thermal_load.delta_temperature))
    return nodal_loads


def _element_displacements(
    result: AnalysisResult, element: CSTElement2D | QuadElement2D
) -> list[float]:
    return [result.displacement(node_id, dof) for node_id, dof in element.dof_keys()]


def thermal_corrected_strain(
    result: AnalysisResult, element: CSTElement2D | QuadElement2D, thermal_load: TemperatureLoad
) -> np.ndarray:
    """Return an element's mechanical strain, with the free thermal strain removed.

    ``epsilon_mechanical = epsilon_total - epsilon_thermal``, where
    ``epsilon_total`` comes from :meth:`element.strain_from_dofs` and
    ``epsilon_thermal = [alpha*dT, alpha*dT, 0]``. See the module
    docstring for why this differs from
    :meth:`~femtoolkit.results.analysis_result.AnalysisResult.element_strain`.

    Args:
        result: The solved analysis result to read displacements from.
        element: The CST or Q4 element to query (must be part of ``result``).
        thermal_load: The uniform temperature change that was applied.

    Returns:
        A length-3 NumPy array, ``[epsilon_x, epsilon_y, gamma_xy]``.

    Raises:
        ValidationError: If ``element``'s material has no thermal
            expansion coefficient set.
    """
    alpha = _element_expansion_coefficient(element)
    epsilon_thermal = _thermal_strain_vector(alpha, thermal_load.delta_temperature)
    total_strain = element.strain_from_dofs(_element_displacements(result, element))
    return total_strain - epsilon_thermal


def thermal_corrected_stress(
    result: AnalysisResult, element: CSTElement2D | QuadElement2D, thermal_load: TemperatureLoad
) -> np.ndarray:
    """Return an element's mechanical stress, with the free thermal strain removed.

    ``sigma = D @ (epsilon_total - epsilon_thermal)`` -- the physically
    correct stress under thermal loading (zero for a fully unrestrained
    element under a uniform temperature change, since it is then free to
    accommodate the thermal strain without developing stress). See the
    module docstring for why this differs from
    :meth:`~femtoolkit.results.analysis_result.AnalysisResult.element_stress`.

    Args:
        result: The solved analysis result to read displacements from.
        element: The CST or Q4 element to query (must be part of ``result``).
        thermal_load: The uniform temperature change that was applied.

    Returns:
        A length-3 NumPy array, ``[sigma_x, sigma_y, tau_xy]``, in pascals.

    Raises:
        ValidationError: If ``element``'s material has no thermal
            expansion coefficient set.
    """
    mechanical_strain = thermal_corrected_strain(result, element, thermal_load)
    return element.material.constitutive_matrix @ mechanical_strain
