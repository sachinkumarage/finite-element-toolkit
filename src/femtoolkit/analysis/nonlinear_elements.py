"""Nonlinear internal force and tangent stiffness for CST and Q4 continuum elements.

For a nonlinear material, the element stiffness matrix used by the
static/dynamic solvers (``Ke = t*A*B^T*D*B`` for CST, the Gauss-quadrature
equivalent for Q4) no longer applies -- there is no fixed ``D`` a
displacement can be multiplied through. Instead, an element must report
its **internal resisting force** and a **tangent stiffness** consistent
with whatever the current trial displacement implies about strain,
stress, and (via the material) the local stress-strain slope:

.. code-block:: text

    epsilon = B @ u                 (strain, from displacement -- unchanged from linear analysis)
    sigma = material.trial_state(epsilon, committed_state).stress   (stress, from the material)
    F_int = integral( B^T @ sigma ) t dA                            (internal force)
    K_t   = integral( B^T @ D_t @ B ) t dA,  D_t = material.tangent_modulus(...)

This module computes exactly that, reusing every piece of existing
machinery it can:

* the strain-displacement matrix ``B`` for CST
  (:attr:`~femtoolkit.mesh.cst_element.CSTElement2D.b_matrix`, an
  existing public property) and, for Q4, the same isoparametric/Gauss
  machinery :class:`~femtoolkit.mesh.quad_element.QuadElement2D` itself
  uses internally
  (:mod:`femtoolkit.continuum.shape_functions`/:mod:`femtoolkit.continuum.jacobian`/:mod:`femtoolkit.continuum.gauss`),
  recomputed here from public functions rather than reaching into the
  element's private helpers -- the same non-invasive pattern used by
  :mod:`femtoolkit.analysis.thermal_load` and
  :mod:`femtoolkit.analysis.mass` in earlier versions;
* :func:`~femtoolkit.continuum.strain.strain_from_displacements` for
  strain recovery (identical formula, linear or nonlinear);
* the same element geometry (:attr:`~femtoolkit.mesh.cst_element.CSTElement2D.area`,
  ``thickness``) used by the existing linear stiffness formulas.

Neither :class:`~femtoolkit.mesh.cst_element.CSTElement2D` nor
:class:`~femtoolkit.mesh.quad_element.QuadElement2D` is modified: their
existing ``stiffness_matrix`` property (built from ``element.material``,
a *linear* :class:`~femtoolkit.materials.linear_elastic_2d.LinearElastic2D`)
continues to work exactly as before for
:class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis` and
:class:`~femtoolkit.analysis.dynamic_analysis.DynamicAnalysis`. Nonlinear
analysis (:mod:`femtoolkit.analysis.nonlinear_analysis`) supplies its
own, separate
:class:`~femtoolkit.materials.nonlinear.NonlinearMaterial` per element
(see that module for why) and never reads ``element.material`` at all.

**Scope: CST and Q4 require a 2D (3-component) nonlinear material.**
Both elements' strain is always a length-3 Voigt vector
(``[epsilon_x, epsilon_y, gamma_xy]``); the material supplied for them
must accept and return that shape (e.g.
:meth:`~femtoolkit.materials.nonlinear.ElasticMaterialAdapter.from_linear_elastic_2d`).
Genuine 2D plasticity (a multiaxial yield *surface* and a return-mapping
algorithm) is out of scope for this version -- see the module docstring
for :mod:`femtoolkit.materials.nonlinear` for why validating this
architecture with the elastic adapter is the correct, honest choice
here, with the 1D
:class:`~femtoolkit.materials.nonlinear.ElasticPerfectlyPlasticMaterial1D`
model reserved for genuinely 1D (bar-like) use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from femtoolkit.continuum.gauss import GAUSS_2X2_POINTS
from femtoolkit.continuum.jacobian import physical_shape_function_derivatives
from femtoolkit.continuum.shape_functions import quad_shape_function_derivatives
from femtoolkit.continuum.strain import quad_strain_displacement_matrix, strain_from_displacements
from femtoolkit.exceptions import InvalidElementError, ValidationError
from femtoolkit.materials.nonlinear import MaterialState, NonlinearMaterial
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.quad_element import QuadElement2D

NONLINEAR_CAPABLE_ELEMENT_TYPES = (CSTElement2D, QuadElement2D)
"""Element types :mod:`femtoolkit.analysis.nonlinear_analysis` can drive."""


@dataclass(frozen=True)
class NonlinearElementState:
    """One element's material state: one entry for CST, one per Gauss point for Q4.

    Attributes:
        states: A length-1 tuple for a
            :class:`~femtoolkit.mesh.cst_element.CSTElement2D` (a single
            constant-strain state for the whole element), or a length-4
            tuple for a :class:`~femtoolkit.mesh.quad_element.QuadElement2D`
            (one independent state per 2x2 Gauss point -- see the module
            docstring on why these are never shared between points).
    """

    states: tuple[MaterialState, ...]


def _require_2d_material(
    element: CSTElement2D | QuadElement2D, material: NonlinearMaterial
) -> None:
    """Reject a scalar-strain (1D) material assigned to a CST/Q4 element early and clearly.

    CST and Q4 strain is always a length-3 Voigt vector (see the module
    docstring); a material built for 1D use (e.g.
    :class:`~femtoolkit.materials.nonlinear.ElasticPerfectlyPlasticMaterial1D`)
    would otherwise fail deep inside a Newton iteration with a confusing
    NumPy broadcasting error instead of a clear, actionable message.

    Raises:
        ValidationError: If ``material``'s initial state is scalar-shaped.
    """
    initial_strain = material.initial_state().strain
    if np.isscalar(initial_strain) or np.asarray(initial_strain).shape != (3,):
        raise ValidationError(
            f"{type(element).__name__} (id={element.id}) requires a 2D, 3-component "
            f"NonlinearMaterial (strain shape (3,)), got {type(material).__name__} "
            "with scalar strain. Use ElasticMaterialAdapter.from_linear_elastic_2d() "
            "or another 2D-capable material instead."
        )


def initial_element_state(
    element: CSTElement2D | QuadElement2D, material: NonlinearMaterial
) -> NonlinearElementState:
    """Build an element's zero-strain initial :class:`NonlinearElementState`.

    Args:
        element: The element to build a state for.
        material: The nonlinear material assigned to it.

    Returns:
        A :class:`NonlinearElementState` with one (CST) or four (Q4)
        independent copies of ``material.initial_state()``.

    Raises:
        InvalidElementError: If ``element`` is not a supported type.
        ValidationError: If ``material`` is not a 2D (3-component)
            material (see :func:`_require_2d_material`).
    """
    if isinstance(element, CSTElement2D):
        _require_2d_material(element, material)
        return NonlinearElementState(states=(material.initial_state(),))
    if isinstance(element, QuadElement2D):
        _require_2d_material(element, material)
        return NonlinearElementState(states=tuple(material.initial_state() for _ in range(4)))
    raise InvalidElementError(
        "Nonlinear analysis only supports CSTElement2D and QuadElement2D, got "
        f"{type(element).__name__} (id={element.id})."
    )


def cst_internal_force_and_tangent(
    element: CSTElement2D,
    material: NonlinearMaterial,
    displacements: Sequence[float],
    committed_state: NonlinearElementState,
) -> tuple[np.ndarray, np.ndarray, NonlinearElementState]:
    """Compute a CST element's internal force and tangent stiffness at a trial displacement.

    .. code-block:: text

        epsilon = B @ u                        (constant over the element)
        sigma = material.trial_state(epsilon, committed_state).stress
        F_int = t * A * B^T @ sigma
        K_t   = t * A * B^T @ D_t @ B

    Args:
        element: The CST element.
        material: Its assigned nonlinear (2D, 3-component) material.
        displacements: Trial nodal displacements, ordered per
            ``element.dof_keys()`` (6 entries: ``[u1,v1,u2,v2,u3,v3]``).
        committed_state: The element's state at the start of the current
            load step (length-1 :class:`NonlinearElementState`).

    Returns:
        ``(f_int, k_t, trial_state)``: the 6-entry internal force
        vector, the 6x6 tangent stiffness matrix, and the element's new
        (not yet committed) :class:`NonlinearElementState`.
    """
    b_matrix = element.b_matrix
    strain = strain_from_displacements(b_matrix, displacements)

    trial_material_state = material.trial_state(strain, committed_state.states[0])
    tangent = material.tangent_modulus(trial_material_state)

    scale = element.thickness * element.area
    f_int = scale * (b_matrix.T @ trial_material_state.stress)
    k_t = scale * (b_matrix.T @ tangent @ b_matrix)

    return f_int, k_t, NonlinearElementState(states=(trial_material_state,))


def quad_internal_force_and_tangent(
    element: QuadElement2D,
    material: NonlinearMaterial,
    displacements: Sequence[float],
    committed_state: NonlinearElementState,
) -> tuple[np.ndarray, np.ndarray, NonlinearElementState]:
    """Compute a Q4 element's internal force and tangent stiffness at a trial displacement.

    Evaluated independently at each of the four 2x2 Gauss points (see
    the module docstring): at each point, the strain-displacement matrix
    ``B`` varies (a Q4 element's strain is not constant), so strain,
    trial stress, and the tangent modulus are all recomputed there from
    that point's own committed :class:`~femtoolkit.materials.nonlinear.MaterialState`.

    .. code-block:: text

        F_int = sum over Gauss points of:  weight * t * det(J) * B^T @ sigma
        K_t   = sum over Gauss points of:  weight * t * det(J) * B^T @ D_t @ B

    Args:
        element: The Q4 element.
        material: Its assigned nonlinear (2D, 3-component) material.
        displacements: Trial nodal displacements, ordered per
            ``element.dof_keys()`` (8 entries).
        committed_state: The element's state at the start of the current
            load step (length-4 :class:`NonlinearElementState`, one per
            Gauss point).

    Returns:
        ``(f_int, k_t, trial_state)``: the 8-entry internal force
        vector, the 8x8 tangent stiffness matrix, and the element's new
        (not yet committed) :class:`NonlinearElementState`.
    """
    x_coords = tuple(node.x for node in element.nodes)
    y_coords = tuple(node.y for node in element.nodes)

    f_int = np.zeros(8)
    k_t = np.zeros((8, 8))
    trial_states: list[MaterialState] = []

    for index, point in enumerate(GAUSS_2X2_POINTS):
        dn_dxi, dn_deta = quad_shape_function_derivatives(point.xi, point.eta)
        dn_dx, dn_dy, det_j = physical_shape_function_derivatives(
            dn_dxi, dn_deta, x_coords, y_coords
        )
        b_matrix = quad_strain_displacement_matrix(dn_dx, dn_dy)
        strain = strain_from_displacements(b_matrix, displacements)

        trial_material_state = material.trial_state(strain, committed_state.states[index])
        tangent = material.tangent_modulus(trial_material_state)

        scale = point.weight * element.thickness * det_j
        f_int += scale * (b_matrix.T @ trial_material_state.stress)
        k_t += scale * (b_matrix.T @ tangent @ b_matrix)
        trial_states.append(trial_material_state)

    return f_int, k_t, NonlinearElementState(states=tuple(trial_states))


def element_internal_force_and_tangent(
    element: CSTElement2D | QuadElement2D,
    material: NonlinearMaterial,
    displacements: Sequence[float],
    committed_state: NonlinearElementState,
) -> tuple[np.ndarray, np.ndarray, NonlinearElementState]:
    """Dispatch to :func:`cst_internal_force_and_tangent`/:func:`quad_internal_force_and_tangent`.

    Args:
        element: The element (CST or Q4).
        material: Its assigned nonlinear material.
        displacements: Trial nodal displacements, ordered per
            ``element.dof_keys()``.
        committed_state: The element's state at the start of the current
            load step.

    Returns:
        ``(f_int, k_t, trial_state)``, see the element-specific functions.

    Raises:
        InvalidElementError: If ``element`` is not a supported type.
    """
    if isinstance(element, CSTElement2D):
        return cst_internal_force_and_tangent(element, material, displacements, committed_state)
    if isinstance(element, QuadElement2D):
        return quad_internal_force_and_tangent(element, material, displacements, committed_state)
    raise InvalidElementError(
        "Nonlinear analysis only supports CSTElement2D and QuadElement2D, got "
        f"{type(element).__name__} (id={element.id})."
    )
