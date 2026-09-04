"""Geometrically nonlinear internal force and tangent stiffness (Version 16).

Every nonlinear element function in :mod:`femtoolkit.analysis.nonlinear_elements`
(Versions 13-15) assumes **geometric linearity**: strain is `B @ u` with a
`B` matrix computed once from an element's fixed node coordinates, and only
the *material* response (via a
:class:`~femtoolkit.materials.nonlinear.NonlinearMaterial`) is nonlinear.
This module adds the complementary, independent kind of nonlinearity:
**geometric nonlinearity**, where the *strain-displacement relationship
itself* is nonlinear because the structure's changing shape affects
equilibrium and stiffness -- distinct from, and freely combinable with,
material nonlinearity (see :mod:`femtoolkit.continuum.deformation` for the
conceptual distinction). This module is entirely new and does not modify
:mod:`femtoolkit.analysis.nonlinear_elements`; both dispatch modules can be
selected independently by :class:`~femtoolkit.analysis.nonlinear_analysis.NonlinearAnalysis`
(via its ``geometric_nonlinearity`` flag), so every Version 13-15 nonlinear
analysis is completely unaffected by this module's existence.

**Total Lagrangian formulation.** All three supported elements
(:class:`~femtoolkit.mesh.truss_element.TrussElement2D`,
:class:`~femtoolkit.mesh.tet4_element.Tet4Element3D`,
:class:`~femtoolkit.mesh.hex8_element.Hex8Element3D`) use the *reference*
(undeformed) configuration for integration, geometry, and volume -- exactly
what :class:`~femtoolkit.mesh.node.Node`'s immutability already guarantees,
since ``element.nodes[i].x/y/z`` never change. The "current configuration"
needed at each Newton-Raphson iteration is simply reference-plus-
displacement, recomputed fresh from the trial displacement vector every
call -- there is no mutable "deformed mesh" anywhere in this toolkit, and
none is needed.

**Internal force.** For TET4/HEX8, the internal force at node ``a`` uses
the standard, compact virtual-work formula

.. code-block:: text

    f_int_a = integral_V0 ( F @ S @ Grad0(Na) ) dV0

(``F`` the deformation gradient, ``S`` the second Piola-Kirchhoff stress,
``Grad0(Na)`` node ``a``'s reference-configuration shape-function
gradient) -- deliberately used *instead of* hand-assembling a nonlinear
strain-displacement (``B_NL``) matrix, since this compact form is safer to
implement correctly (fewer places for a sign or index error to hide) while
being exactly equivalent to it.

**Tangent stiffness: material vs. geometric.** ``K_t = K_material +
K_geometric``. ``K_geometric`` (the "initial stress" stiffness) has a
simple, well-known closed form depending only on the *current stress
state*, not on how that stress responds to further strain:

.. code-block:: text

    K_geometric[a,b] = integral_V0 ( Grad0(Na) . S . Grad0(Nb) ) dV0 * I

(a scalar times the identity, since it does not couple different spatial
directions). ``K_material``, in contrast, is not implemented from a
separately hand-derived closed form: a material tangent obtained by
differentiating ``F_int`` through both ``F`` and ``S(E(F))`` is a genuine
source of easy, silent sign/index errors (the same concern that led
:class:`~femtoolkit.materials.j2_plasticity.J2Plasticity3D`'s tangent to
use numerical differentiation in Version 15). Here, ``K_material`` is
instead obtained by **subtraction from a numerically differentiated total
tangent**: ``K_t`` is computed once via central differences of the
(already-verified, closed-form) internal-force formula above with respect
to each nodal DOF, and ``K_material = K_t - K_geometric``. This makes
``K_t`` self-consistent with the actual internal-force implementation *by
construction*, and still yields the two separately meaningful, documented
quantities the assignment asks for.

**Truss.** The 2-node truss is simple enough that its *exact* tangent
stiffness was derived directly (not recalled from memory) by
differentiating the truss's own compact internal-force formula
symbolically -- see :func:`truss_geometric_internal_force_and_tangent`'s
docstring for the full derivation and the resulting closed forms for
``K_material``/``K_geometric``, then cross-validated numerically in the
test suite. For a 2-node truss, the Total Lagrangian and **corotational**
formulations coincide exactly (there is only one possible current
orientation to separate "rigid rotation" from "deformation" against, and
Green-Lagrange strain is objective by construction -- see
:mod:`femtoolkit.continuum.deformation`), so this single implementation
satisfies both.

**Rigid-body objectivity.** For all three elements, a pure rigid-body
translation or rotation of the reference configuration produces `E = 0`
(exactly, to floating-point precision) and therefore `S = 0` and
`F_int = 0` -- not by any special-cased check in this module, but as a
direct, unavoidable consequence of Green-Lagrange strain's mathematical
definition. This is verified explicitly in the test suite
(``tests/validation/test_rigid_body_motion.py``) as the single most
important correctness check for this module.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from femtoolkit.continuum.deformation import (
    deformation_gradient,
    displacement_gradient,
    green_lagrange_strain_voigt,
    validate_deformation_gradient,
)
from femtoolkit.continuum.gauss import GAUSS_2X2X2_POINTS
from femtoolkit.continuum.jacobian import (
    jacobian_matrix_3d,
    physical_shape_function_derivatives_3d,
)
from femtoolkit.continuum.shape_functions import (
    hex8_shape_function_derivatives,
    tet4_shape_function_derivatives,
)
from femtoolkit.continuum.tensor import voigt_stress_to_tensor
from femtoolkit.exceptions import InvalidDeformationGradientError, InvalidElementError
from femtoolkit.materials.nonlinear import MaterialState, NonlinearMaterial
from femtoolkit.mesh.hex8_element import Hex8Element3D
from femtoolkit.mesh.tet4_element import Tet4Element3D
from femtoolkit.mesh.truss_element import TrussElement2D

GEOMETRIC_NONLINEAR_CAPABLE_ELEMENT_TYPES = (TrussElement2D, Tet4Element3D, Hex8Element3D)
"""Element types :class:`~femtoolkit.analysis.nonlinear_analysis.NonlinearAnalysis`
can drive with ``geometric_nonlinearity=True``."""

GeometricNonlinearCapableElement = TrussElement2D | Tet4Element3D | Hex8Element3D

_TANGENT_RELATIVE_STEP: float = 1e-6
"""Relative finite-difference step size for numerical tangent differentiation,
scaled by each DOF's own displacement magnitude -- see
:data:`_TANGENT_ABSOLUTE_FLOOR` for the floor applied near zero."""

_TANGENT_ABSOLUTE_FLOOR: float = 1e-6
"""Absolute floor (meters) for the per-DOF finite-difference step, preventing
a zero step when a displacement component is (numerically) exactly zero."""


@dataclass(frozen=True)
class NonlinearElementState:
    """One element's material state -- re-exported for dispatch-signature parity.

    Identical in shape and purpose to
    :class:`~femtoolkit.analysis.nonlinear_elements.NonlinearElementState`
    (a length-1 tuple for the truss and TET4, length-8 for HEX8's 8 Gauss
    points); kept as a separate class only so this module never needs to
    import from :mod:`femtoolkit.analysis.nonlinear_elements` (avoiding any
    coupling between the two independent dispatch pathways). The two
    classes are structurally interchangeable.
    """

    states: tuple[MaterialState, ...]


def _finite_difference_step(value: float) -> float:
    return _TANGENT_RELATIVE_STEP * max(abs(value), _TANGENT_ABSOLUTE_FLOOR)


# --------------------------------------------------------------------------
# Truss
# --------------------------------------------------------------------------


def _truss_current_geometry(
    element: TrussElement2D, displacements: Sequence[float]
) -> tuple[float, np.ndarray]:
    """Return ``(L, n)``: current length and current unit direction, from displaced nodes.

    Deliberately does **not** use ``element.length``/``element.direction_cosines``
    (both computed from the element's fixed *reference* coordinates) --
    the whole point of geometric nonlinearity is that these quantities
    change with the current trial displacement.
    """
    node_1, node_2 = element.nodes
    ux1, uy1, ux2, uy2 = displacements
    dx = (node_2.x + ux2) - (node_1.x + ux1)
    dy = (node_2.y + uy2) - (node_1.y + uy1)
    length = math.hypot(dx, dy)
    if not math.isfinite(length) or length <= 0.0:
        raise InvalidDeformationGradientError(
            f"TrussElement2D {element.id} has collapsed to zero (or non-finite) current "
            f"length ({length}); the applied displacement is not physically admissible."
        )
    return length, np.array([dx / length, dy / length])


def _truss_internal_force(
    element: TrussElement2D,
    material: NonlinearMaterial,
    displacements: Sequence[float],
    committed_state: NonlinearElementState,
) -> tuple[np.ndarray, MaterialState, float, float, np.ndarray]:
    """Return ``(f_int, trial_state, T, L, n)`` for one trial displacement."""
    reference_length = element.length
    current_length, direction = _truss_current_geometry(element, displacements)

    green_strain = (current_length**2 - reference_length**2) / (2.0 * reference_length**2)
    trial_material_state = material.trial_state(green_strain, committed_state.states[0])
    second_piola_kirchhoff_stress = trial_material_state.stress

    axial_force = (
        second_piola_kirchhoff_stress
        * element.cross_section.area
        * (current_length / reference_length)
    )
    f_int = axial_force * np.array(
        [-direction[0], -direction[1], direction[0], direction[1]]
    )
    return f_int, trial_material_state, axial_force, current_length, direction


def truss_geometric_internal_force_and_tangent(
    element: TrussElement2D,
    material: NonlinearMaterial,
    displacements: Sequence[float],
    committed_state: NonlinearElementState,
) -> tuple[np.ndarray, np.ndarray, NonlinearElementState]:
    """Compute a large-displacement truss element's internal force and tangent stiffness.

    **Kinematics.** For a 2-node truss, the deformation gradient reduces
    to the scalar stretch ratio ``lambda = L / L0`` along the member's own
    axis, and Green-Lagrange strain to its familiar 1D form:

    .. code-block:: text

        E = (L^2 - L0^2) / (2 * L0^2)

    with ``L0``/``L`` the reference/current length. ``S = material.trial_state(E,
    ...).stress`` (second Piola-Kirchhoff, here just a scalar), and the
    internal axial force conjugate to ``E`` (derived from the virtual-work
    identity ``delta_W = S * A0 * L0 * delta_E`` and ``delta_E = (L/L0^2) *
    delta_L``) is:

    .. code-block:: text

        T = S * A0 * (L / L0)

    acting along the *current* unit direction ``n``: ``f_int = T * [-n, n]``.

    **Tangent stiffness -- derived directly, not recalled.** Writing
    ``d = current relative position vector`` (``L = |d|``, ``n = d/L``),
    ``f = (T/L) * P @ d`` for a fixed connectivity matrix ``P``, and
    differentiating through ``T(E(L(d(u))))`` component by component gives,
    after the ``S*A0/(L0*L) - T/L^2`` cancellation that occurs along the
    way (worked in full in the module's test suite docstrings):

    .. code-block:: text

        K_total_local  = C_T * A0 * (L/L0)^2 / L0 * (n (x) n)  +  (T/L) * I
        K_geometric    = (T/L) * I                              (matches the
                          same "initial stress" formula used for TET4/HEX8:
                          Grad0(N1)*S*Grad0(N2)*V0*I, with Grad0(N)=+/-1/L0,
                          S*A0 = T*L0/L, V0=A0*L0)
        K_material     = K_total_local - K_geometric
                       = C_T * A0 * (L/L0)^2 / L0 * (n (x) n)

    where ``C_T = material.tangent_modulus(...)`` (``dS/dE``). At the
    reference configuration (``L=L0``, ``T=0``), this reduces exactly to
    the familiar small-displacement truss stiffness ``(C_T*A0/L0)*(n (x)
    n)`` -- the standard linear result -- confirming the large-displacement
    generalization is dimensionally and physically consistent. Expanded to
    the full 4x4 element matrix with the standard ``[[K,-K],[-K,K]]``
    truss sign pattern (:func:`~femtoolkit.analysis.stiffness.truss_element_stiffness_2d`'s
    same pattern).

    Args:
        element: The truss element.
        material: Its assigned nonlinear (scalar) material, e.g.
            :class:`~femtoolkit.materials.finite_strain.SaintVenantKirchhoff1D`.
        displacements: Trial nodal displacements ``[ux1, uy1, ux2, uy2]``.
        committed_state: The element's state at the start of the current
            load step.

    Returns:
        ``(f_int, k_t, trial_state)``: the 4-entry internal force vector,
        the 4x4 tangent stiffness matrix, and the element's new (not yet
        committed) state.
    """
    f_int, trial_material_state, axial_force, current_length, direction = _truss_internal_force(
        element, material, displacements, committed_state
    )
    tangent_modulus = material.tangent_modulus(trial_material_state)
    reference_length = element.length

    direction_outer = np.outer(direction, direction)
    identity_2 = np.eye(2)

    k_geometric_local = (axial_force / current_length) * identity_2
    k_material_local = (
        tangent_modulus
        * element.cross_section.area
        * (current_length / reference_length) ** 2
        / reference_length
    ) * direction_outer
    k_local = k_material_local + k_geometric_local

    k_t = np.block([[k_local, -k_local], [-k_local, k_local]])

    return f_int, k_t, NonlinearElementState(states=(trial_material_state,))


def truss_geometric_stiffness_split(
    element: TrussElement2D,
    material: NonlinearMaterial,
    displacements: Sequence[float],
    committed_state: NonlinearElementState,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(k_material, k_geometric)`` separately, for testing/documentation.

    See :func:`truss_geometric_internal_force_and_tangent`'s docstring for
    the derivation; ``k_material + k_geometric`` equals that function's
    ``k_t`` exactly.
    """
    _, trial_material_state, axial_force, current_length, direction = _truss_internal_force(
        element, material, displacements, committed_state
    )
    tangent_modulus = material.tangent_modulus(trial_material_state)
    reference_length = element.length

    direction_outer = np.outer(direction, direction)
    identity_2 = np.eye(2)

    k_geometric_local = (axial_force / current_length) * identity_2
    k_material_local = (
        tangent_modulus
        * element.cross_section.area
        * (current_length / reference_length) ** 2
        / reference_length
    ) * direction_outer

    k_geometric = np.block(
        [[k_geometric_local, -k_geometric_local], [-k_geometric_local, k_geometric_local]]
    )
    k_material = np.block(
        [[k_material_local, -k_material_local], [-k_material_local, k_material_local]]
    )
    return k_material, k_geometric


# --------------------------------------------------------------------------
# TET4
# --------------------------------------------------------------------------


def _tet4_reference_gradients(element: Tet4Element3D) -> np.ndarray:
    """Return the four TET4 nodes' constant reference-configuration shape gradients, ``(4, 3)``."""
    dn_dxi, dn_deta, dn_dzeta = tet4_shape_function_derivatives()
    x = tuple(node.x for node in element.nodes)
    y = tuple(node.y for node in element.nodes)
    z = tuple(node.z for node in element.nodes)
    jacobian = jacobian_matrix_3d(dn_dxi, dn_deta, dn_dzeta, x, y, z)
    jacobian_inverse = np.linalg.inv(jacobian)
    natural_derivatives = np.array([dn_dxi, dn_deta, dn_dzeta], dtype=float)
    return (jacobian_inverse @ natural_derivatives).T  # (4, 3)


def _tet4_internal_force(
    element: Tet4Element3D,
    material: NonlinearMaterial,
    displacements: Sequence[float],
    committed_state: NonlinearElementState,
    reference_gradients: np.ndarray,
) -> tuple[np.ndarray, MaterialState]:
    """Return ``(f_int, trial_state)`` for one trial displacement, at fixed reference geometry."""
    reference_volume = element.volume
    gradient_h = displacement_gradient(displacements, reference_gradients)
    deformation_gradient_tensor = deformation_gradient(gradient_h)
    validate_deformation_gradient(deformation_gradient_tensor)

    green_strain_voigt = green_lagrange_strain_voigt(deformation_gradient_tensor)
    trial_material_state = material.trial_state(green_strain_voigt, committed_state.states[0])
    stress_tensor = voigt_stress_to_tensor(trial_material_state.stress)

    f_int = np.zeros(12)
    for node_index in range(4):
        nodal_force = reference_volume * (
            deformation_gradient_tensor @ stress_tensor @ reference_gradients[node_index]
        )
        f_int[3 * node_index : 3 * node_index + 3] = nodal_force
    return f_int, trial_material_state


def _initial_stress_geometric_stiffness(
    reference_gradients: np.ndarray, stress_tensor: np.ndarray, reference_volume: float
) -> np.ndarray:
    """Assemble the standard "initial stress" geometric stiffness for an N-node solid.

    .. code-block:: text

        K_geometric[a,b] = V0 * (Grad0(Na) . S . Grad0(Nb)) * I_3

    Shared by TET4 (one point) and HEX8 (summed over Gauss points).

    Args:
        reference_gradients: ``(n_nodes, 3)`` reference-configuration
            shape-function gradients.
        stress_tensor: The symmetric 3x3 second Piola-Kirchhoff stress at
            this point.
        reference_volume: The integration weight (``V0`` for TET4, or
            ``weight * det(J0)`` for one HEX8 Gauss point).

    Returns:
        A ``(3*n_nodes, 3*n_nodes)`` NumPy array.
    """
    n_nodes = reference_gradients.shape[0]
    k_geometric = np.zeros((3 * n_nodes, 3 * n_nodes))
    for a in range(n_nodes):
        for b in range(n_nodes):
            scalar = reference_volume * (
                reference_gradients[a] @ stress_tensor @ reference_gradients[b]
            )
            k_geometric[3 * a : 3 * a + 3, 3 * b : 3 * b + 3] = scalar * np.eye(3)
    return k_geometric


def _numerical_tangent(
    internal_force_fn, displacements: Sequence[float], n_dofs: int
) -> np.ndarray:
    """Central-difference tangent of ``internal_force_fn(displacements) -> f_int`` w.r.t. each DOF.

    Shared by TET4 and HEX8's total-tangent computation (see the module
    docstring for why numerical differentiation is used here rather than a
    hand-derived material tangent).
    """
    displacements_array = np.asarray(displacements, dtype=float)
    jacobian = np.zeros((n_dofs, n_dofs))
    for dof_index in range(n_dofs):
        step = _finite_difference_step(displacements_array[dof_index])
        perturbation = np.zeros(n_dofs)
        perturbation[dof_index] = step

        f_plus = internal_force_fn(displacements_array + perturbation)
        f_minus = internal_force_fn(displacements_array - perturbation)
        jacobian[:, dof_index] = (f_plus - f_minus) / (2.0 * step)

    # The exact tangent is symmetric (conservative elasticity); central
    # differencing only approximates that, so symmetrizing removes purely
    # numerical asymmetry -- the same regularization used for J2Plasticity3D's
    # tangent in Version 15.
    return 0.5 * (jacobian + jacobian.T)


def tet4_geometric_stiffness_split(
    element: Tet4Element3D,
    material: NonlinearMaterial,
    displacements: Sequence[float],
    committed_state: NonlinearElementState,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(k_material, k_geometric)`` separately, for testing/documentation.

    ``k_material + k_geometric`` equals
    :func:`tet4_internal_force_and_tangent`'s ``k_t`` exactly, by
    construction (``k_material`` is defined as the numerically
    differentiated total tangent minus this closed-form ``k_geometric``).
    """
    reference_gradients = _tet4_reference_gradients(element)

    def internal_force(trial_displacements: np.ndarray) -> np.ndarray:
        f_int, _ = _tet4_internal_force(
            element, material, trial_displacements, committed_state, reference_gradients
        )
        return f_int

    _, trial_material_state = _tet4_internal_force(
        element, material, displacements, committed_state, reference_gradients
    )
    stress_tensor = voigt_stress_to_tensor(trial_material_state.stress)

    k_geometric = _initial_stress_geometric_stiffness(
        reference_gradients, stress_tensor, element.volume
    )
    k_total = _numerical_tangent(internal_force, displacements, n_dofs=12)
    k_material = k_total - k_geometric
    return k_material, k_geometric


def tet4_geometric_internal_force_and_tangent(
    element: Tet4Element3D,
    material: NonlinearMaterial,
    displacements: Sequence[float],
    committed_state: NonlinearElementState,
) -> tuple[np.ndarray, np.ndarray, NonlinearElementState]:
    """Compute a TET4 element's Total Lagrangian internal force and tangent stiffness.

    See the module docstring for the compact internal-force formula and
    the material/geometric tangent split (:func:`tet4_geometric_stiffness_split`
    exposes the split explicitly; this function returns their sum, matching
    the standard dispatch signature).

    Args:
        element: The TET4 element.
        material: Its assigned nonlinear (3D, 6-component) finite-strain
            material, e.g.
            :class:`~femtoolkit.materials.finite_strain.SaintVenantKirchhoff3D`.
        displacements: Trial nodal displacements, ordered per
            ``element.dof_keys()`` (12 entries).
        committed_state: The element's state at the start of the current
            load step (length-1 state, TET4's strain/stress being
            constant over the element).

    Returns:
        ``(f_int, k_t, trial_state)``.

    Raises:
        InvalidDeformationGradientError: If the deformation gradient is
            non-finite or non-positive-determinant (element inversion).
    """
    reference_gradients = _tet4_reference_gradients(element)
    f_int, trial_material_state = _tet4_internal_force(
        element, material, displacements, committed_state, reference_gradients
    )

    def internal_force(trial_displacements: np.ndarray) -> np.ndarray:
        f, _ = _tet4_internal_force(
            element, material, trial_displacements, committed_state, reference_gradients
        )
        return f

    k_t = _numerical_tangent(internal_force, displacements, n_dofs=12)

    return f_int, k_t, NonlinearElementState(states=(trial_material_state,))


def tet4_deformation_gradient(
    element: Tet4Element3D, displacements: Sequence[float]
) -> np.ndarray:
    """Return the (constant, single-point) deformation gradient ``F`` for a TET4.

    A small, purely additive reporting helper (Version 17): composes the
    same reference-gradient/displacement-gradient machinery already used
    internally by :func:`tet4_geometric_internal_force_and_tangent`, so a
    caller can recover ``F`` -- and from it, via
    :class:`~femtoolkit.materials.hyperelastic.HyperelasticMaterial`'s
    public methods, strain energy density, the Jacobian ``J``, and any of
    the PK1/PK2/Cauchy stress measures -- without this module needing to
    extend :class:`~femtoolkit.results.nonlinear_result.NonlinearAnalysisResult`
    (which stores only Voigt strain/stress, unchanged since Version 16).

    Args:
        element: The TET4 element.
        displacements: Trial nodal displacements, ordered per
            ``element.dof_keys()`` (12 entries).

    Returns:
        The 3x3 deformation gradient ``F``.

    Raises:
        InvalidDeformationGradientError: If ``F`` is non-finite or
            non-positive-determinant (element inversion).
    """
    reference_gradients = _tet4_reference_gradients(element)
    gradient_h = displacement_gradient(displacements, reference_gradients)
    deformation_gradient_tensor = deformation_gradient(gradient_h)
    validate_deformation_gradient(deformation_gradient_tensor)
    return deformation_gradient_tensor


# --------------------------------------------------------------------------
# HEX8
# --------------------------------------------------------------------------


def _hex8_reference_gradients_at_gauss_points(
    element: Hex8Element3D,
) -> list[tuple[np.ndarray, float]]:
    """Return, per 2x2x2 Gauss point, ``(reference_gradients (8, 3), reference_volume_weight)``."""
    x = tuple(node.x for node in element.nodes)
    y = tuple(node.y for node in element.nodes)
    z = tuple(node.z for node in element.nodes)

    data = []
    for point in GAUSS_2X2X2_POINTS:
        dn_dxi, dn_deta, dn_dzeta = hex8_shape_function_derivatives(point.xi, point.eta, point.zeta)
        dn_dx, dn_dy, dn_dz, det_j = physical_shape_function_derivatives_3d(
            dn_dxi, dn_deta, dn_dzeta, x, y, z
        )
        reference_gradients = np.column_stack([dn_dx, dn_dy, dn_dz])  # (8, 3)
        data.append((reference_gradients, point.weight * det_j))
    return data


def _hex8_internal_force(
    element: Hex8Element3D,
    material: NonlinearMaterial,
    displacements: Sequence[float],
    committed_state: NonlinearElementState,
    gauss_point_data: list[tuple[np.ndarray, float]],
) -> tuple[np.ndarray, list[MaterialState]]:
    """Return ``(f_int, trial_states)`` for one trial displacement, at fixed reference geometry."""
    f_int = np.zeros(24)
    trial_states: list[MaterialState] = []

    for point_index, (reference_gradients, reference_weight) in enumerate(gauss_point_data):
        gradient_h = displacement_gradient(displacements, reference_gradients)
        deformation_gradient_tensor = deformation_gradient(gradient_h)
        validate_deformation_gradient(deformation_gradient_tensor)

        green_strain_voigt = green_lagrange_strain_voigt(deformation_gradient_tensor)
        trial_material_state = material.trial_state(
            green_strain_voigt, committed_state.states[point_index]
        )
        stress_tensor = voigt_stress_to_tensor(trial_material_state.stress)

        for node_index in range(8):
            nodal_force = reference_weight * (
                deformation_gradient_tensor @ stress_tensor @ reference_gradients[node_index]
            )
            f_int[3 * node_index : 3 * node_index + 3] += nodal_force

        trial_states.append(trial_material_state)

    return f_int, trial_states


def hex8_geometric_stiffness_split(
    element: Hex8Element3D,
    material: NonlinearMaterial,
    displacements: Sequence[float],
    committed_state: NonlinearElementState,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(k_material, k_geometric)`` separately, for testing/documentation.

    See :func:`tet4_geometric_stiffness_split` for the analogous TET4
    derivation; ``k_geometric`` here is the sum of the same closed-form
    "initial stress" formula over the 8 Gauss points.
    """
    gauss_point_data = _hex8_reference_gradients_at_gauss_points(element)

    def internal_force(trial_displacements: np.ndarray) -> np.ndarray:
        f_int, _ = _hex8_internal_force(
            element, material, trial_displacements, committed_state, gauss_point_data
        )
        return f_int

    _, trial_states = _hex8_internal_force(
        element, material, displacements, committed_state, gauss_point_data
    )

    k_geometric = np.zeros((24, 24))
    for (reference_gradients, reference_weight), state in zip(
        gauss_point_data, trial_states, strict=True
    ):
        stress_tensor = voigt_stress_to_tensor(state.stress)
        k_geometric += _initial_stress_geometric_stiffness(
            reference_gradients, stress_tensor, reference_weight
        )

    k_total = _numerical_tangent(internal_force, displacements, n_dofs=24)
    k_material = k_total - k_geometric
    return k_material, k_geometric


def hex8_geometric_internal_force_and_tangent(
    element: Hex8Element3D,
    material: NonlinearMaterial,
    displacements: Sequence[float],
    committed_state: NonlinearElementState,
) -> tuple[np.ndarray, np.ndarray, NonlinearElementState]:
    """Compute a HEX8 element's Total Lagrangian internal force and tangent stiffness.

    Evaluated independently at each of the 8 Gauss points, each with its
    own committed/trial :class:`~femtoolkit.materials.nonlinear.MaterialState`
    (see the module docstring for the shared TET4/HEX8 formulation).

    Args:
        element: The HEX8 element.
        material: Its assigned nonlinear (3D, 6-component) finite-strain
            material.
        displacements: Trial nodal displacements, ordered per
            ``element.dof_keys()`` (24 entries).
        committed_state: The element's state at the start of the current
            load step (length-8 state, one per Gauss point).

    Returns:
        ``(f_int, k_t, trial_state)``.

    Raises:
        InvalidDeformationGradientError: If any Gauss point's deformation
            gradient is non-finite or non-positive-determinant.
    """
    gauss_point_data = _hex8_reference_gradients_at_gauss_points(element)
    f_int, trial_states = _hex8_internal_force(
        element, material, displacements, committed_state, gauss_point_data
    )

    def internal_force(trial_displacements: np.ndarray) -> np.ndarray:
        f, _ = _hex8_internal_force(
            element, material, trial_displacements, committed_state, gauss_point_data
        )
        return f

    k_t = _numerical_tangent(internal_force, displacements, n_dofs=24)

    return f_int, k_t, NonlinearElementState(states=tuple(trial_states))


def hex8_deformation_gradients(
    element: Hex8Element3D, displacements: Sequence[float]
) -> list[np.ndarray]:
    """Return the deformation gradient ``F`` at each of a HEX8's 8 Gauss points.

    See :func:`tet4_deformation_gradient` for why this reporting helper
    exists; the HEX8 analogue, one ``F`` per Gauss point (matching the
    per-Gauss-point granularity already used for strain/stress state
    throughout this module).

    Args:
        element: The HEX8 element.
        displacements: Trial nodal displacements, ordered per
            ``element.dof_keys()`` (24 entries).

    Returns:
        A length-8 list of 3x3 deformation gradients, in Gauss-point order.

    Raises:
        InvalidDeformationGradientError: If any Gauss point's deformation
            gradient is non-finite or non-positive-determinant.
    """
    gauss_point_data = _hex8_reference_gradients_at_gauss_points(element)
    deformation_gradients = []
    for reference_gradients, _ in gauss_point_data:
        gradient_h = displacement_gradient(displacements, reference_gradients)
        deformation_gradient_tensor = deformation_gradient(gradient_h)
        validate_deformation_gradient(deformation_gradient_tensor)
        deformation_gradients.append(deformation_gradient_tensor)
    return deformation_gradients


# --------------------------------------------------------------------------
# Shared dispatch
# --------------------------------------------------------------------------


def _require_geometric_material(
    element: GeometricNonlinearCapableElement, material: NonlinearMaterial, expected_shape: tuple
) -> None:
    """Reject a material whose strain shape does not match ``element``, early and clearly."""
    from femtoolkit.exceptions import ValidationError

    initial_strain = material.initial_state().strain
    actual_shape = () if np.isscalar(initial_strain) else np.asarray(initial_strain).shape
    if actual_shape != expected_shape:
        raise ValidationError(
            f"{type(element).__name__} (id={element.id}) requires a NonlinearMaterial with "
            f"strain shape {expected_shape}, got {type(material).__name__} with strain shape "
            f"{actual_shape}."
        )


def initial_element_state(
    element: GeometricNonlinearCapableElement, material: NonlinearMaterial
) -> NonlinearElementState:
    """Build an element's zero-strain initial :class:`NonlinearElementState`.

    Args:
        element: The element to build a state for.
        material: The nonlinear material assigned to it.

    Returns:
        A :class:`NonlinearElementState` with one (truss, TET4) or eight
        (HEX8) independent copies of ``material.initial_state()``.

    Raises:
        InvalidElementError: If ``element`` is not a supported type.
        ValidationError: If ``material``'s strain shape does not match
            the element.
    """
    if isinstance(element, TrussElement2D):
        _require_geometric_material(element, material, ())
        return NonlinearElementState(states=(material.initial_state(),))
    if isinstance(element, Tet4Element3D):
        _require_geometric_material(element, material, (6,))
        return NonlinearElementState(states=(material.initial_state(),))
    if isinstance(element, Hex8Element3D):
        _require_geometric_material(element, material, (6,))
        return NonlinearElementState(states=tuple(material.initial_state() for _ in range(8)))
    raise InvalidElementError(
        "Geometrically nonlinear analysis only supports TrussElement2D, Tet4Element3D, and "
        f"Hex8Element3D, got {type(element).__name__} (id={element.id})."
    )


def element_internal_force_and_tangent(
    element: GeometricNonlinearCapableElement,
    material: NonlinearMaterial,
    displacements: Sequence[float],
    committed_state: NonlinearElementState,
) -> tuple[np.ndarray, np.ndarray, NonlinearElementState]:
    """Dispatch to the element-type-specific geometrically nonlinear function.

    Mirrors :func:`femtoolkit.analysis.nonlinear_elements.element_internal_force_and_tangent`'s
    signature exactly, so
    :class:`~femtoolkit.analysis.nonlinear_analysis.NonlinearAnalysis` can
    select between the two dispatch modules with a single flag.

    Args:
        element: The element (truss, TET4, or HEX8).
        material: Its assigned nonlinear material.
        displacements: Trial nodal displacements, ordered per
            ``element.dof_keys()``.
        committed_state: The element's state at the start of the current
            load step.

    Returns:
        ``(f_int, k_t, trial_state)``.

    Raises:
        InvalidElementError: If ``element`` is not a supported type.
    """
    if isinstance(element, TrussElement2D):
        return truss_geometric_internal_force_and_tangent(
            element, material, displacements, committed_state
        )
    if isinstance(element, Tet4Element3D):
        return tet4_geometric_internal_force_and_tangent(
            element, material, displacements, committed_state
        )
    if isinstance(element, Hex8Element3D):
        return hex8_geometric_internal_force_and_tangent(
            element, material, displacements, committed_state
        )
    raise InvalidElementError(
        "Geometrically nonlinear analysis only supports TrussElement2D, Tet4Element3D, and "
        f"Hex8Element3D, got {type(element).__name__} (id={element.id})."
    )
