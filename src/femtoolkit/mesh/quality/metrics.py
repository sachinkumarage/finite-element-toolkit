"""Per-element shape-quality metrics (Version 8, extended in Version 25).

**2D continuum elements** (:func:`compute_element_quality`, unchanged
since Version 8): area, edge lengths, aspect ratio, equiangle skewness,
and (for a Q4 element only) the Jacobian determinant at the element
center.

**3D solid elements** (:func:`compute_solid_element_quality`, new in
Version 25): the same family of ideas -- edge-length-based aspect ratio
and quality, plus a Jacobian-based distortion measure -- extended to
:class:`~femtoolkit.mesh.tet4_element.Tet4Element3D` and
:class:`~femtoolkit.mesh.hex8_element.Hex8Element3D`. Only metrics that
are mathematically meaningful for each element's own formulation are
computed:

* **TET4** has linear shape functions, so its Jacobian is *constant*
  over the element -- there is exactly one value to report, not a
  minimum over several points. It also follows a documented
  "either node winding is valid" orientation policy (see
  :mod:`femtoolkit.mesh.tet4_element`'s module docstring), so unlike
  HEX8, a negative Jacobian determinant does **not** by itself indicate
  an invalid element for TET4 -- only a near-zero one does (already
  guarded by ``MIN_TETRAHEDRON_VOLUME`` at construction). The reported
  value keeps its sign for transparency; validity is judged by
  magnitude, not sign.
* **HEX8** is isoparametric with *trilinear* shape functions, so its
  Jacobian varies from point to point; the metric reported is the
  **minimum** determinant over the same 2x2x2 Gauss points used by the
  element's own stiffness integration (:data:`~femtoolkit.continuum.gauss.GAUSS_2X2X2_POINTS`).
  HEX8 construction already rejects a non-positive Jacobian at any Gauss
  point (see :mod:`femtoolkit.continuum.jacobian`), so for any live
  ``Hex8Element3D`` instance this value is always positive -- it serves
  as a *distortion* measure (how close to degenerate), not a validity
  gate, which is already enforced at construction time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from femtoolkit.continuum.gauss import GAUSS_2X2X2_POINTS
from femtoolkit.continuum.jacobian import (
    jacobian_determinant as _jacobian_determinant,
)
from femtoolkit.continuum.jacobian import (
    jacobian_matrix,
    jacobian_matrix_3d,
    physical_shape_function_derivatives_3d,
)
from femtoolkit.continuum.shape_functions import (
    hex8_shape_function_derivatives,
    quad_shape_function_derivatives,
    tet4_shape_function_derivatives,
)
from femtoolkit.exceptions import UnsupportedQualityMetricError
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.hex8_element import Hex8Element3D
from femtoolkit.mesh.quad_element import QuadElement2D
from femtoolkit.mesh.tet4_element import Tet4Element3D

_TET4_EDGE_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
"""Local node-index pairs for TET4's 6 edges (every pair of the 4 corners)."""

_HEX8_EDGE_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
)
"""Local node-index pairs for HEX8's 12 edges, per
:data:`~femtoolkit.continuum.shape_functions._HEX8_NATURAL_COORDS`'s
bottom-face-then-top-face node ordering."""


@dataclass(frozen=True)
class ElementQuality:
    """Shape-quality metrics for a single 2D continuum element.

    Attributes:
        element_id: ID of the element these metrics describe.
        area: Element area, in square meters.
        min_edge_length: Shortest boundary edge, in meters.
        max_edge_length: Longest boundary edge, in meters.
        aspect_ratio: ``max_edge_length / min_edge_length`` (``>= 1.0``).
        skewness: Equiangle skew, in ``[0, 1]`` (0 = ideal shape).
        quality: ``min_edge_length / max_edge_length``, in ``(0, 1]``
            (1.0 = best possible shape).
        jacobian_determinant: Jacobian determinant at the element center,
            for a Q4 element; ``None`` for a CST element.
    """

    element_id: int
    area: float
    min_edge_length: float
    max_edge_length: float
    aspect_ratio: float
    skewness: float
    quality: float
    jacobian_determinant: float | None


@dataclass(frozen=True)
class SolidElementQuality:
    """Shape-quality metrics for a single 3D solid (TET4/HEX8) element.

    Attributes:
        element_id: ID of the element these metrics describe.
        volume: Element volume, in cubic meters.
        min_edge_length: Shortest edge, in meters.
        max_edge_length: Longest edge, in meters.
        aspect_ratio: ``max_edge_length / min_edge_length`` (``>= 1.0``).
        quality: ``min_edge_length / max_edge_length``, in ``(0, 1]``
            (1.0 = best possible shape).
        jacobian_determinant: TET4's single constant Jacobian
            determinant (sign-preserving -- see the module docstring for
            why a negative value is not, by itself, invalid for TET4),
            or HEX8's *minimum* Jacobian determinant over its 8 Gauss
            points (always positive for a successfully constructed
            element -- a distortion measure, not a validity gate).
        characteristic_size: ``volume ** (1/3)``, a representative linear
            dimension for the element.
    """

    element_id: int
    volume: float
    min_edge_length: float
    max_edge_length: float
    aspect_ratio: float
    quality: float
    jacobian_determinant: float
    characteristic_size: float


def _polygon_interior_angles(coordinates: list[tuple[float, float]]) -> list[float]:
    """Interior angle, in degrees, at each vertex of a (convex) polygon."""
    n = len(coordinates)
    angles = []
    for i in range(n):
        previous_point = coordinates[(i - 1) % n]
        current_point = coordinates[i]
        next_point = coordinates[(i + 1) % n]

        v1 = (previous_point[0] - current_point[0], previous_point[1] - current_point[1])
        v2 = (next_point[0] - current_point[0], next_point[1] - current_point[1])

        magnitude_1 = math.hypot(*v1)
        magnitude_2 = math.hypot(*v2)
        cosine = (v1[0] * v2[0] + v1[1] * v2[1]) / (magnitude_1 * magnitude_2)
        cosine = max(-1.0, min(1.0, cosine))  # clamp for floating-point safety
        angles.append(math.degrees(math.acos(cosine)))
    return angles


def _equiangle_skewness(angles: list[float], ideal_angle: float) -> float:
    """Equiangle skew: how far the most-distorted angle is from the ideal one.

    .. code-block:: text

        skew = max(
            (theta_max - theta_ideal) / (180 - theta_ideal),
            (theta_ideal - theta_min) / theta_ideal,
        )

    Zero for a perfectly regular shape (all angles equal to ``theta_ideal``);
    approaches 1 as any angle approaches 0 or 180 degrees (a degenerate,
    nearly flat element).
    """
    theta_max = max(angles)
    theta_min = min(angles)
    skew_from_max = (theta_max - ideal_angle) / (180.0 - ideal_angle)
    skew_from_min = (ideal_angle - theta_min) / ideal_angle
    return max(skew_from_max, skew_from_min, 0.0)


def compute_element_quality(element: CSTElement2D | QuadElement2D) -> ElementQuality:
    """Compute shape-quality metrics for a single CST or Q4 element.

    Args:
        element: The element to evaluate.

    Returns:
        The element's :class:`ElementQuality` metrics.

    Raises:
        UnsupportedQualityMetricError: If ``element`` is not a
            :class:`~femtoolkit.mesh.cst_element.CSTElement2D` or
            :class:`~femtoolkit.mesh.quad_element.QuadElement2D`.
    """
    if isinstance(element, CSTElement2D):
        ideal_angle = 60.0
        jacobian_det: float | None = None
    elif isinstance(element, QuadElement2D):
        ideal_angle = 90.0
        x_coords = [node.x for node in element.nodes]
        y_coords = [node.y for node in element.nodes]
        dn_dxi, dn_deta = quad_shape_function_derivatives(0.0, 0.0)
        jacobian_det = _jacobian_determinant(jacobian_matrix(dn_dxi, dn_deta, x_coords, y_coords))
    else:
        raise UnsupportedQualityMetricError(
            f"compute_element_quality does not support {type(element).__name__}; "
            "expected CSTElement2D or QuadElement2D."
        )

    coordinates = [(node.x, node.y) for node in element.nodes]
    n = len(coordinates)
    edge_lengths = [
        math.hypot(
            coordinates[(i + 1) % n][0] - coordinates[i][0],
            coordinates[(i + 1) % n][1] - coordinates[i][1],
        )
        for i in range(n)
    ]
    min_edge_length = min(edge_lengths)
    max_edge_length = max(edge_lengths)

    angles = _polygon_interior_angles(coordinates)

    return ElementQuality(
        element_id=element.id,
        area=element.area,
        min_edge_length=min_edge_length,
        max_edge_length=max_edge_length,
        aspect_ratio=max_edge_length / min_edge_length,
        skewness=_equiangle_skewness(angles, ideal_angle),
        quality=min_edge_length / max_edge_length,
        jacobian_determinant=jacobian_det,
    )


def _edge_lengths(nodes: tuple, edge_pairs: tuple[tuple[int, int], ...]) -> list[float]:
    return [
        math.dist((nodes[i].x, nodes[i].y, nodes[i].z), (nodes[j].x, nodes[j].y, nodes[j].z))
        for i, j in edge_pairs
    ]


def _tet4_quality(element: Tet4Element3D) -> SolidElementQuality:
    x_coords = [node.x for node in element.nodes]
    y_coords = [node.y for node in element.nodes]
    z_coords = [node.z for node in element.nodes]
    dn_dxi, dn_deta, dn_dzeta = tet4_shape_function_derivatives()
    jacobian = jacobian_matrix_3d(dn_dxi, dn_deta, dn_dzeta, x_coords, y_coords, z_coords)
    # TET4's own orientation policy accepts either winding, so evaluate the raw
    # (sign-preserving) determinant directly rather than the validated helper,
    # which would reject a negative value that TET4 itself treats as valid.
    jacobian_det = float(np.linalg.det(jacobian))

    edge_lengths = _edge_lengths(element.nodes, _TET4_EDGE_PAIRS)
    min_edge_length = min(edge_lengths)
    max_edge_length = max(edge_lengths)
    volume = element.volume

    return SolidElementQuality(
        element_id=element.id,
        volume=volume,
        min_edge_length=min_edge_length,
        max_edge_length=max_edge_length,
        aspect_ratio=max_edge_length / min_edge_length,
        quality=min_edge_length / max_edge_length,
        jacobian_determinant=jacobian_det,
        characteristic_size=volume ** (1.0 / 3.0),
    )


def _hex8_quality(element: Hex8Element3D) -> SolidElementQuality:
    x_coords = [node.x for node in element.nodes]
    y_coords = [node.y for node in element.nodes]
    z_coords = [node.z for node in element.nodes]

    jacobian_dets = []
    for point in GAUSS_2X2X2_POINTS:
        dn_dxi, dn_deta, dn_dzeta = hex8_shape_function_derivatives(point.xi, point.eta, point.zeta)
        _, _, _, det_j = physical_shape_function_derivatives_3d(
            dn_dxi, dn_deta, dn_dzeta, x_coords, y_coords, z_coords
        )
        jacobian_dets.append(det_j)

    edge_lengths = _edge_lengths(element.nodes, _HEX8_EDGE_PAIRS)
    min_edge_length = min(edge_lengths)
    max_edge_length = max(edge_lengths)
    volume = element.volume

    return SolidElementQuality(
        element_id=element.id,
        volume=volume,
        min_edge_length=min_edge_length,
        max_edge_length=max_edge_length,
        aspect_ratio=max_edge_length / min_edge_length,
        quality=min_edge_length / max_edge_length,
        jacobian_determinant=min(jacobian_dets),
        characteristic_size=volume ** (1.0 / 3.0),
    )


def compute_solid_element_quality(element: Tet4Element3D | Hex8Element3D) -> SolidElementQuality:
    """Compute shape-quality metrics for a single TET4 or HEX8 element.

    Args:
        element: The element to evaluate.

    Returns:
        The element's :class:`SolidElementQuality` metrics.

    Raises:
        UnsupportedQualityMetricError: If ``element`` is not a
            :class:`~femtoolkit.mesh.tet4_element.Tet4Element3D` or
            :class:`~femtoolkit.mesh.hex8_element.Hex8Element3D`.
    """
    if isinstance(element, Tet4Element3D):
        return _tet4_quality(element)
    if isinstance(element, Hex8Element3D):
        return _hex8_quality(element)
    raise UnsupportedQualityMetricError(
        f"compute_solid_element_quality does not support {type(element).__name__}; "
        "expected Tet4Element3D or Hex8Element3D."
    )


def compute_any_element_quality(element: object) -> ElementQuality | SolidElementQuality:
    """Compute shape-quality metrics for any supported element type.

    Dispatches to :func:`compute_element_quality` (CST/Q4) or
    :func:`compute_solid_element_quality` (TET4/HEX8) by element type.

    Args:
        element: The element to evaluate.

    Returns:
        An :class:`ElementQuality` or :class:`SolidElementQuality`.

    Raises:
        UnsupportedQualityMetricError: If ``element``'s type has no
            defined shape-quality metric (e.g. a bar, truss, or frame
            element, which has no area/volume concept).
    """
    if isinstance(element, (CSTElement2D, QuadElement2D)):
        return compute_element_quality(element)
    if isinstance(element, (Tet4Element3D, Hex8Element3D)):
        return compute_solid_element_quality(element)
    raise UnsupportedQualityMetricError(
        f"No shape-quality metric is defined for {type(element).__name__}."
    )


__all__ = [
    "ElementQuality",
    "SolidElementQuality",
    "compute_any_element_quality",
    "compute_element_quality",
    "compute_solid_element_quality",
]
