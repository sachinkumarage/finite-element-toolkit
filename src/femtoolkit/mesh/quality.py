"""Element and mesh quality metrics.

A finite element mesh's numerical accuracy depends not just on how many
elements it has, but on their *shape*. A poorly shaped element -- one
that is extremely elongated, or whose corners are far from the "ideal"
right angle (Q4) or equilateral angle (CST) -- can locally reduce
solution accuracy and, in more extreme cases, cause conditioning
problems in the global stiffness matrix. This module provides a small,
descriptive set of per-element and whole-mesh metrics for spotting such
elements, deliberately kept simple rather than building out an elaborate
mesh-quality framework.

**Metrics computed per element** (:func:`compute_element_quality`):

* **Area** -- the element's physical area (already available on both
  :class:`~femtoolkit.mesh.cst_element.CSTElement2D` and
  :class:`~femtoolkit.mesh.quad_element.QuadElement2D` as ``.area``).
* **Edge lengths** -- the minimum and maximum of the element's boundary
  edge lengths.
* **Aspect ratio** -- ``max_edge_length / min_edge_length``. For a
  rectangle this is exactly the ratio of its side lengths; for a general
  (non-rectangular) quadrilateral or a non-equilateral triangle it is
  only an *approximation* of shape distortion, since it does not account
  for skew independent of edge-length variation (see below). A value of
  1.0 is a square/equilateral triangle; large values indicate a
  "needle-like" element, which can be numerically problematic because the
  element's stiffness becomes very different in different directions.
* **Quality** -- ``min_edge_length / max_edge_length``, the reciprocal of
  aspect ratio, rescaled to ``(0, 1]`` so that **larger is better**
  (1.0 = best possible shape, values near 0 = degenerate). This is the
  scalar used for :class:`MeshQualitySummary`'s min/max/average.
* **Skewness** -- the *equiangle skew*, a standard, simple skewness
  measure: how far the element's most-distorted interior angle is from
  the "ideal" angle for its shape (60 degrees for an equilateral
  triangle, 90 degrees for a square), normalized to ``[0, 1]`` (0 = ideal
  shape, 1 = degenerate/flat).
* **Jacobian determinant** -- for a Q4 element only (evaluated at the
  element's natural-coordinate center), the same quantity validated by
  :class:`~femtoolkit.mesh.quad_element.QuadElement2D` at construction
  time (see :mod:`femtoolkit.continuum.jacobian`); ``None`` for a CST
  element, which has no isoparametric Jacobian in the same sense (its
  strain-displacement matrix is built directly from physical
  coordinates, not through a natural-coordinate mapping).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from femtoolkit.continuum.jacobian import jacobian_determinant as _jacobian_determinant
from femtoolkit.continuum.jacobian import jacobian_matrix
from femtoolkit.continuum.shape_functions import quad_shape_function_derivatives
from femtoolkit.exceptions import ValidationError
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.mesh.quad_element import QuadElement2D


@dataclass(frozen=True)
class ElementQuality:
    """Shape-quality metrics for a single continuum element.

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
class MeshQualitySummary:
    """Whole-mesh shape-quality summary, aggregated over its continuum elements.

    Attributes:
        num_nodes: Total number of nodes in the mesh.
        num_elements: Total number of elements in the mesh (of any type).
        min_area: Smallest element area, in square meters.
        max_area: Largest element area, in square meters.
        min_edge_length: Shortest edge across all elements, in meters.
        max_edge_length: Longest edge across all elements, in meters.
        min_quality: Worst (smallest) per-element quality value.
        max_quality: Best (largest) per-element quality value.
        average_quality: Mean per-element quality value.
        num_invalid_elements: Number of elements with non-positive or
            non-finite area. Elements are validated at construction time
            (see :mod:`femtoolkit.mesh.cst_element`/:mod:`femtoolkit.mesh.quad_element`),
            so for any mesh built through normal APIs this is expected to
            always be 0; the check remains as defense-in-depth.
    """

    num_nodes: int
    num_elements: int
    min_area: float
    max_area: float
    min_edge_length: float
    max_edge_length: float
    min_quality: float
    max_quality: float
    average_quality: float
    num_invalid_elements: int


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
        ValidationError: If ``element`` is not a :class:`~femtoolkit.mesh.cst_element.CSTElement2D`
            or :class:`~femtoolkit.mesh.quad_element.QuadElement2D`.
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
        raise ValidationError(
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


def compute_mesh_quality_summary(mesh: Mesh) -> MeshQualitySummary:
    """Compute a whole-mesh shape-quality summary over its continuum elements.

    Non-continuum elements (bar, truss, frame) are counted in
    ``num_nodes``/``num_elements`` but have no shape-quality concept and
    are excluded from the area/edge/quality statistics.

    Args:
        mesh: The mesh to summarize.

    Returns:
        The mesh's :class:`MeshQualitySummary`.

    Raises:
        ValidationError: If the mesh contains no CST or Q4 elements.
    """
    qualities = [
        compute_element_quality(element)
        for element in mesh.elements
        if isinstance(element, (CSTElement2D, QuadElement2D))
    ]
    if not qualities:
        raise ValidationError(
            "Mesh contains no continuum (CST or Q4) elements to compute a quality summary for."
        )

    num_invalid_elements = sum(
        1 for quality in qualities if not math.isfinite(quality.area) or quality.area <= 0
    )
    areas = [quality.area for quality in qualities]
    quality_values = [quality.quality for quality in qualities]

    return MeshQualitySummary(
        num_nodes=len(mesh.nodes),
        num_elements=len(mesh.elements),
        min_area=min(areas),
        max_area=max(areas),
        min_edge_length=min(quality.min_edge_length for quality in qualities),
        max_edge_length=max(quality.max_edge_length for quality in qualities),
        min_quality=min(quality_values),
        max_quality=max(quality_values),
        average_quality=sum(quality_values) / len(quality_values),
        num_invalid_elements=num_invalid_elements,
    )
