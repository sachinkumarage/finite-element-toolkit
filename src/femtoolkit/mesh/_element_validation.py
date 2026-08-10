"""Shared validation for two-node structural elements.

:class:`~femtoolkit.mesh.bar_element.BarElement` and
:class:`~femtoolkit.mesh.truss_element.TrussElement2D` both connect exactly
two distinct nodes and reference a material and a cross-section. This
private helper centralizes that common validation so it is not duplicated
between the two element types; each element still validates its own
geometry (element length) separately, since a bar's length depends only on
X while a truss element's depends on both X and Y.
"""

from __future__ import annotations

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import Material
from femtoolkit.mesh.node import Node
from femtoolkit.sections import CrossSection


def validate_two_node_element(
    element_type_name: str,
    id: int,  # noqa: A002 (mirrors the element's own `id` field name)
    nodes: tuple[Node, Node],
    material: Material,
    cross_section: CrossSection,
) -> None:
    """Validate the shared fields of a two-node structural element.

    Args:
        element_type_name: Name of the calling element class, used in
            error messages (e.g. ``"BarElement"``).
        id: Candidate element ID.
        nodes: Candidate pair of nodes.
        material: Candidate material.
        cross_section: Candidate cross-section.

    Raises:
        ValidationError: If ``id`` is not a positive integer, ``nodes`` is
            not a pair of distinct :class:`Node` instances, ``material``
            is not a :class:`Material`, or ``cross_section`` is not a
            :class:`CrossSection`.
    """
    if not isinstance(id, int) or isinstance(id, bool) or id <= 0:
        raise ValidationError(f"{element_type_name} id must be a positive integer, got {id!r}.")

    if len(nodes) != 2 or not all(isinstance(node, Node) for node in nodes):
        raise ValidationError(
            f"{element_type_name} nodes must be a pair of Node instances, got {nodes!r}."
        )
    if nodes[0].id == nodes[1].id:
        raise ValidationError(f"{element_type_name} requires two distinct nodes.")

    if not isinstance(material, Material):
        raise ValidationError(
            f"{element_type_name} material must be a Material instance, got {material!r}."
        )
    if not isinstance(cross_section, CrossSection):
        raise ValidationError(
            f"{element_type_name} cross_section must be a CrossSection instance, "
            f"got {cross_section!r}."
        )
