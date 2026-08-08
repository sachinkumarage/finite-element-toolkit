"""Element data model.

This module defines :class:`Element`, a finite element that connects a set
of nodes and references a material. Version 1 stores this topology only;
it does not compute stiffness matrices, shape functions, or any other
element-level mathematics. Those belong to future versions.
"""

from __future__ import annotations

from dataclasses import dataclass

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import Material
from femtoolkit.mesh.node import Node


@dataclass
class Element:
    """A finite element connecting two or more nodes through a material.

    Version 1 represents an element purely as topology: which nodes it
    connects and which material it is made of. It performs no numerical
    computation; stiffness matrices, shape functions, and stress/strain
    calculations belong to future versions of the toolkit.

    Attributes:
        id: Positive integer identifying the element uniquely within a mesh.
        nodes: Ordered list of :class:`~femtoolkit.mesh.node.Node` instances
            the element connects. Must contain at least one node.
        material: :class:`~femtoolkit.materials.material.Material` assigned
            to the element.

    Raises:
        ValidationError: If ``id`` is invalid, ``nodes`` is empty or
            contains a non-:class:`Node` object, or ``material`` is not a
            :class:`Material` instance.

    Example:
        >>> element = Element(id=1, nodes=[node_1, node_2], material=steel)
    """

    id: int
    nodes: list[Node]
    material: Material

    def __post_init__(self) -> None:
        """Validate element topology immediately after construction.

        Raises:
            ValidationError: If ``id`` is not a positive integer, ``nodes``
                is empty or contains a non-:class:`Node` object, or
                ``material`` is not a :class:`Material` instance.
        """
        if not isinstance(self.id, int) or isinstance(self.id, bool) or self.id <= 0:
            raise ValidationError(f"Element id must be a positive integer, got {self.id!r}.")

        if not self.nodes:
            raise ValidationError("Element must connect at least one node.")

        for node in self.nodes:
            if not isinstance(node, Node):
                raise ValidationError(f"Element nodes must be Node instances, got {node!r}.")

        if not isinstance(self.material, Material):
            raise ValidationError(
                f"Element material must be a Material instance, got {self.material!r}."
            )
