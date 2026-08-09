"""Mesh container.

This module defines :class:`Mesh`, a simple container that manages the
nodes and elements of a finite element model. It provides only storage,
retrieval, and referential-integrity validation; it does not generate,
refine, or otherwise process geometry.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from femtoolkit.exceptions import DuplicateIDError, EntityNotFoundError, ValidationError
from femtoolkit.mesh.node import Node


@runtime_checkable
class MeshElement(Protocol):
    """Structural type for anything a :class:`Mesh` can store as an element.

    The mesh only needs an element's ID and the nodes it connects; it does
    not care which concrete element type is used. Both the generic
    :class:`~femtoolkit.mesh.element.Element` (Version 1) and
    :class:`~femtoolkit.mesh.bar_element.BarElement` (Version 3) satisfy
    this protocol, so a single :class:`Mesh` implementation can hold
    either without a shared base class.
    """

    id: int
    nodes: Sequence[Node]


class Mesh:
    """A container for the nodes and elements that make up a finite element model.

    The mesh keeps nodes and elements indexed by their integer ID and
    guards against duplicate IDs and elements that reference nodes not
    present in the mesh. Version 1 does not generate or refine meshes; it
    is purely a management structure for entities the caller constructs.

    Example:
        >>> mesh = Mesh()
        >>> mesh.add_node(Node(id=1, x=0.0, y=0.0, z=0.0))
        >>> mesh.add_node(Node(id=2, x=1.0, y=0.0, z=0.0))
        >>> mesh.add_element(Element(id=1, nodes=[node_1, node_2], material=steel))
    """

    def __init__(self) -> None:
        """Create an empty mesh with no nodes or elements."""
        self._nodes: dict[int, Node] = {}
        self._elements: dict[int, MeshElement] = {}

    def add_node(self, node: Node) -> None:
        """Add a node to the mesh.

        Args:
            node: The node to add.

        Raises:
            DuplicateIDError: If a node with the same ID already exists in
                the mesh.
        """
        if node.id in self._nodes:
            raise DuplicateIDError(f"A node with id {node.id} already exists in the mesh.")
        self._nodes[node.id] = node

    def get_node(self, node_id: int) -> Node:
        """Retrieve a node by its ID.

        Args:
            node_id: The ID of the node to retrieve.

        Returns:
            The node registered under ``node_id``.

        Raises:
            EntityNotFoundError: If no node with the given ID exists.
        """
        try:
            return self._nodes[node_id]
        except KeyError as error:
            raise EntityNotFoundError(f"No node with id {node_id} found in the mesh.") from error

    def add_element(self, element: MeshElement) -> None:
        """Add an element to the mesh.

        Args:
            element: The element to add. All nodes referenced by the
                element must already have been added to the mesh.

        Raises:
            DuplicateIDError: If an element with the same ID already exists
                in the mesh.
            ValidationError: If the element references a node that is not
                registered in the mesh.
        """
        if element.id in self._elements:
            raise DuplicateIDError(f"An element with id {element.id} already exists in the mesh.")

        for node in element.nodes:
            if self._nodes.get(node.id) is not node:
                raise ValidationError(
                    f"Element {element.id} references node {node.id}, "
                    "which has not been added to the mesh."
                )

        self._elements[element.id] = element

    def get_element(self, element_id: int) -> MeshElement:
        """Retrieve an element by its ID.

        Args:
            element_id: The ID of the element to retrieve.

        Returns:
            The element registered under ``element_id``.

        Raises:
            EntityNotFoundError: If no element with the given ID exists.
        """
        try:
            return self._elements[element_id]
        except KeyError as error:
            raise EntityNotFoundError(
                f"No element with id {element_id} found in the mesh."
            ) from error

    @property
    def nodes(self) -> list[Node]:
        """All nodes currently in the mesh, in insertion order."""
        return list(self._nodes.values())

    @property
    def elements(self) -> list[MeshElement]:
        """All elements currently in the mesh, in insertion order."""
        return list(self._elements.values())
