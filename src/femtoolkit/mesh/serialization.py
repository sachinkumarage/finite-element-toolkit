"""JSON mesh export and import.

A lightweight, self-contained serialization format for meshes of
:class:`~femtoolkit.mesh.cst_element.CSTElement2D` and
:class:`~femtoolkit.mesh.quad_element.QuadElement2D` elements -- the two
element types :mod:`femtoolkit.mesh.generator` produces. Each element's
own material and thickness are stored alongside it, so a round trip
(:func:`export_mesh` then :func:`import_mesh`) fully reconstructs the
mesh without the caller needing to re-supply anything.

.. code-block:: text

    {
        "metadata": {"format_version": "1.0", "node_count": N, "element_count": M},
        "nodes": [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0}, ...],
        "elements": [
            {
                "id": 1,
                "type": "CSTElement2D" | "QuadElement2D",
                "node_ids": [1, 2, 3] | [1, 2, 3, 4],
                "thickness": 0.01,
                "material": {
                    "youngs_modulus": 200e9,
                    "poisson_ratio": 0.3,
                    "formulation": "plane_stress"
                }
            },
            ...
        ]
    }

**Limitations.** This is a foundation, not an industry mesh format: it
supports only :class:`~femtoolkit.mesh.cst_element.CSTElement2D` and
:class:`~femtoolkit.mesh.quad_element.QuadElement2D` elements (bar,
truss, and frame elements are out of scope for this version) and does
not implement Abaqus INP, ANSYS, Gmsh, or VTK formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.mesh.node import Node
from femtoolkit.mesh.quad_element import QuadElement2D
from femtoolkit.mesh.validation import validate_mesh

FORMAT_VERSION = "1.0"

_SUPPORTED_ELEMENT_TYPES = (CSTElement2D, QuadElement2D)


def _material_to_dict(material: LinearElastic2D) -> dict[str, Any]:
    return {
        "youngs_modulus": material.youngs_modulus,
        "poisson_ratio": material.poisson_ratio,
        "formulation": material.formulation,
    }


def _material_from_dict(data: dict[str, Any]) -> LinearElastic2D:
    return LinearElastic2D(
        youngs_modulus=data["youngs_modulus"],
        poisson_ratio=data["poisson_ratio"],
        formulation=data["formulation"],
    )


def mesh_to_dict(mesh: Mesh) -> dict[str, Any]:
    """Serialize a mesh of CST/Q4 elements into a plain JSON-compatible dict.

    Args:
        mesh: The mesh to serialize.

    Returns:
        A dict with ``"metadata"``, ``"nodes"``, and ``"elements"`` keys
        (see the module docstring for the schema).

    Raises:
        ValidationError: If the mesh contains an element type other than
            :class:`~femtoolkit.mesh.cst_element.CSTElement2D` or
            :class:`~femtoolkit.mesh.quad_element.QuadElement2D`.
    """
    nodes = [{"id": node.id, "x": node.x, "y": node.y, "z": node.z} for node in mesh.nodes]

    elements = []
    for element in mesh.elements:
        if not isinstance(element, _SUPPORTED_ELEMENT_TYPES):
            raise ValidationError(
                f"mesh_to_dict does not support element {element.id} "
                f"({type(element).__name__}); only CSTElement2D and QuadElement2D "
                "are supported."
            )
        elements.append(
            {
                "id": element.id,
                "type": type(element).__name__,
                "node_ids": [node.id for node in element.nodes],
                "thickness": element.thickness,
                "material": _material_to_dict(element.material),
            }
        )

    return {
        "metadata": {
            "format_version": FORMAT_VERSION,
            "node_count": len(nodes),
            "element_count": len(elements),
        },
        "nodes": nodes,
        "elements": elements,
    }


def mesh_from_dict(data: dict[str, Any]) -> Mesh:
    """Reconstruct a mesh from the dict produced by :func:`mesh_to_dict`.

    Args:
        data: A dict matching the schema documented in the module docstring.

    Returns:
        The reconstructed :class:`~femtoolkit.mesh.mesh.Mesh`.

    Raises:
        ValidationError: If ``data`` contains an unsupported element
            ``"type"``, or if the reconstructed mesh fails validation
            (see :func:`~femtoolkit.mesh.validation.validate_mesh`).
        KeyError: If a required field is missing from ``data``.
    """
    mesh = Mesh()
    for node_data in data["nodes"]:
        mesh.add_node(
            Node(id=node_data["id"], x=node_data["x"], y=node_data["y"], z=node_data["z"])
        )

    nodes_by_id = {node.id: node for node in mesh.nodes}

    for element_data in data["elements"]:
        element_type = element_data["type"]
        material = _material_from_dict(element_data["material"])
        element_nodes = tuple(nodes_by_id[node_id] for node_id in element_data["node_ids"])

        if element_type == "CSTElement2D":
            element = CSTElement2D(
                id=element_data["id"],
                nodes=element_nodes,  # type: ignore[arg-type]
                material=material,
                thickness=element_data["thickness"],
            )
        elif element_type == "QuadElement2D":
            element = QuadElement2D(
                id=element_data["id"],
                nodes=element_nodes,  # type: ignore[arg-type]
                material=material,
                thickness=element_data["thickness"],
            )
        else:
            raise ValidationError(
                f'Unsupported element type {element_type!r}; only "CSTElement2D" '
                'and "QuadElement2D" are supported.'
            )
        mesh.add_element(element)

    validate_mesh(mesh)
    return mesh


def export_mesh(mesh: Mesh, path: str | Path) -> None:
    """Export a mesh to a JSON file.

    Args:
        mesh: The mesh to export.
        path: Destination file path.

    Raises:
        ValidationError: If the mesh contains an unsupported element type.
    """
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(mesh_to_dict(mesh), file, indent=2)


def import_mesh(path: str | Path) -> Mesh:
    """Import a mesh from a JSON file written by :func:`export_mesh`.

    Args:
        path: Source file path.

    Returns:
        The reconstructed :class:`~femtoolkit.mesh.mesh.Mesh`.

    Raises:
        ValidationError: If the file's element types are unsupported or
            the reconstructed mesh fails validation.
    """
    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)
    return mesh_from_dict(data)
