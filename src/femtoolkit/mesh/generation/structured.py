"""Structured mesh generators, adapted to the
:class:`~femtoolkit.mesh.generation.base.MeshGenerator` protocol.

Both classes are thin adapters: all the actual node/element generation
logic remains exactly where it was in Version 8
(:func:`~femtoolkit.mesh.generator.create_quad_mesh`/
:func:`~femtoolkit.mesh.generator.create_triangular_mesh`), unchanged
and unduplicated. These classes only translate a
:class:`~femtoolkit.geometry.rectangle.Rectangle` +
:class:`~femtoolkit.mesh.sizing.MeshSizingParameters` pair into the
``(width, height, nx, ny)`` call those functions already expect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from femtoolkit.mesh.generator import DiagonalDirection, create_quad_mesh, create_triangular_mesh

if TYPE_CHECKING:
    from femtoolkit.geometry.rectangle import Rectangle
    from femtoolkit.materials.linear_elastic_2d import LinearElastic2D
    from femtoolkit.mesh.mesh import Mesh
    from femtoolkit.mesh.sizing import MeshSizingParameters


@dataclass(frozen=True)
class StructuredQuadMeshGenerator:
    """Generates a structured Q4 mesh (see :func:`~femtoolkit.mesh.generator.create_quad_mesh`)."""

    def generate(
        self,
        geometry: Rectangle,
        sizing: MeshSizingParameters,
        material: LinearElastic2D,
        thickness: float,
    ) -> Mesh:
        """Generate a structured Q4 mesh over ``geometry`` at the requested ``sizing``."""
        nx, ny = sizing.subdivisions(geometry.width, geometry.height)
        return create_quad_mesh(geometry.width, geometry.height, nx, ny, material, thickness)


@dataclass(frozen=True)
class StructuredTriangularMeshGenerator:
    """Generates a structured CST mesh (see
    :func:`~femtoolkit.mesh.generator.create_triangular_mesh`).

    Attributes:
        diagonal: Which diagonal splits each grid cell -- ``"forward"``
            or ``"backward"``. See
            :func:`~femtoolkit.mesh.generator.create_triangular_mesh`.
    """

    diagonal: DiagonalDirection = "forward"

    def generate(
        self,
        geometry: Rectangle,
        sizing: MeshSizingParameters,
        material: LinearElastic2D,
        thickness: float,
    ) -> Mesh:
        """Generate a structured CST mesh over ``geometry`` at the requested ``sizing``."""
        nx, ny = sizing.subdivisions(geometry.width, geometry.height)
        return create_triangular_mesh(
            geometry.width, geometry.height, nx, ny, material, thickness, diagonal=self.diagonal
        )


__all__ = ["StructuredQuadMeshGenerator", "StructuredTriangularMeshGenerator"]
