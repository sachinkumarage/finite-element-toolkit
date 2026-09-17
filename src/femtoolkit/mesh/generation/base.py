"""The pluggable mesh-generator interface (Version 25, spec sections 11-12).

.. code-block:: text

    Geometry (e.g. femtoolkit.geometry.Rectangle)
        -> Meshing Parameters (MeshSizingParameters)
        -> Mesh Generator (this module's MeshGenerator protocol)
        -> Finite Element Mesh (femtoolkit.mesh.mesh.Mesh)

:class:`MeshGenerator` is a :class:`typing.Protocol`, not an abstract
base class: any object with a matching ``generate`` method satisfies it
structurally, the same "duck typing with a static-checkable shape"
pattern :class:`~femtoolkit.mesh.mesh.MeshElement` already uses for
elements. This lets :mod:`femtoolkit.mesh.generation.structured`'s two
concrete generators (and any future one -- unstructured triangular,
tetrahedral, hexahedral) plug in without a shared base class or without
:mod:`femtoolkit.mesh.mesh` or the core solver needing to know any
concrete generator exists.

This version implements only **structured** generators (thin adapters
over the existing, unchanged :mod:`femtoolkit.mesh.generator` functions
from Version 8) -- no unstructured, CAD-driven, or automatic mesh
generation. The geometry input is this toolkit's existing
:class:`~femtoolkit.geometry.rectangle.Rectangle` (Version 9); this
version does not introduce a new geometry representation, only reuses
what already exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from femtoolkit.geometry.rectangle import Rectangle
    from femtoolkit.materials.linear_elastic_2d import LinearElastic2D
    from femtoolkit.mesh.mesh import Mesh
    from femtoolkit.mesh.sizing import MeshSizingParameters


@runtime_checkable
class MeshGenerator(Protocol):
    """Structural interface for anything that can turn geometry + sizing into a mesh.

    A future unstructured, tetrahedral, or hexahedral generator
    satisfies this protocol without inheriting from anything -- the
    core mesh data structures (:class:`~femtoolkit.mesh.mesh.Mesh`) and
    the solver never need to change to accommodate a new generator.
    """

    def generate(
        self,
        geometry: Rectangle,
        sizing: MeshSizingParameters,
        material: LinearElastic2D,
        thickness: float,
    ) -> Mesh:
        """Generate a mesh over ``geometry`` at the requested ``sizing``.

        Args:
            geometry: The domain to mesh.
            sizing: The requested element sizing.
            material: The material assigned to every generated element.
            thickness: The thickness assigned to every generated element.

        Returns:
            The generated, already-validated
            :class:`~femtoolkit.mesh.mesh.Mesh`.
        """
        ...


__all__ = ["MeshGenerator"]
