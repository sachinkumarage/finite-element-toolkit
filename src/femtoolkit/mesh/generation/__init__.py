"""Pluggable mesh generation (Version 25, spec sections 11-12).

.. code-block:: text

    MeshGenerator (protocol)
    |-- StructuredQuadMeshGenerator        (this version)
    |-- StructuredTriangularMeshGenerator  (this version)
    `-- future unstructured/TET4/HEX8 generators

See :mod:`femtoolkit.mesh.generation.base` for the protocol and
:mod:`femtoolkit.mesh.generation.structured` for this version's two
concrete generators, both thin adapters over the unchanged Version 8
:mod:`femtoolkit.mesh.generator` functions.
"""

from femtoolkit.mesh.generation.base import MeshGenerator
from femtoolkit.mesh.generation.structured import (
    StructuredQuadMeshGenerator,
    StructuredTriangularMeshGenerator,
)

__all__ = ["MeshGenerator", "StructuredQuadMeshGenerator", "StructuredTriangularMeshGenerator"]
