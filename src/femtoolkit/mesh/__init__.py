"""Mesh domain model: nodes, elements, the mesh container, structured mesh
generation, and (Version 25) quality/validation/statistics/refinement.
"""

from femtoolkit.mesh.bar_element import BarElement
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.edges import ElementEdge, element_edges, find_boundary_edges
from femtoolkit.mesh.element import Element
from femtoolkit.mesh.frame_element import FrameElement2D
from femtoolkit.mesh.generation import (
    MeshGenerator,
    StructuredQuadMeshGenerator,
    StructuredTriangularMeshGenerator,
)
from femtoolkit.mesh.generator import create_quad_mesh, create_triangular_mesh
from femtoolkit.mesh.hex8_element import Hex8Element3D
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.mesh.node import Node
from femtoolkit.mesh.quad_element import QuadElement2D
from femtoolkit.mesh.quality import (
    ElementQuality,
    MeshQualityReport,
    MeshQualitySummary,
    QualityEvaluator,
    SolidElementQuality,
)
from femtoolkit.mesh.refinement import refine_uniform
from femtoolkit.mesh.serialization import export_mesh, import_mesh, mesh_from_dict, mesh_to_dict
from femtoolkit.mesh.sizing import MeshSizingParameters
from femtoolkit.mesh.statistics import MeshStatistics, compute_mesh_statistics
from femtoolkit.mesh.tet4_element import Tet4Element3D
from femtoolkit.mesh.truss_element import TrussElement2D
from femtoolkit.mesh.validation import (
    MeshValidationReport,
    generate_validation_report,
    validate_mesh,
)

__all__ = [
    "BarElement",
    "CSTElement2D",
    "Element",
    "ElementEdge",
    "ElementQuality",
    "FrameElement2D",
    "Hex8Element3D",
    "Mesh",
    "MeshGenerator",
    "MeshQualityReport",
    "MeshQualitySummary",
    "MeshSizingParameters",
    "MeshStatistics",
    "MeshValidationReport",
    "Node",
    "QualityEvaluator",
    "QuadElement2D",
    "SolidElementQuality",
    "StructuredQuadMeshGenerator",
    "StructuredTriangularMeshGenerator",
    "Tet4Element3D",
    "TrussElement2D",
    "compute_mesh_statistics",
    "create_quad_mesh",
    "create_triangular_mesh",
    "element_edges",
    "export_mesh",
    "find_boundary_edges",
    "generate_validation_report",
    "import_mesh",
    "mesh_from_dict",
    "mesh_to_dict",
    "refine_uniform",
    "validate_mesh",
]
