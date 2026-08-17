"""Basic FEA mathematical foundation: DOFs, loads, boundary conditions,
stiffness matrices, assembly, and linear system solving.
"""

from femtoolkit.analysis.assembly import ElementStiffnessContribution, assemble_global_stiffness
from femtoolkit.analysis.boundary_conditions import (
    BoundaryCondition,
    boundary_conditions_for_region,
)
from femtoolkit.analysis.distributed_load import DistributedLoad, distributed_load_to_nodal_loads
from femtoolkit.analysis.dof import DOFMap, RotationDOF, TranslationDOF
from femtoolkit.analysis.load_case import LoadCase
from femtoolkit.analysis.loads import NodalLoad
from femtoolkit.analysis.static_linear import StaticLinearAnalysis
from femtoolkit.analysis.stiffness import (
    bar_element_stiffness,
    cst_element_stiffness,
    frame_element_stiffness_2d,
    frame_element_stiffness_local,
    quad_element_stiffness,
    truss_element_stiffness_2d,
)
from femtoolkit.analysis.system import LinearSystem, build_force_vector, solve
from femtoolkit.analysis.transformation import frame_transformation_matrix_2d

__all__ = [
    "BoundaryCondition",
    "DOFMap",
    "DistributedLoad",
    "ElementStiffnessContribution",
    "LinearSystem",
    "LoadCase",
    "NodalLoad",
    "RotationDOF",
    "StaticLinearAnalysis",
    "TranslationDOF",
    "assemble_global_stiffness",
    "bar_element_stiffness",
    "boundary_conditions_for_region",
    "build_force_vector",
    "cst_element_stiffness",
    "distributed_load_to_nodal_loads",
    "frame_element_stiffness_2d",
    "frame_element_stiffness_local",
    "frame_transformation_matrix_2d",
    "quad_element_stiffness",
    "solve",
    "truss_element_stiffness_2d",
]
