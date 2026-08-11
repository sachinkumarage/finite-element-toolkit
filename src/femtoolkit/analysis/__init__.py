"""Basic FEA mathematical foundation: DOFs, loads, boundary conditions,
stiffness matrices, assembly, and linear system solving.
"""

from femtoolkit.analysis.assembly import ElementStiffnessContribution, assemble_global_stiffness
from femtoolkit.analysis.boundary_conditions import BoundaryCondition
from femtoolkit.analysis.dof import DOFMap, RotationDOF, TranslationDOF
from femtoolkit.analysis.loads import NodalLoad
from femtoolkit.analysis.static_linear import StaticLinearAnalysis
from femtoolkit.analysis.stiffness import (
    bar_element_stiffness,
    frame_element_stiffness_2d,
    frame_element_stiffness_local,
    truss_element_stiffness_2d,
)
from femtoolkit.analysis.system import LinearSystem, build_force_vector, solve
from femtoolkit.analysis.transformation import frame_transformation_matrix_2d

__all__ = [
    "BoundaryCondition",
    "DOFMap",
    "ElementStiffnessContribution",
    "LinearSystem",
    "NodalLoad",
    "RotationDOF",
    "StaticLinearAnalysis",
    "TranslationDOF",
    "assemble_global_stiffness",
    "bar_element_stiffness",
    "build_force_vector",
    "frame_element_stiffness_2d",
    "frame_element_stiffness_local",
    "frame_transformation_matrix_2d",
    "solve",
    "truss_element_stiffness_2d",
]
