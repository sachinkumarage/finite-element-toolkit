"""Basic FEA mathematical foundation: DOFs, loads, boundary conditions,
stiffness matrices, assembly, and linear system solving.
"""

from femtoolkit.analysis.assembly import ElementStiffnessContribution, assemble_global_stiffness
from femtoolkit.analysis.boundary_conditions import BoundaryCondition
from femtoolkit.analysis.dof import DOFMap, TranslationDOF
from femtoolkit.analysis.loads import NodalLoad
from femtoolkit.analysis.static_linear import StaticLinearAnalysis
from femtoolkit.analysis.stiffness import bar_element_stiffness
from femtoolkit.analysis.system import LinearSystem, build_force_vector, solve

__all__ = [
    "BoundaryCondition",
    "DOFMap",
    "ElementStiffnessContribution",
    "LinearSystem",
    "NodalLoad",
    "StaticLinearAnalysis",
    "TranslationDOF",
    "assemble_global_stiffness",
    "bar_element_stiffness",
    "build_force_vector",
    "solve",
]
