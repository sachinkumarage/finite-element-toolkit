"""Basic FEA mathematical foundation: DOFs, loads, boundary conditions,
stiffness matrices, assembly, and linear system solving.
"""

from femtoolkit.analysis.assembly import ElementStiffnessContribution, assemble_global_stiffness
from femtoolkit.analysis.body_load import GravityLoad, gravity_load_to_nodal_loads
from femtoolkit.analysis.boundary_conditions import (
    BoundaryCondition,
    boundary_conditions_for_region,
)
from femtoolkit.analysis.distributed_load import DistributedLoad, distributed_load_to_nodal_loads
from femtoolkit.analysis.dof import DOFMap, RotationDOF, TranslationDOF
from femtoolkit.analysis.load_case import LoadCase
from femtoolkit.analysis.load_combination import LoadCombination
from femtoolkit.analysis.load_manager import LoadManager
from femtoolkit.analysis.loads import NodalLoad
from femtoolkit.analysis.multi_point_constraint import (
    MultiPointConstraint,
    apply_multi_point_constraints,
)
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
from femtoolkit.analysis.thermal_load import (
    TemperatureLoad,
    thermal_corrected_strain,
    thermal_corrected_stress,
    thermal_load_to_nodal_loads,
)
from femtoolkit.analysis.transformation import frame_transformation_matrix_2d

__all__ = [
    "BoundaryCondition",
    "DOFMap",
    "DistributedLoad",
    "ElementStiffnessContribution",
    "GravityLoad",
    "LinearSystem",
    "LoadCase",
    "LoadCombination",
    "LoadManager",
    "MultiPointConstraint",
    "NodalLoad",
    "RotationDOF",
    "StaticLinearAnalysis",
    "TemperatureLoad",
    "TranslationDOF",
    "apply_multi_point_constraints",
    "assemble_global_stiffness",
    "bar_element_stiffness",
    "boundary_conditions_for_region",
    "build_force_vector",
    "cst_element_stiffness",
    "distributed_load_to_nodal_loads",
    "frame_element_stiffness_2d",
    "frame_element_stiffness_local",
    "frame_transformation_matrix_2d",
    "gravity_load_to_nodal_loads",
    "quad_element_stiffness",
    "solve",
    "thermal_corrected_strain",
    "thermal_corrected_stress",
    "thermal_load_to_nodal_loads",
    "truss_element_stiffness_2d",
]
