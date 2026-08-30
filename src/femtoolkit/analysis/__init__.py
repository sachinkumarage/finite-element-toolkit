"""Basic FEA mathematical foundation: DOFs, loads, boundary conditions,
stiffness/mass matrices, assembly, static, dynamic, and frequency-domain solving.
"""

from femtoolkit.analysis.assembly import (
    ElementForceContribution,
    ElementMassContribution,
    ElementStiffnessContribution,
    assemble_global_internal_force,
    assemble_global_mass,
    assemble_global_stiffness,
)
from femtoolkit.analysis.body_load import GravityLoad, gravity_load_to_nodal_loads
from femtoolkit.analysis.boundary_conditions import (
    BoundaryCondition,
    boundary_conditions_for_region,
)
from femtoolkit.analysis.convergence import (
    NORM_FLOOR,
    ConvergenceCriterion,
    displacement_correction_ratio,
    has_converged,
    residual_norm_ratio,
)
from femtoolkit.analysis.damping import (
    ModalDamping,
    RayleighDamping,
    modal_damping_ratios_from_matrix,
)
from femtoolkit.analysis.distributed_load import DistributedLoad, distributed_load_to_nodal_loads
from femtoolkit.analysis.dof import DOFMap, RotationDOF, TranslationDOF
from femtoolkit.analysis.dynamic_analysis import DynamicAnalysis
from femtoolkit.analysis.dynamic_loads import (
    ConstantLoad,
    SinusoidalLoad,
    StepLoad,
    TimeDependentLoad,
    TimeDependentNodalLoad,
)
from femtoolkit.analysis.dynamic_system import (
    DynamicSystem,
    build_dynamic_system,
    free_and_constrained_indices,
)
from femtoolkit.analysis.harmonic import (
    FrequencyResponseResult,
    HarmonicResult,
    frequency_response,
    harmonic_response,
)
from femtoolkit.analysis.load_case import LoadCase
from femtoolkit.analysis.load_combination import LoadCombination
from femtoolkit.analysis.load_manager import LoadManager
from femtoolkit.analysis.loads import NodalLoad
from femtoolkit.analysis.mass import MassMatrixType, element_mass_matrix, element_total_mass
from femtoolkit.analysis.modal import (
    DEFAULT_RIGID_BODY_TOLERANCE,
    ModalAnalysisResult,
    ModalResult,
    compute_periods,
    effective_modal_mass,
    effective_modal_mass_ratio,
    influence_vector,
    mass_normalize_mode_shapes,
    modal_analysis,
    modal_analysis_of_system,
    modal_participation_factors,
    natural_frequencies,
    natural_frequencies_of_system,
)
from femtoolkit.analysis.modal_superposition import modal_superposition
from femtoolkit.analysis.multi_point_constraint import (
    MultiPointConstraint,
    apply_multi_point_constraints,
)
from femtoolkit.analysis.newmark import (
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    effective_force,
    effective_stiffness,
    newmark_step,
    update_velocity_acceleration,
)
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.analysis.nonlinear_elements import (
    NONLINEAR_CAPABLE_ELEMENT_TYPES,
    NonlinearElementState,
    cst_internal_force_and_tangent,
    element_internal_force_and_tangent,
    initial_element_state,
    quad_internal_force_and_tangent,
)
from femtoolkit.analysis.spectrum import (
    ModalSpectralResponseResult,
    ResponseSpectrum,
    modal_spectral_response,
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
    "DEFAULT_BETA",
    "DEFAULT_GAMMA",
    "DEFAULT_RIGID_BODY_TOLERANCE",
    "NONLINEAR_CAPABLE_ELEMENT_TYPES",
    "NORM_FLOOR",
    "BoundaryCondition",
    "ConstantLoad",
    "ConvergenceCriterion",
    "DOFMap",
    "DistributedLoad",
    "DynamicAnalysis",
    "DynamicSystem",
    "ElementForceContribution",
    "ElementMassContribution",
    "ElementStiffnessContribution",
    "FrequencyResponseResult",
    "GravityLoad",
    "HarmonicResult",
    "LinearSystem",
    "LoadCase",
    "LoadCombination",
    "LoadManager",
    "MassMatrixType",
    "ModalAnalysisResult",
    "ModalDamping",
    "ModalResult",
    "ModalSpectralResponseResult",
    "MultiPointConstraint",
    "NodalLoad",
    "NonlinearAnalysis",
    "NonlinearElementState",
    "NonlinearSolverSettings",
    "RayleighDamping",
    "ResponseSpectrum",
    "RotationDOF",
    "SinusoidalLoad",
    "StaticLinearAnalysis",
    "StepLoad",
    "TemperatureLoad",
    "TimeDependentLoad",
    "TimeDependentNodalLoad",
    "TranslationDOF",
    "apply_multi_point_constraints",
    "assemble_global_internal_force",
    "assemble_global_mass",
    "assemble_global_stiffness",
    "bar_element_stiffness",
    "boundary_conditions_for_region",
    "build_dynamic_system",
    "build_force_vector",
    "compute_periods",
    "cst_element_stiffness",
    "cst_internal_force_and_tangent",
    "displacement_correction_ratio",
    "distributed_load_to_nodal_loads",
    "effective_force",
    "effective_modal_mass",
    "effective_modal_mass_ratio",
    "effective_stiffness",
    "element_internal_force_and_tangent",
    "element_mass_matrix",
    "element_total_mass",
    "frame_element_stiffness_2d",
    "frame_element_stiffness_local",
    "frame_transformation_matrix_2d",
    "free_and_constrained_indices",
    "frequency_response",
    "gravity_load_to_nodal_loads",
    "harmonic_response",
    "has_converged",
    "influence_vector",
    "initial_element_state",
    "mass_normalize_mode_shapes",
    "modal_analysis",
    "modal_analysis_of_system",
    "modal_damping_ratios_from_matrix",
    "modal_participation_factors",
    "modal_spectral_response",
    "modal_superposition",
    "natural_frequencies",
    "natural_frequencies_of_system",
    "newmark_step",
    "quad_element_stiffness",
    "quad_internal_force_and_tangent",
    "residual_norm_ratio",
    "solve",
    "thermal_corrected_strain",
    "thermal_corrected_stress",
    "thermal_load_to_nodal_loads",
    "truss_element_stiffness_2d",
    "update_velocity_acceleration",
]
