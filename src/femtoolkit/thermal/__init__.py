"""Heat-transfer finite element framework (Version 20).

Kept as a separate top-level package from :mod:`femtoolkit.materials`/
:mod:`femtoolkit.analysis` on purpose: thermal conduction is a distinct
physical problem (a scalar temperature field, governed by the heat
equation) from the mechanical problems the rest of this toolkit solves,
and this module's thermal elements never read a mechanical element's
``material`` attribute (see :mod:`femtoolkit.thermal.thermal_elements`'s
module docstring). It reuses the existing continuum math (shape
functions, Jacobians, Gauss quadrature) and DOF/assembly/linear-solve
machinery wherever the underlying mathematics is identical, and connects
back to :mod:`femtoolkit.analysis.temperature_field` (Version 19) for the
sequential thermal -> mechanical workflow.
"""

from femtoolkit.thermal.thermal_analysis import (
    SteadyStateThermalAnalysis,
    TimeDependentThermalLoad,
    TransientThermalAnalysis,
    build_thermal_force_vector,
)
from femtoolkit.thermal.thermal_boundary_conditions import PrescribedHeatFlux, PrescribedTemperature
from femtoolkit.thermal.thermal_elements import (
    ThermalCapableElement,
    bar_capacity_matrix,
    bar_conductivity_matrix,
    capacity_contribution,
    conductivity_contribution,
    cst_capacity_matrix,
    cst_conductivity_matrix,
    element_heat_flux,
    element_temperature_at_centroid,
    element_temperature_gradient,
    hex8_capacity_matrix,
    hex8_conductivity_matrix,
    quad_capacity_matrix,
    quad_conductivity_matrix,
    tet4_capacity_matrix,
    tet4_conductivity_matrix,
    thermal_dof_keys,
)
from femtoolkit.thermal.thermal_loads import (
    HeatGeneration,
    ThermalLoad,
    element_heat_generation_to_thermal_loads,
    heat_generation_to_thermal_loads,
)
from femtoolkit.thermal.thermal_material import ThermalMaterial
from femtoolkit.thermal.thermal_result import SteadyStateThermalResult, TransientThermalResult

__all__ = [
    "HeatGeneration",
    "PrescribedHeatFlux",
    "PrescribedTemperature",
    "SteadyStateThermalAnalysis",
    "SteadyStateThermalResult",
    "ThermalCapableElement",
    "ThermalLoad",
    "ThermalMaterial",
    "TimeDependentThermalLoad",
    "TransientThermalAnalysis",
    "TransientThermalResult",
    "bar_capacity_matrix",
    "bar_conductivity_matrix",
    "build_thermal_force_vector",
    "capacity_contribution",
    "conductivity_contribution",
    "cst_capacity_matrix",
    "cst_conductivity_matrix",
    "element_heat_flux",
    "element_heat_generation_to_thermal_loads",
    "element_temperature_at_centroid",
    "element_temperature_gradient",
    "heat_generation_to_thermal_loads",
    "hex8_capacity_matrix",
    "hex8_conductivity_matrix",
    "quad_capacity_matrix",
    "quad_conductivity_matrix",
    "tet4_capacity_matrix",
    "tet4_conductivity_matrix",
    "thermal_dof_keys",
]
