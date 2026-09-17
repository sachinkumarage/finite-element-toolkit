"""The engineering application/service layer for the GUI (Version 24).

Sits strictly between the GUI (:mod:`femtoolkit.gui`) and the core FEA
engine:

.. code-block:: text

    Streamlit UI (femtoolkit.gui)
         |
    Application Service (femtoolkit.application)
         |
    Existing FEA Engine (femtoolkit.materials/mesh/analysis/thermal)
         |
    Results (femtoolkit.postprocessing)

Nothing in this package assembles a stiffness matrix, integrates a
shape function, or solves a linear system -- it only builds/validates
plain configuration (:mod:`femtoolkit.application.project`) and
orchestrates the existing, unmodified solvers. Every class here is
usable, and independently testable, without ever importing
:mod:`streamlit`.
"""

from femtoolkit.application.analysis_types import (
    SUPPORTED_ANALYSIS_TYPES,
    AnalysisTypeInfo,
    available_analysis_types,
    get_analysis_type,
)
from femtoolkit.application.exceptions_display import describe_error, describe_visualization_error
from femtoolkit.application.materials_catalog import (
    CUSTOM_MATERIAL_KEY,
    MATERIAL_CATALOG,
    get_material_preset,
    material_options,
)
from femtoolkit.application.mesh_preparation_service import MeshPreparationService
from femtoolkit.application.model_service import MeshSummary, ModelService
from femtoolkit.application.project import (
    BoundaryConditionConfig,
    LoadConfig,
    MaterialConfig,
    MeshConfig,
    Project,
    SolverConfig,
)
from femtoolkit.application.project_service import ProjectService
from femtoolkit.application.results_service import FieldRange, ResultsService
from femtoolkit.application.simulation_service import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_INVALID,
    SimulationRunResult,
    SimulationService,
)
from femtoolkit.application.validation import ValidationResult, validate_project

__all__ = [
    "CUSTOM_MATERIAL_KEY",
    "MATERIAL_CATALOG",
    "RUN_STATUS_COMPLETED",
    "RUN_STATUS_FAILED",
    "RUN_STATUS_INVALID",
    "SUPPORTED_ANALYSIS_TYPES",
    "AnalysisTypeInfo",
    "BoundaryConditionConfig",
    "FieldRange",
    "LoadConfig",
    "MaterialConfig",
    "MeshConfig",
    "MeshPreparationService",
    "MeshSummary",
    "ModelService",
    "Project",
    "ProjectService",
    "ResultsService",
    "SimulationRunResult",
    "SimulationService",
    "SolverConfig",
    "ValidationResult",
    "available_analysis_types",
    "describe_error",
    "describe_visualization_error",
    "get_analysis_type",
    "get_material_preset",
    "material_options",
    "validate_project",
]
