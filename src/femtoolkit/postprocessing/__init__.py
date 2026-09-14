"""Post-processing and visualization framework for FEA results (Version 22).

Turns any of this toolkit's existing result classes
(:class:`~femtoolkit.results.analysis_result.AnalysisResult`,
:class:`~femtoolkit.results.nonlinear_result.NonlinearAnalysisResult`,
:class:`~femtoolkit.results.dynamic_result.DynamicResult`,
:class:`~femtoolkit.thermal.thermal_result.SteadyStateThermalResult`,
:class:`~femtoolkit.thermal.thermal_result.TransientThermalResult`) into
a single, uniform representation suitable for querying, plotting, and
exporting -- without changing, duplicating, or depending on any solver
internals. See each submodule's docstring for its part of the pipeline:

.. code-block:: text

    Solver (unchanged)
        -> existing result class
        -> adapters.from_*(...)              (Result Model)
        -> field_calculator.with_derived_fields(...)  (Field Calculator)
        -> post_processor.PostProcessor(...)   (Post Processor / queries)
        -> visualization.plot_*(...)           (Visualization)
        -> export.export_to_csv/json(...)      (Export)
"""

from femtoolkit.postprocessing.adapters import (
    from_dynamic,
    from_nonlinear,
    from_static_linear,
    from_thermal_steady_state,
    from_thermal_transient,
    merge_thermomechanical,
)
from femtoolkit.postprocessing.export import (
    ExportFormat,
    export_result,
    export_to_csv,
    export_to_json,
)
from femtoolkit.postprocessing.field_calculator import (
    EngineeringSummary,
    deformed_coordinates,
    equivalent_strain,
    equivalent_stress,
    summarize,
    vector_magnitude,
    with_derived_fields,
)
from femtoolkit.postprocessing.post_processor import PostProcessor
from femtoolkit.postprocessing.result_model import (
    FieldValue,
    MeshTopology,
    ResultStep,
    SimulationResult,
)
from femtoolkit.postprocessing.visualization import (
    plot_deformed_shape_2d,
    plot_element_scatter_2d,
    plot_heat_flux_vectors_2d,
    plot_line,
    plot_nodal_contour_2d,
    plot_time_history,
)

__all__ = [
    "EngineeringSummary",
    "ExportFormat",
    "FieldValue",
    "MeshTopology",
    "PostProcessor",
    "ResultStep",
    "SimulationResult",
    "deformed_coordinates",
    "equivalent_strain",
    "equivalent_stress",
    "export_result",
    "export_to_csv",
    "export_to_json",
    "from_dynamic",
    "from_nonlinear",
    "from_static_linear",
    "from_thermal_steady_state",
    "from_thermal_transient",
    "merge_thermomechanical",
    "plot_deformed_shape_2d",
    "plot_element_scatter_2d",
    "plot_heat_flux_vectors_2d",
    "plot_line",
    "plot_nodal_contour_2d",
    "plot_time_history",
    "summarize",
    "vector_magnitude",
    "with_derived_fields",
]
