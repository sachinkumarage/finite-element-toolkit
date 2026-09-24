"""Parameter studies: scenarios, sweeps, comparison, and sensitivity (Version 30).

Turns "Simulation Project" (:mod:`femtoolkit.application.project`) and
"Simulation Run" (:mod:`femtoolkit.runs`) into "Simulation Scenario ->
Multiple Runs -> Result Comparison -> Sensitivity" -- an engineer
defines what varies (:class:`~femtoolkit.studies.scenarios.Scenario`,
:class:`~femtoolkit.studies.parameter_sweep.ParameterDefinition`), a
:class:`~femtoolkit.studies.runner.StudyRunner` executes every variation
sequentially through the existing solver pipeline, and the result is a
serializable :class:`~femtoolkit.studies.results.StudyResult` that can
be compared, checked for sensitivity, and reported on. See
``docs/studies.md`` for the full guide.
"""

from __future__ import annotations

from femtoolkit.studies.comparison import (
    DEFAULT_EPSILON,
    ComparisonEntry,
    ComparisonResult,
    absolute_difference,
    compare_runs,
    percentage_change,
    relative_difference,
)
from femtoolkit.studies.extractors import EXTRACTORS, Extractor, get_extractor
from femtoolkit.studies.parameter_sweep import (
    DEFAULT_MAX_SCENARIOS,
    ParameterDefinition,
    count_combinations,
    generate_scenarios,
)
from femtoolkit.studies.plots import plot_study_quantity
from femtoolkit.studies.report import (
    StudyReport,
    build_study_report,
    render_study_report_html,
    render_study_report_markdown,
    save_study_report,
)
from femtoolkit.studies.results import StudyResult
from femtoolkit.studies.runner import SimulationStudy, StudyRunner
from femtoolkit.studies.scenarios import (
    Scenario,
    apply_scenario,
    get_by_path,
    validate_unique_scenario_ids,
)
from femtoolkit.studies.sensitivity import (
    SensitivityResult,
    compute_sensitivity,
    compute_sensitivity_series,
)

__all__ = [
    "DEFAULT_EPSILON",
    "DEFAULT_MAX_SCENARIOS",
    "EXTRACTORS",
    "ComparisonEntry",
    "ComparisonResult",
    "Extractor",
    "ParameterDefinition",
    "Scenario",
    "SensitivityResult",
    "SimulationStudy",
    "StudyReport",
    "StudyResult",
    "StudyRunner",
    "absolute_difference",
    "apply_scenario",
    "build_study_report",
    "compare_runs",
    "compute_sensitivity",
    "compute_sensitivity_series",
    "count_combinations",
    "generate_scenarios",
    "get_by_path",
    "get_extractor",
    "percentage_change",
    "plot_study_quantity",
    "relative_difference",
    "render_study_report_html",
    "render_study_report_markdown",
    "save_study_report",
    "validate_unique_scenario_ids",
]
