"""Structural analysis result representations."""

from femtoolkit.results.analysis_result import AnalysisResult
from femtoolkit.results.dynamic_result import DynamicResult
from femtoolkit.results.element_results import FrameElementForces, FrameEndForces
from femtoolkit.results.nonlinear_result import LoadStepResult, NonlinearAnalysisResult
from femtoolkit.results.result_set import ResultSet

__all__ = [
    "AnalysisResult",
    "DynamicResult",
    "FrameElementForces",
    "FrameEndForces",
    "LoadStepResult",
    "NonlinearAnalysisResult",
    "ResultSet",
]
