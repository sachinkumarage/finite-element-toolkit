"""Structural analysis result representations."""

from femtoolkit.results.analysis_result import AnalysisResult
from femtoolkit.results.element_results import FrameElementForces, FrameEndForces
from femtoolkit.results.result_set import ResultSet

__all__ = ["AnalysisResult", "FrameElementForces", "FrameEndForces", "ResultSet"]
