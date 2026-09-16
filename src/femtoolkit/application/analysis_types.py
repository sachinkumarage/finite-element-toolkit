"""The registry of analysis types the GUI exposes (Version 24).

Section 6 of this version's brief is explicit: "Only expose analysis
types that are actually supported by the current codebase... If a
feature is not yet implemented in the core toolkit, clearly mark it as
unavailable, or disable it in the GUI. Do not silently simulate
unsupported physics." This module is the single source of truth for
that decision, so :mod:`femtoolkit.gui` never has to guess.

Every analysis type listed here genuinely exists in the core toolkit
(``StaticLinearAnalysis`` since Version 3, ``SteadyStateThermalAnalysis``
since Version 20, ``NonlinearAnalysis`` since Version 13, the sequential
thermomechanical workflow since Version 19-21). Two of the four are
marked ``available=False`` not because the toolkit lacks them, but
because this version's GUI workflow (a single 2D structured mesh, a
region-based boundary-condition/load model) does not yet wire them up
-- an honest, narrower claim than "unsupported," made explicit in each
entry's ``reason``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisTypeInfo:
    """Describes one analysis type the GUI can offer.

    Attributes:
        key: The stable identifier stored in
            :attr:`~femtoolkit.application.project.Project.analysis_type`.
        label: The human-readable name shown in the GUI.
        available: Whether this version's GUI workflow can actually run it.
        reason: If ``available`` is ``False``, why -- shown to the user
            instead of a disabled control with no explanation.
    """

    key: str
    label: str
    available: bool
    reason: str | None = None


SUPPORTED_ANALYSIS_TYPES: dict[str, AnalysisTypeInfo] = {
    "linear_static": AnalysisTypeInfo(
        key="linear_static",
        label="Linear Static (Mechanical)",
        available=True,
    ),
    "thermal_steady_state": AnalysisTypeInfo(
        key="thermal_steady_state",
        label="Steady-State Thermal",
        available=True,
    ),
    "nonlinear_static": AnalysisTypeInfo(
        key="nonlinear_static",
        label="Nonlinear Static (Mechanical)",
        available=False,
        reason=(
            "Supported by the core toolkit (femtoolkit.analysis.nonlinear_analysis) "
            "since Version 13, but this version's GUI workflow does not yet wire up "
            "nonlinear material selection and load stepping."
        ),
    ),
    "thermomechanical": AnalysisTypeInfo(
        key="thermomechanical",
        label="Sequential Thermomechanical",
        available=False,
        reason=(
            "Supported by the core toolkit (the sequential thermal-to-mechanical "
            "workflow from Versions 19-21) but this version's GUI workflow does not "
            "yet wire up a combined thermal-then-mechanical project configuration."
        ),
    ),
}


def get_analysis_type(key: str) -> AnalysisTypeInfo:
    """Look up one analysis type by its key.

    Args:
        key: An analysis type key.

    Returns:
        The matching :class:`AnalysisTypeInfo`.

    Raises:
        KeyError: If ``key`` is not a known analysis type.
    """
    if key not in SUPPORTED_ANALYSIS_TYPES:
        raise KeyError(
            f"Unknown analysis type {key!r}; known types: "
            f"{sorted(SUPPORTED_ANALYSIS_TYPES)}."
        )
    return SUPPORTED_ANALYSIS_TYPES[key]


def available_analysis_types() -> list[AnalysisTypeInfo]:
    """Return every analysis type this version's GUI can actually run."""
    return [info for info in SUPPORTED_ANALYSIS_TYPES.values() if info.available]
