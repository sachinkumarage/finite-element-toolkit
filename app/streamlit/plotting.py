"""Small, dependency-light chart helpers for the quickstart Streamlit app.

Every function here returns a plain :class:`pandas.DataFrame` shaped for one of
Streamlit's native chart widgets (``st.line_chart``/``st.scatter_chart``/
``st.bar_chart``) -- no new plotting dependency is introduced; pandas already
ships as a transitive dependency of Streamlit itself.
"""

from __future__ import annotations

import pandas as pd


def predicted_vs_actual_frame(predicted: list[float], actual: list[float]) -> pd.DataFrame:
    """A predicted-vs-actual scatter frame, with the ideal ``y = x`` line for reference."""
    return pd.DataFrame({"Surrogate Prediction": predicted, "High-Fidelity FEA": actual})


def convergence_frame(values: list[float | None], label: str) -> pd.DataFrame:
    """An iteration-indexed convergence frame, skipping any ``None`` entries."""
    rows = [(i, v) for i, v in enumerate(values) if v is not None]
    return pd.DataFrame(
        {"Iteration": [row[0] for row in rows], label: [row[1] for row in rows]}
    ).set_index("Iteration")


def objective_history_frame(objective_values: list[float], label: str) -> pd.DataFrame:
    """An evaluation-indexed objective-history frame for an optimization run."""
    frame = pd.DataFrame({"Evaluation": range(len(objective_values)), label: objective_values})
    return frame.set_index("Evaluation")
