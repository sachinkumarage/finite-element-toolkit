"""Home page: what this toolkit is, and what this quickstart app can do."""

from __future__ import annotations

import streamlit as st

from femtoolkit.config import __version__

_GITHUB_URL = "https://github.com/sachinkumarage/finite-element-toolkit"


def render() -> None:
    """Render the Home page."""
    st.title("Finite Element Toolkit")
    st.caption(f"Engineering Simulation & Surrogate-Assisted Design -- v{__version__}")

    st.markdown(
        "An open-source Python toolkit for finite element analysis, built "
        "incrementally from the ground up: 1D/2D/3D elements, linear and "
        "nonlinear materials, static/dynamic/thermal analysis, verification & "
        "validation, engineering optimization, uncertainty quantification, "
        "parallel execution, and -- as of this version -- reduced-order "
        "modeling and surrogate-assisted adaptive design."
    )

    st.subheader("Major capabilities")
    st.markdown(
        "- **FEA core** -- bar/truss/frame/CST/Q4/TET4/HEX8 elements, linear "
        "and nonlinear materials (plasticity, hyperelasticity), static, "
        "dynamic, and thermal analysis.\n"
        "- **Verification & validation** -- analytical benchmarks, mesh "
        "convergence studies, equilibrium checks.\n"
        "- **Engineering optimization** -- bounded random search, "
        "coordinate search, differential evolution, genetic algorithm, "
        "particle swarm, NSGA-II, with optional uncertainty-aware (robust) "
        "design.\n"
        "- **Uncertainty quantification** -- Monte Carlo sampling, "
        "confidence intervals, sensitivity, reliability.\n"
        "- **Parallel execution** -- independent simulation tasks spread "
        "across local worker processes.\n"
        "- **Surrogate modeling & reduced-order models** -- polynomial/RBF "
        "surrogates, Proper Orthogonal Decomposition, always verified "
        "against real FEA before being trusted.\n"
        "- **Adaptive, surrogate-assisted optimization** (this version) -- "
        "candidate search on a cheap surrogate, refined by adding new "
        "high-fidelity samples where they matter most."
    )

    st.subheader("About this app")
    st.markdown(
        "This is a small, example-driven **quickstart** companion to the "
        "toolkit's full engineering GUI (`streamlit run src/femtoolkit/gui/app.py`). "
        "It never implements FEA, optimization, or surrogate logic itself -- "
        "every page here only calls the existing library APIs. Use the "
        "sidebar to try a simple FEA analysis, a plain optimization run, and "
        "the core new feature: surrogate-assisted adaptive optimization."
    )

    st.subheader("Project")
    st.markdown(f"Source code and documentation: {_GITHUB_URL}")


__all__ = ["render"]
