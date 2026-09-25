"""Uncertainty Analysis page (Version 31).

Surfaces the Version 31 uncertainty-quantification framework
(:mod:`femtoolkit.uncertainty`) inside the existing GUI: defining
uncertain parameters and their probability distributions, previewing
each input distribution before running anything, configuring and
executing a Monte Carlo study on top of the existing Version 30 study
infrastructure, and reviewing output statistics, percentiles,
confidence intervals, correlation, limit exceedance, and a downloadable
report -- with no distribution, sampling, Monte Carlo execution, or
statistical logic implemented in this module itself.

**On study cancellation (spec section 34).** This GUI runs each page
synchronously within one Streamlit script execution, the same way the
Version 30 Simulation Studies page does -- there is no in-process
mechanism in this architecture to safely interrupt a study partway
through and preserve its partial results. A genuine cancellation
mechanism is therefore not implemented here; ``max_samples`` and a
deliberately modest default sample count are this page's safeguard
against an unexpectedly long-running study instead.
"""

from __future__ import annotations

import streamlit as st

from femtoolkit.application.exceptions_display import describe_error
from femtoolkit.exceptions import FiniteElementToolkitError
from femtoolkit.gui.components import require_project
from femtoolkit.gui.state import AppState
from femtoolkit.studies.extractors import EXTRACTORS, get_extractor
from femtoolkit.uncertainty.confidence import confidence_interval_mean
from femtoolkit.uncertainty.correlation import correlation_summary
from femtoolkit.uncertainty.distributions import (
    DeterministicDistribution,
    LognormalDistribution,
    NormalDistribution,
    UniformDistribution,
)
from femtoolkit.uncertainty.monte_carlo import (
    DEFAULT_MAX_SAMPLES,
    MonteCarloConfig,
    MonteCarloRunner,
)
from femtoolkit.uncertainty.parameters import UncertainParameter, UncertaintyCategory
from femtoolkit.uncertainty.plots import plot_input_distribution, plot_output_histogram
from femtoolkit.uncertainty.reliability import exceedance_probability
from femtoolkit.uncertainty.report import (
    build_uncertainty_report,
    render_uncertainty_report_markdown,
)
from femtoolkit.uncertainty.statistics import compute_output_statistics

_PARAMETERS_KEY = "femtoolkit_uncertainty_parameters"
_RESULT_KEY = "femtoolkit_uncertainty_result"

_DISTRIBUTION_TYPES = ("deterministic", "uniform", "normal", "lognormal")


def _parameters(state_store) -> list[UncertainParameter]:
    return state_store.setdefault(_PARAMETERS_KEY, [])


def render(state: AppState) -> None:
    """Render the Uncertainty Analysis page."""
    st.header("Uncertainty Analysis")
    st.caption(
        "Uncertain Inputs -> Sampling -> Monte Carlo Runs -> Output Statistics -> "
        "Confidence Intervals -> Correlation -> Limit Exceedance -> Report"
    )

    if not require_project(state):
        return

    _render_parameter_builder(state)
    st.divider()
    _render_execution(state)
    st.divider()
    _render_analysis(state)
    st.divider()
    _render_report_generation(state)


def _render_parameter_builder(state: AppState) -> None:
    st.subheader("1. Uncertain Parameters")
    st.caption(
        "Each parameter is a dotted override path on the current project (e.g. "
        "'material.youngs_modulus') paired with a probability distribution."
    )
    parameters = _parameters(st.session_state)

    # Deliberately not an st.form: the number-input labels below (Value/Low/High vs.
    # Mean/Standard deviation) depend on the Distribution selectbox, and a form defers
    # every widget's rerun until submission -- which would leave the wrong fields shown
    # until after the first, discarded submit. Plain widgets rerun immediately instead.
    col_path, col_label, col_units = st.columns(3)
    path = col_path.text_input("Override path", placeholder="material.youngs_modulus")
    label = col_label.text_input("Label", placeholder="Young's Modulus")
    units = col_units.text_input("Units", placeholder="Pa")

    distribution_type = st.selectbox("Distribution", options=_DISTRIBUTION_TYPES)
    col_a, col_b = st.columns(2)
    high: float | None = None
    if distribution_type == "deterministic":
        value = col_a.number_input("Value", value=1.0, format="%.6g")
    elif distribution_type == "uniform":
        value = col_a.number_input("Low", value=0.0, format="%.6g")
        high = col_b.number_input("High", value=1.0, format="%.6g")
    else:
        value = col_a.number_input("Mean", value=1.0, format="%.6g")
        high = col_b.number_input("Standard deviation", value=0.1, format="%.6g")

    col_category, col_lower, col_upper = st.columns(3)
    category = col_category.selectbox("Category", options=[c.value for c in UncertaintyCategory])
    lower_bound_text = col_lower.text_input("Physical lower bound (optional)", value="")
    upper_bound_text = col_upper.text_input("Physical upper bound (optional)", value="")

    submitted = st.button("Add Parameter")

    if submitted:
        try:
            distribution = _build_distribution(distribution_type, value, high)
            parameters.append(
                UncertainParameter(
                    path=path,
                    label=label or path,
                    distribution=distribution,
                    units=units,
                    category=UncertaintyCategory(category),
                    physical_lower_bound=float(lower_bound_text) if lower_bound_text else None,
                    physical_upper_bound=float(upper_bound_text) if upper_bound_text else None,
                )
            )
            st.success(f"Added uncertain parameter '{label or path}'.")
        except FiniteElementToolkitError as exc:
            st.error(describe_error(exc))
        except ValueError as exc:
            st.error(str(exc))

    if not parameters:
        st.info("No uncertain parameters added yet -- add at least one to run a study.")
        return

    for index, parameter in enumerate(parameters):
        col_info, col_plot, col_remove = st.columns([4, 2, 1])
        col_info.write(
            f"**{parameter.label}** (`{parameter.path}`) -- "
            f"{type(parameter.distribution).__name__}, mean={parameter.distribution.mean():.6g}, "
            f"std={parameter.distribution.std():.6g}"
        )
        if col_plot.button("Preview", key=f"preview_parameter_{index}"):
            st.session_state[f"_preview_{index}"] = True
        if col_remove.button("Remove", key=f"remove_parameter_{index}"):
            parameters.pop(index)
            st.rerun()
        if st.session_state.get(f"_preview_{index}"):
            st.pyplot(plot_input_distribution(parameter))


def _build_distribution(distribution_type: str, value: float, high: float | None):
    if distribution_type == "deterministic":
        return DeterministicDistribution(value)
    if distribution_type == "uniform":
        return UniformDistribution(value, high)
    if distribution_type == "normal":
        return NormalDistribution(value, high)
    return LognormalDistribution.from_mean_std(value, high)


def _render_execution(state: AppState) -> None:
    st.subheader("2. Monte Carlo Study")
    parameters = _parameters(st.session_state)

    col_samples, col_seed, col_max = st.columns(3)
    n_samples = col_samples.number_input("Sample count", min_value=1, value=100, step=1)
    seed = col_seed.number_input("Random seed", min_value=0, value=42, step=1)
    max_samples = col_max.number_input(
        "Max samples", min_value=1, value=DEFAULT_MAX_SAMPLES, step=1
    )

    col_method, col_fail_fast = st.columns(2)
    method = col_method.selectbox("Sampling method", options=["random", "latin_hypercube"])
    fail_fast = col_fail_fast.checkbox("Stop at first failure (fail_fast)", value=False)

    quantities = st.multiselect(
        "Output quantities", options=sorted(EXTRACTORS), default=["maximum_displacement"]
    )
    study_name = st.text_input("Study name", value=f"{state.project.name} Uncertainty Study")

    if st.button("Run Monte Carlo Study", type="primary", disabled=not (parameters and quantities)):
        config = MonteCarloConfig(
            study_id=study_name,
            name=study_name,
            base_project=state.project,
            parameters=list(parameters),
            output_quantities=list(quantities),
            n_samples=int(n_samples),
            seed=int(seed),
            method=method,
            max_samples=int(max_samples),
            fail_fast=fail_fast,
        )
        try:
            with st.spinner(f"Running {config.n_samples} samples..."):
                result = MonteCarloRunner().run(config)
        except FiniteElementToolkitError as exc:
            st.error(describe_error(exc))
            return

        st.session_state[_RESULT_KEY] = result
        st.success(
            f"Completed: {result.n_successful + result.n_failed} / {config.n_samples}. "
            f"Successful: {result.n_successful}. Failed: {result.n_failed}. "
            f"Rejected as physically invalid: {result.n_invalid}."
        )


def _current_result():
    return st.session_state.get(_RESULT_KEY)


def _render_analysis(state: AppState) -> None:
    st.subheader("3. Output Statistics & Analysis")
    result = _current_result()
    if result is None:
        st.info("Run a Monte Carlo study above to see its output analysis.")
        return
    if result.n_successful == 0:
        st.warning("No run completed successfully -- no output analysis is available.")
        return

    quantity_name = st.selectbox(
        "Quantity", options=result.config.output_quantities, key="uncertainty_quantity_select"
    )
    extractor = get_extractor(quantity_name)
    values = result.output_values(extractor)

    stats = compute_output_statistics(
        values,
        quantity_name,
        n_requested=result.config.n_samples,
        n_failed=result.n_failed,
        n_invalid=result.n_invalid,
    )
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Mean", f"{stats.mean:.4e}" if stats.mean is not None else "-")
    col_b.metric("Std", f"{stats.std:.4e}" if stats.std is not None else "-")
    col_c.metric(
        "CV",
        f"{stats.coefficient_of_variation:.3f}" if stats.coefficient_of_variation else "-",
    )
    col_d.metric("N successful", stats.n_successful)

    st.write("**Percentiles** (describe this sample, not a confidence interval)")
    st.dataframe(
        [{"Percentile": f"P{p:g}", "Value": v} for p, v in sorted(stats.percentiles.items())],
        width="stretch",
    )

    if len(values) >= 2:
        ci = confidence_interval_mean(values, quantity_name)
        st.write(
            f"**95%-style confidence interval for the mean** ({ci.confidence_level:.0%}): "
            f"[{ci.lower:.4e}, {ci.upper:.4e}] (this is a statement about the estimated "
            "mean, not a percentile of the output distribution)"
        )

    st.pyplot(plot_output_histogram(values, quantity_name))

    st.write("**Correlation (input vs. output, descriptive only)**")
    pairs = {}
    for parameter in result.config.parameters:
        x, y = result.successful_pairs(parameter.path, extractor)
        if len(x) >= 2:
            pairs[parameter.path] = (parameter.label, x, y)
    if pairs:
        summary = correlation_summary(quantity_name, pairs)
        st.dataframe(
            [
                {
                    "Parameter": r.parameter_label,
                    "Pearson r": r.pearson_r,
                    "Spearman rho": r.spearman_rho,
                    "N": r.n_samples,
                }
                for r in summary
            ],
            width="stretch",
        )

    st.write("**Limit exceedance (empirical, from this sample only)**")
    col_threshold, col_direction = st.columns(2)
    threshold = col_threshold.number_input("Threshold", value=float(stats.mean or 0.0))
    direction = col_direction.selectbox("Direction", options=["above", "below"])
    if len(values) > 0:
        exceedance = exceedance_probability(values, threshold, quantity_name, direction)
        st.write(
            f"{exceedance.n_exceeding} / {exceedance.n_samples} samples "
            f"({exceedance.exceedance_frequency:.2%}) {direction} {threshold:.4g}"
        )


def _render_report_generation(state: AppState) -> None:
    st.subheader("4. Uncertainty Report")
    result = _current_result()
    if result is None or result.n_successful == 0:
        st.info("Run a Monte Carlo study with at least one successful run to generate a report.")
        return

    summary = st.text_area(
        "Study summary", value=f"Uncertainty study of project '{state.project.name}'."
    )
    conclusions = st.text_area("Conclusions (optional, free text -- never generated automatically)")

    quantity_name = result.config.output_quantities[0]
    extractor = get_extractor(quantity_name)
    values = result.output_values(extractor)
    output_statistics = []
    confidence_intervals = []
    if len(values) > 0:
        output_statistics.append(
            compute_output_statistics(
                values,
                quantity_name,
                n_requested=result.config.n_samples,
                n_failed=result.n_failed,
                n_invalid=result.n_invalid,
            )
        )
    if len(values) >= 2:
        confidence_intervals.append(confidence_interval_mean(values, quantity_name))

    report = build_uncertainty_report(
        title=f"{result.config.name} Report",
        study_summary=summary,
        base_model_description=f"Project '{state.project.name}', {state.project.analysis_type}.",
        result=result,
        output_statistics=output_statistics,
        confidence_intervals=confidence_intervals,
        conclusions=conclusions,
    )
    markdown_text = render_uncertainty_report_markdown(report)

    st.download_button(
        "Download Markdown Report",
        data=markdown_text,
        file_name="uncertainty_report.md",
        mime="text/markdown",
    )
    st.text_area("Preview (Markdown)", value=markdown_text, height=300)
