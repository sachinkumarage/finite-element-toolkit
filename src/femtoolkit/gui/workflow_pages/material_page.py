"""Material page: select, inspect, and edit material properties (Version 24, spec section 7)."""

from __future__ import annotations

import streamlit as st

from femtoolkit.application.materials_catalog import (
    CUSTOM_MATERIAL_KEY,
    MATERIAL_CATALOG,
    get_material_preset,
    material_options,
)
from femtoolkit.application.project import MaterialConfig
from femtoolkit.application.validation import validate_material
from femtoolkit.gui.components import error_banner, require_project
from femtoolkit.gui.state import AppState

_MECHANICAL_TYPES = ("linear_static", "nonlinear_static", "thermomechanical")
_THERMAL_TYPES = ("thermal_steady_state", "thermomechanical")


def render(state: AppState) -> None:
    """Render the Material configuration page."""
    st.header("Material")
    st.caption("Select a preset engineering material or enter custom properties.")

    if not require_project(state):
        return

    project = state.project
    is_mechanical = project.analysis_type in _MECHANICAL_TYPES
    is_thermal = project.analysis_type in _THERMAL_TYPES

    preset_key = st.selectbox(
        "Material preset",
        options=material_options(),
        format_func=lambda key: MATERIAL_CATALOG[key].name,
    )
    if st.button("Apply Preset"):
        project.material = get_material_preset(preset_key)
        st.success(f"Applied preset '{MATERIAL_CATALOG[preset_key].name}'.")

    st.subheader("Properties")
    material = project.material
    name = st.text_input("Name", value=material.name)

    if is_mechanical:
        col_e, col_nu, col_rho = st.columns(3)
        with col_e:
            youngs_modulus = st.number_input(
                "Young's Modulus E (Pa)",
                value=float(material.youngs_modulus or 0.0),
                format="%.3e",
                help="Must be positive (E > 0).",
            )
        with col_nu:
            poisson_ratio = st.number_input(
                "Poisson's Ratio v",
                value=float(material.poisson_ratio or 0.0),
                min_value=-0.999,
                max_value=0.499,
                help="Must satisfy -1 < v < 0.5.",
            )
        with col_rho:
            density = st.number_input("Density (kg/m^3)", value=float(material.density or 0.0))
    else:
        youngs_modulus = material.youngs_modulus
        poisson_ratio = material.poisson_ratio
        density = material.density

    if is_thermal:
        col_k, col_c = st.columns(2)
        with col_k:
            thermal_conductivity = st.number_input(
                "Thermal Conductivity k (W/m*K)",
                value=float(material.thermal_conductivity or 0.0),
                help="Must be positive (k > 0).",
            )
        with col_c:
            specific_heat = st.number_input(
                "Specific Heat c (J/kg*K)",
                value=float(material.specific_heat or 0.0),
                help="Must be positive (c > 0).",
            )
        if density in (None, 0.0):
            density = st.number_input(
                "Density (kg/m^3)", value=float(material.density or 1000.0), key="thermal_density"
            )
    else:
        thermal_conductivity = material.thermal_conductivity
        specific_heat = material.specific_heat

    if st.button("Save Material", type="primary"):
        project.material = MaterialConfig(
            name=name or CUSTOM_MATERIAL_KEY,
            youngs_modulus=youngs_modulus,
            poisson_ratio=poisson_ratio,
            density=density,
            thermal_conductivity=thermal_conductivity,
            specific_heat=specific_heat,
        )
        errors = validate_material(project)
        if errors:
            error_banner("Material validation failed", errors)
        else:
            st.success("Material saved and validated.")
