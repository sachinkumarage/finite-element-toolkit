"""Tests for femtoolkit.thermal.thermal_elements: conductivity/capacity matrices, gradient, flux."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.materials import LinearElastic2D, LinearElastic3D, Material
from femtoolkit.materials.thermal_properties import TemperatureDependentProperty
from femtoolkit.mesh import Hex8Element3D, Node, Tet4Element3D
from femtoolkit.mesh.bar_element import BarElement
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.quad_element import QuadElement2D
from femtoolkit.sections import CrossSection
from femtoolkit.thermal import thermal_elements as te
from femtoolkit.thermal.thermal_material import ThermalMaterial

_STEEL_THERMAL = ThermalMaterial(thermal_conductivity=50.0, density=7850.0, specific_heat=460.0)
_TET4_COORDS = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
_HEX8_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


@pytest.fixture
def bar() -> BarElement:
    mat = Material(name="steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    n1, n2 = Node(id=1, x=0.0, y=0.0, z=0.0), Node(id=2, x=2.0, y=0.0, z=0.0)
    return BarElement(id=1, nodes=(n1, n2), material=mat, cross_section=CrossSection(area=0.01))


@pytest.fixture
def cst() -> CSTElement2D:
    mat = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    nodes = (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=0.0, y=1.0, z=0.0),
    )
    return CSTElement2D(id=1, nodes=nodes, material=mat, thickness=0.02)


@pytest.fixture
def quad() -> QuadElement2D:
    mat = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    nodes = tuple(
        Node(id=i + 1, x=x, y=y, z=0.0)
        for i, (x, y) in enumerate([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    )
    return QuadElement2D(id=1, nodes=nodes, material=mat, thickness=0.02)


@pytest.fixture
def tet() -> Tet4Element3D:
    mat = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_TET4_COORDS))
    return Tet4Element3D(id=1, nodes=nodes, material=mat)


@pytest.fixture
def hexa() -> Hex8Element3D:
    mat = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_HEX8_COORDS))
    return Hex8Element3D(id=1, nodes=nodes, material=mat)


# --- Bar (1D) -----------------------------------------------------------------


def test_bar_conductivity_matrix_closed_form(bar: BarElement) -> None:
    k_matrix = te.bar_conductivity_matrix(bar, _STEEL_THERMAL, 293.15)
    expected = (50.0 * 0.01 / 2.0) * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert_allclose(k_matrix, expected)


def test_bar_conductivity_matrix_symmetric_and_singular(bar: BarElement) -> None:
    """K_T must be symmetric and singular (a uniform temperature offset gives zero flux)."""
    k_matrix = te.bar_conductivity_matrix(bar, _STEEL_THERMAL, 293.15)
    assert_allclose(k_matrix, k_matrix.T)
    assert_allclose(k_matrix.sum(axis=1), np.zeros(2), atol=1e-10)


def test_bar_capacity_matrix_totals_to_rho_c_v(bar: BarElement) -> None:
    c_matrix = te.bar_capacity_matrix(bar, _STEEL_THERMAL, 293.15)
    assert_allclose(c_matrix, c_matrix.T)
    assert c_matrix.sum() == pytest.approx(7850.0 * 460.0 * 0.01 * 2.0)


# --- CST (2D triangle) ----------------------------------------------------------


def test_cst_conductivity_matrix_symmetric_and_singular(cst: CSTElement2D) -> None:
    k_matrix = te.cst_conductivity_matrix(cst, _STEEL_THERMAL, 293.15)
    assert_allclose(k_matrix, k_matrix.T)
    assert_allclose(k_matrix.sum(axis=1), np.zeros(3), atol=1e-10)


def test_cst_capacity_matrix_totals_to_rho_c_v(cst: CSTElement2D) -> None:
    c_matrix = te.cst_capacity_matrix(cst, _STEEL_THERMAL, 293.15)
    assert c_matrix.sum() == pytest.approx(7850.0 * 460.0 * cst.area * cst.thickness)


# --- Q4 (2D quad) ---------------------------------------------------------------


def test_quad_conductivity_matrix_symmetric_and_singular(quad: QuadElement2D) -> None:
    k_matrix = te.quad_conductivity_matrix(quad, _STEEL_THERMAL, 293.15)
    assert_allclose(k_matrix, k_matrix.T, atol=1e-8)
    assert_allclose(k_matrix.sum(axis=1), np.zeros(4), atol=1e-8)


def test_quad_capacity_matrix_totals_to_rho_c_v(quad: QuadElement2D) -> None:
    c_matrix = te.quad_capacity_matrix(quad, _STEEL_THERMAL, 293.15)
    assert c_matrix.sum() == pytest.approx(7850.0 * 460.0 * 1.0 * quad.thickness, rel=1e-8)


# --- TET4 -----------------------------------------------------------------------


def test_tet4_conductivity_matrix_symmetric_and_singular(tet: Tet4Element3D) -> None:
    k_matrix = te.tet4_conductivity_matrix(tet, _STEEL_THERMAL, 293.15)
    assert_allclose(k_matrix, k_matrix.T)
    assert_allclose(k_matrix.sum(axis=1), np.zeros(4), atol=1e-10)


def test_tet4_capacity_matrix_totals_to_rho_c_v(tet: Tet4Element3D) -> None:
    c_matrix = te.tet4_capacity_matrix(tet, _STEEL_THERMAL, 293.15)
    assert c_matrix.sum() == pytest.approx(7850.0 * 460.0 * tet.volume)


# --- HEX8 -----------------------------------------------------------------------


def test_hex8_conductivity_matrix_symmetric_and_singular(hexa: Hex8Element3D) -> None:
    k_matrix = te.hex8_conductivity_matrix(hexa, _STEEL_THERMAL, 293.15)
    assert_allclose(k_matrix, k_matrix.T, atol=1e-8)
    assert_allclose(k_matrix.sum(axis=1), np.zeros(8), atol=1e-6)


def test_hex8_capacity_matrix_totals_to_rho_c_v(hexa: Hex8Element3D) -> None:
    c_matrix = te.hex8_capacity_matrix(hexa, _STEEL_THERMAL, 293.15)
    assert c_matrix.sum() == pytest.approx(7850.0 * 460.0 * 1.0, rel=1e-8)


# --- Temperature-dependent conductivity changes the matrix ----------------------


def test_temperature_dependent_conductivity_changes_conductivity_matrix(tet: Tet4Element3D) -> None:
    material = ThermalMaterial(
        thermal_conductivity=TemperatureDependentProperty(
            temperatures=(293.15, 493.15), values=(50.0, 25.0)
        ),
        density=7850.0,
        specific_heat=460.0,
    )
    k_cold = te.tet4_conductivity_matrix(tet, material, 293.15)
    k_hot = te.tet4_conductivity_matrix(tet, material, 493.15)
    assert_allclose(k_hot, 0.5 * k_cold)


# --- Fourier's law: q = -k * grad(T) --------------------------------------------


def test_fourier_law_heat_flows_from_hot_to_cold(bar: BarElement) -> None:
    """Hotter at node 1 than node 2 -> heat must flow in the +x direction."""
    flux = te.element_heat_flux(bar, _STEEL_THERMAL, [373.15, 293.15], 293.15)
    assert flux[0] > 0.0


def test_fourier_law_reversed_gradient_reverses_flux_direction(bar: BarElement) -> None:
    flux = te.element_heat_flux(bar, _STEEL_THERMAL, [293.15, 373.15], 293.15)
    assert flux[0] < 0.0


def test_fourier_law_zero_gradient_gives_zero_flux(bar: BarElement) -> None:
    flux = te.element_heat_flux(bar, _STEEL_THERMAL, [310.0, 310.0], 293.15)
    assert_allclose(flux, np.zeros(1), atol=1e-10)


def test_fourier_law_magnitude(bar: BarElement) -> None:
    flux = te.element_heat_flux(bar, _STEEL_THERMAL, [373.15, 293.15], 293.15)
    expected = -50.0 * (293.15 - 373.15) / 2.0
    assert flux[0] == pytest.approx(expected)


def test_fourier_law_with_temperature_dependent_conductivity(bar: BarElement) -> None:
    material = ThermalMaterial(
        thermal_conductivity=TemperatureDependentProperty(
            temperatures=(293.15, 493.15), values=(50.0, 25.0)
        ),
        density=7850.0,
        specific_heat=460.0,
    )
    flux_cold = te.element_heat_flux(bar, material, [373.15, 293.15], 293.15)
    flux_hot = te.element_heat_flux(bar, material, [373.15, 293.15], 493.15)
    assert flux_hot[0] == pytest.approx(0.5 * flux_cold[0])


def test_tet4_gradient_matches_direct_computation(tet: Tet4Element3D) -> None:
    """For F(x,y,z) = a*x, grad(T) should be exactly (a, 0, 0)."""
    temperatures = [0.0, 5.0, 0.0, 0.0]  # T = 5*x
    gradient = te.element_temperature_gradient(tet, temperatures)
    assert_allclose(gradient, [5.0, 0.0, 0.0], atol=1e-10)


def test_hex8_gradient_at_uniform_temperature_is_zero(hexa: Hex8Element3D) -> None:
    gradient = te.element_temperature_gradient(hexa, [300.0] * 8)
    assert_allclose(gradient, np.zeros(3), atol=1e-10)


# --- Temperature interpolation ---------------------------------------------------


def test_bar_temperature_interpolation_at_midpoint(bar: BarElement) -> None:
    assert te.element_temperature_at_centroid(bar, [300.0, 320.0]) == pytest.approx(310.0)


def test_cst_temperature_interpolation_is_average(cst: CSTElement2D) -> None:
    assert te.element_temperature_at_centroid(cst, [0.0, 3.0, 6.0]) == pytest.approx(3.0)


def test_tet4_temperature_interpolation_uniform_field(tet: Tet4Element3D) -> None:
    assert te.element_temperature_at_centroid(tet, [310.0, 310.0, 310.0, 310.0]) == pytest.approx(
        310.0
    )


def test_hex8_temperature_interpolation_uniform_field(hexa: Hex8Element3D) -> None:
    assert te.element_temperature_at_centroid(hexa, [305.0] * 8) == pytest.approx(305.0)


# --- Dispatch / contribution helpers ---------------------------------------------


def test_conductivity_contribution_dof_keys_match_nodes(tet: Tet4Element3D) -> None:
    contribution = te.conductivity_contribution(tet, _STEEL_THERMAL, 293.15)
    assert len(contribution.dof_keys) == 4
    assert contribution.stiffness.shape == (4, 4)


def test_capacity_contribution_dof_keys_match_nodes(hexa: Hex8Element3D) -> None:
    contribution = te.capacity_contribution(hexa, _STEEL_THERMAL, 293.15)
    assert len(contribution.dof_keys) == 8
    assert contribution.mass.shape == (8, 8)
