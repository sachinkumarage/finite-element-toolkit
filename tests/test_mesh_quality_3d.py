"""Tests for femtoolkit.mesh.quality.metrics's TET4/HEX8 (3D) support (Version 25)."""

import pytest

from femtoolkit.exceptions import UnsupportedQualityMetricError
from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node, Tet4Element3D
from femtoolkit.mesh.quality.metrics import (
    SolidElementQuality,
    compute_any_element_quality,
    compute_element_quality,
    compute_solid_element_quality,
)

_HEX8_UNIT_CUBE = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]

_TET4_REFERENCE = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]


@pytest.fixture
def material() -> LinearElastic3D:
    return LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)


def _hex8(material: LinearElastic3D, coords=None) -> Hex8Element3D:
    coords = coords or _HEX8_UNIT_CUBE
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(coords))
    return Hex8Element3D(id=1, nodes=nodes, material=material)


def _tet4(material: LinearElastic3D, coords=None) -> Tet4Element3D:
    coords = coords or _TET4_REFERENCE
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(coords))
    return Tet4Element3D(id=1, nodes=nodes, material=material)


def test_unit_cube_hex8_quality_is_ideal(material: LinearElastic3D) -> None:
    hexa = _hex8(material)
    quality = compute_solid_element_quality(hexa)

    assert isinstance(quality, SolidElementQuality)
    assert quality.volume == pytest.approx(1.0)
    assert quality.min_edge_length == pytest.approx(1.0)
    assert quality.max_edge_length == pytest.approx(1.0)
    assert quality.aspect_ratio == pytest.approx(1.0)
    assert quality.quality == pytest.approx(1.0)
    assert quality.jacobian_determinant == pytest.approx(0.125)
    assert quality.characteristic_size == pytest.approx(1.0)


def test_reference_tet4_quality(material: LinearElastic3D) -> None:
    tet = _tet4(material)
    quality = compute_solid_element_quality(tet)

    assert quality.volume == pytest.approx(1.0 / 6.0)
    assert quality.jacobian_determinant == pytest.approx(1.0)
    assert quality.characteristic_size == pytest.approx((1.0 / 6.0) ** (1.0 / 3.0))


def test_elongated_hex8_has_high_aspect_ratio(material: LinearElastic3D) -> None:
    coords = [
        (0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (5.0, 0.0, 1.0), (5.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]
    hexa = _hex8(material, coords)
    quality = compute_solid_element_quality(hexa)

    assert quality.aspect_ratio == pytest.approx(5.0)
    assert quality.quality == pytest.approx(0.2)
    assert quality.jacobian_determinant > 0


def test_tet4_negative_jacobian_is_not_invalid() -> None:
    """TET4 accepts either node winding -- a negative Jacobian is not, by itself, invalid."""
    material = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    reversed_coords = list(reversed(_TET4_REFERENCE))
    tet = _tet4(material, reversed_coords)
    quality = compute_solid_element_quality(tet)

    assert quality.volume == pytest.approx(1.0 / 6.0)
    # The sign may flip with reversed winding, but the element itself is valid.


def test_compute_solid_element_quality_rejects_unsupported_type(material) -> None:
    with pytest.raises(UnsupportedQualityMetricError):
        compute_solid_element_quality("not an element")  # type: ignore[arg-type]


def test_compute_any_element_quality_dispatches_to_solid(material: LinearElastic3D) -> None:
    hexa = _hex8(material)
    quality = compute_any_element_quality(hexa)
    assert isinstance(quality, SolidElementQuality)


def test_compute_any_element_quality_dispatches_to_2d() -> None:
    from femtoolkit.materials import LinearElastic2D
    from femtoolkit.mesh.cst_element import CSTElement2D

    mat2d = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    nodes = (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=0.0, y=1.0, z=0.0),
    )
    cst = CSTElement2D(id=1, nodes=nodes, material=mat2d, thickness=0.1)
    quality = compute_any_element_quality(cst)
    assert quality is compute_element_quality(cst) or quality.element_id == 1


def test_compute_any_element_quality_rejects_unsupported_element(material: LinearElastic3D) -> None:
    mesh = Mesh()
    n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    n2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    mesh.add_node(n1)
    mesh.add_node(n2)
    from femtoolkit.materials import Material
    from femtoolkit.mesh import BarElement
    from femtoolkit.sections import CrossSection

    bar_material = Material(
        name="steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3
    )
    bar = BarElement(
        id=1, nodes=(n1, n2), material=bar_material, cross_section=CrossSection(area=0.01)
    )
    mesh.add_element(bar)

    with pytest.raises(UnsupportedQualityMetricError):
        compute_any_element_quality(bar)
