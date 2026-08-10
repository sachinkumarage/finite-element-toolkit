"""Tests for the TrussElement2D domain object."""

import math

import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.element import StructuralElement
from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import Material
from femtoolkit.mesh import Node, TrussElement2D
from femtoolkit.sections import CrossSection


@pytest.fixture
def steel() -> Material:
    return Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)


@pytest.fixture
def section() -> CrossSection:
    return CrossSection(area=0.01)


def test_truss_element_satisfies_structural_element_protocol(
    steel: Material, section: CrossSection
) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)
    element = TrussElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)

    assert isinstance(element, StructuralElement)
    assert element.dofs_per_node == 2


def test_truss_element_node_and_material_association(
    steel: Material, section: CrossSection
) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)
    element = TrussElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)

    assert element.nodes == (node_1, node_2)
    assert element.material is steel
    assert element.cross_section is section


def test_horizontal_element_length_and_direction_cosines(
    steel: Material, section: CrossSection
) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)
    element = TrussElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)

    assert_allclose(element.length, 2.0)
    c, s = element.direction_cosines
    assert_allclose(c, 1.0)
    assert_allclose(s, 0.0, atol=1e-12)


def test_vertical_element_length_and_direction_cosines(
    steel: Material, section: CrossSection
) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=0.0, y=2.0, z=0.0)
    element = TrussElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)

    assert_allclose(element.length, 2.0)
    c, s = element.direction_cosines
    assert_allclose(c, 0.0, atol=1e-12)
    assert_allclose(s, 1.0)


def test_diagonal_element_length_and_direction_cosines(
    steel: Material, section: CrossSection
) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=1.0, z=0.0)
    element = TrussElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)

    assert_allclose(element.length, math.sqrt(2))
    c, s = element.direction_cosines
    assert_allclose(c, math.sqrt(2) / 2)
    assert_allclose(s, math.sqrt(2) / 2)


def test_zero_length_element_raises(steel: Material, section: CrossSection) -> None:
    node_1 = Node(id=1, x=1.0, y=1.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=1.0, z=0.0)

    with pytest.raises(ValidationError):
        TrussElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)


def test_stiffness_matrix_shape_and_symmetry(steel: Material, section: CrossSection) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=3.0, y=4.0, z=0.0)
    element = TrussElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)

    stiffness = element.stiffness_matrix
    assert stiffness.shape == (4, 4)
    assert_allclose(stiffness, stiffness.T)


def test_stiffness_matrix_values_match_direct_formula(
    steel: Material, section: CrossSection
) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=1.0, z=0.0)
    element = TrussElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)

    from femtoolkit.analysis.stiffness import truss_element_stiffness_2d

    expected = truss_element_stiffness_2d(
        youngs_modulus=steel.youngs_modulus,
        area=section.area,
        length=element.length,
        cos_theta=math.sqrt(2) / 2,
        sin_theta=math.sqrt(2) / 2,
    )
    assert_allclose(element.stiffness_matrix, expected)


def test_dof_keys_order(steel: Material, section: CrossSection) -> None:
    node_1 = Node(id=5, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=9, x=2.0, y=0.0, z=0.0)
    element = TrussElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)

    from femtoolkit.analysis.dof import TranslationDOF

    assert element.dof_keys() == (
        (5, TranslationDOF.X),
        (5, TranslationDOF.Y),
        (9, TranslationDOF.X),
        (9, TranslationDOF.Y),
    )


def test_strain_projects_displacement_onto_local_axis(
    steel: Material, section: CrossSection
) -> None:
    """For a horizontal element, only the X displacements matter."""
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)
    element = TrussElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)

    strain = element.strain(ux1=0.0, uy1=0.0, ux2=0.002, uy2=999.0)  # uy2 must be ignored (c=1,s=0)
    assert_allclose(strain, 0.001)


def test_strain_for_diagonal_element(steel: Material, section: CrossSection) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=1.0, z=0.0)
    element = TrussElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)

    c = s = math.sqrt(2) / 2
    ux2, uy2 = 0.001, 0.001
    expected_strain = (c * ux2 + s * uy2) / element.length
    strain = element.strain(ux1=0.0, uy1=0.0, ux2=ux2, uy2=uy2)
    assert_allclose(strain, expected_strain)


def test_stress_and_axial_force(steel: Material, section: CrossSection) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)
    element = TrussElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)

    strain = element.strain(ux1=0.0, uy1=0.0, ux2=0.002, uy2=0.0)
    expected_stress = steel.youngs_modulus * strain
    expected_force = expected_stress * section.area

    assert_allclose(element.stress(ux1=0.0, uy1=0.0, ux2=0.002, uy2=0.0), expected_stress)
    assert_allclose(element.axial_force(ux1=0.0, uy1=0.0, ux2=0.002, uy2=0.0), expected_force)


def test_tension_and_compression_signs(steel: Material, section: CrossSection) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)
    element = TrussElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)

    assert element.axial_force(ux1=0.0, uy1=0.0, ux2=0.001, uy2=0.0) > 0  # stretched: tension
    assert element.axial_force(ux1=0.0, uy1=0.0, ux2=-0.001, uy2=0.0) < 0  # shortened: compression


def test_from_dofs_wrappers_match_direct_methods(steel: Material, section: CrossSection) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=1.0, z=0.0)
    element = TrussElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)

    displacements = [0.0, 0.0, 0.001, 0.0005]
    assert_allclose(element.strain_from_dofs(displacements), element.strain(*displacements))
    assert_allclose(element.stress_from_dofs(displacements), element.stress(*displacements))
    assert_allclose(
        element.axial_force_from_dofs(displacements), element.axial_force(*displacements)
    )


def test_invalid_id_raises(steel: Material, section: CrossSection) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)

    with pytest.raises(ValidationError):
        TrussElement2D(id=0, nodes=(node_1, node_2), material=steel, cross_section=section)


def test_duplicate_node_raises(steel: Material, section: CrossSection) -> None:
    node = Node(id=1, x=0.0, y=0.0, z=0.0)

    with pytest.raises(ValidationError):
        TrussElement2D(id=1, nodes=(node, node), material=steel, cross_section=section)


def test_missing_material_raises(section: CrossSection) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)

    with pytest.raises(ValidationError):
        TrussElement2D(id=1, nodes=(node_1, node_2), material=None, cross_section=section)  # type: ignore[arg-type]


def test_missing_cross_section_raises(steel: Material) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)

    with pytest.raises(ValidationError):
        TrussElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=None)  # type: ignore[arg-type]
