"""Validation Case 3a: tension.

A fixed-free bar pulled outward (positive end load) must show positive
strain, stress, and axial force -- consistent elongation under tension.
"""

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis
from femtoolkit.materials import Material
from femtoolkit.mesh import BarElement, Mesh, Node
from femtoolkit.sections import CrossSection


def test_tension_produces_positive_strain_stress_and_force() -> None:
    steel = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    section = CrossSection(area=0.01)
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)

    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_element(
        BarElement(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)
    )

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=0, value=0.0))
    analysis.add_load(NodalLoad(node_id=2, dof=0, value=1000.0))  # pulling outward
    result = analysis.solve()

    assert result.displacement(2) > 0.0
    assert result.element_strain(1) > 0.0
    assert result.element_stress(1) > 0.0
    assert result.element_axial_force(1) > 0.0
