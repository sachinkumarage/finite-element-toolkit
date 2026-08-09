"""Validation Case 3b: compression.

A fixed-free bar pushed inward (negative end load) must show negative
strain, stress, and axial force -- consistent shortening under
compression.
"""

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis
from femtoolkit.materials import Material
from femtoolkit.mesh import BarElement, Mesh, Node
from femtoolkit.sections import CrossSection


def test_compression_produces_negative_strain_stress_and_force() -> None:
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
    analysis.add_load(NodalLoad(node_id=2, dof=0, value=-1000.0))  # pushing inward
    result = analysis.solve()

    assert result.displacement(2) < 0.0
    assert result.element_strain(1) < 0.0
    assert result.element_stress(1) < 0.0
    assert result.element_axial_force(1) < 0.0
