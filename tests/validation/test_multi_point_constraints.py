"""Validation: a multi-point constraint reproduces a single continuous bar.

Two independently defined bar chains -- node 1 (fixed) -- bar -- node 2,
and node 3 (a physically coincident but topologically separate node) --
bar -- node 4 (loaded) -- are tied together with
``MultiPointConstraint(node_id_a=2, node_id_b=3, dof=X)``. If the tie is
enforced correctly, the two chains behave as one continuous bar of the
combined length, with a known closed-form analytical solution:

.. code-block:: text

    u(x) = F * x / (E * A)      for 0 <= x <= L1 + L2
    R(node_1) = -F

This is the standard analytical validation for a bar of length
``L1 + L2`` fixed at one end and loaded with ``F`` at the other -- an
independent check on the multi-point-constraint penalty method that does
not depend on the CST/Q4 continuum elements at all.
"""

import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import (
    BoundaryCondition,
    MultiPointConstraint,
    NodalLoad,
    StaticLinearAnalysis,
    TranslationDOF,
)
from femtoolkit.materials import Material
from femtoolkit.mesh import BarElement, Mesh, Node
from femtoolkit.sections import CrossSection

YOUNGS_MODULUS = 200e9
AREA = 0.01
LENGTH_1 = 1.0
LENGTH_2 = 1.0
FORCE = 10_000.0


@pytest.fixture
def result():
    steel = Material(
        name="Steel", density=7850.0, youngs_modulus=YOUNGS_MODULUS, poissons_ratio=0.3
    )
    section = CrossSection(area=AREA)

    mesh = Mesh()
    mesh.add_node(Node(1, 0.0, 0.0, 0.0))
    mesh.add_node(Node(2, LENGTH_1, 0.0, 0.0))
    mesh.add_node(Node(3, LENGTH_1, 0.0, 0.0))  # coincides with node 2, but topologically separate
    mesh.add_node(Node(4, LENGTH_1 + LENGTH_2, 0.0, 0.0))
    mesh.add_element(BarElement(id=1, nodes=(mesh.get_node(1), mesh.get_node(2)),
                                  material=steel, cross_section=section))
    mesh.add_element(BarElement(id=2, nodes=(mesh.get_node(3), mesh.get_node(4)),
                                  material=steel, cross_section=section))

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(1, TranslationDOF.X, 0.0))
    analysis.add_load(NodalLoad(4, TranslationDOF.X, FORCE))
    analysis.add_multi_point_constraint(
        MultiPointConstraint(node_id_a=2, node_id_b=3, dof=TranslationDOF.X)
    )
    return analysis.solve()


def test_tied_midpoint_displacement_matches_analytical(result) -> None:
    expected = FORCE * LENGTH_1 / (YOUNGS_MODULUS * AREA)
    assert_allclose(result.displacement(2, TranslationDOF.X), expected, rtol=1e-4)
    assert_allclose(result.displacement(3, TranslationDOF.X), expected, rtol=1e-4)


def test_tip_displacement_matches_analytical(result) -> None:
    expected = FORCE * (LENGTH_1 + LENGTH_2) / (YOUNGS_MODULUS * AREA)
    assert_allclose(result.displacement(4, TranslationDOF.X), expected, rtol=1e-4)


def test_fixed_end_reaction_balances_applied_force(result) -> None:
    assert_allclose(result.reaction(1, TranslationDOF.X), -FORCE, rtol=1e-6)


def test_constrained_nodes_agree_to_a_close_tolerance(result) -> None:
    """The penalty method is approximate: node 2 and node 3 (tied to have
    equal ux) will not match to bit-for-bit precision, only to a very
    close tolerance controlled by the constraint's penalty stiffness.
    """
    ux_2 = result.displacement(2, TranslationDOF.X)
    ux_3 = result.displacement(3, TranslationDOF.X)
    assert_allclose(ux_2, ux_3, rtol=1e-4)
