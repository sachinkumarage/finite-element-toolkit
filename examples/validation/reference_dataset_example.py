"""Example: comparing FEA results to a reference dataset (Version 29).

**Validation is not verification.** This example does not claim
experimental validation -- no experimental data ships with this
toolkit, and none is fabricated here. Instead, it demonstrates the
*infrastructure* (:mod:`femtoolkit.validation`) with a dataset honestly
labeled ``source="self-generated: ..."``: values computed from the same
closed-form Euler-Bernoulli beam formula used in
``examples/verification/beam_verification.py``, sampled at several
points along the beam. A real engineering use of this module would
instead load a dataset from a trusted external source (published
handbook data, an experimental report, a prior high-fidelity
simulation) via :func:`~femtoolkit.validation.load_reference_dataset_csv`/
:func:`~femtoolkit.validation.load_reference_dataset_json`.
"""

from __future__ import annotations

import numpy as np

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import Material
from femtoolkit.mesh import FrameElement2D, Mesh, Node
from femtoolkit.sections import CrossSection
from femtoolkit.validation import ReferenceDataset, compare_to_reference_dataset
from femtoolkit.verification.plots import plot_numerical_vs_reference

X, Y, RZ = TranslationDOF.X, TranslationDOF.Y, 2

_LOAD = 1000.0
_YOUNGS_MODULUS = 200e9
_MOMENT_OF_INERTIA = 8.333e-6
_LENGTH = 2.0
_NUM_ELEMENTS = 8


def _analytical_deflection(x: float) -> float:
    """Euler-Bernoulli cantilever deflection at position x (0 <= x <= L)."""
    return -(_LOAD * x**2 * (3 * _LENGTH - x)) / (6 * _YOUNGS_MODULUS * _MOMENT_OF_INERTIA)


def _solve_beam() -> tuple[list[float], np.ndarray]:
    material = Material(
        name="Steel", density=7850.0, youngs_modulus=_YOUNGS_MODULUS, poissons_ratio=0.3
    )
    section = CrossSection(area=0.01, second_moment_of_area=_MOMENT_OF_INERTIA)
    x_coords = np.linspace(0.0, _LENGTH, _NUM_ELEMENTS + 1)
    nodes = [Node(id=i + 1, x=float(x), y=0.0, z=0.0) for i, x in enumerate(x_coords)]

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    for i in range(_NUM_ELEMENTS):
        mesh.add_element(
            FrameElement2D(
                id=i + 1, nodes=(nodes[i], nodes[i + 1]), material=material, cross_section=section
            )
        )

    analysis = StaticLinearAnalysis(mesh)
    for dof in (X, Y, RZ):
        analysis.add_boundary_condition(BoundaryCondition(nodes[0].id, dof, 0.0))
    analysis.add_load(NodalLoad(nodes[-1].id, Y, -_LOAD))
    result = analysis.solve()

    deflections = [result.displacement(node.id, dof=Y) for node in nodes]
    return list(x_coords), np.array(deflections)


def main() -> None:
    """Compare FEA cantilever deflections against a self-generated reference dataset."""
    print("Finite Element Toolkit")
    print("Version 29 -- Reference Dataset Comparison")
    print("=" * 44)

    x_coords, fea_deflections = _solve_beam()
    reference_deflections = np.array([_analytical_deflection(x) for x in x_coords])

    dataset = ReferenceDataset(
        name="Cantilever deflection profile",
        source="self-generated: closed-form Euler-Bernoulli solution (not experimental data)",
        quantity="Deflection",
        units="m",
        independent_variable=np.array(x_coords),
        independent_variable_label="x (m)",
        values=reference_deflections,
        description="Deflection at each node position, computed from beam theory for comparison.",
    )

    result = compare_to_reference_dataset(dataset, fea_deflections)
    print(f"\n{result.message}")
    print(f"Dataset source: {result.dataset_source}")

    figure = plot_numerical_vs_reference(
        np.array(x_coords),
        fea_deflections,
        reference_deflections,
        "Deflection (m)",
        x_label="x (m)",
    )
    figure.savefig("examples/validation/reference_dataset_comparison.png", dpi=100)
    print("Saved reference_dataset_comparison.png")


if __name__ == "__main__":
    main()
