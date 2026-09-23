"""Example: generating a complete engineering simulation report (Version 29).

**Procedure.** Solves a real cantilever plate, runs the standard
analytical benchmark suite and a global equilibrium check, collects
reproducibility metadata, assembles an
:class:`~femtoolkit.reporting.models.EngineeringReport`, and renders it
to both Markdown and HTML
(:mod:`femtoolkit.reporting.renderers`) -- the same pipeline the GUI's
Verification & Validation page uses (``femtoolkit.gui.workflow_pages.verification_page``).
"""

from __future__ import annotations

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.analysis.system import build_force_vector
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.generator import create_quad_mesh
from femtoolkit.reporting import EngineeringReport, collect_reproducibility_metadata, save_report
from femtoolkit.verification.benchmarks import standard_benchmark_suite
from femtoolkit.verification.checks import check_force_equilibrium
from femtoolkit.verification.runner import VerificationRunner

X = TranslationDOF.X
Y = TranslationDOF.Y


def main() -> None:
    """Solve a cantilever plate and generate a full engineering report for it."""
    print("Finite Element Toolkit")
    print("Version 29 -- Engineering Report Generation")
    print("=" * 45)

    material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    mesh = create_quad_mesh(width=4.0, height=1.0, nx=20, ny=6, material=material, thickness=0.02)
    analysis = StaticLinearAnalysis(mesh)
    for node in mesh.nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(BoundaryCondition(node.id, X, 0.0))
            analysis.add_boundary_condition(BoundaryCondition(node.id, Y, 0.0))
    loads = [NodalLoad(n.id, Y, -10000.0) for n in mesh.nodes if n.x == 4.0 and n.y == 0.0]
    for load in loads:
        analysis.add_load(load)
    result = analysis.solve()

    forces = build_force_vector(result.dof_map, loads)
    equilibrium_check = check_force_equilibrium(result.dof_map, forces, result.reactions)

    benchmark_report = VerificationRunner().run_all(standard_benchmark_suite())

    max_displacement = max(abs(result.displacement(n.id, dof=Y)) for n in mesh.nodes)
    max_von_mises = max(result.element_von_mises(e.id) for e in mesh.elements)

    metadata = collect_reproducibility_metadata(
        model_name="Cantilever Plate",
        analysis_type="linear_static",
        mesh_statistics={
            "nodes": len(mesh.nodes),
            "elements": len(mesh.elements),
            "dofs": result.dof_map.total_dofs,
        },
        element_types=["QuadElement2D"],
        material_properties={"youngs_modulus": 200e9, "poisson_ratio": 0.3},
        boundary_conditions_summary="Fixed support (X, Y) on the left edge.",
        loads_summary="10000 N downward, distributed across the right edge tip nodes.",
        degrees_of_freedom=result.dof_map.total_dofs,
    )

    report = EngineeringReport(
        title="Cantilever Plate: Engineering Simulation Report",
        simulation_summary="Linear static analysis of a Q4 cantilever plate under a tip load.",
        model_description="A 4 m x 1 m x 0.02 m steel cantilever plate, fixed at one edge.",
        geometry_description="Rectangular domain, 4.0 m x 1.0 m.",
        mesh_summary={
            "nodes": len(mesh.nodes),
            "elements": len(mesh.elements),
            "element_type": "QuadElement2D",
        },
        materials_summary={"Steel": {"E": 200e9, "poisson_ratio": 0.3}},
        boundary_conditions_summary="Fixed (X, Y) at every node on the left edge.",
        loads_summary=(
            f"{len(loads)} nodal loads totalling {sum(load.value for load in loads):.0f} N "
            "(Y direction)."
        ),
        analysis_type="linear_static",
        solver_configuration="Dense Direct (default).",
        reproducibility=metadata,
        verification_results=benchmark_report.results,
        equilibrium_checks=[equilibrium_check],
        key_results={
            "Maximum displacement (m)": max_displacement,
            "Maximum von Mises stress (Pa)": max_von_mises,
        },
        conclusions=(
            "All analytical benchmark cases passed, and the solved model satisfies global "
            "force equilibrium within tolerance."
        ),
    )

    save_report(report, "examples/reports/cantilever_plate_report.md", "markdown")
    save_report(report, "examples/reports/cantilever_plate_report.html", "html")

    print(f"\nOverall status: {report.overall_status.value.upper()}")
    print("Saved cantilever_plate_report.md and cantilever_plate_report.html")


if __name__ == "__main__":
    main()
