"""Example: the Engineering Simulation Workspace service layer, Version 24.

Demonstrates the exact GUI workflow this version implements --
Project -> Material -> Mesh -> Boundary Conditions -> Loads -> Solver ->
Run -> Results -> Visualization -- driven directly through
:mod:`femtoolkit.application`, the same service layer
:mod:`femtoolkit.gui` calls. No Streamlit is involved: this script is
the concrete proof that the application layer is fully usable on its
own, exactly as spec section 18 requires ("a user should still be able
to run pytest [or a plain script] without requiring the GUI to be
launched").

The problem solved is the toolkit's familiar cantilever beam benchmark
(the same 2m x 0.4m Q4 mesh used throughout the Version 22 examples):
fixed at the left edge, a downward tip load at the right edge.
"""

from pathlib import Path

from femtoolkit.application import (
    BoundaryConditionConfig,
    LoadConfig,
    ModelService,
    ProjectService,
    ResultsService,
    SimulationService,
    get_material_preset,
    validate_project,
)
from femtoolkit.postprocessing.visualization import plot_deformed_shape_2d, plot_element_scatter_2d

OUTPUT_STRESS = Path(__file__).parent / "cantilever_beam_stress.png"
OUTPUT_DEFORMED = Path(__file__).parent / "cantilever_beam_deformed.png"


def main() -> None:
    """Run the full Project -> ... -> Visualization GUI workflow as a plain script."""
    print("Finite Element Toolkit")
    print("Version 24 -- Engineering Simulation Workspace Demo")
    print("=" * 52)

    # 1. Create Project
    project_service = ProjectService()
    project = project_service.create_project("Cantilever Beam", "linear_static")
    print(f"\n1. Created project '{project.name}' ({project.analysis_type}).")

    # 2. Select Material
    project.material = get_material_preset("structural_steel")
    print(
        f"2. Selected material: {project.material.name} "
        f"(E={project.material.youngs_modulus:.3e} Pa)"
    )

    # 3. Inspect Mesh
    project.mesh.width = 2.0
    project.mesh.height = 0.4
    project.mesh.nx = 16
    project.mesh.ny = 4
    project.mesh.thickness = 0.02
    model_service = ModelService()
    mesh = model_service.build_mesh(project)
    summary = model_service.mesh_summary(mesh)
    print(
        f"3. Mesh: {summary.element_type}, {summary.num_nodes} nodes, "
        f"{summary.num_elements} elements, quality={summary.quality.average_quality:.3f}"
    )

    # 4. Apply Fixed Boundary
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    print("4. Applied fixed boundary condition on the 'left' region (ux=uy=0).")

    # 5. Apply Load
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-2000.0)]
    print("5. Applied a -2000 N nodal force (Y) on every node of the 'right' region.")

    # 6. Validate + Run Linear Analysis
    validation = validate_project(project)
    print(f"\n6. Validation: {'PASSED' if validation.is_valid else 'FAILED'}")
    if not validation.is_valid:
        for error in validation.errors:
            print(f"   - {error}")
        return

    run_result = SimulationService().run(project)
    print(f"   Simulation status: {run_result.status}")
    if not run_result.succeeded:
        for error in run_result.errors:
            print(f"   - {error}")
        return

    # 7. View Displacement / Stress
    simulation = run_result.simulation
    results_service = ResultsService()
    summary = results_service.summary(simulation)
    print(f"\n7. Maximum displacement: {summary.maximum_displacement:.6e} m")
    print(f"   Maximum von Mises stress: {summary.maximum_von_mises_stress:.6e} Pa")

    stress_figure = plot_element_scatter_2d(simulation, "von_mises_stress")
    stress_figure.savefig(OUTPUT_STRESS, dpi=150)
    print(f"   Saved von Mises stress plot to {OUTPUT_STRESS}")

    # 8. Inspect Deformed Shape
    deformed_figure = plot_deformed_shape_2d(
        simulation, scale=300.0, color_field="von_mises_stress"
    )
    deformed_figure.savefig(OUTPUT_DEFORMED, dpi=150)
    print(f"8. Saved deformed shape plot to {OUTPUT_DEFORMED}")

    # Save the finished project configuration for reuse.
    project_path = Path(__file__).parent / "cantilever_beam_project.json"
    project_service.save(project, project_path)
    print(f"\nSaved project configuration to {project_path}")


if __name__ == "__main__":
    main()
