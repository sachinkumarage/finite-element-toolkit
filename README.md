# Finite Element Toolkit

An open-source Python toolkit for developing finite element analysis (FEA) capabilities, built incrementally as a series of versioned milestones.

**This is the Version 8 release.** Version 1 established the project's architecture and core domain model. Version 2 added the basic mathematical foundation for FEA. Version 3 turned that into a validated **1D structural analysis** capability (a bar element, `StaticLinearAnalysis`, and results). Version 4 extended the same analysis workflow to **2D truss structures**: two translational DOFs per node, a `TrussElement2D` transformed from local to global coordinates via its direction cosines, and X/Y loads, constraints, displacements, reactions, and member forces. Version 5 added **2D Euler-Bernoulli beam and frame analysis**: a rotational DOF per node, a `FrameElement2D` that resists axial force, shear force, and bending moment, and per-element shear/moment/bending-stress results. Version 6 introduced the toolkit's first true **2D continuum element**: a `CSTElement2D` (3-node constant strain triangle) representing a finite *area* of material rather than a line member, with plane stress/strain constitutive models, a strain-displacement (`B`) matrix, and von Mises/principal stress recovery. Version 7 added a second continuum element, `QuadElement2D` (4-node bilinear quadrilateral, "Q4"): natural coordinates, isoparametric mapping, the Jacobian, and 2x2 Gauss quadrature, needed because -- unlike the CST element -- a Q4 element's strain-displacement matrix has no closed form and varies within the element. Version 8 adds **automatic structured 2D mesh generation**: `create_quad_mesh`/`create_triangular_mesh` turn a rectangular domain and a subdivision count into a fully connected, correctly oriented mesh, plus whole-mesh validation, shape-quality metrics, and a JSON export/import foundation. It does **not** yet contain unstructured or CAD-driven meshing, adaptive refinement, higher-order continuum elements, 3D beams, Timoshenko beams, plate, shell, or 3D solid elements, or nonlinear/dynamic analysis, or visualization.

## Current Features

**Version 1 — domain model**

- **Material** — data model for isotropic material properties (density, Young's modulus, Poisson's ratio)
- **Node** — a point in 3D space, identified by ID and coordinates
- **Element** — connects nodes through a material, with no computation performed
- **Mesh** — a container managing nodes and elements with referential-integrity validation
- **Engineering units foundation** — named SI unit constants
- **Custom exceptions** — domain-specific error types
- **Logging** — a package logger that stays silent unless the host application configures it

**Version 2 — basic FEA mathematics**

- **Degrees of freedom** — `TranslationDOF` and `DOFMap`, mapping `(node_id, dof)` pairs to global DOF indices
- **Boundary conditions** — `BoundaryCondition`, a prescribed-displacement constraint on one DOF
- **Nodal loads** — `NodalLoad`, an externally applied force on one DOF
- **1D bar element stiffness matrix** — `bar_element_stiffness(E, A, L)`
- **Global stiffness matrix assembly** — `assemble_global_stiffness`, mapping element DOFs to global DOFs
- **Linear system representation** — `LinearSystem`, separating `[K]`, `{F}`, and boundary conditions from the numerical solve
- **Basic linear solver** — `solve`, using `numpy.linalg.solve` on the reduced free-DOF system

**Version 3 — 1D structural analysis**

- **Cross-section** — `CrossSection`, the cross-sectional area of a bar
- **Bar element** — `BarElement`, a dedicated two-node axial element computing its own length, stiffness matrix, strain, stress, and axial force
- **Static linear analysis** — `StaticLinearAnalysis`, orchestrating DOF mapping, assembly, and solving for a mesh of bar elements
- **Analysis results** — `AnalysisResult`, exposing nodal displacement, reaction force, and per-element strain/stress/axial force
- **Domain-specific errors** — `InvalidAnalysisError`, `InvalidElementError`, `InsufficientConstraintsError`, `SingularSystemError`

**Version 4 — 2D truss analysis**

- **2D truss element** — `TrussElement2D`, a two-node, two-DOF-per-node axial element computing its own length, direction cosines, stiffness matrix, strain, stress, and axial force
- **Element interface** — `StructuralElement`, a lightweight protocol letting `StaticLinearAnalysis` and `AnalysisResult` work with any element type (bar or truss) without importing it by name
- **Reusable numerical infrastructure** — assembly, the linear system, and the solver were generalized in Version 4 to support any number of DOFs per node, so `BarElement` and `TrussElement2D` share the exact same solve path
- **2D results** — `result.node_displacement(node_id)` and `result.node_reaction(node_id)` return `(x, y)` tuples; `displacement()`/`reaction()` gained an optional `dof` argument
- **Tests** — a pytest suite covering all four versions, including a validation suite (`tests/validation/`) with a horizontal-truss backward check against the Version 3 bar formula, a triangular truss solved independently by the method of joints, a symmetric-truss consistency check, and a multi-material truss

**Version 5 — 2D beam and frame analysis**

- **Rotational DOF** — `RotationDOF.RZ`, activated alongside `TranslationDOF.X`/`Y` by frame elements (`dofs_per_node=3`); `DOFMap` already supported three DOFs per node since Version 2
- **Cross-section bending properties** — `CrossSection` gained optional `second_moment_of_area` (required for a frame element's bending stiffness) and `extreme_fiber_distance` (optional, only used by bending-stress post-processing)
- **2D frame element** — `FrameElement2D`, a two-node, three-DOF-per-node element combining a bar element's axial stiffness (`EA/L`) with Euler-Bernoulli bending stiffness (`EI`), transformed from local to global coordinates via a 6x6 transformation matrix
- **Frame stiffness math** — `frame_element_stiffness_local`, `frame_transformation_matrix_2d`, and `frame_element_stiffness_2d` (`Kg = Tᵀ·Kl·T`) in `femtoolkit.analysis`
- **Frame results** — `element_end_forces`, `element_shear_force`, `element_bending_moment`, and `element_bending_stress` on `AnalysisResult`, gated to elements satisfying the new `FrameStructuralElement` protocol; `node_displacement`/`node_reaction` generalized to return one component per active DOF (`(ux, uy, rz)` for a frame analysis, unchanged `(ux, uy)` for a truss analysis)
- **Backward compatible** — `BarElement` and `TrussElement2D` are untouched; assembly, the linear system, and the solver required no frame-specific changes at all, since they were already DOF-count-agnostic since Version 4
- **Engineering validation** — five validation cases plus a portal frame in `tests/validation/`, each checked against a classical Euler-Bernoulli analytical solution or an independently derived equilibrium/consistency invariant (see [Version 5](#version-5) below)

**Version 6 — 2D continuum elements**

- **Continuum math foundation** (`femtoolkit.continuum`) — independently testable pure functions for triangle geometry, linear shape functions, the strain-displacement (`B`) matrix, plane stress/strain constitutive (`D`) matrices, and stress recovery (including von Mises and principal stress), kept separate from the element class that coordinates them
- **2D linear elastic material** — `LinearElastic2D`, a dedicated constitutive model (Young's modulus, Poisson's ratio, and a `"plane_stress"`/`"plane_strain"` formulation) distinct from the structural-element `Material`
- **CST element** — `CSTElement2D`, a three-node, two-DOF-per-node continuum element (`ux`, `uy` only — no rotational DOF) with a constant strain-displacement matrix and stiffness `Ke = t*A*Bᵀ*D*B`; either clockwise or counter-clockwise node order is accepted and gives identical results
- **Element interface split** — `AssemblableElement` (the minimal interface `StaticLinearAnalysis` needs: DOF mapping and a stiffness matrix) is now the base protocol; `StructuralElement` (scalar axial results) and `ContinuumElement` (vector strain/stress, von Mises, principal stress) are two independent extensions of it, so a continuum element and a structural member can both be assembled and solved through the exact same code path
- **Continuum results** — `element_strain`/`element_stress` are reused across element types (returning a scalar for a structural member, a 3-component array `[x, y, xy]` for a continuum element); `element_von_mises` and `element_principal_stresses` are new, gated to continuum elements
- **New exception** — `DegenerateElementError`, raised for collinear, nearly collinear, or duplicate-coordinate triangle nodes
- **Backward compatible** — `BarElement`, `TrussElement2D`, and `FrameElement2D` are untouched; assembly, the linear system, and the solver required no continuum-specific changes
- **Engineering validation** — a patch test (exact constant-strain reproduction across three differently shaped triangles), an independently hand-derived plane-stress/plane-strain constitutive matrix and single-triangle stiffness matrix, a boundary-artifact-free uniaxial-stress recovery check, and a two-triangle assembly/equilibrium/continuity check (see [Version 6](#version-6) below)

**Version 7 — 2D quadrilateral elements**

- **Natural coordinates and bilinear shape functions** — `quad_shape_functions(xi, eta)` / `quad_shape_function_derivatives(xi, eta)`, defined on `[-1,1] x [-1,1]`, extending `femtoolkit.continuum.shape_functions`
- **Isoparametric mapping and the Jacobian** (`femtoolkit.continuum.jacobian`) — `jacobian_matrix`, `jacobian_determinant`, `inverse_jacobian`, and `physical_shape_function_derivatives`, converting natural-coordinate shape function derivatives to physical (`x`, `y`) ones
- **2x2 Gauss quadrature** (`femtoolkit.continuum.gauss`) — the four points and unit weights needed because a Q4 element's stiffness integral has no closed form (unlike CST's)
- **Q4 element** — `QuadElement2D`, a four-node, two-DOF-per-node continuum element (`ux`, `uy` only) with an 8x8 stiffness matrix assembled by numerically integrating `Ke = integral(t*Bᵀ*D*B) dA` over the four Gauss points; node order is fixed counter-clockwise (unlike CST, a general quadrilateral has no single "signed area" to normalize clockwise input against)
- **Representative strain/stress** — since a Q4 element's strain varies within the element (unlike CST's constant strain), `element_strain`/`element_stress`/`element_von_mises`/`element_principal_stresses` report the value at the element's natural-coordinate center
- **Zero wiring changes** — `QuadElement2D` satisfies the exact same `AssemblableElement`/`ContinuumElement` protocols introduced in Version 6, so `StaticLinearAnalysis`, `AnalysisResult`, assembly, and the solver required **no** Version 7-specific code at all
- **Backward compatible** — `BarElement`, `TrussElement2D`, `FrameElement2D`, and `CSTElement2D` are untouched
- **Engineering validation** — a patch test (exact constant-strain reproduction across three differently shaped quadrilaterals, proving a linear field is an exact special case of the bilinear interpolation), a single-element uniaxial-tension case cross-checked against a from-scratch independent stiffness re-derivation, and a two-element assembly/equilibrium/continuity check (see [Version 7](#version-7) below)

**Version 8 — 2D mesh generation**

- **Structured mesh generators** (`femtoolkit.mesh.generator`) — `create_quad_mesh(width, height, nx, ny, material, thickness)` and `create_triangular_mesh(..., diagonal="forward"|"backward")` turn a rectangular domain into a fully connected `Mesh` of `QuadElement2D` or `CSTElement2D` elements, with deterministic row-major node numbering and left-to-right/bottom-to-top element numbering (both documented in the module)
- **Kept separate from `Mesh`** — generation logic lives in its own module rather than as `Mesh` methods, matching the existing separation between domain containers and the math/algorithms that populate them
- **Self-validating** — both generators call `validate_mesh` on their own output before returning, so they can never hand back a degenerate or inverted mesh
- **Mesh validation** (`femtoolkit.mesh.validation`) — `validate_mesh(mesh)` checks for duplicate node coordinates (a new `DuplicateNodeCoordinatesError`) and element/geometry integrity; duplicate IDs and missing node references were already caught by `Mesh.add_node`/`add_element`, and degenerate/inverted element geometry was already caught by `CSTElement2D`/`QuadElement2D` at construction -- Version 8 closes the one gap fail-fast construction cannot catch on its own
- **Mesh quality metrics** (`femtoolkit.mesh.quality`) — per-element area, min/max edge length, aspect ratio, equiangle skewness, and (for Q4) Jacobian determinant, via `Mesh.element_area()`/`Mesh.element_quality()`; a whole-mesh `Mesh.quality_summary()` aggregates node/element counts, area and edge-length ranges, and min/max/average quality
- **JSON mesh export/import** (`femtoolkit.mesh.serialization`) — `export_mesh`/`import_mesh` and the underlying `mesh_to_dict`/`mesh_from_dict`, storing each element's own material and thickness for a fully self-contained round trip
- **Mesh refinement through subdivision** — increasing `nx`/`ny` densifies a mesh over the same domain; no adaptive or error-based refinement (out of scope for this version)
- **Reuses the existing solver unchanged** — a generated mesh is passed straight into `StaticLinearAnalysis` exactly like a hand-built one; no Version 8-specific solver code exists (see [Version 8](#version-8) below)

## Installation

Clone the repository and install it in editable mode:

```bash
git clone <repository-url>
cd finite-element-toolkit
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```python
from femtoolkit.materials import Material
from femtoolkit.mesh import Element, Mesh, Node

steel = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)

node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
element = Element(id=1, nodes=[node_1, node_2], material=steel)

mesh = Mesh()
mesh.add_node(node_1)
mesh.add_node(node_2)
mesh.add_element(element)

print(mesh.get_node(1))
print(mesh.get_element(1))
```

Solving a minimal axial bar problem (see [Version 2](#version-2) below for the theory):

```python
from femtoolkit.analysis import (
    BoundaryCondition,
    DOFMap,
    ElementStiffnessContribution,
    LinearSystem,
    NodalLoad,
    assemble_global_stiffness,
    bar_element_stiffness,
    build_force_vector,
    solve,
)
from femtoolkit.materials import Material
from femtoolkit.mesh import Node

steel = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
node_1, node_2 = Node(id=1, x=0.0, y=0.0, z=0.0), Node(id=2, x=2.0, y=0.0, z=0.0)

dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
local_k = bar_element_stiffness(youngs_modulus=steel.youngs_modulus, area=0.01, length=2.0)
dof_keys = ((1, 0), (2, 0))  # (node_id, dof); dof 0 = X (axial direction)
global_k = assemble_global_stiffness(dof_map, [ElementStiffnessContribution(dof_keys, local_k)])

forces = build_force_vector(dof_map, [NodalLoad(node_id=2, dof=0, value=1000.0)])
system = LinearSystem(
    dof_map=dof_map,
    stiffness=global_k,
    forces=forces,
    boundary_conditions=[BoundaryCondition(node_id=1, dof=0, value=0.0)],
)

displacements = solve(system)  # displacements[1] == 1e-06 m
```

Solving the same problem with the Version 3 structural analysis workflow (see [Version 3](#version-3) below for the theory):

```python
from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis
from femtoolkit.materials import Material
from femtoolkit.mesh import BarElement, Mesh, Node
from femtoolkit.sections import CrossSection

steel = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
section = CrossSection(area=0.01)
node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)
bar = BarElement(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)

mesh = Mesh()
mesh.add_node(node_1)
mesh.add_node(node_2)
mesh.add_element(bar)

analysis = StaticLinearAnalysis(mesh)
analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=0, value=0.0))
analysis.add_load(NodalLoad(node_id=2, dof=0, value=1000.0))
result = analysis.solve()

print(result.displacement(2))  # 1e-06 m
print(result.reaction(1))  # -1000.0 N
print(result.element_stress(1))  # 100000.0 Pa
print(result.element_strain(1))  # 5e-07
print(result.element_axial_force(1))  # 1000.0 N
```

Solving a small 2D truss with Version 4 (see [Version 4](#version-4) below for the theory):

```python
from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import Material
from femtoolkit.mesh import Mesh, Node, TrussElement2D
from femtoolkit.sections import CrossSection

steel = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
section = CrossSection(area=0.01)
node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)
truss = TrussElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)

mesh = Mesh()
mesh.add_node(node_1)
mesh.add_node(node_2)
mesh.add_element(truss)

analysis = StaticLinearAnalysis(mesh)
analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.X, value=0.0))
analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.Y, value=0.0))
analysis.add_boundary_condition(BoundaryCondition(node_id=2, dof=TranslationDOF.Y, value=0.0))
analysis.add_load(NodalLoad(node_id=2, dof=TranslationDOF.X, value=1000.0))
result = analysis.solve()

print(result.node_displacement(2))  # (1e-06, 0.0) m -- matches the 1D bar solution
print(result.node_reaction(1))  # (-1000.0, 0.0) N
print(result.element_axial_force(1))  # 1000.0 N
```

Solving a cantilever beam with Version 5 (see [Version 5](#version-5) below for the theory):

```python
from femtoolkit.analysis import BoundaryCondition, NodalLoad, RotationDOF, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import Material
from femtoolkit.mesh import FrameElement2D, Mesh, Node
from femtoolkit.sections import CrossSection

steel = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
section = CrossSection(area=0.01, second_moment_of_area=8.333e-6)
node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)
beam = FrameElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)

mesh = Mesh()
mesh.add_node(node_1)
mesh.add_node(node_2)
mesh.add_element(beam)

analysis = StaticLinearAnalysis(mesh)
for dof in (TranslationDOF.X, TranslationDOF.Y, RotationDOF.RZ):
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=dof, value=0.0))
analysis.add_load(NodalLoad(node_id=2, dof=TranslationDOF.Y, value=-1000.0))
result = analysis.solve()

print(result.node_displacement(2))  # (0.0, -1.600064e-03, -1.200048e-03) -- (ux, uy, rz)
print(result.node_reaction(1))  # (0.0, 1000.0, 2000.0) -- (Rx, Ry, Mz)
print(result.element_bending_moment(1))  # 2000.0 N*m, the fixed-end moment
print(result.element_shear_force(1))  # 1000.0 N, the fixed-end shear
```

Solving a single CST continuum element with Version 6 (see [Version 6](#version-6) below for the theory):

```python
from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import CSTElement2D, Mesh, Node

material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
node_3 = Node(id=3, x=0.0, y=1.0, z=0.0)
triangle = CSTElement2D(id=1, nodes=(node_1, node_2, node_3), material=material, thickness=0.01)

mesh = Mesh()
mesh.add_node(node_1)
mesh.add_node(node_2)
mesh.add_node(node_3)
mesh.add_element(triangle)

analysis = StaticLinearAnalysis(mesh)
analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.X, value=0.0))
analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.Y, value=0.0))
analysis.add_boundary_condition(BoundaryCondition(node_id=3, dof=TranslationDOF.X, value=0.0))
analysis.add_load(NodalLoad(node_id=2, dof=TranslationDOF.X, value=5000.0))
result = analysis.solve()

print(result.element_strain(1))  # [5e-06, -1.5e-06, 0.0] -- [epsilon_x, epsilon_y, gamma_xy]
print(result.element_stress(1))  # [1e+06, ~0.0, ~0.0] Pa -- [sigma_x, sigma_y, tau_xy]
print(result.element_von_mises(1))  # 1e+06 Pa
```

Solving a single Q4 continuum element with Version 7 (see [Version 7](#version-7) below for the theory):

```python
from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, Node, QuadElement2D

material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
node_3 = Node(id=3, x=1.0, y=1.0, z=0.0)
node_4 = Node(id=4, x=0.0, y=1.0, z=0.0)
quad = QuadElement2D(id=1, nodes=(node_1, node_2, node_3, node_4), material=material, thickness=0.01)

mesh = Mesh()
mesh.add_node(node_1)
mesh.add_node(node_2)
mesh.add_node(node_3)
mesh.add_node(node_4)
mesh.add_element(quad)

analysis = StaticLinearAnalysis(mesh)
analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.X, value=0.0))
analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.Y, value=0.0))
analysis.add_boundary_condition(BoundaryCondition(node_id=4, dof=TranslationDOF.X, value=0.0))
analysis.add_load(NodalLoad(node_id=2, dof=TranslationDOF.X, value=5000.0))
result = analysis.solve()

print(result.element_strain(1))  # [5e-06, -1.5e-06, ~0.0] -- at the element's natural-coordinate center
print(result.element_stress(1))  # [1e+06, ~0.0, ~0.0] Pa
print(result.element_von_mises(1))  # 1e+06 Pa
```

Generating and solving a mesh automatically with Version 8 (see [Version 8](#version-8) below for the theory):

```python
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import create_quad_mesh

material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
mesh = create_quad_mesh(width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01)

print(len(mesh.nodes), len(mesh.elements))  # 15 8
print(sum(mesh.element_area(e.id) for e in mesh.elements))  # 2.0 -- matches width * height

summary = mesh.quality_summary()
print(summary.average_quality)  # 1.0 -- a regular grid of squares is ideal shape

# The generated mesh works with the existing solver unchanged:
from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF

analysis = StaticLinearAnalysis(mesh)
for node in mesh.nodes:
    if node.x == 0.0:
        analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.X, 0.0))
analysis.add_boundary_condition(BoundaryCondition(1, TranslationDOF.Y, 0.0))
for node in mesh.nodes:
    if node.x == 2.0:
        analysis.add_load(NodalLoad(node.id, TranslationDOF.X, 10000.0 / 3))
result = analysis.solve()
print(result.element_stress(1)[0])  # ~1e+06 Pa
```

## Version 2

Version 2 introduces the mathematical foundation that future versions will build a full solver on top of.

### Degrees of freedom

A **degree of freedom (DOF)** is an independent unknown at a node — for the 1D axial bar problem, each node has a single translational DOF: its axial displacement. `TranslationDOF` names the available directions (`X`, `Y`, `Z`); Version 2 only activates `X`. Rather than storing solver state on `Node` itself, a `DOFMap` externally assigns each `(node_id, dof)` pair a **global DOF index** — the row/column position it occupies in the global system of equations.

### Element stiffness matrix

An element's **stiffness matrix** relates its nodal displacements to the nodal forces required to produce them. For a two-node axial bar of Young's modulus `E`, cross-sectional area `A`, and length `L`:

```text
k = EA / L

[ k  -k ]
[ -k  k ]
```

This matrix exists because the bar behaves like a linear spring along its axis: stretching one end while holding the other fixed requires a force proportional to `k`.

### Global stiffness matrix and assembly

A structure is built from multiple elements sharing nodes. Each element only knows its own local stiffness matrix; **assembly** maps each element's local DOFs onto the structure's global DOFs and sums overlapping contributions into a single global stiffness matrix `[K]`. Without this step, elements could not "communicate" the forces they transmit through shared nodes.

### Boundary conditions

The global system `[K]{u} = {F}` cannot be solved as-is: an unconstrained structure is free to translate as a rigid body, which makes `[K]` singular. A **boundary condition** prescribes a known displacement for a DOF (most commonly `u = 0`, a fixed support), removing that rigid-body freedom so the system has a unique solution.

### Linear system

The structural problem reduces to:

```text
[K]{u} = {F}
```

where `[K]` is the global stiffness matrix, `{u}` is the vector of unknown nodal displacements, and `{F}` is the vector of applied nodal forces. `LinearSystem` holds `[K]`, `{F}`, and the boundary conditions together; `solve()` partitions the DOFs into free and constrained sets, solves the reduced free-free system with `numpy.linalg.solve`, and reassembles the full displacement vector.

### Analytical validation

For a fixed-free axial bar under a tip load, the exact solution is:

```text
u = F * L / (E * A)
```

The test suite checks the FEA displacement result against this formula (`tests/test_system.py`), verifying both the numerical implementation and its engineering correctness.

### Basic bar example

See [`examples/basic_bar_analysis.py`](examples/basic_bar_analysis.py) for the complete workflow: Material → Nodes → Element → element stiffness matrix → global stiffness matrix → boundary condition → force vector → linear system → displacement, validated against the analytical solution above.

## Version 3

Version 3 turns the Version 2 mathematics into a first useful engineering capability: linear static analysis of 1D axial bar structures.

**Engineering assumptions:** linear elastic material behavior (`sigma = E * epsilon`), small deformation (infinitesimal strain, unchanged geometry), static loading (time-invariant), purely axial (1D) bar behavior, and a consistent SI unit system throughout (length in m, area in m², force in N, stress and Young's modulus in Pa, density in kg/m³).

### Bar element

A `BarElement` connects two nodes along the structural X axis and behaves as a linear elastic spring with stiffness `k = EA/L`. Unlike the generic Version 1 `Element` (topology only), a `BarElement` performs its own local engineering calculations — it knows its `length`, `stiffness_matrix`, and how to compute `strain`, `stress`, and `axial_force` from nodal displacements. Y and Z coordinates are not used in Version 3; nodes should be laid out along X.

### Cross-section

`CrossSection` holds the one geometric property a 1D bar needs: `area`. Different elements may use different cross-sections and materials in the same mesh — see `examples/two_element_bar_analysis.py`.

### Static linear analysis and results

`StaticLinearAnalysis` wraps a `Mesh` of `BarElement` instances and orchestrates the Version 2 building blocks (DOF mapping, assembly, the linear system, and the solver) without duplicating any of their logic. `solve()` returns an `AnalysisResult`, a read-only view providing:

```text
result.displacement(node_id)      # u, in meters
result.reaction(node_id)          # R = [K]{u} - {F}, in newtons
result.element_strain(element_id) # epsilon = (u2 - u1) / L
result.element_stress(element_id) # sigma = E * epsilon
result.element_axial_force(element_id)  # N = sigma * A
```

Positive strain, stress, and axial force mean **tension**; negative means **compression**.

### The governing equations

```text
[K]{u} = {F}          Global equilibrium: stiffness times displacement equals applied force
epsilon = (u2-u1)/L    Axial strain from nodal displacements
sigma = E * epsilon    Hooke's law for a linear elastic material
N = sigma * A          Internal axial force from stress and area
u = F * L / (E * A)    Analytical displacement of a fixed-free bar under a tip load
```

### Reaction forces and equilibrium

Reactions are computed as `R = [K]{u} - {F}` over the full (unreduced) system. For a fixed-free bar under a tip load `F`, this gives a reaction of `-F` at the fixed support — so `F + R ≈ 0`, the global equilibrium check used throughout the test suite and both example scripts.

### Singular systems

An insufficiently constrained structure (for example, a bar chain with no fixed support, or a node left disconnected from the rest of the mesh) produces a singular reduced stiffness matrix. `StaticLinearAnalysis.solve()` detects this and raises `SingularSystemError` with a clear engineering explanation, rather than surfacing a raw NumPy linear-algebra error. Calling `solve()` with no boundary conditions at all raises the more specific `InsufficientConstraintsError`.

### Engineering validation

Every major numerical feature is checked against an independently derived expected result in `tests/validation/`:

- **Single bar** — displacement vs. `u = FL/(EA)`, reaction vs. equilibrium, strain/stress/axial force, and the strain-energy identity `U = 1/2 u^T K u = W = 1/2 u^T F` (valid here because the only support has zero prescribed displacement)
- **Two-element bar** — a series chain where statics requires both elements to carry the same axial force as the applied end load
- **Tension / compression** — sign consistency of strain, stress, and axial force under outward vs. inward loads
- **Mixed materials** — a steel + aluminum chain, confirming each element uses its own `E` and `A` while still sharing a common axial force

### Limitations

Version 3 does not include beam elements, 2D or 3D elements, nonlinear or dynamic analysis, thermal loads, automatic mesh generation, or visualization. See the [Roadmap](#roadmap).

## Version 4

Version 4 extends the toolkit's structural analysis capability from 1D bars to **2D pin-jointed trusses**: straight two-node members that carry only axial force (no bending, shear, or torsion), connected at joints that transmit force but not moment.

**Engineering assumptions:** linear elastic material behavior, small deformation, static loading, pin-jointed connections (axial force only, no bending), constant cross-sectional area per member, and the same consistent SI unit system as Version 3.

### 2D degrees of freedom

Every node now has two translational DOFs, `ux` and `uy`. `DOFMap` already supported an arbitrary number of DOFs per node since Version 2 (`dofs_per_node`); Version 4 simply uses `dofs_per_node=2` and activates `TranslationDOF.Y` alongside `TranslationDOF.X`.

### Geometry and direction cosines

For a truss element from `(x1, y1)` to `(x2, y2)`:

```text
L = sqrt((x2-x1)^2 + (y2-y1)^2)
c = (x2-x1) / L   (cosine of the element's orientation angle)
s = (y2-y1) / L   (sine of the element's orientation angle)
```

### 2D truss stiffness matrix

The element's local axial stiffness `k = EA/L` is transformed into global X/Y coordinates using its direction cosines. For nodal DOFs ordered `[ux1, uy1, ux2, uy2]`:

```text
EA/L *
[ c²    cs   -c²   -cs ]
[ cs    s²   -cs   -s² ]
[-c²   -cs    c²    cs ]
[-cs   -s²    cs    s² ]
```

A horizontal element (`c=1, s=0`) reduces this exactly to the Version 3 bar stiffness matrix in its `ux1/ux2` sub-block, with zero coupling to `uy` — verified directly in `tests/test_stiffness.py`.

### Reusable numerical infrastructure

Rather than duplicating assembly and solving for a second element type, Version 4 generalized the shared machinery: every element (bar or truss) exposes a `dof_keys()` method returning the `(node_id, dof)` pair for each row/column of its `stiffness_matrix`, plus `dofs_per_node` and `strain_from_dofs`/`stress_from_dofs`/`axial_force_from_dofs`. This is the `StructuralElement` protocol (`femtoolkit.analysis.element`). `assemble_global_stiffness`, `LinearSystem`, `build_force_vector`, and `solve` were already DOF-count-agnostic or became so with this change, so `StaticLinearAnalysis` and `AnalysisResult` work identically for `BarElement` and `TrussElement2D` without referencing either class by name.

### Strain, stress, and axial force

Global nodal displacements are projected onto the element's local axis using its direction cosines before computing strain:

```text
u1' = c*ux1 + s*uy1
u2' = c*ux2 + s*uy2
epsilon = (u2' - u1') / L
sigma = E * epsilon
N = sigma * A
```

Positive strain, stress, and axial force mean **tension**; negative means **compression** — the same convention as Version 3.

### 2D loads, boundary conditions, and results

`NodalLoad` and `BoundaryCondition` already accepted a `dof` argument since Version 2; for 2D work, pass `TranslationDOF.X` or `TranslationDOF.Y` instead of a bare `0`/`1` for clarity. `AnalysisResult.displacement()` and `.reaction()` gained an optional `dof` parameter (defaulting to `X`, preserving the Version 3 single-argument call), and `node_displacement()`/`node_reaction()` return `(x, y)` tuples directly.

### Structural instability

An insufficiently constrained 2D structure — for example, a single truss member pinned at one end with the other end completely free — has zero stiffness against motion perpendicular to its own axis, exactly like a real pin-jointed member. `StaticLinearAnalysis.solve()` detects the resulting singular matrix and raises `SingularSystemError`, reusing the same exception introduced in Version 3 rather than adding a redundant one.

### Engineering validation

Version 4 adds four validation cases in `tests/validation/`:

- **Horizontal truss** — a purely axial 2D truss member must reproduce the Version 3 analytical bar solution `u = FL/(EA)` exactly; an important backward-compatibility check for the shared infrastructure
- **Triangular truss** — a three-member "tent" truss, statically determinate, solved independently by the method of joints (member forces, reactions, and apex displacement cross-checked via the unit-load/virtual-work method)
- **Symmetric truss** — a consistency check: symmetric geometry, supports, and loading must produce mirror-symmetric displacements, reactions, and member forces
- **Multi-material truss** — a steel + aluminum truss confirming each element uses its own `E` and `A` while sharing a common axial force where statics requires it

### Limitations

Version 4 does not include beam, frame, plate, shell, 2D continuum, or 3D elements, nonlinear or dynamic analysis, thermal loads, automatic mesh generation, or visualization. See the [Roadmap](#roadmap).

## Version 5

Version 5 extends the toolkit from axial-only truss members to **2D Euler-Bernoulli beam and frame elements**: straight two-node members that resist axial force, shear force, *and* bending moment, connected at joints that transmit moment as well as force.

**Engineering assumptions:** linear elastic material behavior, small deformation, small strain, static loading, Euler-Bernoulli beam theory (plane sections remain plane and perpendicular to the neutral axis, so shear deformation is neglected), plane (2D) frame behavior, a constant prismatic cross-section per element, and the same consistent SI unit system as every prior version (rotation and curvature use radians, never degrees).

### Euler-Bernoulli beam theory

A truss element's stiffness is governed entirely by `EA`, its axial rigidity. A frame element adds `EI`, its **flexural rigidity** (`E` = Young's modulus, `I` = second moment of area), which governs how much the member resists bending. The Euler-Bernoulli assumption -- *plane sections remain plane and perpendicular to the neutral axis after deformation* -- is what lets bending stiffness be derived purely from `EI` and length, without a separate shear-stiffness term (that refinement, Timoshenko beam theory, is out of scope for this version).

### Cross-section

`CrossSection` gained two optional properties on top of `area`:

```text
second_moment_of_area          # I, in m^4 -- required by FrameElement2D (bending stiffness)
extreme_fiber_distance         # c, in m   -- optional, only used by bending-stress post-processing
```

`BarElement` and `TrussElement2D` still only need `area`; a `CrossSection` without `second_moment_of_area` set is rejected by `FrameElement2D` at construction time.

### 2D frame degrees of freedom

Every node now has three DOFs when used with a frame element: `ux`, `uy`, and `rz` (rotation about the out-of-plane Z axis, counter-clockwise positive, in radians). `DOFMap` already supported an arbitrary number of DOFs per node since Version 2; Version 5 simply uses `dofs_per_node=3` and activates a new `RotationDOF.RZ` (numeric value 2) alongside `TranslationDOF.X`/`Y`. A frame element therefore has 6 DOFs total: `[ux1, uy1, rz1, ux2, uy2, rz2]`.

### Local frame stiffness matrix

In local coordinates `[u1, v1, theta1, u2, v2, theta2]` (`u` = axial, `v` = transverse, `theta` = rotation), the axial and bending terms are uncoupled -- a frame element is a bar element and a beam element sharing the same two nodes:

```text
[ EA/L        0             0        -EA/L        0             0      ]
[ 0       12EI/L³       6EI/L²        0      -12EI/L³       6EI/L²    ]
[ 0        6EI/L²       4EI/L         0       -6EI/L²       2EI/L     ]
[-EA/L       0             0         EA/L         0             0      ]
[ 0      -12EI/L³      -6EI/L²        0       12EI/L³      -6EI/L²    ]
[ 0        6EI/L²       2EI/L         0       -6EI/L²       4EI/L     ]
```

`12EI/L³` relates transverse displacement to shear force, `6EI/L²` couples transverse displacement and rotation, and `4EI/L`/`2EI/L` are the direct and cross rotational stiffness terms. Setting `I` aside entirely (e.g. via `axial_indices` slicing) reduces the axial rows/columns to exactly the Version 2 bar stiffness matrix -- verified directly in `tests/test_stiffness.py`.

### Coordinate transformation

The local stiffness matrix is transformed into global `[ux1, uy1, rz1, ux2, uy2, rz2]` coordinates with a 6x6 transformation matrix built from the element's direction cosines `c = (x2-x1)/L`, `s = (y2-y1)/L` (identical to the Version 4 truss orientation):

```text
[ c   s   0   0   0   0 ]
[-s   c   0   0   0   0 ]
[ 0   0   1   0   0   0 ]
[ 0   0   0   c   s   0 ]
[ 0   0   0  -s   c   0 ]
[ 0   0   0   0   0   1 ]
```

Each node contributes an independent 2x2 in-plane rotation block for `ux`/`uy` plus an identity entry for `rz` (a planar rotation is the same in local and global coordinates). The global stiffness matrix is `Kg = Tᵀ · Kl · T`, verified for horizontal, vertical, 45-degree, and arbitrary orientations in `tests/test_stiffness.py` and `tests/test_transformation.py`.

### Element end forces and sign convention

The local end-force vector is recovered directly from the local stiffness matrix and the local displacement vector:

```text
f_local = Kl @ u_local             # u_local = T @ u_global
[N1, V1, M1, N2, V2, M2] = f_local
```

- **Axial force `N`** — positive is tension (matches `BarElement`/`TrussElement2D`).
- **Shear force `V`** and **bending moment `M`** — the standard finite-element end-force convention: the forces each node must apply to the element to hold its deformed shape. By construction these satisfy element equilibrium (`N1 = -N2`, `V1 = -V2`, and a moment balance about either end) -- verified in `tests/test_frame_element.py`.
- **Rotation** — always radians internally; no unit-conversion framework was introduced (a `math.degrees()` call is enough for display, see the example scripts).

`AnalysisResult.element_end_forces(element_id)` returns a `FrameElementForces(node_1, node_2)` with the axial force, shear force, and bending moment at each end; `element_shear_force(element_id, end="node_1")` and `element_bending_moment(element_id, end="node_1")` are single-value convenience wrappers over it.

### Stress utilities

```text
sigma_axial = N / A                                     # element_stress(element_id)
sigma_bending = M * extreme_fiber_distance / I           # element_bending_stress(element_id, end=...)
```

`element_bending_stress` requires `extreme_fiber_distance` to be set on the element's `CrossSection`; Version 5 does not build a full stress field or a combined `sigma_total = N/A ± Mc/I` utility, since the two separate utilities above are sufficient for extreme-fiber checks.

### Reusable numerical infrastructure

No frame-specific changes were needed in assembly, `LinearSystem`, or `solve()` -- they were already DOF-count-agnostic since Version 4. `FrameElement2D` satisfies the same `StructuralElement` protocol as `BarElement` and `TrussElement2D` (plus a new `FrameStructuralElement` protocol adding `end_forces_from_dofs`), so `StaticLinearAnalysis` assembles and solves a mesh of frame elements through the exact same code path, without a single `if isinstance(element, FrameElement2D)` branch anywhere in the solver. `node_displacement()`/`node_reaction()` were generalized to return one component per active DOF (`dof_map.dofs_per_node`), so they still return `(ux, uy)` for a truss analysis and now return `(ux, uy, rz)` for a frame analysis.

### Structural instability

A frame with insufficient boundary conditions (for example, a single member with only its axial DOF constrained, free to translate transversely and rotate as a rigid body) produces a singular reduced stiffness matrix. `StaticLinearAnalysis.solve()` detects this and raises `SingularSystemError` -- the same exception introduced in Version 3 and reused for trusses in Version 4, since the underlying failure (a singular linear system) is identical regardless of element type.

### Engineering validation

Six validation cases in `tests/validation/`, each checked against a classical closed-form solution or an independently derived invariant:

- **Bar regression** — a horizontal frame element under a purely axial load reproduces the Version 3 bar solution `u = FL/(EA)` exactly, and induces zero shear/moment, proving the frame element's axial and bending behavior are correctly uncoupled
- **Cantilever, tip load** — `delta = PL³/(3EI)`, `theta = PL²/(2EI)`, fixed-end moment `M = PL`, fixed-end shear `V = P`
- **Cantilever, end moment** — `theta = ML/(EI)`, `delta = ML²/(2EI)`, and the reaction moment exactly balances the applied moment
- **Simply supported beam** — a two-element beam with a midspan point load: `R = P/2` at each support, `M_max = PL/4` at midspan, `delta_mid = PL³/(48EI)`, zero moment at both simple supports
- **Two-element beam** — the same cantilever as above, re-meshed into two elements, reproduces the identical tip deflection/rotation and fixed-end forces, validating assembly and DOF continuity between elements
- **Portal frame** — two columns and a beam under an asymmetric lateral load (statically indeterminate, so validated the way Version 4's symmetric truss is: global force/moment equilibrium and physically sensible sway direction, not a closed-form target)

### Limitations

Version 5 does not include 3D beams, Timoshenko beams, distributed beam loads, plate, shell, 2D continuum, or 3D elements, nonlinear or dynamic analysis, buckling, modal analysis, thermal analysis, contact, automatic mesh generation, or visualization. See the [Roadmap](#roadmap).

## Version 6

Every element through Version 5 represents a structural *member*: a line between two nodes. Version 6 introduces the toolkit's first **2D continuum element** -- the 3-node constant strain triangle (CST) -- which represents a finite *area* of material. Instead of `Node --- Node`, a structure is now (optionally) modeled as a mesh of triangles occupying a 2D region, with two translational DOFs per node (`ux`, `uy`) and no rotational DOF, since a continuum point has no orientation to rotate.

**Engineering assumptions:** linear elastic, isotropic material; small deformation; small strain; static loading; a constant strain field per element (see below); a constant thickness per element; and the same consistent SI unit system as every prior version. **Sign convention:** engineering shear strain, `gamma_xy = du/dy + dv/dx` (not tensorial shear strain `epsilon_xy = gamma_xy / 2`) -- every formula below and throughout `femtoolkit.continuum` uses this convention consistently.

### Continuum mechanics vs. structural elements

A truss or frame element's kinematics reduce to a single number along its axis (or three, for a frame's DOFs) -- there is no "field" to speak of. A continuum element approximates a genuinely 2D displacement field, `u(x, y)` and `v(x, y)`, from nodal values via shape functions:

```text
u(x, y) = N1(x,y)*u1 + N2(x,y)*u2 + N3(x,y)*u3
v(x, y) = N1(x,y)*v1 + N2(x,y)*v2 + N3(x,y)*v3
```

with `N1 + N2 + N3 = 1` everywhere (partition of unity) and `Ni(node_j) = 1` if `i == j` else `0` (the nodal/Kronecker-delta property) -- both verified directly in `tests/test_shape_functions.py`.

### Plane stress and plane strain

A 2D model must reduce the full 3D stress state with one of two assumptions:

- **Plane stress** (`sigma_z = 0`): thin, flat bodies loaded in their own plane (e.g. a thin plate).
- **Plane strain** (`epsilon_z = 0`): bodies long in the out-of-plane direction and restrained from extending along it (e.g. a long dam cross-section).

```text
D_plane_stress =
E/(1-v^2) *
[ 1    v       0    ]
[ v    1       0    ]
[ 0    0   (1-v)/2  ]

D_plane_strain =
E/((1+v)(1-2v)) *
[ 1-v    v        0    ]
[ v     1-v       0    ]
[ 0      0    (1-2v)/2 ]
```

Both give the same `sigma = D @ epsilon` relation, but with meaningfully different values -- `LinearElastic2D(youngs_modulus, poisson_ratio, formulation="plane_stress" | "plane_strain")` selects which. This is a small, dedicated constitutive model, not a new responsibility bolted onto the structural-element `Material`.

### The constant strain triangle (CST)

For a 3-node triangle, each shape function is linear in `x` and `y`, so its gradient -- and therefore the strain field `epsilon = B @ d` -- is **constant** over the entire element for any fixed nodal displacement vector. This is the origin of the element's name, and its main limitation: a single CST element cannot represent a strain gradient (e.g. bending), only a uniform strain state.

```text
epsilon =
[ epsilon_x ]     B =
[ epsilon_y ]     1/(2A) *
[ gamma_xy  ]     [ b1   0    b2   0    b3   0  ]
                  [ 0    c1   0    c2   0    c3 ]
                  [ c1   b1   c2   b2   c3   b3 ]

b1 = y2-y3, b2 = y3-y1, b3 = y1-y2
c1 = x3-x2, c2 = x1-x3, c3 = x2-x1
```

`CSTElement2D` connects three nodes (DOFs ordered `[ux1,uy1,ux2,uy2,ux3,uy3]`), a `LinearElastic2D` material, and a `thickness`. **Orientation policy:** nodes may be listed clockwise or counter-clockwise -- both are accepted and give identical strain, stress, and stiffness, because the *signed* area is used consistently inside `B` (matching the literal shape-function-derivative formulas) while the *absolute* area is used wherever a physical area is needed. A triangle with (near-)zero area -- collinear, nearly collinear, or duplicate-coordinate nodes -- raises `DegenerateElementError`.

### Element stiffness matrix

```text
Ke = t * A * B^T * D * B
```

where `t` is thickness, `A` is the (always positive) physical area, `B` is the strain-displacement matrix, and `D` is the constitutive matrix. Unlike a truss or frame element, no coordinate transformation is needed -- `ux`/`uy` are already expressed in global coordinates for a continuum element, so this formula produces the element's *global* stiffness matrix directly. `Ke` scales linearly with both thickness and Young's modulus, and is positive semidefinite before boundary conditions are applied (verified in `tests/test_stiffness.py`).

### Strain, stress, von Mises, and principal stress

```text
epsilon = B @ d              (strain from nodal displacements)
sigma = D @ epsilon           (stress from strain, Hooke's law)

sigma_vm (plane stress) = sqrt(sigma_x^2 - sigma_x*sigma_y + sigma_y^2 + 3*tau_xy^2)

sigma_1, sigma_2 = (sigma_x+sigma_y)/2 +/- sqrt(((sigma_x-sigma_y)/2)^2 + tau_xy^2)
```

Under plane strain, the out-of-plane stress `sigma_z = v*(sigma_x+sigma_y)` is generally *nonzero* (unlike plane stress, where it is zero by definition); reusing the plane-stress von Mises formula for a plane-strain result would understate the true equivalent stress. `femtoolkit.continuum.stress.von_mises_3d` is the minimal, reusable core both `von_mises_plane_stress` and `von_mises_plane_strain` reduce to once the correct `sigma_z` is substituted.

### Element interface split

`StaticLinearAnalysis` only ever needs an element's ID, `dofs_per_node`, `dof_keys()`, and `stiffness_matrix` to assemble and solve -- captured by the new minimal `AssemblableElement` protocol. `StructuralElement` (scalar axial strain/stress/force -- bar, truss, frame) and `ContinuumElement` (vector strain/stress, von Mises, principal stress -- CST) are two independent extensions of it, not one a subtype of the other: a continuum element has no meaningful "axial force," and a structural member has no meaningful "principal stress." `StaticLinearAnalysis` itself required **zero** continuum-specific code -- it already worked against the minimal protocol, so a mesh of `CSTElement2D` instances assembles and solves through the exact same path as a mesh of `TrussElement2D` instances.

`AnalysisResult.element_strain`/`element_stress` are reused across both kinds of element (returning a scalar for a structural member, a 3-element `[x, y, xy]` array for a continuum element) rather than introducing new method names, since both ultimately answer "what deformation/stress is this element experiencing." `element_von_mises` and `element_principal_stresses` are new, and (like `element_axial_force` for structural members) raise `InvalidElementError` if called on the wrong kind of element.

### Engineering validation

Five validation cases in `tests/validation/`:

- **Patch test** (`test_cst_patch.py`) — for any linear displacement field `u=ax+by, v=cx+dy` prescribed at every node (so the solver has nothing left to solve for), the recovered strain must be the exact `[a, d, b+c]`, on three differently shaped triangles
- **Plane stress** (`test_plane_stress.py`) — the constitutive matrix independently hand-derived for a steel material, plus a boundary-artifact-free uniaxial stress recovery: a displacement field derived analytically from a target uniaxial stress state must reproduce that exact stress (`sigma_x = applied`, `sigma_y ≈ 0`, `tau_xy ≈ 0`)
- **Plane strain** (`test_plane_strain.py`) — the constitutive matrix independently hand-derived, and shown to meaningfully differ from (and be stiffer than) the plane-stress matrix for the same material
- **Single triangle** (`test_single_triangle.py`) — the full 6x6 stiffness matrix for `E=1, v=0.3, t=1` on nodes `(0,0),(1,0),(0,1)`, computed by an independent from-scratch formula implementation (not the library code under test) and compared directly
- **Two-triangle plate** (`test_two_triangle_plate.py`) — a rectangle split into two CST elements under a statically-equivalent uniform edge traction; validates assembly, displacement continuity at the shared edge, reaction equilibrium, and that both elements recover the identical uniform stress state

### Limitations

Version 6 does not include quadrilateral elements, 6-node (quadratic) triangles, higher-order elements, 3D solid elements, nonlinear material models, plasticity, hyperelasticity, large deformation, dynamic analysis, modal analysis, buckling, contact, fracture, adaptive meshing, automatic mesh generation, mesh refinement, or visualization. See the [Roadmap](#roadmap).

## Version 7

Version 6's CST element has a closed-form stiffness matrix only because its shape functions are linear -- its `B` matrix is constant, so `Ke = t*A*Bᵀ*D*B` needs no integration. Version 7 introduces the 4-node bilinear quadrilateral (**Q4**), whose shape functions are *bilinear*: `B` varies from point to point within the element, and the stiffness integral has no closed form. This is the central new mathematical machinery of Version 7: **natural coordinates**, the **isoparametric Jacobian**, and **numerical (Gauss) integration** -- none of which the CST element needed.

**Engineering assumptions:** linear elastic, isotropic material; small deformation; small strain; static loading; a constant thickness per element; the same consistent SI unit system as every prior version; engineering shear strain (`gamma_xy = du/dy + dv/dx`), matching Version 6.

### Natural coordinates and bilinear shape functions

A Q4 element's shape functions are defined on a natural-coordinate square, `xi, eta in [-1, 1]`, not directly on the element's physical coordinates:

```text
Node 1: (xi,eta) = (-1,-1)      Node 4 ------- Node 3
Node 2: (xi,eta) = ( 1,-1)        |               |
Node 3: (xi,eta) = ( 1, 1)        |               |
Node 4: (xi,eta) = (-1, 1)      Node 1 ------- Node 2

N1 = (1-xi)(1-eta)/4
N2 = (1+xi)(1-eta)/4
N3 = (1+xi)(1+eta)/4
N4 = (1-xi)(1+eta)/4
```

Each `Ni` is *bilinear* -- linear in `xi` and `eta` separately, but containing an `xi*eta` cross term -- so, unlike CST, its gradient is **not** constant: a Q4 element's strain varies within the element. `N1+N2+N3+N4 = 1` everywhere (partition of unity) and `Ni(node_j) = 1` if `i==j` else `0` (the nodal property), both verified in `tests/test_shape_functions.py`.

### Isoparametric mapping and the Jacobian

An isoparametric element uses the *same* shape functions for both geometry and displacement: `x(xi,eta) = sum(Ni*xi_coord)`, `y(xi,eta) = sum(Ni*yi_coord)`. The **Jacobian matrix** relates natural-coordinate derivatives (where the shape functions are simple) to physical-coordinate derivatives (where strain is defined):

```text
J =
[ dx/dxi   dy/dxi  ]
[ dx/deta  dy/deta ]

dNi/dx, dNi/dy  =  J^-1 @ [ dNi/dxi, dNi/deta ]
```

`det(J)` is the area-scaling factor between the natural-coordinate square and the physical element. **Node order is fixed counter-clockwise**: unlike CST (which accepts either winding order via a signed/absolute-area distinction), a Q4 element has no single "signed area" for a general quadrilateral to normalize clockwise input against -- clockwise or self-intersecting node order gives a negative `det(J)` and is rejected with `DegenerateElementError`.

### 2x2 Gauss integration

```text
Ke = integral( B^T D B t ) dA  =  integral( B^T D B t det(J) ) dxi deta
   ~= sum over 4 points of: weight * t * B(xi,eta)^T D B(xi,eta) det(J(xi,eta))
```

The four points, `(+/-1/sqrt(3), +/-1/sqrt(3))`, each with weight `1.0`, are the standard 2-point-per-direction Gauss-Legendre rule -- exact for the bilinear geometric mapping, and the standard choice for a 4-node quadrilateral (`femtoolkit.continuum.gauss`).

### Q4 element and stiffness matrix

`QuadElement2D` connects four nodes (DOFs ordered `[ux1,uy1,ux2,uy2,ux3,uy3,ux4,uy4]`, 8 total), a `LinearElastic2D` material, and a `thickness`, giving an 8x8 stiffness matrix (`femtoolkit.analysis.stiffness.quad_element_stiffness`) that scales linearly with thickness and Young's modulus, and is positive semidefinite with exactly 3 zero eigenvalues (2 translation + 1 rotation rigid-body mode) before boundary conditions.

### Strain, stress, and representative reporting

```text
epsilon = B(xi,eta) @ d        (varies with position, unlike CST)
sigma = D @ epsilon
```

Since a Q4 element's strain is generally different at every point, `element_strain`/`element_stress`/`element_von_mises`/`element_principal_stresses` report the value at the element's natural-coordinate center (`xi=eta=0`) as a single representative value -- the standard simplified reporting convention (reporting all four Gauss-point values separately is out of scope for this version).

### Zero wiring changes

`QuadElement2D` satisfies the exact same `AssemblableElement` and `ContinuumElement` protocols introduced in Version 6 -- `strain_from_dofs`/`stress_from_dofs` returning a 3-element array, plus `von_mises_from_dofs` and `principal_stresses_from_dofs`. `StaticLinearAnalysis`, `AnalysisResult`, assembly, and the solver required **zero** Version 7-specific code: a mesh of `QuadElement2D` instances is assembled and solved through the exact same path as a mesh of `CSTElement2D` instances, confirming the Version 6 protocol split was the right generalization rather than a CST-specific one.

### Engineering validation

Three validation cases in `tests/validation/`:

- **Patch test** (`test_quad_patch.py`) — a linear displacement field `u=ax+by, v=cx+dy` (a special case of the bilinear interpolation with a zero `xi*eta` coefficient) prescribed at every node must give the exact constant strain `[a, d, b+c]`, on three differently shaped quadrilaterals
- **Single element** (`test_single_quad.py`) — a unit-square Q4 element under uniaxial tension recovers the exact analytical stress state, and its 8x8 stiffness matrix is independently cross-checked against a from-scratch NumPy re-derivation of the isoparametric formula (not calling `femtoolkit.continuum` or `femtoolkit.analysis.stiffness`)
- **Two-element plate** (`test_two_quad_plate.py`) — a rectangle split into two Q4 elements under a statically-equivalent uniform edge traction; validates assembly, displacement continuity at the shared edge, reaction equilibrium, and that both elements recover the identical uniform stress state

### Limitations

Version 7 does not include higher-order (8-node or 9-node) quadrilaterals, 3D solid elements (tetrahedral, hexahedral), nonlinear material models, plasticity, dynamic analysis, contact, fracture, or visualization. See the [Roadmap](#roadmap).

## Version 8

Every mesh through Version 7 was built by hand: one `Node` and one `CSTElement2D`/`QuadElement2D` call at a time. That does not scale past a handful of elements. Version 8 adds **structured 2D mesh generation** -- turning a rectangular domain and a subdivision count into a fully connected, correctly oriented mesh automatically -- plus the supporting infrastructure a generated mesh needs: whole-mesh validation, shape-quality metrics, and a JSON export/import foundation.

```text
Geometry (width, height, nx, ny)
    v
Mesh Generator
    v
Nodes + Elements
    v
Existing FEM solver (unchanged)
```

### Why meshing matters

A finite element mesh divides a continuous physical domain into a finite number of simply shaped pieces (elements) so the continuum equations can be approximated by a solvable linear system. **Mesh density** (how many elements) trades accuracy for computational cost: more elements generally resolve the true stress/strain field more closely, at the cost of a larger system to assemble and solve. **Element quality** -- how close each element's shape is to "ideal" -- matters independently of density: a mesh with enough elements but poorly shaped ones can still give inaccurate or numerically unstable results (see below).

### Structured mesh

Version 8 generates a **structured** mesh: a regular grid of identically-arranged cells, each becoming one Q4 element or two CST elements. This is deliberately simple -- no CAD geometry import, no unstructured (e.g. Delaunay) triangulation, no adaptive refinement -- and is the natural scope for a rectangular domain. Its advantage is predictability: node and element numbering are fully deterministic (see below), which makes generated meshes easy to reason about and to target with boundary conditions (e.g. "every node with `x == 0`"). Its limitation is equally direct: it only covers domains that decompose into a regular grid -- an arbitrary or curved boundary needs unstructured meshing, out of scope for this version.

### Node and element numbering

**Node numbering** is row-major, bottom-to-top, left-to-right, 1-indexed: `node_id(row, col) = row * (nx+1) + col + 1`. For `nx=2, ny=1`:

```text
Node4 -------- Node5 -------- Node6      row 1 (top)
  |              |              |
  |              |              |
Node1 -------- Node2 -------- Node3      row 0 (bottom)
```

**Element numbering** is left-to-right within a row, then bottom-to-top across rows, 1-indexed: `element_id(row, col) = row*nx + col + 1` for Q4 (one element per cell), or `2*(row*nx + col) + 1`/`+2` for CST (two triangles per cell). Both are fully deterministic and reproducible -- generating the same mesh twice gives identical IDs and coordinates every time.

### Connectivity and orientation

Every generated element lists its nodes **counter-clockwise**, starting from its cell's bottom-left corner -- `QuadElement2D` requires this outright (see Version 7), and `CSTElement2D` accepts either winding but the generator always emits counter-clockwise for consistency. A **negative Jacobian determinant** (Q4) or **negative signed area** (CST) means the element's node order is inverted -- physically, the element would have negative area, which makes its stiffness matrix meaningless. The generator's cell-corner-based connectivity makes an inverted element structurally impossible to produce; both generators additionally call `validate_mesh()` on their own output as a final, explicit self-check before returning.

### Triangular mesh: diagonal direction

Splitting a cell into two CST elements requires choosing a diagonal:

```text
diagonal="forward" (bottom-left to top-right)     diagonal="backward" (bottom-right to top-left)

top_left ------ top_right                          top_left ------ top_right
   | \               |                                  |             /  |
   |   \             |                                  |           /    |
   |     \           |                                  |         /      |
bottom_left ---- bottom_right                       bottom_left ---- bottom_right
```

The diagonal is fixed per mesh (uniform across every cell) -- per-cell adaptive diagonal selection is out of scope for this version.

### Mesh validation

Most validity checks happen **at construction time**, before an invalid object can exist: `Node` rejects non-finite coordinates, `Mesh.add_node`/`add_element` reject duplicate IDs and dangling node references, and `CSTElement2D`/`QuadElement2D` reject degenerate or inverted geometry. `validate_mesh()` (`femtoolkit.mesh.validation`) covers the one thing fail-fast construction cannot catch on its own: **duplicate node coordinates** -- two distinct node IDs placed at the same physical location, each individually valid, but geometrically wrong together -- raising the new `DuplicateNodeCoordinatesError`.

### Mesh quality metrics

For each continuum element, `Mesh.element_quality(element_id)` computes:

```text
aspect_ratio = max_edge_length / min_edge_length          (>= 1.0; 1.0 = square/equilateral)
quality = min_edge_length / max_edge_length                (in (0,1]; 1.0 = best shape)
skewness = max(
    (theta_max - theta_ideal) / (180 - theta_ideal),
    (theta_ideal - theta_min) / theta_ideal,
)                                                            (equiangle skew, in [0,1]; 0 = ideal)
```

where `theta_ideal` is 90 degrees for a Q4 element or 60 degrees for a CST element. **Aspect ratio** approximates shape distortion via edge-length variation; for a rectangle it is exact, but for a general (non-rectangular) quadrilateral or non-equilateral triangle it does not fully capture skew independent of edge length -- a documented approximation, not a complete shape descriptor. **Why extremely elongated ("needle") elements are problematic**: their stiffness becomes very different in different directions, which can degrade solution accuracy and, in severe cases, harm the conditioning of the global stiffness matrix. `Mesh.quality_summary()` aggregates these across the whole mesh (node/element counts, area and edge-length ranges, min/max/average quality, and a count of elements with non-positive area -- expected to always be zero given fail-fast construction, retained as defense-in-depth).

### JSON mesh export/import

`export_mesh`/`import_mesh` (`femtoolkit.mesh.serialization`) write/read a lightweight, self-contained JSON format -- each element's own material and thickness are stored alongside it, so a round trip fully reconstructs the mesh. This is a foundation, not an industry format: no Abaqus INP, ANSYS, Gmsh, or VTK export (future-version scope), and only `CSTElement2D`/`QuadElement2D` elements are supported.

### Mesh refinement

Increasing `nx`/`ny` densifies a mesh over the same domain -- `coarse mesh -> finer mesh` through a larger subdivision count, nothing more. There is no adaptive or error-based refinement in this version: refinement is a modeling choice the caller makes up front, not something the mesh adjusts itself.

### Material assignment stays separate from geometry

The mesh generator never chooses or defaults a material -- `material` and `thickness` are required parameters the caller supplies, kept as data flowing *through* the generator rather than a decision the generator makes. This preserves the existing architecture's separation: `Geometry -> Mesh -> Elements -> Material assignment -> Analysis`, with material assignment happening exactly where it already did in Versions 6-7 (at element construction), not moved into the generator.

### Reuses the existing solver, unchanged

A mesh from `create_quad_mesh`/`create_triangular_mesh` is a plain `Mesh` of ordinary `QuadElement2D`/`CSTElement2D` elements -- `StaticLinearAnalysis`, `AnalysisResult`, assembly, and the solver required **zero** Version 8-specific code (see `examples/mesh_and_analysis.py`).

### Limitations

Version 8 does not include unstructured meshing, Delaunay triangulation, CAD geometry import, Gmsh integration, adaptive or error-based mesh refinement, 3D mesh generation, tetrahedral/hexahedral meshing, Abaqus/ANSYS/VTK export, or visualization. See the [Roadmap](#roadmap).

## Project Structure

```text
finite-element-toolkit/
├── src/femtoolkit/
│   ├── materials/          # Material, LinearElastic2D (2D constitutive model)
│   ├── mesh/               # Node, Element, BarElement, TrussElement2D,
│   │                       # FrameElement2D, CSTElement2D, QuadElement2D, Mesh,
│   │                       # generator.py (structured mesh generation),
│   │                       # quality.py (shape-quality metrics),
│   │                       # validation.py (whole-mesh checks),
│   │                       # serialization.py (JSON export/import)
│   ├── sections/           # CrossSection
│   ├── continuum/          # Reusable 2D continuum math: geometry, shape
│   │                       # functions (triangle + Q4), isoparametric
│   │                       # Jacobian, 2x2 Gauss quadrature,
│   │                       # strain-displacement (B) matrix, plane
│   │                       # stress/strain constitutive (D) matrices,
│   │                       # stress/von Mises/principal stress recovery
│   ├── analysis/           # DOFs (incl. RotationDOF), boundary conditions,
│   │                       # loads, stiffness matrix, transformation
│   │                       # matrix, assembly, linear system, the
│   │                       # AssemblableElement/StructuralElement/
│   │                       # FrameStructuralElement/ContinuumElement
│   │                       # protocols, and the StaticLinearAnalysis workflow
│   ├── results/            # AnalysisResult, FrameEndForces, FrameElementForces
│   ├── units/               # SI unit constants
│   ├── exceptions/          # Custom exception types (incl. DegenerateElementError,
│   │                       # DuplicateNodeCoordinatesError)
│   ├── config.py             # Package metadata and defaults
│   └── logging_config.py     # Package logger configuration
├── examples/                  # Runnable example scripts
└── tests/
    ├── ...                      # Unit tests, one file per module
    └── validation/                # Engineering validation against
                                    # analytical solutions
```

## Testing

Run the test suite with pytest:

```bash
pytest
```

## Code Quality

Check code style and lint rules with Ruff:

```bash
ruff check .
```

## Example

Run the example scripts:

```bash
python examples/create_basic_model.py        # Version 1: build and print a minimal model
python examples/basic_bar_analysis.py         # Version 2: solve a minimal axial bar problem
python examples/basic_1d_bar_analysis.py      # Version 3: full single-bar analysis workflow
python examples/two_element_bar_analysis.py   # Version 3: two-element, mixed-material bar chain
python examples/basic_2d_truss_analysis.py    # Version 4: triangular 2D truss analysis
python examples/multi_material_truss.py       # Version 4: multi-material, multi-orientation truss
python examples/cantilever_beam.py            # Version 5: cantilever beam under a tip load
python examples/cantilever_end_moment.py      # Version 5: cantilever beam under a pure end moment
python examples/portal_frame.py               # Version 5: two-column, one-beam portal frame
python examples/single_cst_element.py         # Version 6: single CST triangle under uniaxial tension
python examples/two_triangle_plate.py         # Version 6: two-triangle plate, assembly + equilibrium
python examples/cst_patch_test.py             # Version 6: constant-strain patch test
python examples/single_quad_element.py        # Version 7: single Q4 element under uniaxial tension
python examples/two_element_plate.py          # Version 7: two-Q4-element plate, assembly + equilibrium
python examples/quad_patch_test.py            # Version 7: Q4 constant-strain patch test
python examples/generate_quad_mesh.py         # Version 8: automatic Q4 mesh generation + quality summary
python examples/generate_tri_mesh.py          # Version 8: automatic CST mesh generation + quality summary
python examples/mesh_and_analysis.py          # Version 8: generated mesh solved by the existing solver
```

## Roadmap

Future versions will build a more complete FEA solver on top of this foundation. None of the following is implemented yet:

- **Version 9** — Advanced mesh generation and boundary/load assignment: geometry entities, named boundary regions, boundary-condition assignment by geometry, distributed/surface loads, mesh refinement regions
- **Later** — Unstructured/CAD-driven meshing, 3D elements, higher-order continuum elements, nonlinear analysis, GUI, visualization, reporting, and more

## License

Released under the [MIT License](LICENSE).
