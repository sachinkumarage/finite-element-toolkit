# Finite Element Toolkit

An open-source Python toolkit for developing finite element analysis (FEA) capabilities, built incrementally as a series of versioned milestones.

**This is the Version 3 release.** Version 1 established the project's architecture and core domain model. Version 2 added the basic mathematical foundation for FEA: degrees of freedom, boundary conditions, nodal loads, the 1D bar element stiffness matrix, global stiffness matrix assembly, and a basic linear solver for `[K]{u} = {F}`. Version 3 turns that foundation into a useful, validated **1D structural analysis** capability: a dedicated bar element with cross-sectional area, a `StaticLinearAnalysis` workflow, and results giving displacement, reaction, strain, stress, and axial force. It does **not** yet contain beam, truss, 2D, or 3D elements, nonlinear or dynamic analysis, or visualization.

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
- **Tests** — a pytest suite covering all three versions, including a dedicated engineering-validation suite (`tests/validation/`) checking displacement, reactions, strain, stress, axial force, tension/compression signs, mixed materials, global equilibrium, and the strain-energy identity against independently derived analytical results

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
global_k = assemble_global_stiffness(dof_map, [ElementStiffnessContribution((1, 2), local_k)])

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

## Project Structure

```text
finite-element-toolkit/
├── src/femtoolkit/
│   ├── materials/          # Material data model
│   ├── mesh/               # Node, Element, BarElement, and Mesh
│   ├── sections/           # CrossSection
│   ├── analysis/           # DOFs, boundary conditions, loads, stiffness
│   │                       # matrix, assembly, linear system, and the
│   │                       # StaticLinearAnalysis workflow
│   ├── results/            # AnalysisResult
│   ├── units/               # SI unit constants
│   ├── exceptions/          # Custom exception types
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
```

## Roadmap

Future versions will build a more complete FEA solver on top of this foundation. None of the following is implemented yet:

- **Version 4** — Beam analysis (bending, rotational DOFs, moment, shear, deflection)
- **Version 5** — 2D finite elements
- **Version 6** — Advanced mesh and visualization
- **Later** — 3D elements, GUI, reporting, and more

## License

Released under the [MIT License](LICENSE).
