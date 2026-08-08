# Finite Element Toolkit

An open-source Python toolkit for developing finite element analysis (FEA) capabilities, built incrementally as a series of versioned milestones.

**This is the Version 2 release.** Version 1 established the project's architecture and core domain model — materials, nodes, elements, and a mesh container. Version 2 adds the basic mathematical foundation for FEA: degrees of freedom, boundary conditions, nodal loads, the 1D bar element stiffness matrix, global stiffness matrix assembly, and a basic linear solver for `[K]{u} = {F}`. It does **not** yet contain dedicated bar/truss/beam element abstractions, 2D or 3D elements, stress/strain recovery, or visualization.

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
- **Tests** — a pytest suite covering both the Version 1 domain model and the Version 2 mathematical foundation, including validation against the analytical axial bar solution

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

## Project Structure

```text
finite-element-toolkit/
├── src/femtoolkit/
│   ├── materials/          # Material data model
│   ├── mesh/               # Node, Element, and Mesh
│   ├── analysis/           # DOFs, boundary conditions, loads,
│   │                       # stiffness matrix, assembly, linear system
│   ├── units/               # SI unit constants
│   ├── exceptions/          # Custom exception types
│   ├── config.py             # Package metadata and defaults
│   └── logging_config.py     # Package logger configuration
├── examples/                  # Runnable example scripts
└── tests/                      # pytest test suite
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
python examples/create_basic_model.py     # Version 1: build and print a minimal model
python examples/basic_bar_analysis.py      # Version 2: solve a minimal axial bar problem
```

## Roadmap

Future versions will build a more complete FEA solver on top of this foundation. None of the following is implemented yet:

- **Version 3** — Dedicated 1D bar/truss element abstractions, cross-sectional properties, reaction forces, axial force/stress/strain results
- **Version 4** — Beam analysis
- **Version 5** — 2D finite elements
- **Version 6** — Advanced mesh and visualization
- **Later** — 3D elements, GUI, reporting, and more

## License

Released under the [MIT License](LICENSE).
